# Benchmarking Framework Design

## Overview

This document describes the design decisions and architecture of the UQLM benchmarking framework.

## Design Goals

1. **Separation of Concerns**: Running benchmarks and analyzing results are distinct workflows
2. **No Code Duplication**: Each method exists in exactly one place
3. **Explicit Parameters**: All configuration visible at call site (no hidden config objects)
4. **Caching & Persistence**: Avoid duplicate computations and enable crash recovery
5. **Extensibility**: Easy to add new benchmarks and analysis methods

## Architecture

### Component Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    User Interface Layer                      │
├──────────────────────────┬──────────────────────────────────┤
│   BenchmarkRunner        │      BenchmarkAnalyzer           │
│   - run_benchmark()      │      - compare_llms()            │
│   - get_run()            │      - compare_scorers()         │
│   - delete_run()         │      - aggregate_metrics()       │
│                          │      - plot_results()            │
│                          │      - export_report()           │
│                          │      - list_runs()               │
└──────────────────────────┴──────────────────────────────────┘
                          │
                          ▼
              ┌───────────────────────┐
              │  BenchmarkResultsDB   │
              │  (SQLite Backend)     │
              └───────────────────────┘
                          │
                          ▼
              ┌───────────────────────┐
              │   Pydantic Models     │
              │   - BenchmarkConfig   │
              │   - BenchmarkRun      │
              │   - RunMetadata       │
              │   - PromptResult      │
              └───────────────────────┘
```

### File Structure

```
uqlm/benchmarks/
├── __init__.py              # Public API exports
├── models.py                # Pydantic data models
├── runner.py                # BenchmarkRunner
├── analyzer.py              # BenchmarkAnalyzer
├── storage.py               # BenchmarkResultsDB (internal)
├── README.md                # User documentation
├── DESIGN.md                # This file
└── implementations/
    ├── __init__.py
    ├── base.py              # BaseBenchmark
    └── factscore.py         # FactScoreBenchmark
```

## Design Decisions

### 1. Separate Runner and Analyzer Classes

**Decision**: Use two independent classes instead of a unified client.

**Rationale**:
- Running and analyzing are distinct workflows with different contexts
- Users typically do one or the other, not both simultaneously
- No need for artificial coupling through a facade
- Cleaner mental model: "I need to run? Use BenchmarkRunner. I need to analyze? Use BenchmarkAnalyzer."

**Alternative Considered**: Unified `BenchmarkClient` with namespaces (`client.run.benchmark()`, `client.analyze.compare()`)
- Rejected because it added unnecessary indirection without clear benefit

### 2. Explicit Parameters Over Config Objects

**Decision**: All parameters explicit in `run_benchmark()` method signature.

**Rationale**:
- IDE-friendly: autocomplete shows all available options
- Discoverable: users see what's configurable
- Type-safe: parameters get validated at call site
- Config object created internally for caching/persistence

**Example**:
```python
# Good: Explicit parameters
results = await runner.run_benchmark(
    benchmark_name="factscore",
    llm_names=["gemini-2.5-flash"],
    scorer_names=["LUQ"],
    dataset_name="dskar/FActScore",
    sampling_temperature=0.4,
    num_responses=5,
    use_cache=True,
    save_results=True
)

# Avoided: Opaque config object
config = BenchmarkConfig(...)  # User must know how to construct this
results = await runner.run_benchmark(config)  # What parameters are available?
```

### 3. Cache Control at Method Level

**Decision**: `use_cache` and `save_results` as method parameters, not instance-level settings.

**Rationale**:
- Different runs may have different requirements
- Test runs typically want `use_cache=False, save_results=False`
- Production runs want `use_cache=True, save_results=True`
- More flexible than instance-level setting

**Example**:
```python
# Quick test - don't cache or save
test_results = await runner.run_benchmark(..., use_cache=False, save_results=False)

# Production run - use cache and save
prod_results = await runner.run_benchmark(..., use_cache=True, save_results=True)
```

### 4. SQLite Backend Only

**Decision**: Single SQLite backend, no pluggable database abstraction.

**Rationale**:
- Simple implementation
- No dependencies (SQLite included with Python)
- Excellent read/write performance for local workflows
- Good query capabilities for analysis
- Single-file portability

**Alternative Considered**: Abstract database interface supporting JSON, SQLite, PostgreSQL
- Rejected as YAGNI (You Aren't Gonna Need It)
- Can add later if needed

### 5. Pydantic Models

**Decision**: Use Pydantic for all data structures.

**Rationale**:
- Type safety and validation
- Easy serialization/deserialization
- Self-documenting schemas
- Clear contracts between components
- Validation happens automatically

**Models**:
- `BenchmarkConfig`: Configuration for a run
- `RunMetadata`: Metadata about a run (status, timestamps, etc.)
- `PromptResult`: Results for a single prompt/LLM combination
- `BenchmarkRun`: Complete run with metadata and results

### 6. Hash-Based Caching

**Decision**: Use deterministic hash of configuration for cache lookup.

**Implementation**:
```python
def compute_hash(self) -> str:
    hash_dict = {
        "benchmark_name": self.benchmark_name,
        "llm_names": sorted(self.llm_names),
        "scorer_names": sorted(self.scorer_names),
        "dataset_name": self.dataset_name,
        "dataset_version": self.dataset_version,
        "sampling_temperature": self.sampling_temperature,
        "num_responses": self.num_responses,
        "additional_params": self.additional_params,
    }
    return hashlib.sha256(json.dumps(hash_dict, sort_keys=True).encode()).hexdigest()
```

**Rationale**:
- Deterministic: same config always produces same hash
- Collision-resistant: SHA256 ensures uniqueness
- Sortable: lists are sorted for consistent ordering

### 7. Incremental Saving

**Decision**: Support `save_interval` for periodic progress saves.

**Rationale**:
- Long-running benchmarks may crash/timeout
- Incremental saves prevent data loss
- Can resume from partial results
- User controls frequency with `save_interval`

### 8. Benchmark Implementations as Separate Module

**Decision**: Benchmark implementations in `implementations/` subfolder.

**Rationale**:
- Clear organization: framework vs. benchmarks
- Easy to add new benchmarks
- Can have benchmark-specific dependencies
- Optional `BaseBenchmark` provides interface guidance

## Data Flow

### Running a Benchmark

```
User Call
    ↓
runner.run_benchmark()
    ↓
Create BenchmarkConfig
    ↓
Compute config hash
    ↓
Check cache (if use_cache=True)
    ├─ Found → Return cached BenchmarkRun
    └─ Not found → Continue
        ↓
    Create RunMetadata
        ↓
    Save initial run (if save_results=True)
        ↓
    Execute benchmark implementation
        ↓
    Save incremental results (every N prompts)
        ↓
    Update run with results
        ↓
    Mark as completed
        ↓
    Save final results
        ↓
    Return BenchmarkRun
```

### Analyzing Results

```
User Call
    ↓
analyzer.compare_llms()
    ↓
Query runs by filters
    ↓
Load complete runs from DB
    ↓
Group/aggregate results
    ↓
Build pandas DataFrame
    ↓
Return to user
    ↓
Optional: plot_results()
Optional: export_report()
```

## Database Schema

### `runs` Table
```sql
CREATE TABLE runs (
    run_id TEXT PRIMARY KEY,
    config_hash TEXT NOT NULL UNIQUE,
    benchmark_name TEXT NOT NULL,
    created_at TEXT NOT NULL,
    completed_at TEXT,
    status TEXT NOT NULL,
    config_json TEXT NOT NULL,
    error_message TEXT
)
```

### `results` Table
```sql
CREATE TABLE results (
    result_id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    prompt_id INTEGER NOT NULL,
    llm_name TEXT NOT NULL,
    prompt TEXT NOT NULL,
    original_response TEXT NOT NULL,
    sampled_responses_json TEXT NOT NULL,
    scores_json TEXT NOT NULL,
    metadata_json TEXT NOT NULL,
    FOREIGN KEY (run_id) REFERENCES runs (run_id) ON DELETE CASCADE
)
```

### Indices
- `idx_runs_config_hash`: Fast cache lookups
- `idx_runs_benchmark`: Filter by benchmark name
- `idx_runs_created`: Sort by creation time
- `idx_results_run_id`: Fast result loading

## Extension Points

### Adding New Benchmarks

1. Subclass `BaseBenchmark` (optional but recommended)
2. Implement `evaluate()` method
3. Return `List[PromptResult]`
4. Add to `implementations/__init__.py`

### Adding New Analysis Methods

1. Add method to `BenchmarkAnalyzer`
2. Use `self.db` to query runs
3. Return pandas DataFrame or dict
4. No changes needed elsewhere

### Custom Storage Backend (Future)

If needed, could abstract `BenchmarkResultsDB`:
1. Define interface (ABC)
2. Create implementations (SQLite, PostgreSQL, etc.)
3. Pass to Runner/Analyzer constructors

## Testing Strategy

1. **Unit Tests**: Test each component independently
   - Models: Validation, serialization
   - Storage: CRUD operations, queries
   - Runner: Execution logic, caching
   - Analyzer: Aggregation, comparison

2. **Integration Tests**: Test end-to-end workflows
   - Run → Save → Load → Analyze
   - Cache hit/miss behavior
   - Incremental saving

3. **Mock Benchmarks**: Simple test implementations
   - Fast execution for CI/CD
   - Predictable results for assertions

## Future Enhancements

1. **Progress Callbacks**: Real-time updates during execution
2. **Distributed Execution**: Run across multiple machines
3. **Result Comparison**: Diff between runs
4. **Export Formats**: JSON, CSV, Parquet
5. **Integration**: W&B, MLflow, etc.
6. **Benchmark Registry**: Discover available benchmarks
7. **Result Streaming**: Process results as they arrive

## Lessons Learned

1. **Simplicity wins**: Started with unified client, ended with two simple classes
2. **Explicit is better**: Method parameters > config objects for discoverability
3. **YAGNI applies**: Single backend sufficient, can add more later
4. **Pydantic is great**: Type safety + validation + serialization in one
5. **SQLite is underrated**: Perfect for local workflows, no extra dependencies


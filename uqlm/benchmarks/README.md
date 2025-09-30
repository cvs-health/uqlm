# UQLM Benchmarking Framework

A comprehensive framework for running and analyzing uncertainty quantification benchmarks with built-in caching, persistence, and analysis capabilities.

## Overview

The benchmarking framework consists of three main components:

1. **BenchmarkRunner** - Execute benchmarks with automatic caching and incremental saving
2. **BenchmarkAnalyzer** - Analyze results, compare models/scorers, and generate visualizations
3. **Benchmark Implementations** - Specific benchmark datasets and evaluation logic (e.g., FactScore)

## Architecture

```
uqlm/benchmarks/
├── models.py                    # Pydantic data models
├── runner.py                    # BenchmarkRunner class
├── analyzer.py                  # BenchmarkAnalyzer class
├── storage.py                   # SQLite database backend
└── implementations/             # Benchmark implementations
    ├── base.py                  # BaseBenchmark abstract class
    └── factscore.py            # FactScore benchmark
```

## Quick Start

### Running a Benchmark

```python
from uqlm.benchmarks import BenchmarkRunner, FactScoreBenchmark
from langchain_google_vertexai import ChatVertexAI

# Initialize runner
runner = BenchmarkRunner(storage_path="~/.uqlm/benchmark_results")

# Setup benchmark
bench_llm = ChatVertexAI(model="gemini-2.5-pro")
fs_benchmark = FactScoreBenchmark(judge_llm=bench_llm)

# Run with automatic caching
results = await runner.run_benchmark(
    benchmark_name="factscore",
    benchmark_implementation=fs_benchmark,
    llm_names=["gemini-2.5-flash", "gemini-2.5-pro"],
    scorer_names=["LongFormUQ"],
    dataset_name="dskar/FActScore",
    sampling_temperature=0.4,
    num_responses=5,
    use_cache=True,      # Use cached results if available
    save_results=True    # Save to database
)
```

### Analyzing Results

```python
from uqlm.benchmarks import BenchmarkAnalyzer

# Initialize analyzer
analyzer = BenchmarkAnalyzer(storage_path="~/.uqlm/benchmark_results")

# List recent runs
recent_runs = analyzer.list_runs(benchmark_name="factscore", limit=10)

# Compare LLMs
comparison = analyzer.compare_llms(
    benchmark_name="factscore",
    scorer_names=["LongFormUQ"]
)

# Visualize
analyzer.plot_results(comparison, plot_type="bar")

# Export report
analyzer.export_report(
    run_ids=[results.metadata.run_id],
    output_path="./report.html",
    format="html"
)
```

## Key Features

### 1. Automatic Caching

The runner automatically detects duplicate benchmark runs based on configuration hash:
- Same benchmark name
- Same LLM names
- Same scorer names
- Same dataset and version
- Same hyperparameters

When `use_cache=True`, it returns cached results instead of re-running.

### 2. Incremental Saving

For long-running benchmarks, use `save_interval` to save progress periodically:

```python
results = await runner.run_benchmark(
    ...,
    save_results=True,
    save_interval=10  # Save every 10 prompts
)
```

### 3. Flexible Analysis

The analyzer supports multiple ways to query and compare results:

```python
# Compare by run IDs
comparison = analyzer.compare_llms(run_ids=["run1", "run2"])

# Compare by filters
comparison = analyzer.compare_scorers(
    benchmark_name="factscore",
    llm_names=["gemini-2.5-flash"],
    date_range=(start_date, end_date)
)

# Aggregate metrics
aggregates = analyzer.aggregate_metrics(
    run_id="12345",
    groupby="llm"  # or "scorer"
)
```

### 4. SQLite Backend

All results are stored in a SQLite database for efficient querying:
- Fast read/write operations
- Support for complex queries
- Indexed for common access patterns
- Portable single-file database

## Data Models

### BenchmarkConfig

Configuration for a benchmark run:

```python
config = BenchmarkConfig(
    benchmark_name="factscore",
    llm_names=["gemini-2.5-flash"],
    scorer_names=["LongFormUQ"],
    dataset_name="dskar/FActScore",
    dataset_version="1.0",
    sampling_temperature=0.4,
    num_responses=5,
    additional_params={"max_length": 2000}
)
```

### BenchmarkRun

Complete benchmark run with metadata and results:

```python
run = BenchmarkRun(
    metadata=RunMetadata(...),
    results=[PromptResult(...), ...]
)

# Access data
run.metadata.run_id
run.metadata.status  # pending, running, completed, failed
run.results[0].scores  # Dict[scorer_name, score]
```

## Creating Custom Benchmarks

Extend `BaseBenchmark` to create custom benchmark implementations:

```python
from uqlm.benchmarks.implementations import BaseBenchmark
from uqlm.benchmarks.models import BenchmarkConfig, PromptResult

class MyBenchmark(BaseBenchmark):
    def __init__(self, judge_llm):
        self.judge_llm = judge_llm
    
    async def evaluate(
        self,
        config: BenchmarkConfig,
        progress_callback=None
    ) -> List[PromptResult]:
        # Load dataset
        dataset = load_my_dataset(config.dataset_name)
        
        # Generate responses and compute scores
        results = []
        for prompt in dataset:
            # ... evaluation logic ...
            result = PromptResult(
                prompt_id=i,
                prompt=prompt,
                llm_name=llm_name,
                original_response=response,
                sampled_responses=sampled,
                scores={"my_scorer": score}
            )
            results.append(result)
        
        return results
```

## Storage Location

By default, benchmark results are stored in:
- **Database**: `~/.uqlm/benchmark_results/benchmarks.db`
- **SQLite format**: Single file, portable

You can customize the storage location:

```python
runner = BenchmarkRunner(storage_path="/custom/path/to/storage")
analyzer = BenchmarkAnalyzer(storage_path="/custom/path/to/storage")
```

## Advanced Usage

### Query Specific Runs

```python
# By benchmark name
runs = analyzer.db.query_runs(benchmark_name="factscore")

# By LLM names
runs = analyzer.db.query_runs(llm_names=["gemini-2.5-flash"])

# By date range
from datetime import datetime, timedelta
end = datetime.now()
start = end - timedelta(days=7)
runs = analyzer.db.query_runs(date_range=(start, end))

# By status
runs = analyzer.db.query_runs(status="completed")
```

### Delete Runs

```python
runner.delete_run(run_id="12345")
```

### Load Specific Run

```python
run = runner.get_run(run_id="12345")
print(run.metadata.config)
print(len(run.results))
```

## Testing Without Saving

For quick tests, disable caching and saving:

```python
results = await runner.run_benchmark(
    ...,
    use_cache=False,    # Don't check cache
    save_results=False  # Don't save to database
)
```

## Examples

See the [benchmark_demo.ipynb](../../examples/benchmark_demo.ipynb) for complete examples.

## Requirements

- Python 3.8+
- SQLite 3.x (included with Python)
- pandas (for analysis)
- matplotlib, seaborn (optional, for plotting)

## Future Enhancements

- [ ] Support for distributed benchmarking
- [ ] Real-time progress tracking with callbacks
- [ ] More sophisticated report generation
- [ ] Export to multiple formats (JSON, CSV, etc.)
- [ ] Integration with experiment tracking platforms (W&B, MLflow)


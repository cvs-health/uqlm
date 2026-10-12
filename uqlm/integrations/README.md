# `uqlm/integrations` — Agent-Framework Integrations

This package connects every UQLM scorer to external agent frameworks without modifying
the scorers themselves. The design is **composition over inheritance**: a thin adapter
layer translates each scorer's native API into a uniform contract, and a per-framework
wrapper calls that contract from whatever shape the framework expects (LangGraph node,
tool, plugin, etc.).

---

## Directory Layout

```
uqlm/integrations/
│
├── README.md                  ← you are here
├── __init__.py                ← re-exports UQLMNode, make_uqlm_node from langgraph
│
├── base.py                    ← ScorerAdapter protocol + shared registry
│
└── langgraph/                 ← LangGraph integration (first shipped framework)
    ├── __init__.py            ← exports UQLMNode, make_uqlm_node
    ├── node.py                ← UQLMNode class + make_uqlm_node factory
    └── adapters/
        ├── __init__.py        ← imports all three submodules, triggering registration
        ├── shortform.py       ← BlackBoxUQ, WhiteBoxUQ, LLMPanel, UQEnsemble, SemanticEntropy
        ├── longform.py        ← LongTextUQ, LongTextQA, LongTextGraph
        └── codegen.py         ← CodeGenUQ

# reserved for upcoming integrations (not yet created):
# ├── claude_code/
# ├── google_adk/
```

### What each file does

| File | Responsibility |
|---|---|
| `base.py` | Defines the `ScorerAdapter` protocol and the global `_REGISTRY`. Any integration reads from here; none write directly to scorer classes. |
| `langgraph/node.py` | `UQLMNode` — reads `state[prompt]`, calls the right adapter, writes `{output_key: payload}` back. Also contains `make_uqlm_node`, a factory that returns a bare coroutine function. |
| `langgraph/adapters/shortform.py` | One adapter class per short-form scorer. Each knows whether the scorer is sync or async, whether it needs sampled responses, and how to call `generate_and_score` vs `score`. |
| `langgraph/adapters/longform.py` | Same pattern for long-form scorers. Extracts `claims_data` and `refined_response` into the `extra` dict. |
| `langgraph/adapters/codegen.py` | Adapter for `CodeGenUQ`. |

---

## Architecture

```
┌──────────────────────────────────────────────────────┐
│  Framework layer   (e.g. LangGraph)                  │
│                                                      │
│   StateGraph.add_node("uq", UQLMNode(scorer))        │
│                      │                               │
│            UQLMNode.__acall__(state)                 │
│                      │                               │
│            resolve_adapter(scorer)  ─────────────┐  │
│                      │                            │  │
│            adapter.run(scorer, prompt, ...)       │  │
└──────────────────────────────────────────────────┼──┘
                                                   │
┌──────────────────────────────────────────────────▼──┐
│  Shared layer   (base.py)                           │
│                                                      │
│   _REGISTRY: list[ScorerAdapter]                     │
│                                                      │
│   ScorerAdapter.run(scorer, *, prompt, response,     │
│                     mode, num_responses, **kwargs)   │
│       → {"scores": {...},                            │
│           "responses": [...],                        │
│           "extra": {...}}                            │
└──────────────────────────────────────────────────────┘
                       │
         (adapter calls the scorer's native API)
                       │
┌──────────────────────▼──────────────────────────────┐
│  Scorer layer   (existing UQLM scorers)              │
│                                                      │
│   BlackBoxUQ, WhiteBoxUQ, LLMPanel, ...              │
│   — completely unmodified                            │
└──────────────────────────────────────────────────────┘
```

### Data flow

1. The framework calls `UQLMNode.__acall__(state)`.
2. `UQLMNode` reads `state[prompt]` (and optionally `state[response]`,
   `state["sampled_responses"]`, `state["logprobs_results"]`,
   `state["sampled_logprobs_results"]`).
3. It calls `resolve_adapter(scorer)` which walks `_REGISTRY` and returns the first
   adapter whose `scorer_type` matches `type(scorer)` (via `isinstance`).
4. The adapter calls the scorer's native method (`generate_and_score` or `score`) and
   returns a normalised payload dict.
5. `UQLMNode` stamps `"scorer": type(scorer).__name__` onto the payload and writes
   `{output_key: payload}` back into the graph state.

### Payload shape (all adapters must return this)

```python
{
    "scores":    {"cosine_sim": 0.87, "noncontradiction": 0.92, ...},
    "responses": ["The capital of France is Paris."],
    "extra":     {},   # adapter-specific extras (e.g. claims_data, refined_response)
}
```

`"scorer"` is added by `UQLMNode` after the adapter returns; adapters do not set it.

---

## Adding a New Platform Integration

Follow these six steps to wire up a new framework (e.g. `my_framework`).

### Step 1 — Create the package skeleton

```
uqlm/integrations/my_framework/
    __init__.py          ← public exports
    node.py              ← framework-specific wrapper class or function
    adapters/
        __init__.py      ← import all adapter submodules to trigger registration
        shortform.py     ← (can be copied verbatim — adapters are framework-agnostic)
        longform.py
        codegen.py
```

> **Note on the adapters:** The adapter classes in `langgraph/adapters/` contain no
> LangGraph-specific code. You can copy them unchanged into your new package — only
> the node wrapper in `node.py` knows about the framework.

### Step 2 — Reuse or copy the adapters

The simplest path is to import directly from the LangGraph adapters package, since the
adapter implementations are framework-independent:

```python
# uqlm/integrations/my_framework/adapters/__init__.py

# Importing these triggers registration in the shared _REGISTRY in base.py.
from uqlm.integrations.langgraph.adapters import shortform  # noqa: F401
from uqlm.integrations.langgraph.adapters import longform   # noqa: F401
from uqlm.integrations.langgraph.adapters import codegen    # noqa: F401
```

If you need different behaviour for a specific scorer in your framework (e.g. a different
sampling strategy), write a new adapter class for that scorer only and register it **before**
importing the shared ones — `resolve_adapter` returns the first match.

### Step 3 — Write the framework wrapper (`node.py`)

The wrapper is the only file that knows about the target framework. Its job is to:

1. Accept a scorer instance and configuration.
2. Translate the framework's invocation convention into a call to `adapter.run(...)`.
3. Translate the returned payload back into whatever the framework expects.

```python
# uqlm/integrations/my_framework/node.py

import uqlm.integrations.my_framework.adapters  # noqa: F401 — triggers registration

from uqlm.integrations.base import resolve_adapter


class MyFrameworkUQLMNode:
    """Wraps any UQLM scorer as a MyFramework tool/node/plugin."""

    def __init__(
        self,
        scorer,
        *,
        output_key: str = "uq",
        prompt: str = "prompt",
        response: str = "response",
        mode: str = "score",
        num_responses: int = 5,
        adapter_kwargs: dict | None = None,
    ):
        self.scorer = scorer
        self.output_key = output_key
        self.prompt = prompt
        self.response = response
        self.mode = mode
        self.num_responses = num_responses
        self.adapter_kwargs = adapter_kwargs

    async def __call__(self, state: dict) -> dict:
        adapter = resolve_adapter(self.scorer)
        prompt = state[self.prompt]
        response = state.get(self.response)

        extra_kwargs = dict(self.adapter_kwargs or {})
        for key in ("sampled_responses", "logprobs_results", "sampled_logprobs_results"):
            if key in state and key not in extra_kwargs:
                extra_kwargs[key] = state[key]

        payload = await adapter.run(
            self.scorer,
            prompt=prompt,
            response=response,
            mode=self.mode,
            num_responses=self.num_responses,
            **extra_kwargs,
        )
        payload["scorer"] = type(self.scorer).__name__

        # Translate payload into whatever shape MyFramework expects.
        # For a framework that reads a flat dict, you might return payload directly.
        return {self.output_key: payload}
```

### Step 4 — Export from the package `__init__.py`

```python
# uqlm/integrations/my_framework/__init__.py

from uqlm.integrations.my_framework.node import MyFrameworkUQLMNode

__all__ = ["MyFrameworkUQLMNode"]
```

### Step 5 — Expose at the top-level integrations `__init__.py`

Add your new symbol to `uqlm/integrations/__init__.py`:

```python
# uqlm/integrations/__init__.py  (existing file — add the new import)

from uqlm.integrations.langgraph import UQLMNode, make_uqlm_node
from uqlm.integrations.my_framework import MyFrameworkUQLMNode  # ← new

__all__ = ["UQLMNode", "make_uqlm_node", "MyFrameworkUQLMNode"]
```

And add it to `uqlm/__init__.py` so users can do `from uqlm import MyFrameworkUQLMNode`:

```python
# uqlm/__init__.py  (add alongside the existing UQLMNode import)
from uqlm.integrations.my_framework import MyFrameworkUQLMNode
```

### Step 6 — Add an optional dependency to `pyproject.toml`

Keep the framework package out of the base dependencies so users who don't need it don't
pay the install cost:

```toml
# pyproject.toml
[project.optional-dependencies]
langgraph    = ["langgraph>=0.2"]
my_framework = ["my-framework-package>=1.0"]
```

---

## Writing a New Adapter for a Custom Scorer

If you have a scorer that is not yet covered by the existing adapters (e.g. a new scorer
class you're adding to `uqlm/scorers/`), write a dedicated adapter class and register it.

```python
# uqlm/integrations/langgraph/adapters/my_scorer.py
# (or inside shortform.py / a new submodule — register order is first-match)

from uqlm.scorers.shortform.my_scorer import MyScorer
from uqlm.integrations.base import register_adapter

_SKIP_KEYS = frozenset({
    "prompts", "responses", "sampled_responses",
    "raw_responses", "raw_sampled_responses",
    "logprob", "sampled_logprob",
})


class MyScorerAdapter:
    scorer_type = MyScorer  # resolve_adapter matches on isinstance(scorer, scorer_type)

    async def run(self, scorer, *, prompt, response, mode, num_responses, **kwargs):
        # Call the scorer's native API.
        result = await scorer.generate_and_score(
            prompts=[prompt],
            num_responses=num_responses,
            show_progress_bars=False,
        )
        # Normalise UQResult.data into the standard payload shape.
        data = result.data
        responses_list = data.get("responses", [])
        scores = {
            k: v[0]
            for k, v in data.items()
            if k not in _SKIP_KEYS and isinstance(v, list) and len(v) > 0
        }
        return {
            "scores":    scores,
            "responses": [responses_list[0]] if responses_list else [],
            "extra":     {},
        }


register_adapter(MyScorerAdapter())
```

Then import the new submodule inside
`uqlm/integrations/langgraph/adapters/__init__.py` so registration happens at import time:

```python
# uqlm/integrations/langgraph/adapters/__init__.py
from uqlm.integrations.langgraph.adapters import shortform   # noqa: F401
from uqlm.integrations.langgraph.adapters import longform    # noqa: F401
from uqlm.integrations.langgraph.adapters import codegen     # noqa: F401
from uqlm.integrations.langgraph.adapters import my_scorer   # noqa: F401  ← add this
```

---

## Adapter Contract Reference

Every adapter class must satisfy this interface (enforced by the `ScorerAdapter`
`runtime_checkable` Protocol in `base.py`):

```python
class MyScorerAdapter:
    scorer_type: type               # the scorer class this adapter handles

    async def run(
        self,
        scorer,                     # the live scorer instance
        *,
        prompt: str,                # the user prompt
        response: str | None,       # existing LLM response (may be None)
        mode: str,                  # "score" | "generate_and_score"
        num_responses: int,         # how many samples to draw if needed
        **kwargs,                   # extra state pass-throughs (sampled_responses, etc.)
    ) -> dict:                      # must return {"scores": ..., "responses": ..., "extra": ...}
        ...
```

### `mode` semantics

| `mode` | What the adapter should do |
|---|---|
| `"generate_and_score"` | Always call `scorer.generate_and_score(...)`. Ignore any pre-existing `response`. |
| `"score"` | If `response` and (where needed) `sampled_responses` / `logprobs_results` are available, call the scorer's lighter-weight `score(...)` method. Fall back to `generate_and_score` if the required inputs are missing. |

### `_SKIP_KEYS` convention

When building the `scores` dict from `UQResult.data`, exclude these keys — they are
metadata or raw responses, not scores:

```python
_SKIP_KEYS = frozenset({
    "prompts", "responses", "sampled_responses",
    "raw_responses", "raw_sampled_responses",
    "logprob", "sampled_logprob",
})
```

Long-form adapters also route `"claims_data"` and `"refined_response"` to `extra` instead
of `scores`.

---

## Tests

Place tests under `tests/integrations/<framework>/`. Mirror the structure used for
the LangGraph tests:

```
tests/integrations/
    __init__.py
    langgraph/
        __init__.py
        test_node.py
        test_adapters_shortform.py
        test_adapters_longform.py
        test_adapters_codegen.py
    my_framework/            ← new
        __init__.py
        test_node.py
        test_adapters.py
```

All tests should stub the scorer with `unittest.mock` — no network, no model loads,
no real framework install required. See the existing LangGraph tests for patterns.

Run only the new tests:

```bash
uv run pytest tests/integrations/my_framework -q
```

Run the full suite to check for regressions:

```bash
uv run pytest tests/ -q
uv run ruff check uqlm/integrations tests/integrations
```

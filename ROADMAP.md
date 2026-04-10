# UQLM Roadmap

This document outlines the planned directions for uqlm. It is not a commitment to specific timelines but rather a transparent signal of where the project is headed.

---

## Agentic UQ

As LLMs are increasingly deployed in agentic settings, uncertainty quantification needs to evolve beyond single-turn question answering.

- **LangGraph integration** — first-class support for uncertainty-aware LangGraph workflows, making it easier to wire uqlm scorers into graph-based pipelines
- **Agentic use case coverage** — testing and handling of agentic-specific scenarios: tool calling, multi-step reasoning, retrieval-augmented generation, and multi-agent coordination
- **Agent-specific demos** — new example notebooks covering common agentic patterns with uncertainty quantification

---

## OSS Quality Standards

- **Extended linting** — expand ruff rule set (e.g. `I`, `B`, `UP`) and enforce stricter code quality checks in CI
- **Type checking** — integrate `basedpyright` or `pyright` for static type checking across the package
- **Optional dependency extras** — reduce install footprint by making heavy dependencies (`transformers`, `torch`, `bert-score`) optional, so users who only need API-based scorers don't pay the ~500MB install cost
- **Integration tests** — add orchestration-level tests that catch schema mismatches and cross-scorer bugs that unit tests miss

---

## Community

- **Slack / Discord** — dedicated community channel for users and contributors to ask questions, share use cases, and discuss research
- **Monthly meetups** — recurring virtual meetups for contributors and users to sync on roadmap, demos, and research
- **GitHub Discussions** — promote Discussions for async Q&A and roadmap feedback

---

## New Scorer Integration

`uqlm` tracks the UQ research literature and aims to integrate new scorers as the field evolves. Contributions of new scorers backed by published research are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for guidance.

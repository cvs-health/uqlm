# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.5.8] - 2026-03-01
### Fixed
- Fixed noncontradiction formula in NLI scoring
- Fixed confidence bounds in `sets` scorer
- Resolved Dependabot security vulnerabilities

## [0.5.7] - 2026-02-01
### Fixed
- Fixed incorrect scorer definitions and broken references in documentation site

## [0.5.6] - 2026-01-01
### Changed
- Relaxed upper bounds on `rich` dependency
- Updated README badges and publication links
- Dependency updates via Dependabot (pytest-asyncio, sphinx-gallery, nbsphinx)

## [0.5.5] - 2025-12-01
### Changed
- Switched package manager from Poetry to `uv`
- Relaxed upper bounds on `langchain` and `langchain-model-profiles`
### Fixed
- Handle numpy arrays in `sampled_responses` to prevent type errors with Parquet data

## [0.5.4] - 2025-11-01
### Added
- `nli_model_name` and `sentence_transformer` parameters in `WhiteBoxUQ`
- All white-box scorers enabled in `UQEnsemble`

## [0.5.3] - 2025-10-01
### Fixed
- Fixed broken reference links for graph-based scorers in documentation
- Security upgrades via Dependabot

## [0.5.2] - 2025-09-01
### Added
- LLM-based NLI entailment as an alternative to model-based NLI

## [0.5.1] - 2025-08-01
### Added
- `longform` subpackage: `LongTextUQ`, `LongTextQA`, `LongTextGraph` scorers for claim-level hallucination detection
- Notebook pointers and long-text scorer definitions in documentation

## [0.5.0] - 2025-07-01
### Changed
- Deprecated `normalized_probability` scorer in favor of `sequence_probability`
### Fixed
- Scorer validation for white-box methods
- Security vulnerabilities via Dependabot

## [0.4.5] - 2025-06-01
### Fixed
- Fixed logprob model string checking

## [0.4.4] - 2025-05-01
### Changed
- Internal improvements and dependency updates

## [0.4.3] - 2025-04-01
### Changed
- Internal improvements and dependency updates

## [0.4.2] - 2025-03-01
### Changed
- Internal improvements and dependency updates

## [0.4.1] - 2025-02-01
### Changed
- Internal improvements and dependency updates

## [0.4.0] - 2025-01-01
### Added
- Ensemble tuning with Optuna-based hyperparameter optimization
- Score calibration via `ScoreCalibrator` (Platt scaling, isotonic regression)

## [0.3.1] - 2024-12-01
### Fixed
- Bug fixes and stability improvements

## [0.3.0] - 2024-11-01
### Added
- LLM-as-a-judge scoring via `LLMPanel`
- Multimodal (image + text) uncertainty support

## [0.2.7] - 2024-10-01
### Fixed
- Bug fixes and dependency updates

## [0.2.0] - 2024-08-01
### Added
- `SemanticEntropy` scorer (state-of-the-art semantic entropy method)
- `SemanticDensity` scorer

## [0.1.0] - 2024-06-01
### Added
- Initial release
- `BlackBoxUQ`: consistency-based hallucination detection (cosine, BERT, exact match)
- `WhiteBoxUQ`: token-probability-based uncertainty (logprobs, p-true, sampled)
- `UQEnsemble`: ensemble scoring with off-the-shelf and tuned configurations
- LangChain integration for all major LLM providers

[Unreleased]: https://github.com/cvs-health/uqlm/compare/v0.5.8...HEAD
[0.5.8]: https://github.com/cvs-health/uqlm/compare/v0.5.7...v0.5.8
[0.5.7]: https://github.com/cvs-health/uqlm/compare/v0.5.6...v0.5.7
[0.5.6]: https://github.com/cvs-health/uqlm/compare/v0.5.5...v0.5.6
[0.5.5]: https://github.com/cvs-health/uqlm/compare/v0.5.4...v0.5.5
[0.5.4]: https://github.com/cvs-health/uqlm/compare/v0.5.3...v0.5.4
[0.5.3]: https://github.com/cvs-health/uqlm/compare/v0.5.2...v0.5.3
[0.5.2]: https://github.com/cvs-health/uqlm/compare/v0.5.1...v0.5.2
[0.5.1]: https://github.com/cvs-health/uqlm/compare/v0.5.0...v0.5.1
[0.5.0]: https://github.com/cvs-health/uqlm/compare/v0.4.5...v0.5.0
[0.4.5]: https://github.com/cvs-health/uqlm/compare/v0.4.4...v0.4.5
[0.4.4]: https://github.com/cvs-health/uqlm/compare/v0.4.3...v0.4.4
[0.4.3]: https://github.com/cvs-health/uqlm/compare/v0.4.2...v0.4.3
[0.4.2]: https://github.com/cvs-health/uqlm/compare/v0.4.1...v0.4.2
[0.4.1]: https://github.com/cvs-health/uqlm/compare/v0.4.0...v0.4.1
[0.4.0]: https://github.com/cvs-health/uqlm/compare/v0.3.1...v0.4.0
[0.3.1]: https://github.com/cvs-health/uqlm/compare/v0.3.0...v0.3.1
[0.3.0]: https://github.com/cvs-health/uqlm/compare/v0.2.7...v0.3.0
[0.2.7]: https://github.com/cvs-health/uqlm/compare/v0.2.0...v0.2.7
[0.2.0]: https://github.com/cvs-health/uqlm/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/cvs-health/uqlm/releases/tag/v0.1.0

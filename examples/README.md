# Examples

This directory contains a comprehensive collection of Jupyter notebooks demonstrating how to use UQLM for various uncertainty quantification and hallucination detection tasks.

## Overview of Example Notebooks

The notebooks are organized into core methods, long-form techniques, and advanced approaches to help you select the best method for your needs.

### Tutorials for Core Uncertainty Quantification Methods

| Tutorial | Great fit for... | LLM Compatibility | Added Cost/Latency |
|----------|-------------|-------------------|--------------|
| [Black-Box UQ](https://github.com/cvs-health/uqlm/blob/main/examples/black_box_demo.ipynb) | Quick setup with any LLM; no need for model internals | All LLMs (API-only access) | Medium-High (multiple generations and comparisons) |
| [White-Box UQ (Single-Generation)](https://github.com/cvs-health/uqlm/blob/main/examples/white_box_single_generation_demo.ipynb) | Fastest and most efficient UQ when you have token probabilities | Requires token probability access | Negligible (single generation) |
| [White-Box UQ (Multi-Generation)](https://github.com/cvs-health/uqlm/blob/main/examples/white_box_multi_generation_demo.ipynb) | Higher accuracy UQ when compute budget allows | Requires token probability access | Medium-High (multiple generations) |
| [LLM-as-a-Judge](https://github.com/cvs-health/uqlm/blob/main/examples/judges_demo.ipynb) | Leveraging one or more LLMs to assess hallucination likelihood | All LLMs (API-only access) | Low-Medium (depends on which judge(s)) |
| [Train a UQ Ensemble](https://github.com/cvs-health/uqlm/blob/main/examples/ensemble_tuning_demo.ipynb) | Maximizing performance by combining multiple UQ methods | Depends on ensemble components | Low-High (depends on selected components) |

### Tutorials for Long-Form Uncertainty Quantification Methods (for long-text outputs)

| Tutorial | Great fit for... | LLM Compatibility | Added Cost/Latency |
|----------|-------------|-------------------|--------------|
| [LUQ method](https://github.com/cvs-health/uqlm/blob/main/examples/long_text_uq_demo.ipynb) | Detecting claim-level hallucinations in long-form text without model internals | All LLMs (API-only access) | Medium-High (operates over all claims/sentences in original response) |
| [Graph-based method](https://github.com/cvs-health/uqlm/blob/main/examples/long_text_graph_demo.ipynb) | Analyzing claim relationships in complex responses | All LLMs (API-only access) | Very High (operates over all claims/sentences in original response and sampled responses) |
| [Generalized Long-form semantic entropy](https://github.com/cvs-health/uqlm/blob/main/examples/long_text_qa_demo.ipynb) | Reflexlive, detailed approach to claim-level hallucination detection | All LLMs (API-only access) | High (operates over all claims/sentences in original response) |

### Other Tutorials and SOTA Method Examples

| Tutorial | Great fit for... | LLM Compatibility | Added Cost/Latency |
|----------|-------------|-------------------|--------------|
| [UQ for Code Generation](https://github.com/cvs-health/uqlm/blob/main/examples/codegen_demo.ipynb) | Uncertainty quantification for code generation | Varies by method | Varies by method |
| [Multimodal UQ](https://github.com/cvs-health/uqlm/blob/main/examples/multimodal_demo.ipynb) | Uncertainty quantification with image+text inputs | Requires image-to-text model | Varies by method |
| [Score Calibration](https://github.com/cvs-health/uqlm/blob/main/examples/score_calibration_demo.ipynb) | Converting raw scores to calibrated probabilities as a postprocessing step | Works with any UQ method | Negligible |
| [Semantic Entropy](https://github.com/cvs-health/uqlm/blob/main/examples/semantic_entropy_demo.ipynb) | State-of-the-art UQ when token probabilities are available | Requires token probability access | Medium-High (multiple generations and comparisons) |
| [Semantic Density](https://github.com/cvs-health/uqlm/blob/main/examples/semantic_density_demo.ipynb) | Newest SOTA method for high-accuracy UQ | Requires token probability access | Medium-High (multiple generations and comparisons) |
| [BS Detector Off-the-Shelf UQ Ensemble](https://github.com/cvs-health/uqlm/blob/main/examples/ensemble_off_the_shelf_demo.ipynb) | Ready-to-use ensemble without training | Depends on ensemble components | Medium-High (multiple generations and comparisons) |


## Where should I start?

We recommend starting with the [Black-Box UQ](https://github.com/cvs-health/uqlm/blob/main/examples/black_box_demo.ipynb) notebook if you're new to uncertainty quantification or don't have access to model internals.

For the most efficient approach with minimal compute requirements, try the [White-Box UQ (Single-Generation)](https://github.com/cvs-health/uqlm/blob/main/examples/white_box_single_generation_demo.ipynb) notebook if you have access to token probabilities.

For long-form text evaluation, the [LUQ method](https://github.com/cvs-health/uqlm/blob/main/examples/long_text_uq_demo.ipynb) provides a good starting point that works with any LLM API.
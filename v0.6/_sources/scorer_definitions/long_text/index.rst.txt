Long-Text Scorers
=================

Long-form uncertainty quantification implements a three-stage pipeline after response generation:

1. Response Decomposition: The response :math:`y` is decomposed into units (claims or sentences), where a unit as denoted as :math:`s`.

2. Unit-Level Confidence Scoring: Confidence scores are computed using a unit-level scoring function with values in :math:`[0, 1]`. Higher scores indicate greater likelihood of factual correctness. Units with scores below threshold :math:`\tau` are flagged as potential hallucinations.

3. Response-Level Aggregation: Unit scores are combined to provide an overall response confidence.

**Key Characteristics:**

- **Universal Compatibility:** Works with any LLM without requiring token probability access
- **Fine-Grained Scoring:** Score at sentence or claim-level to localize likely hallucinations
- **Uncertainty-aware decoding:** Improve factual precision by dropping high-uncertainty claims

**Trade-offs:**

- **Higher Cost:** Requires multiple generations per prompt
- **Limited Compatibility:** Multiple generations and comparison calculations increase latency


Long-Text Scoring Methods
-------------------------
 
There are three main categories of long-text scoring methods offered by UQLM:

.. toctree::
   :maxdepth: 1

   luq
   graph
   qa
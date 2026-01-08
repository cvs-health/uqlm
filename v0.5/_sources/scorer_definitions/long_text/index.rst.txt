Long-Text Scorers
=================

Long-form uncertainty quantification implements a three-stage pipeline after response generation:

1. Response Decomposition: The response $y$ is decomposed into units (claims or sentences), where a unit as denoted as $s$.

2. Unit-Level Confidence Scoring: Confidence scores are computed using function $c_g(s;\cdot) \in [0, 1]$. Higher scores indicate greater likelihood of factual correctness. Units with scores below threshold $\tau$ are flagged as potential hallucinations.

3. Response-Level Aggregation: Unit scores are combined to provide an overall response confidence.

**Key Characteristics:**

- **Universal Compatibility:** Works with any LLM without requiring token probability access
- **Fine-Grained Scoring:** Score at sentence or claim-level to localize likely hallucinations
- **Uncertainty-aware decoding:** Improve factual precision by dropping high-uncertainty claims

**Trade-offs:**

- **Higher Cost:** Requires multiple generations per prompt
- **Limited Compatibility:** Multiple generations and comparison calculations increase latency


Claim-Response Scorers
----------------------

These scorers directly compare claims or sentences in the original responses with sampled responses generated from the same prompt.

.. toctree::
   :maxdepth: 1

   entailment
   noncontradiction
   contrasted_entailment

Graph-Based Scorers
-------------------

These scorers decompose original and sampled responses into claims, obtain the union of unique claims across all responses, and compute graph centrality metrics on the bipartite graph of claim-response entailment to measure uncertainty.

.. toctree::
   :maxdepth: 1

   closeness_centrality
   harmonic_centrality
   degree_centrality
   betweenness_centrality
   laplacian_centrality
   page_rank


Claim-QA Scorers
----------------

These scorers decompose responses into granular units (sentences or claims), convert each claim or sentence to a question, sample LLM responses to those questions, and measure consistency among those answers to score the claim.

.. toctree::
   :maxdepth: 1

   semantic_negentropy
   semantic_sets_confidence
   noncontradiction
   entailment
   exact_match
   bert_score
   cosine_sim
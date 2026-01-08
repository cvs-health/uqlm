White-Box Scorers
=================

White-box Uncertainty Quantification (UQ) methods leverage token probabilities to estimate uncertainty.
These scorers offer single-generation scoring, which is significantly faster and cheaper than black-box
methods, but require access to the LLM's internal probabilities.

**Key Characteristics:**

- **Minimal Latency:** Token probabilities are already returned by the LLM
- **No Added Cost:** Doesn't require additional LLM calls (for single-generation scorers)
- **High Performance:** Access to internal model states provides rich uncertainty signals

**Trade-offs:**

- **Limited Compatibility:** Requires access to token probabilities, not available for all LLMs/APIs

**Notation:**

Let the tokenization of LLM response :math:`y_i` be denoted as :math:`\{t_1,...,t_{L_i}\}`, where
:math:`L_i` denotes the number of tokens in the response. Let :math:`p_t` denote the token probability
for token :math:`t`.

Single-Generation Scorers
-------------------------

These scorers require only one LLM generation and use the token probabilities from that single response.

.. toctree::
   :maxdepth: 1

   sequence_probability
   normalized_probability
   min_probability
   mean_token_negentropy
   min_token_negentropy
   probability_margin

Multi-Generation Scorers
------------------------

These scorers generate multiple responses from the same prompt, combining the sampling approach of
black-box UQ with token-probability-based signals.

.. toctree::
   :maxdepth: 1

   monte_carlo_probability
   consistency_and_confidence
   semantic_negentropy_whitebox
   semantic_density
   p_true


Black-Box Scorers
=================

Black-box Uncertainty Quantification (UQ) methods treat the LLM as a black box and evaluate
consistency of multiple responses generated from the same prompt to estimate response-level confidence.
These scorers are compatible with any LLM and don't require access to internal model states or token probabilities.

**Key Characteristics:**

- **Universal Compatibility:** Works with any LLM
- **Intuitive:** Easy to understand and implement
- **No Internal Access Required:** Doesn't need token probabilities or model internals

**Trade-offs:**

- **Higher Cost:** Requires multiple generations per prompt
- **Slower:** Multiple generations and comparison calculations increase latency

**Notation:**

For a given prompt :math:`x_i`, these approaches involve generating :math:`m` responses
:math:`\tilde{\mathbf{y}}_i = \{ \tilde{y}_{i1},...,\tilde{y}_{im}\}`, using a non-zero temperature,
from the same prompt and comparing these responses to the original response :math:`y_{i}`.

.. toctree::
   :maxdepth: 1
   :caption: Available Scorers

   semantic_negentropy
   semantic_sets_confidence
   noncontradiction
   entailment
   exact_match
   bert_score
   cosine_sim


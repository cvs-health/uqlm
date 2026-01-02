Ensemble Scorers
================

Ensemble scorers leverage a weighted average of multiple individual scorers to provide a more robust
uncertainty/confidence estimate. They offer high flexibility and customizability, allowing you to
tailor the ensemble to specific use cases.

**Key Characteristics:**

- **Flexible:** Combine any mix of black-box, white-box, and LLM-as-a-Judge scorers
- **Customizable:** Tune weights for your specific use case and data
- **Off-the-Shelf Options:** Pre-configured ensembles like BS Detector available

**Trade-offs:**

- **Inherited Costs:** Ensemble inherits latency and cost from component scorers
- **Tuning Requirements:** Optimal performance may require weight tuning on labeled data

**Mathematical Framework:**

Given a set of :math:`n` component scorers with scores :math:`s_1, s_2, ..., s_n` and weights
:math:`w_1, w_2, ..., w_n` (where :math:`\sum w_i = 1`), the ensemble score is:

.. math::

    \text{Ensemble}(y_i) = \sum_{k=1}^n w_k \cdot s_k(y_i)

.. toctree::
   :maxdepth: 1
   :caption: Available Ensemble Methods

   bs_detector
   generalized_ensemble


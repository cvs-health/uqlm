Monte Carlo Sequence Probability
================================

.. currentmodule:: uqlm.scorers

``monte_carlo_probability``

Monte Carlo Sequence Probability (MCSP) computes the average length-normalized sequence probability
across multiple sampled responses.

Definition
----------

Let :math:`y_1, y_2, ..., y_m` denote :math:`m` sampled responses generated from the same prompt.
Monte Carlo Sequence Probability is defined as:

.. math::

    MCSP(y_1, y_2, ..., y_m) = \frac{1}{m} \sum_{i=1}^m \prod_{t \in y_i} p_t^{\frac{1}{L_i}}

where :math:`L_i` is the number of tokens in response :math:`y_i` and :math:`p_t` is the token probability.

**Key Properties:**

- Combines multiple response samples for more robust probability estimation
- Length-normalized to allow fair comparison across responses
- Score range: :math:`[0, 1]`

How It Works
------------

1. Generate multiple responses from the same prompt with logprobs enabled
2. For each response, compute the length-normalized sequence probability (geometric mean of token probabilities)
3. Average across all sampled responses

This scorer combines the sampling approach of black-box methods with token probability information,
providing a more robust estimate than single-response probability.

Parameters
----------

When using :class:`WhiteBoxUQ`, specify ``"monte_carlo_probability"`` in the ``scorers`` list.

Example
-------

.. code-block:: python

    from uqlm import WhiteBoxUQ

    # Initialize with monte_carlo_probability scorer
    wbuq = WhiteBoxUQ(
        llm=llm,
        scorers=["monte_carlo_probability"],
        sampling_temperature=1.0
    )

    # Generate responses and compute scores (requires multiple samples)
    results = await wbuq.generate_and_score(prompts=prompts, num_responses=5)

    # Access the monte_carlo_probability scores
    print(results.to_df()["monte_carlo_probability"])

References
----------

- Kuhn, L., et al. (2023). `Semantic Uncertainty: Linguistic Invariances for Uncertainty Estimation in Natural Language Generation <https://arxiv.org/abs/2302.09664>`_. *arXiv*.

See Also
--------

- :class:`WhiteBoxUQ` - Main class for white-box uncertainty quantification
- :doc:`consistency_and_confidence` - Alternative multi-generation scorer
- :doc:`normalized_probability` - Single-generation length-normalized probability


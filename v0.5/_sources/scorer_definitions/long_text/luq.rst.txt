Long-Text Uncertainty Quantification (LUQ)
==========================================

.. currentmodule:: uqlm.scorers

Definition
----------

The Long-text UQ (LUQ) approach demonstrated here is adapted from Zhang et al. (2024). Similar to standard black-box UQ, this approach requires generating a original response and sampled candidate responses to the same prompt. The original response :math:`y` is then decomposed into units (claims or sentences). A confidence score for each unit :math:`s` is then obtained by averaging entailment probabilities across candidate responses:

.. math::

    c_g(s; \mathbf{y}_{\text{cand}}) = \frac{1}{m} \sum_{j=1}^m P(\text{entail}|y_j, s)

where :math:`\mathbf{y}^{(s)}_{\text{cand}} = \{y_1^{(s)}, ..., y_m^{(s)}\}` are :math:`m` candidate responses, and :math:`P(\text{entail}|y_j, s)` denotes the NLI-estimated probability that :math:`s` is entailed in :math:`y_j`.

**Key Properties:**

- Claim or sententence-level scoring
- Less complex (cost and latency) than other long-form scoring methods
- Score range: :math:`[0, 1]`

How It Works
------------

1. Generate an original response and sampled responses
2. Decompose original response into units (claims or sentences)
3. Obtain entailment probabilities of units in original response with respect to sampled responses
4. For each unit, average entailment probabilities across sampled responses

Parameters
----------

When using :class:`LongTextUQ`, specify ``"entailment"`` (or alternative scoring function) in the ``scorers`` list.

Example
-------

.. code-block:: python

    from uqlm import LongTextUQ

    # Initialize 
    luq = LongTextUQ(
        llm=original_llm,
        claim_decomposition_llm=claim_decomposition_llm,
        scorers=["entailment"],
        sampling_temperature=1.0
    )

    # Generate responses and compute scores
    results = await luq.generate_and_score(prompts=prompts, num_responses=5)

    # Access the claim-level scores
    print(results.to_df()["claims_data"])

References
----------

- Zhang, C., et al. (2024). `LUQ: Long-text Uncertainty Quantification for LLMs <https://arxiv.org/abs/2403.20279>`_. *arXiv*.

See Also
--------

- :class:`LongTextUQ` - Class for LUQ-style scoring for long-form generations


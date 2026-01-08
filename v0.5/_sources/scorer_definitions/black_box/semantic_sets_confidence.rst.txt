Semantic Sets Confidence
========================

.. currentmodule:: uqlm.scorers

``semantic_sets_confidence``

Semantic Sets Confidence (SSC) counts the number of unique response sets (clusters) obtained during
the computation of semantic entropy and normalizes this count to obtain a confidence score.

Definition
----------

Let :math:`N_C` denote the number of unique semantic clusters and :math:`m` denote the number of
sampled responses. We normalize this count to obtain a confidence score in :math:`[0,1]` as follows:

.. math::

    SSC(y_i; \tilde{\mathbf{y}}_i) = \frac{m - N_C}{m - 1}

**Interpretation:**

- When :math:`N_C = 1`: All sampled responses are semantically equivalent, so the confidence score is **1**
- When :math:`N_C = m`: All responses are semantically distinct, so the confidence score is **0**

How It Works
------------

1. Generate multiple responses :math:`\tilde{\mathbf{y}}_i` from the same prompt
2. Use an NLI model to cluster semantically equivalent responses based on mutual entailment
3. Count the number of unique semantic clusters :math:`N_C`
4. Normalize using the formula above to get a score in :math:`[0,1]`

Fewer semantic clusters indicate higher consistency among responses, which typically correlates
with higher confidence in the response accuracy.

Parameters
----------

When using :class:`BlackBoxUQ`, specify ``"semantic_sets_confidence"`` in the ``scorers`` list.

Example
-------

.. code-block:: python

    from uqlm import BlackBoxUQ

    # Initialize with semantic_sets_confidence scorer
    bbuq = BlackBoxUQ(
        llm=llm,
        scorers=["semantic_sets_confidence"],
        nli_model_name="microsoft/deberta-large-mnli"
    )

    # Generate responses and compute scores
    results = await bbuq.generate_and_score(prompts=prompts, num_responses=5)

    # Access the semantic_sets_confidence scores
    print(results.to_df()["semantic_sets_confidence"])

References
----------

- Lin, Z., et al. (2024). `Generating with Confidence: Uncertainty Quantification for Black-box Large Language Models <https://arxiv.org/abs/2305.19187>`_. *arXiv*.
- Vashurin, R., et al. (2025). `Benchmarking LLM Uncertainty Quantification Methods for Agentic AI <https://arxiv.org/abs/2406.15627>`_. *arXiv*.
- Kuhn, L., et al. (2023). `Semantic Uncertainty: Linguistic Invariances for Uncertainty Estimation in Natural Language Generation <https://arxiv.org/abs/2302.09664>`_. *arXiv*.

See Also
--------

- :class:`BlackBoxUQ` - Main class for black-box uncertainty quantification
- :doc:`semantic_negentropy` - Related scorer based on semantic entropy


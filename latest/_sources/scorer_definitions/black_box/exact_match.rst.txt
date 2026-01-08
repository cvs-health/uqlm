Exact Match Rate
================

.. currentmodule:: uqlm.scorers

``exact_match``

Exact Match Rate (EMR) computes the proportion of candidate responses that are identical to the
original response.

Definition
----------

The Exact Match Rate is defined as:

.. math::

    EMR(y_i; \tilde{\mathbf{y}}_i) = \frac{1}{m} \sum_{j=1}^m \mathbb{I}(y_i = \tilde{y}_{ij})

where :math:`\mathbb{I}(\cdot)` is the indicator function that equals 1 if the condition is true and 0 otherwise.

**Key Properties:**

- Simple string comparison - no semantic analysis required
- Particularly effective for short, factual answers (e.g., names, numbers, single words)
- Score range: :math:`[0, 1]` where 1 indicates all sampled responses exactly match the original

How It Works
------------

1. Generate multiple candidate responses :math:`\tilde{\mathbf{y}}_i` from the same prompt
2. Compare each candidate response exactly (string-wise) to the original response
3. Calculate the proportion of exact matches

This scorer is most effective for:

- Short-answer questions
- Factual queries with deterministic answers
- Multiple-choice questions

For longer, more nuanced responses, semantic similarity scorers like :doc:`noncontradiction` or
:doc:`semantic_negentropy` may be more appropriate.

Parameters
----------

When using :class:`BlackBoxUQ`, specify ``"exact_match"`` in the ``scorers`` list.

Example
-------

.. code-block:: python

    from uqlm import BlackBoxUQ

    # Initialize with exact_match scorer
    bbuq = BlackBoxUQ(
        llm=llm,
        scorers=["exact_match"]
    )

    # Generate responses and compute scores
    results = await bbuq.generate_and_score(prompts=prompts, num_responses=5)

    # Access the exact_match scores
    print(results.to_df()["exact_match"])

References
----------

- Cole, J., et al. (2023). `Selectively Answering Ambiguous Questions <https://arxiv.org/abs/2305.14613>`_. *arXiv*.
- Chen, J. & Mueller, J. (2023). `Quantifying Uncertainty in Answers from any Language Model and Enhancing their Trustworthiness <https://arxiv.org/abs/2308.16175>`_. *arXiv*.

See Also
--------

- :class:`BlackBoxUQ` - Main class for black-box uncertainty quantification
- :doc:`cosine_sim` - Semantic similarity using sentence embeddings
- :doc:`bert_score` - Semantic similarity using BERT embeddings


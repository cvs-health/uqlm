Consistency and Confidence (CoCoA)
==================================

.. currentmodule:: uqlm.scorers

``consistency_and_confidence``

Consistency and Confidence Approach (CoCoA) leverages two distinct signals: (1) similarity between
an original response and sampled responses, and (2) token probabilities from the original response.

Definition
----------

Let :math:`y_0` be the original response and :math:`y_1, ..., y_m` be :math:`m` sampled responses.

**Step 1: Compute Length-Normalized Token Probability**

.. math::

    LNTP(y_0) = \prod_{t \in y_0} p_t^{\frac{1}{L_0}}

**Step 2: Compute Normalized Cosine Similarity**

Average cosine similarity across pairings of the original response with all sampled responses,
normalized to :math:`[0,1]`:

.. math::

    NCS(y_0; y_1, ..., y_m) = \frac{1}{m} \sum_{i=1}^m \frac{\cos(y_0, y_i) + 1}{2}

**Step 3: Compute CoCoA Score**

CoCoA is the product of these two terms:

.. math::

    CoCoA(y_0; y_1, ..., y_m) = LNTP(y_0) \cdot NCS(y_0; y_1, ..., y_m)

**Key Properties:**

- Combines token-level confidence with response-level consistency
- Multiplicative combination ensures both signals must be high for high confidence
- Score range: :math:`[0, 1]`

How It Works
------------

1. Generate an original response with logprobs enabled
2. Generate multiple sampled responses from the same prompt
3. Compute the length-normalized probability of the original response
4. Encode all responses using a sentence transformer and compute cosine similarities
5. Multiply the probability and average similarity score

This approach is particularly effective because it requires both:

- The model to be confident in its token predictions (high probability)
- The responses to be semantically consistent (high similarity)

Parameters
----------

When using :class:`WhiteBoxUQ`, specify ``"consistency_and_confidence"`` in the ``scorers`` list.

Example
-------

.. code-block:: python

    from uqlm import WhiteBoxUQ

    # Initialize with consistency_and_confidence scorer
    wbuq = WhiteBoxUQ(
        llm=llm,
        scorers=["consistency_and_confidence"],
        sampling_temperature=1.0
    )

    # Generate responses and compute scores
    results = await wbuq.generate_and_score(prompts=prompts, num_responses=5)

    # Access the consistency_and_confidence scores
    print(results.to_df()["consistency_and_confidence"])

References
----------

- Vashurin, R., et al. (2025). `CoCoA: Towards Efficient Multi-Criteria Conformal Calibration of Large Language Models <https://arxiv.org/abs/2502.04964>`_. *arXiv*.

See Also
--------

- :class:`WhiteBoxUQ` - Main class for white-box uncertainty quantification
- :doc:`monte_carlo_probability` - Alternative multi-generation scorer
- :doc:`/scorer_definitions/black_box/cosine_sim` - Black-box cosine similarity scorer


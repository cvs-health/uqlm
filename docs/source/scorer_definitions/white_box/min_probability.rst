Minimum Token Probability
=========================

.. currentmodule:: uqlm.scorers

``min_probability``

Minimum Token Probability (MTP) uses the minimum among token probabilities for a given response
as a confidence score.

Definition
----------

Minimum token probability is defined as:

.. math::

    MTP(y_i) = \min_{t \in y_i} p_t

where :math:`t` iterates over all tokens in response :math:`y_i` and :math:`p_t` denotes the token probability.

**Key Properties:**

- Identifies the "weakest link" - the least confident token in the response
- Robust to long responses with mostly high-confidence tokens
- Score range: :math:`[0, 1]`

How It Works
------------

1. Generate a response with logprobs enabled
2. Extract the probability for each token in the response
3. Return the minimum probability across all tokens

This scorer is particularly useful for detecting responses where the model is uncertain about
specific parts, even if most of the response is generated with high confidence.

Parameters
----------

When using :class:`WhiteBoxUQ`, specify ``"min_probability"`` in the ``scorers`` list.

Example
-------

.. code-block:: python

    from uqlm import WhiteBoxUQ

    # Initialize with min_probability scorer
    wbuq = WhiteBoxUQ(
        llm=llm,
        scorers=["min_probability"]
    )

    # Generate responses and compute scores
    results = await wbuq.generate_and_score(prompts=prompts)

    # Access the min_probability scores
    print(results.to_df()["min_probability"])

References
----------

- Manakul, P., et al. (2023). `SelfCheckGPT: Zero-Resource Black-Box Hallucination Detection for Generative Large Language Models <https://arxiv.org/abs/2303.08896>`_. *arXiv*.

See Also
--------

- :class:`WhiteBoxUQ` - Main class for white-box uncertainty quantification
- :doc:`normalized_probability` - Geometric mean of token probabilities
- :doc:`sequence_probability` - Joint probability of all tokens


Length-Normalized Sequence Probability
======================================

.. currentmodule:: uqlm.scorers

``normalized_probability``

Length-Normalized Token Probability (LNTP) computes a length-normalized analog of joint token probability,
making it invariant to response length.

Definition
----------

Length-normalized token (sequence) probability computes a length-normalized analog of joint token probability:

.. math::

    LNTP(y_i) = \prod_{t \in y_i} p_t^{\frac{1}{L_i}}

where :math:`L_i` is the number of tokens in response :math:`y_i`.

**Key Properties:**

- Equivalent to the **geometric mean** of token probabilities for response :math:`y_i`
- Length-invariant, making it suitable for comparing responses of different lengths
- Score range: :math:`[0, 1]`

How It Works
------------

1. Generate a response with logprobs enabled
2. Extract the probability for each token in the response
3. Compute the geometric mean of all token probabilities

This normalization addresses the issue that sequence probability decreases with response length,
allowing fair comparison across responses of varying lengths.

Parameters
----------

.. note::

    **Deprecated.** This scorer is deprecated as of v0.5. Use ``sequence_probability`` with
    ``length_normalize=True`` (the default) instead. Specifying ``"normalized_probability"`` in the
    ``scorers`` list will raise a :class:`ValueError`.

Example
-------

Use :doc:`sequence_probability` with the default ``length_normalize=True`` for the same behavior:

.. code-block:: python

    from uqlm import WhiteBoxUQ

    # Use sequence_probability (length-normalized by default) instead of normalized_probability
    wbuq = WhiteBoxUQ(
        llm=llm,
        scorers=["sequence_probability"]
    )

    # Generate responses and compute scores
    results = await wbuq.generate_and_score(prompts=prompts)

    # Access the sequence_probability scores (equivalent to former normalized_probability)
    print(results.to_df()["sequence_probability"])

References
----------

- Malinin, A. & Gales, M. (2021). `Uncertainty Estimation in Autoregressive Structured Prediction <https://arxiv.org/abs/2002.07650>`_. *arXiv*.

See Also
--------

- :class:`WhiteBoxUQ` - Main class for white-box uncertainty quantification
- :doc:`sequence_probability` - Sequence probability (length-normalized by default; replaces this scorer)
- :doc:`min_probability` - Minimum token probability across the response


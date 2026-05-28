Sequence Probability
====================

.. currentmodule:: uqlm.scorers

``sequence_probability``

Sequence Probability (SP) computes a probability measure over the tokens in the generated response.
By default it is length-normalized (geometric mean of token probabilities); with
``length_normalize=False`` it is the raw joint probability.

Definition
----------

**Default (length-normalized):** With :class:`WhiteBoxUQ` default ``length_normalize=True``, sequence
probability is the geometric mean of token probabilities (length-normalized token probability, LNTP):

.. math::

    SP(y_i) = \prod_{t \in y_i} p_t^{\frac{1}{L_i}}

where :math:`L_i` is the number of tokens in response :math:`y_i` and :math:`p_t` denotes the token
probability. This is length-invariant and in :math:`[0, 1]`.

**Non-normalized:** With ``length_normalize=False``, sequence probability is the joint probability:

.. math::

    SP(y_i) = \prod_{t \in y_i} p_t

which tends to decrease with longer responses and is typically very small for longer sequences.

**Key Properties:**

- Direct measure of how likely the model considers its own output
- Default is length-normalized (geometric mean) for fair comparison across response lengths
- Score range: :math:`[0, 1]`

How It Works
------------

1. Generate a response with logprobs enabled
2. Extract the probability for each token in the response
3. With default ``length_normalize=True``, compute the geometric mean of token probabilities; otherwise compute the product

Parameters
----------

When using :class:`WhiteBoxUQ`, specify ``"sequence_probability"`` in the ``scorers`` list. Use
``length_normalize=False`` when you want the raw joint probability instead of the default
length-normalized form.

Example
-------

.. code-block:: python

    from uqlm import WhiteBoxUQ

    # Initialize with sequence_probability scorer
    wbuq = WhiteBoxUQ(
        llm=llm,
        scorers=["sequence_probability"]
    )

    # Generate responses and compute scores
    results = await wbuq.generate_and_score(prompts=prompts)

    # Access the sequence_probability scores
    print(results.to_df()["sequence_probability"])

References
----------

- Vashurin, R., et al. (2024). `Benchmarking Uncertainty Quantification Methods for Large Language Models with LM-Polygraph <https://arxiv.org/abs/2406.15627>`_. *arXiv*.

See Also
--------

- :class:`WhiteBoxUQ` - Main class for white-box uncertainty quantification
- :doc:`min_probability` - Minimum token probability across the response
- :doc:`monte_carlo_probability` - Multi-generation average of length-normalized sequence probability


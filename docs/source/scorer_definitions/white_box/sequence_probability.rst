Sequence Probability
====================

.. currentmodule:: uqlm.scorers

``sequence_probability``

Sequence Probability (SP) computes the joint probability of all tokens in the generated response.

Definition
----------

Sequence probability is the joint probability of all tokens:

.. math::

    SP(y_i) = \prod_{t \in y_i} p_t

where :math:`p_t` denotes the token probability for token :math:`t`.

**Key Properties:**

- Direct measure of how likely the model considers its own output
- Not length-normalized, so tends to decrease with longer responses
- Score range: :math:`[0, 1]` but typically very small for longer sequences

How It Works
------------

1. Generate a response with logprobs enabled
2. Extract the probability for each token in the response
3. Multiply all token probabilities together

Note that due to the multiplicative nature, sequence probability decreases rapidly with response
length. For length-invariant scoring, consider :doc:`normalized_probability`.

Parameters
----------

When using :class:`WhiteBoxUQ`, specify ``"sequence_probability"`` in the ``scorers`` list.

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

- Vashurin, R., et al. (2024). `Benchmarking LLM Uncertainty Quantification Methods for Agentic AI <https://arxiv.org/abs/2406.15627>`_. *arXiv*.

See Also
--------

- :class:`WhiteBoxUQ` - Main class for white-box uncertainty quantification
- :doc:`normalized_probability` - Length-normalized version of sequence probability
- :doc:`min_probability` - Minimum token probability across the response


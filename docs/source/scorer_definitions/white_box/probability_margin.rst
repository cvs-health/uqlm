Probability Margin
==================

.. currentmodule:: uqlm.scorers

``probability_margin``

Probability Margin (PM) computes the average difference between the top two token probabilities
for each token in the response.

Definition
----------

This scorer requires accessing the top-K logprobs per token (K ≥ 2). Let the top-K token probabilities
for token :math:`t_j` be denoted as :math:`\{p_{t_{jk}}\}_{k=1}^K`, ordered by decreasing probability.

Probability Margin is defined as:

.. math::

    PM(y_i) = \frac{1}{L_i}\sum_{j=1}^{L_i} (p_{t_{j1}} - p_{t_{j2}})

where :math:`p_{t_{j1}}` is the probability of the selected (top) token and :math:`p_{t_{j2}}` is
the probability of the second-best token.

**Key Properties:**

- Measures how "decisively" the model chose each token
- Large margins indicate clear preference; small margins indicate ambiguity
- Score range: :math:`[0, 1]`

How It Works
------------

1. Generate a response with top-K logprobs enabled (K ≥ 2)
2. For each token position:

   - Get the probabilities of the top two candidate tokens
   - Compute the difference (margin) between them

3. Average the margins across all token positions

A high probability margin indicates that the model was confident in its token choices throughout
the response, while a low margin suggests the model was uncertain between alternatives.

Parameters
----------

When using :class:`WhiteBoxUQ`, specify ``"probability_margin"`` in the ``scorers`` list.

.. note::

    This scorer requires the LLM to support returning top-K logprobs (e.g., OpenAI models with ``top_logprobs`` parameter).

Example
-------

.. code-block:: python

    from uqlm import WhiteBoxUQ

    # Initialize with probability_margin scorer
    wbuq = WhiteBoxUQ(
        llm=llm,
        scorers=["probability_margin"],
        top_k_logprobs=5  # At least 2 required
    )

    # Generate responses and compute scores
    results = await wbuq.generate_and_score(prompts=prompts)

    # Access the probability_margin scores
    print(results.to_df()["probability_margin"])

References
----------

- Farr, C., et al. (2024). `Measuring Uncertainty in Large Language Models through Semantic Embeddings <https://arxiv.org/abs/2408.08217>`_. *arXiv*.

See Also
--------

- :class:`WhiteBoxUQ` - Main class for white-box uncertainty quantification
- :doc:`mean_token_negentropy` - Alternative top-K based scorer
- :doc:`min_probability` - Simpler single-logprob scorer


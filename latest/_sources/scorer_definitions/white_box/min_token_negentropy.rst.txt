Minimum Token Negentropy
========================

.. currentmodule:: uqlm.scorers

``min_token_negentropy``

Minimum Token Negentropy (MinTN) uses the minimum among token-level negentropies for a given response
as a confidence score.

Definition
----------

This scorer requires accessing the top-K logprobs per token. Let the top-K token probabilities for
token :math:`t_j` be denoted as :math:`\{p_{t_{jk}}\}_{k=1}^K`.

First, we compute the Token Negentropy for each token position (see :doc:`mean_token_negentropy` for details):

.. math::

    TN@K(t_j) = 1 - \frac{TE@K(t_j)}{\log K}

where :math:`TE@K(t_j) = -\sum_{k=1}^{K} p_{t_{jk}} \log p_{t_{jk}}` is the Top-K Token Entropy.

Minimum Token Negentropy is then:

.. math::

    MinTN(y_i) = \min_{j \in \{1,...,L_i\}} TN@K(t_j)

**Key Properties:**

- Identifies the token position with highest uncertainty (lowest confidence)
- Acts as a "weakest link" detector for token-level confidence
- Score range: :math:`[0, 1]`

How It Works
------------

1. Generate a response with top-K logprobs enabled
2. For each token position:

   - Compute the entropy across the top-K candidate tokens
   - Normalize to get a negentropy (confidence) score

3. Return the minimum negentropy across all token positions

This scorer is useful for detecting responses where the model is uncertain about specific tokens,
even if most tokens are generated with high confidence.

Parameters
----------

When using :class:`WhiteBoxUQ`, specify ``"min_token_negentropy"`` in the ``scorers`` list.

.. note::

    This scorer requires the LLM to support returning top-K logprobs (e.g., OpenAI models with ``top_logprobs`` parameter).

Example
-------

.. code-block:: python

    from uqlm import WhiteBoxUQ

    # Initialize with min_token_negentropy scorer
    wbuq = WhiteBoxUQ(
        llm=llm,
        scorers=["min_token_negentropy"],
        top_k_logprobs=15  # Number of top logprobs to use
    )

    # Generate responses and compute scores
    results = await wbuq.generate_and_score(prompts=prompts)

    # Access the min_token_negentropy scores
    print(results.to_df()["min_token_negentropy"])

References
----------

- Scalena, D., et al. (2025). `TrustScore: Reference-Free Evaluation of LLM Response Trustworthiness <https://arxiv.org/abs/2510.11170>`_. *arXiv*.
- Manakul, P., et al. (2023). `SelfCheckGPT: Zero-Resource Black-Box Hallucination Detection for Generative Large Language Models <https://arxiv.org/abs/2303.08896>`_. *arXiv*.

See Also
--------

- :class:`WhiteBoxUQ` - Main class for white-box uncertainty quantification
- :doc:`mean_token_negentropy` - Average negentropy across all tokens
- :doc:`min_probability` - Minimum token probability (simpler alternative)


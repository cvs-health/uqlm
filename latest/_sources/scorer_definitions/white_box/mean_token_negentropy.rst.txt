Mean Token Negentropy
=====================

.. currentmodule:: uqlm.scorers

``mean_token_negentropy``

Mean Token Negentropy (MTN) computes the entropy of each token using the top-K logprobs, transforms
them to normalized negentropy scores, and averages these scores to obtain a confidence score for
each response.

Definition
----------

This scorer requires accessing the top-K logprobs per token. Let the top-K token probabilities for
token :math:`t_j` be denoted as :math:`\{p_{t_{jk}}\}_{k=1}^K`.

We first define Top-K Token Entropy for token :math:`j` as:

.. math::

    TE@K(t_j) = -\sum_{k=1}^{K} p_{t_{jk}} \log p_{t_{jk}}

The Token Negentropy (TN) transformation normalizes entropy to a confidence score in :math:`[0,1]`:

.. math::

    TN@K(t_j) = 1 - \frac{TE@K(t_j)}{\log K}

Finally, Mean Token Negentropy is the simple average across all tokens:

.. math::

    MTN(y_i) = \frac{1}{L_i}\sum_{j=1}^{L_i} TN@K(t_j)

**Key Properties:**

- Higher values indicate lower entropy (higher confidence)
- Uses top-K logprobs to estimate uncertainty at each token position
- Score range: :math:`[0, 1]`

How It Works
------------

1. Generate a response with top-K logprobs enabled
2. For each token position:

   - Compute the entropy across the top-K candidate tokens
   - Normalize to get a negentropy (confidence) score

3. Average the negentropy scores across all token positions

Parameters
----------

When using :class:`WhiteBoxUQ`, specify ``"mean_token_negentropy"`` in the ``scorers`` list.

.. note::

    This scorer requires the LLM to support returning top-K logprobs (e.g., OpenAI models with ``top_logprobs`` parameter).

Example
-------

.. code-block:: python

    from uqlm import WhiteBoxUQ

    # Initialize with mean_token_negentropy scorer
    wbuq = WhiteBoxUQ(
        llm=llm,
        scorers=["mean_token_negentropy"],
        top_k_logprobs=15  # Number of top logprobs to use
    )

    # Generate responses and compute scores
    results = await wbuq.generate_and_score(prompts=prompts)

    # Access the mean_token_negentropy scores
    print(results.to_df()["mean_token_negentropy"])

References
----------

- Scalena, D., et al. (2025). `TrustScore: Reference-Free Evaluation of LLM Response Trustworthiness <https://arxiv.org/abs/2510.11170>`_. *arXiv*.
- Manakul, P., et al. (2023). `SelfCheckGPT: Zero-Resource Black-Box Hallucination Detection for Generative Large Language Models <https://arxiv.org/abs/2303.08896>`_. *arXiv*.
- Bouchard, D. & Chauhan, M. S. (2025). `Generalized Ensembles for Robust Uncertainty Quantification of LLMs <https://arxiv.org/abs/2504.19254>`_. *arXiv*.

See Also
--------

- :class:`WhiteBoxUQ` - Main class for white-box uncertainty quantification
- :doc:`min_token_negentropy` - Minimum negentropy across all tokens
- :doc:`probability_margin` - Difference between top-2 token probabilities


Semantic Negentropy (Token-Probability-Based)
=============================================

.. currentmodule:: uqlm.scorers

``semantic_negentropy`` (via WhiteBoxUQ)

Token-probability-based Semantic Negentropy extends the discrete semantic entropy approach by using
token probabilities to weight the cluster probabilities, providing a more nuanced uncertainty estimate.

Definition
----------

Under this approach, responses are clustered using an NLI model based on mutual entailment. After
obtaining the set of clusters :math:`\mathcal{C}`, semantic entropy is computed as:

.. math::

    SE(y_i; \tilde{\mathbf{y}}_i) = - \sum_{C \in \mathcal{C}} P(C|y_i, \tilde{\mathbf{y}}_i)\log P(C|y_i, \tilde{\mathbf{y}}_i)

where :math:`P(C|y_i, \tilde{\mathbf{y}}_i)` is calculated as the **average across response-level
sequence probabilities** (normalized or otherwise), rather than uniform probabilities as in the
discrete version.

Normalized Semantic Negentropy (NSN) is then:

.. math::

    NSN(y_i; \tilde{\mathbf{y}}_i) = 1 - \frac{SE(y_i; \tilde{\mathbf{y}}_i)}{\log m}

where :math:`\log m` normalizes the support.

**Key Differences from Discrete Version:**

- Uses token probabilities to weight each response's contribution to its cluster
- Clusters with high-probability responses have larger weights
- Provides finer-grained uncertainty estimation

How It Works
------------

1. Generate multiple responses with logprobs enabled from the same prompt
2. Use an NLI model to cluster semantically equivalent responses
3. Compute each response's probability from its token logprobs
4. Weight cluster probabilities by the sum of response probabilities within each cluster
5. Compute entropy and normalize to get a confidence score

Parameters
----------

When using :class:`WhiteBoxUQ`, specify ``"semantic_negentropy"`` in the ``scorers`` list.

Example
-------

.. code-block:: python

    from uqlm import WhiteBoxUQ

    # Initialize with semantic_negentropy scorer (token-probability-based)
    wbuq = WhiteBoxUQ(
        llm=llm,
        scorers=["semantic_negentropy"],
        sampling_temperature=1.0,
        length_normalize=True
    )

    # Generate responses and compute scores
    results = await wbuq.generate_and_score(prompts=prompts, num_responses=5)

    # Access the semantic_negentropy scores
    print(results.to_df()["semantic_negentropy"])

References
----------

- Farquhar, S., et al. (2024). `Detecting hallucinations in large language models using semantic entropy <https://www.nature.com/articles/s41586-024-07421-0>`_. *Nature*.
- Kuhn, L., et al. (2023). `Semantic Uncertainty: Linguistic Invariances for Uncertainty Estimation in Natural Language Generation <https://arxiv.org/abs/2302.09664>`_. *arXiv*.
- Bouchard, D. & Chauhan, M. S. (2025). `Generalized Ensembles for Robust Uncertainty Quantification of LLMs <https://arxiv.org/abs/2504.19254>`_. *arXiv*.

See Also
--------

- :class:`WhiteBoxUQ` - Main class for white-box uncertainty quantification
- :class:`SemanticEntropy` - Dedicated class for semantic entropy computation
- :doc:`/scorer_definitions/black_box/semantic_negentropy` - Discrete (black-box) version
- :doc:`semantic_density` - Related semantic-based scorer


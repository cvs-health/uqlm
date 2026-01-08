Normalized Semantic Negentropy
==============================

.. currentmodule:: uqlm.scorers

``semantic_negentropy``

Normalized Semantic Negentropy (NSN) normalizes the standard computation of discrete semantic entropy
to be increasing with higher confidence and have :math:`[0,1]` support.

Definition
----------

In contrast to exact match and non-contradiction scorers, semantic entropy does not distinguish between
an original response and candidate responses. Instead, this approach computes a single metric value on
a list of responses generated from the same prompt.

Under this approach, responses are clustered using an NLI model based on mutual entailment.
The discrete version of Semantic Entropy (SE) is defined as:

.. math::

    SE(y_i; \tilde{\mathbf{y}}_i) = - \sum_{C \in \mathcal{C}} P(C|y_i, \tilde{\mathbf{y}}_i)\log P(C|y_i, \tilde{\mathbf{y}}_i)

where :math:`P(C|y_i, \tilde{\mathbf{y}}_i)` is calculated as the probability a randomly selected
response :math:`y \in \{y_i\} \cup \tilde{\mathbf{y}}_i` belongs to cluster :math:`C`, and
:math:`\mathcal{C}` denotes the full set of clusters of :math:`\{y_i\} \cup \tilde{\mathbf{y}}_i`.

To ensure that we have a normalized confidence score with :math:`[0,1]` support and with higher values
corresponding to higher confidence, we implement the following normalization to arrive at
*Normalized Semantic Negentropy* (NSN):

.. math::

    NSN(y_i; \tilde{\mathbf{y}}_i) = 1 - \frac{SE(y_i; \tilde{\mathbf{y}}_i)}{\log m}

where :math:`\log m` is included to normalize the support.

How It Works
------------

1. Generate multiple responses :math:`\tilde{\mathbf{y}}_i` from the same prompt
2. Use an NLI model to cluster semantically equivalent responses based on mutual entailment
3. Compute the entropy of the cluster distribution
4. Normalize the entropy to obtain a confidence score in :math:`[0,1]`

Higher NSN values indicate that responses are more semantically consistent (fewer clusters),
suggesting higher confidence in the response.

Parameters
----------

When using :class:`BlackBoxUQ`, specify ``"semantic_negentropy"`` in the ``scorers`` list.

Example
-------

.. code-block:: python

    from uqlm import BlackBoxUQ

    # Initialize with semantic_negentropy scorer
    bbuq = BlackBoxUQ(
        llm=llm,
        scorers=["semantic_negentropy"],
        nli_model_name="microsoft/deberta-large-mnli"
    )

    # Generate responses and compute scores
    results = await bbuq.generate_and_score(prompts=prompts, num_responses=5)

    # Access the semantic_negentropy scores
    print(results.to_df()["semantic_negentropy"])

References
----------

- Farquhar, S., et al. (2024). `Detecting hallucinations in large language models using semantic entropy <https://www.nature.com/articles/s41586-024-07421-0>`_. *Nature*.
- Kuhn, L., et al. (2023). `Semantic Uncertainty: Linguistic Invariances for Uncertainty Estimation in Natural Language Generation <https://arxiv.org/abs/2302.09664>`_. *arXiv*.
- Bouchard, D. & Chauhan, M. S. (2025). `Generalized Ensembles for Robust Uncertainty Quantification of LLMs <https://arxiv.org/abs/2504.19254>`_. *arXiv*.

See Also
--------

- :class:`BlackBoxUQ` - Main class for black-box uncertainty quantification
- :class:`SemanticEntropy` - Dedicated class for semantic entropy computation
- :doc:`semantic_sets_confidence` - Related scorer based on number of semantic clusters


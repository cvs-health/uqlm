Entailment Probability
======================

.. currentmodule:: uqlm.scorers

``entailment``

Entailment Probability (EP) computes the mean entailment probability estimated by a natural language
inference (NLI) model.

Definition
----------

This score is formally defined as follows:

.. math::

    EP(y_i; \tilde{\mathbf{y}}_i) = \frac{1}{m} \sum_{j=1}^m\frac{p_{\text{entail}}(y_i, \tilde{y}_{ij}) + p_{\text{entail}}(\tilde{y}_{ij}, y_i)}{2}

where :math:`p_{\text{entail}}(y_i, \tilde{y}_{ij})` denotes the (asymmetric) entailment probability
estimated by the NLI model for response :math:`y_i` and candidate :math:`\tilde{y}_{ij}`.

**Key Properties:**

- The bidirectional averaging :math:`(p_{\text{entail}}(a, b) + p_{\text{entail}}(b, a))/2` accounts for the asymmetric nature of NLI
- Higher EP values indicate that the original response is more likely to be entailed by (and entail) the sampled responses
- Score range: :math:`[0, 1]` where 1 indicates strong mutual entailment

How It Works
------------

1. Generate multiple candidate responses :math:`\tilde{\mathbf{y}}_i` from the same prompt
2. For each pair of original response :math:`y_i` and candidate :math:`\tilde{y}_{ij}`:

   - Compute entailment probability in both directions using an NLI model
   - Average the bidirectional entailment probabilities

3. Average across all candidates to get the mean entailment probability

Parameters
----------

When using :class:`BlackBoxUQ`, specify ``"entailment"`` in the ``scorers`` list.

Example
-------

.. code-block:: python

    from uqlm import BlackBoxUQ

    # Initialize with entailment scorer
    bbuq = BlackBoxUQ(
        llm=llm,
        scorers=["entailment"],
        nli_model_name="microsoft/deberta-large-mnli"
    )

    # Generate responses and compute scores
    results = await bbuq.generate_and_score(prompts=prompts, num_responses=5)

    # Access the entailment scores
    print(results.to_df()["entailment"])

References
----------

- Chen, J. & Mueller, J. (2023). `Quantifying Uncertainty in Answers from any Language Model and Enhancing their Trustworthiness <https://arxiv.org/abs/2308.16175>`_. *arXiv*.
- Lin, Z., et al. (2024). `Generating with Confidence: Uncertainty Quantification for Black-box Large Language Models <https://arxiv.org/abs/2305.19187>`_. *arXiv*.

See Also
--------

- :class:`BlackBoxUQ` - Main class for black-box uncertainty quantification
- :doc:`noncontradiction` - Related scorer measuring non-contradiction probability


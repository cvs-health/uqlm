Non-Contradiction Probability
=============================

.. currentmodule:: uqlm.scorers

``noncontradiction``

Non-Contradiction Probability (NCP) computes the mean non-contradiction probability estimated by a
natural language inference (NLI) model.

Definition
----------

This score is formally defined as follows:

.. math::

    NCP(y_i; \tilde{\mathbf{y}}_i) = 1 - \frac{1}{m} \sum_{j=1}^m\frac{p_{\text{contra}}(y_i, \tilde{y}_{ij}) + p_{\text{contra}}(\tilde{y}_{ij}, y_i)}{2}

where :math:`p_{\text{contra}}(y_i, \tilde{y}_{ij})` denotes the (asymmetric) contradiction probability
estimated by the NLI model for response :math:`y_i` and candidate :math:`\tilde{y}_{ij}`.

**Key Properties:**

- The bidirectional averaging :math:`(p_{\text{contra}}(a, b) + p_{\text{contra}}(b, a))/2` accounts for the asymmetric nature of NLI
- Higher NCP values indicate that the original response is less likely to contradict the sampled responses
- Score range: :math:`[0, 1]` where 1 indicates no contradictions

How It Works
------------

1. Generate multiple candidate responses :math:`\tilde{\mathbf{y}}_i` from the same prompt
2. For each pair of original response :math:`y_i` and candidate :math:`\tilde{y}_{ij}`:

   - Compute contradiction probability in both directions using an NLI model
   - Average the bidirectional contradiction probabilities

3. Average across all candidates and subtract from 1 to get non-contradiction probability

Parameters
----------

When using :class:`BlackBoxUQ`, specify ``"noncontradiction"`` in the ``scorers`` list.

Example
-------

.. code-block:: python

    from uqlm import BlackBoxUQ

    # Initialize with noncontradiction scorer
    bbuq = BlackBoxUQ(
        llm=llm,
        scorers=["noncontradiction"],
        nli_model_name="microsoft/deberta-large-mnli"
    )

    # Generate responses and compute scores
    results = await bbuq.generate_and_score(prompts=prompts, num_responses=5)

    # Access the noncontradiction scores
    print(results.to_df()["noncontradiction"])

References
----------

- Chen, J. & Mueller, J. (2023). `Quantifying Uncertainty in Answers from any Language Model and Enhancing their Trustworthiness <https://arxiv.org/abs/2308.16175>`_. *arXiv*.
- Lin, Z., et al. (2024). `Generating with Confidence: Uncertainty Quantification for Black-box Large Language Models <https://arxiv.org/abs/2305.19187>`_. *arXiv*.
- Manakul, P., et al. (2023). `SelfCheckGPT: Zero-Resource Black-Box Hallucination Detection for Generative Large Language Models <https://arxiv.org/abs/2303.08896>`_. *arXiv*.

See Also
--------

- :class:`BlackBoxUQ` - Main class for black-box uncertainty quantification
- :doc:`entailment` - Related scorer measuring entailment probability


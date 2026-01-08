Binary Judge (True/False)
=========================

.. currentmodule:: uqlm.judges

``true_false``

The binary judge template instructs an LLM to classify a question-response as either *correct* or
*incorrect*.

Definition
----------

This template modifies the ternary approach to include only two categories:

.. math::

    J(y_i) = \begin{cases}
        0 & \text{LLM states response is incorrect} \\
        1 & \text{LLM states response is correct}
    \end{cases}

The judge function :math:`J: \mathcal{Y} \rightarrow \{0, 1\}` maps responses to binary scores.

**Key Properties:**

- Simpler binary classification without uncertain category
- Forces the judge to make a definitive decision
- Useful when you want clear-cut correct/incorrect labels

How It Works
------------

1. Present the judge LLM with the original question and response
2. Ask the judge to classify the response as "correct" or "incorrect"
3. Map the classification to a numerical score (1 or 0)

Use this template when you prefer binary decisions without an intermediate uncertainty category.

Parameters
----------

When using :class:`~uqlm.judges.LLMJudge` or :class:`~uqlm.scorers.LLMPanel`, specify
``scoring_template="true_false"``.

Example
-------

.. code-block:: python

    from uqlm.judges import LLMJudge

    # Initialize with binary template
    judge = LLMJudge(
        llm=judge_llm,
        scoring_template="true_false"
    )

    # Score responses
    result = await judge.judge_responses(
        prompts=prompts,
        responses=responses
    )

Using with LLMPanel:

.. code-block:: python

    from uqlm import LLMPanel

    # Create a panel with binary scoring
    panel = LLMPanel(
        llm=original_llm,
        judges=[judge_llm1, judge_llm2],
        scoring_templates=["true_false"] * 2
    )

    results = await panel.generate_and_score(prompts=prompts)

References
----------

- Chen, J. & Mueller, J. (2023). `Quantifying Uncertainty in Answers from any Language Model and Enhancing their Trustworthiness <https://arxiv.org/abs/2308.16175>`_. *arXiv*.
- Luo, H., et al. (2023). `ChatGPT as a Factual Inconsistency Evaluator for Text Summarization <https://arxiv.org/abs/2303.15621>`_. *arXiv*.

See Also
--------

- :class:`~uqlm.judges.LLMJudge` - Single LLM judge class
- :class:`~uqlm.scorers.LLMPanel` - Panel of multiple judges
- :doc:`true_false_uncertain` - Ternary scoring template with uncertainty


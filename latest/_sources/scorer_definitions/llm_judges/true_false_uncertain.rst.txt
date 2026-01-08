Ternary Judge (True/False/Uncertain)
====================================

.. currentmodule:: uqlm.judges

``true_false_uncertain``

The ternary judge template instructs an LLM to score a question-response concatenation as either
*incorrect*, *uncertain*, or *correct* using a carefully constructed prompt.

Definition
----------

We follow the approach proposed by Chen & Mueller (2023), where an LLM is instructed to score a
question-response as one of three categories. These categories are mapped to numerical scores:

.. math::

    J(y_i) = \begin{cases}
        0 & \text{LLM states response is incorrect} \\
        0.5 & \text{LLM states that it is uncertain} \\
        1 & \text{LLM states response is correct}
    \end{cases}

The judge function :math:`J: \mathcal{Y} \rightarrow \{0, 0.5, 1\}` maps responses to confidence scores.

**Key Properties:**

- Three-way classification allows expressing uncertainty
- Intermediate score (0.5) useful for ambiguous cases
- Can be used with self-judging or external judge LLMs

How It Works
------------

1. Present the judge LLM with the original question and response
2. Ask the judge to classify the response as "incorrect", "uncertain", or "correct"
3. Map the classification to a numerical score (0, 0.5, or 1)

The ternary format is the default template in UQLM and is recommended for most use cases where
distinguishing between definitely wrong, uncertain, and definitely correct responses is valuable.

Parameters
----------

When using :class:`~uqlm.judges.LLMJudge` or :class:`~uqlm.scorers.LLMPanel`, specify
``scoring_template="true_false_uncertain"``.

Example
-------

.. code-block:: python

    from uqlm.judges import LLMJudge

    # Initialize with ternary template (default)
    judge = LLMJudge(
        llm=judge_llm,
        scoring_template="true_false_uncertain"
    )

    # Score responses
    result = await judge.judge_responses(
        prompts=prompts,
        responses=responses
    )

Using with LLMPanel for multiple judges:

.. code-block:: python

    from uqlm import LLMPanel

    # Create a panel with multiple judges using ternary template
    panel = LLMPanel(
        llm=original_llm,
        judges=[judge_llm1, judge_llm2, judge_llm3],
        scoring_templates=["true_false_uncertain"] * 3
    )

    # Generate and score
    results = await panel.generate_and_score(prompts=prompts)

References
----------

- Chen, J. & Mueller, J. (2023). `Quantifying Uncertainty in Answers from any Language Model and Enhancing their Trustworthiness <https://arxiv.org/abs/2308.16175>`_. *arXiv*.
- Manakul, P., et al. (2023). `SelfCheckGPT: Zero-Resource Black-Box Hallucination Detection for Generative Large Language Models <https://arxiv.org/abs/2303.08896>`_. *arXiv*.

See Also
--------

- :class:`~uqlm.judges.LLMJudge` - Single LLM judge class
- :class:`~uqlm.scorers.LLMPanel` - Panel of multiple judges
- :doc:`true_false` - Binary (simpler) scoring template


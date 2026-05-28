Continuous Judge
================

.. currentmodule:: uqlm.judges

``continuous``

The continuous judge template instructs an LLM to directly score a question-response concatenation's
correctness on a scale of 0 to 1.

Definition
----------

For the continuous template, the LLM is asked to provide a numerical score:

.. math::

    J(y_i) \in [0, 1]

where 0 indicates completely incorrect and 1 indicates completely correct.

**Key Properties:**

- Fine-grained scoring without discrete categories
- Allows nuanced assessment of partial correctness
- Score range: :math:`[0, 1]` continuous

How It Works
------------

1. Present the judge LLM with the original question and response
2. Ask the judge to assign a correctness score between 0 and 1
3. Parse and return the numerical score

This template is useful when you want more granular assessments than binary or ternary classifications,
allowing the judge to express partial correctness (e.g., 0.7 for mostly correct responses).

Parameters
----------

When using :class:`~uqlm.judges.LLMJudge` or :class:`~uqlm.scorers.LLMPanel`, specify
``scoring_template="continuous"``.

Example
-------

.. code-block:: python

    from uqlm.judges import LLMJudge

    # Initialize with continuous template
    judge = LLMJudge(
        llm=judge_llm,
        scoring_template="continuous"
    )

    # Score responses
    result = await judge.judge_responses(
        prompts=prompts,
        responses=responses
    )

    # Scores will be continuous values between 0 and 1
    print(result["scores"])

Using with LLMPanel:

.. code-block:: python

    from uqlm import LLMPanel

    # Create a panel with continuous scoring
    panel = LLMPanel(
        llm=original_llm,
        judges=[judge_llm1, judge_llm2],
        scoring_templates=["continuous"] * 2
    )

    results = await panel.generate_and_score(prompts=prompts)

References
----------

- Xiong, M., et al. (2024). `Can LLMs Express Their Uncertainty? An Empirical Evaluation of Confidence Elicitation in LLMs <https://arxiv.org/abs/2306.13063>`_. *arXiv*.

See Also
--------

- :class:`~uqlm.judges.LLMJudge` - Single LLM judge class
- :class:`~uqlm.scorers.LLMPanel` - Panel of multiple judges
- :doc:`likert` - Structured 5-point scale alternative


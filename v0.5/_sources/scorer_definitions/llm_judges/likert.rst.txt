Likert Scale Judge
==================

.. currentmodule:: uqlm.judges

``likert``

The Likert scale judge template instructs an LLM to score a question-response on a 5-point scale,
which is then normalized to the :math:`[0, 1]` range.

Definition
----------

The judge is asked to score on a 5-point Likert scale, which is converted to normalized scores:

.. math::

    J(y_i) = \begin{cases}
        0 & \text{LLM states response is completely incorrect} \\
        0.25 & \text{LLM states response is mostly incorrect} \\
        0.5 & \text{LLM states response is partially correct} \\
        0.75 & \text{LLM states response is mostly correct} \\
        1 & \text{LLM states response is completely correct}
    \end{cases}

**Key Properties:**

- Structured 5-point scale familiar from survey research
- Balanced granularity between binary and continuous scoring
- Normalized to :math:`[0, 1]` for consistency with other scorers

How It Works
------------

1. Present the judge LLM with the original question and response
2. Ask the judge to rate on a 5-point scale:

   - 1: Completely incorrect
   - 2: Mostly incorrect
   - 3: Partially correct
   - 4: Mostly correct
   - 5: Completely correct

3. Normalize the score to :math:`[0, 1]` by mapping 1→0, 2→0.25, 3→0.5, 4→0.75, 5→1

The Likert scale provides more structure than continuous scoring while offering more granularity
than ternary classification.

Parameters
----------

When using :class:`~uqlm.judges.LLMJudge` or :class:`~uqlm.scorers.LLMPanel`, specify
``scoring_template="likert"``.

Example
-------

.. code-block:: python

    from uqlm.judges import LLMJudge

    # Initialize with likert template
    judge = LLMJudge(
        llm=judge_llm,
        scoring_template="likert"
    )

    # Score responses
    result = await judge.judge_responses(
        prompts=prompts,
        responses=responses
    )

    # Scores will be one of: 0, 0.25, 0.5, 0.75, 1
    print(result["scores"])

Using with LLMPanel:

.. code-block:: python

    from uqlm import LLMPanel

    # Create a panel with likert scoring
    panel = LLMPanel(
        llm=original_llm,
        judges=[judge_llm1, judge_llm2],
        scoring_templates=["likert"] * 2
    )

    results = await panel.generate_and_score(prompts=prompts)

References
----------

- Bai, Y., et al. (2023). `Benchmarking ChatGPT for Retrieving and Recommending Medical Information <https://arxiv.org/abs/2306.04181>`_. *arXiv*.

See Also
--------

- :class:`~uqlm.judges.LLMJudge` - Single LLM judge class
- :class:`~uqlm.scorers.LLMPanel` - Panel of multiple judges
- :doc:`continuous` - Continuous scoring alternative
- :doc:`true_false_uncertain` - Simpler 3-point classification


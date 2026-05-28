Panel of LLM Judges
===================

.. currentmodule:: uqlm.scorers

``LLMPanel``

The Panel of LLM Judges aggregates scores from multiple LLM judges using various aggregation methods
to provide a more robust confidence estimate.

Overview
--------

The :class:`LLMPanel` class coordinates multiple :class:`~uqlm.judges.LLMJudge` instances, allowing
you to leverage diverse LLM perspectives for improved evaluation reliability.

**Aggregation Methods:**

- **Average (``avg``):** Mean of all judge scores
- **Maximum (``max``):** Most optimistic judge assessment
- **Minimum (``min``):** Most conservative judge assessment
- **Median (``median``):** Middle value, robust to outliers

Definition
----------

Let :math:`J_1, J_2, ..., J_n` be :math:`n` judges and :math:`s_k = J_k(y_i)` be the score from
judge :math:`k` for response :math:`y_i`.

**Average:**

.. math::

    \text{Panel}_{avg}(y_i) = \frac{1}{n} \sum_{k=1}^n s_k

**Maximum:**

.. math::

    \text{Panel}_{max}(y_i) = \max_{k \in \{1,...,n\}} s_k

**Minimum:**

.. math::

    \text{Panel}_{min}(y_i) = \min_{k \in \{1,...,n\}} s_k

**Median:**

.. math::

    \text{Panel}_{median}(y_i) = \text{median}(s_1, s_2, ..., s_n)

How It Works
------------

1. Configure multiple LLM judges (can be different models or same model with different prompts)
2. For each response, obtain scores from all judges in the panel
3. Aggregate scores using your preferred method
4. Return individual judge scores and aggregated scores

Benefits of using a panel:

- **Diversity:** Different LLMs may catch different types of errors
- **Robustness:** Aggregation reduces impact of individual judge mistakes
- **Flexibility:** Mix models of different sizes and capabilities

Parameters
----------

The :class:`LLMPanel` class accepts:

- ``judges``: List of :class:`~uqlm.judges.LLMJudge` instances or :class:`~langchain_core.language_models.chat_models.BaseChatModel` instances
- ``scoring_templates``: List of scoring templates for each judge
- ``explanations``: Whether to include judge explanations

Example
-------

.. code-block:: python

    from uqlm import LLMPanel

    # Create a panel with multiple judges
    panel = LLMPanel(
        llm=original_llm,  # LLM to generate responses
        judges=[gpt4, claude, gemini],  # Panel of judge LLMs
        scoring_templates=["true_false_uncertain"] * 3,  # Same template for all
        explanations=True  # Include judge reasoning
    )

    # Generate responses and get panel scores
    results = await panel.generate_and_score(prompts=prompts)

    # Access aggregated scores
    df = results.to_df()
    print(df["avg"])     # Average of all judges
    print(df["median"])  # Median score
    print(df["min"])     # Most conservative
    print(df["max"])     # Most optimistic

    # Access individual judge scores
    print(df["judge_1"])
    print(df["judge_2"])
    print(df["judge_3"])

Mixed Templates Example:

.. code-block:: python

    from uqlm import LLMPanel

    # Use different templates for different judges
    panel = LLMPanel(
        llm=original_llm,
        judges=[gpt4, claude, gemini],
        scoring_templates=["true_false_uncertain", "continuous", "likert"]
    )

    results = await panel.generate_and_score(prompts=prompts)

References
----------

- Verga, P., et al. (2024). `Replacing Judges with Juries: Evaluating LLM Generations with a Panel of Diverse Models <https://arxiv.org/abs/2404.18796>`_. *arXiv*.

See Also
--------

- :class:`LLMPanel` - Main panel class documentation
- :class:`~uqlm.judges.LLMJudge` - Individual judge class
- :doc:`true_false_uncertain` - Default scoring template


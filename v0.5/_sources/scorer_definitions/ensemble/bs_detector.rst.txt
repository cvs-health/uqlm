BS Detector
===========

.. currentmodule:: uqlm.scorers

The BS Detector is an off-the-shelf ensemble method proposed by Chen & Mueller (2023) that combines
three components: exact match rate, non-contradiction probability, and LLM-as-a-Judge (self-reflection).

Definition
----------

The BS Detector ensemble score is computed as:

.. math::

    \text{BSDetector}(y_i) = w_1 \cdot \text{NCP}(y_i) + w_2 \cdot \text{EMR}(y_i) + w_3 \cdot J(y_i)

where:

- :math:`\text{NCP}(y_i)` is the Non-Contradiction Probability score
- :math:`\text{EMR}(y_i)` is the Exact Match Rate score
- :math:`J(y_i)` is the LLM-as-a-Judge (self-reflection) score

**Default Weights:**

The default weights from Chen & Mueller (2023) are:

- :math:`w_1 = 0.56` (non-contradiction, 80% of black-box weight)
- :math:`w_2 = 0.14` (exact match, 20% of black-box weight)
- :math:`w_3 = 0.30` (self-judge)

.. note::

    The black-box components (NCP + EMR) receive 70% of the total weight, while the self-judge
    receives 30%.

Components
----------

**1. Non-Contradiction Probability (56%)**

Uses an NLI model to measure semantic consistency. See :doc:`/scorer_definitions/black_box/noncontradiction`.

**2. Exact Match Rate (14%)**

Simple string matching for identical responses. See :doc:`/scorer_definitions/black_box/exact_match`.

**3. Self-Reflection Judge (30%)**

Uses the same LLM to evaluate its own response. See :doc:`/scorer_definitions/llm_judges/true_false_uncertain`.

How It Works
------------

1. Generate multiple candidate responses from the same prompt
2. Compute NCP and EMR using the candidates
3. Use the LLM as a self-judge to evaluate the original response
4. Combine scores using the weighted average

The combination of consistency-based (NCP, EMR) and self-reflection (judge) signals provides
robust hallucination detection across diverse question types.

Parameters
----------

When using :class:`UQEnsemble` without specifying ``scorers``, the BS Detector configuration is
used by default.

Example
-------

.. code-block:: python

    from uqlm import UQEnsemble

    # BS Detector is the default when no scorers specified
    bsd = UQEnsemble(llm=llm)

    # Generate and score
    results = await bsd.generate_and_score(prompts=prompts, num_responses=5)

    # Access ensemble scores
    df = results.to_df()
    print(df["ensemble_scores"])

    # Individual component scores are also available
    print(df["noncontradiction"])
    print(df["exact_match"])
    print(df["judge_1"])

Custom Weights:

.. code-block:: python

    from uqlm import UQEnsemble

    # Modify weights
    bsd = UQEnsemble(
        llm=llm,
        scorers=["noncontradiction", "exact_match", llm],  # Same components
        weights=[0.5, 0.2, 0.3]  # Custom weights
    )

    results = await bsd.generate_and_score(prompts=prompts)

References
----------

- Chen, J. & Mueller, J. (2023). `Quantifying Uncertainty in Answers from any Language Model and Enhancing their Trustworthiness <https://arxiv.org/abs/2308.16175>`_. *arXiv*.

See Also
--------

- :class:`UQEnsemble` - Main ensemble class
- :doc:`generalized_ensemble` - Flexible ensemble with custom components
- :doc:`/scorer_definitions/black_box/noncontradiction` - NCP scorer
- :doc:`/scorer_definitions/black_box/exact_match` - EMR scorer


Generalized Ensemble
====================

.. currentmodule:: uqlm.scorers

The Generalized Ensemble allows you to create custom combinations of any black-box, white-box, and
LLM-as-a-Judge scorers with configurable weights.

Definition
----------

Given a set of :math:`n` component scorers with scores :math:`s_1, s_2, ..., s_n` and weights
:math:`w_1, w_2, ..., w_n`, the ensemble score is:

.. math::

    \text{Ensemble}(y_i) = \sum_{k=1}^n w_k \cdot s_k(y_i)

where weights are normalized such that :math:`\sum_{k=1}^n w_k = 1`.

**Weight Tuning:**

Weights can be optimized using labeled data:

.. math::

    \mathbf{w}^* = \arg\max_{\mathbf{w}} \text{Objective}(\text{Ensemble}_{\mathbf{w}}, \mathbf{y}_{true})

where the objective can be ROC-AUC, F1-score, accuracy, or other classification metrics.

Available Components
--------------------

The generalized ensemble can include any combination of:

**Black-Box Scorers:**

- ``semantic_negentropy``
- ``semantic_sets_confidence``
- ``noncontradiction``
- ``entailment``
- ``exact_match``
- ``bert_score``
- ``cosine_sim``

**White-Box Scorers:**

- ``normalized_probability``
- ``min_probability``

**LLM-as-a-Judge:**

- Any :class:`~langchain_core.language_models.chat_models.BaseChatModel` instance
- Any :class:`~uqlm.judges.LLMJudge` instance

How It Works
------------

1. Specify the components to include in the ensemble
2. Optionally specify initial weights (defaults to equal weights)
3. Generate responses and compute all component scores
4. Combine using weighted average
5. Optionally tune weights on labeled data

Weight Tuning Methods
---------------------

:class:`UQEnsemble` supports automatic weight tuning using:

- **Optuna optimization:** Bayesian optimization over weight space
- **Grid search:** For threshold optimization

**Supported Objectives:**

- ``roc_auc``: Area under ROC curve (default for weights)
- ``fbeta_score``: F-beta score (default for threshold, uses F1 when beta=1)
- ``accuracy_score``: Classification accuracy
- ``balanced_accuracy_score``: Balanced accuracy for imbalanced data
- ``log_loss``: Logarithmic loss
- ``average_precision``: Average precision score
- ``brier_score``: Brier score

Example
-------

Basic custom ensemble:

.. code-block:: python

    from uqlm import UQEnsemble

    # Create ensemble with custom components
    ensemble = UQEnsemble(
        llm=llm,
        scorers=[
            "semantic_negentropy",
            "noncontradiction",
            "cosine_sim",
            judge_llm  # LLM-as-a-Judge component
        ],
        weights=[0.3, 0.3, 0.2, 0.2]  # Custom weights
    )

    # Generate and score
    results = await ensemble.generate_and_score(prompts=prompts, num_responses=5)
    print(results.to_df()["ensemble_scores"])

Weight tuning example:

.. code-block:: python

    from uqlm import UQEnsemble

    # Initialize ensemble (weights will be tuned)
    ensemble = UQEnsemble(
        llm=llm,
        scorers=["semantic_negentropy", "noncontradiction", llm]
    )

    # Tune weights using labeled data
    results = await ensemble.tune(
        prompts=prompts,
        ground_truth_answers=answers,
        num_responses=5,
        weights_objective="roc_auc",
        thresh_objective="fbeta_score",
        n_trials=100
    )

    # View optimized weights
    ensemble.print_ensemble_weights()

    # Save configuration for later use
    ensemble.save_config("my_ensemble_config.json")

Loading a saved configuration:

.. code-block:: python

    from uqlm import UQEnsemble

    # Load previously tuned ensemble
    ensemble = UQEnsemble.load_config("my_ensemble_config.json", llm=llm)

    # Use with new data
    results = await ensemble.generate_and_score(prompts=new_prompts)

References
----------

- Bouchard, D. & Chauhan, M. S. (2025). `Generalized Ensembles for Robust Uncertainty Quantification of LLMs <https://arxiv.org/abs/2504.19254>`_. *arXiv*.

See Also
--------

- :class:`UQEnsemble` - Main ensemble class
- :doc:`bs_detector` - Pre-configured BS Detector ensemble
- :class:`~uqlm.utils.Tuner` - Weight optimization utilities


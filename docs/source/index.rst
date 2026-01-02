.. image:: ./_static/images/uqlm_flow_ds.png
   :class: only-light no-scaled-link responsive-img
   :align: center

.. image:: ./_static/images/uqlm_flow_ds_dark.png
   :class: only-dark no-scaled-link responsive-img
   :align: center

uqlm: Uncertainty Quantification for Language Models
====================================================

:doc:`Get Started → <getstarted>` | :doc:`View Examples → <_notebooks/index>`

UQLM is a Python library for Large Language Model (LLM) hallucination detection using state-of-the-art uncertainty quantification techniques.


Hallucination Detection
-----------------------

UQLM provides a suite of response-level scorers for quantifying the uncertainty of Large Language Model (LLM) outputs. Each scorer returns a confidence score between 0 and 1, where higher scores indicate a lower likelihood of errors or hallucinations.  We categorize these scorers into four main types:

.. list-table:: Comparison of Scorer Types
   :header-rows: 1
   :widths: 20 20 20 20 20

   * - Scorer Type
     - Added Latency
     - Added Cost
     - Compatibility
     - Off-the-Shelf / Effort
   * - :ref:`Black-Box Scorers <black-box-scorers>`
     - ⏱️ Medium-High (multiple generations & comparisons)
     - 💸 High (multiple LLM calls)
     - 🌍 Universal (works with any LLM)
     - ✅ Off-the-shelf
   * - :ref:`White-Box Scorers <white-box-scorers>`
     - ⚡ Minimal (token probabilities already returned)
     - ✔️ None (no extra LLM calls)
     - 🔒 Limited (requires access to token probabilities)
     - ✅ Off-the-shelf
   * - :ref:`LLM-as-a-Judge Scorers <llm-as-a-judge-scorers>`
     - ⏳ Low-Medium (additional judge calls add latency)
     - 💵 Low-High (depends on number of judges)
     - 🌍 Universal (any LLM can serve as judge)
     - ✅ Off-the-shelf; Can be customized
   * - :ref:`Ensemble Scorers <ensemble-scorers>`
     - 🔀 Flexible (combines various scorers)
     - 🔀 Flexible (combines various scorers)
     - 🔀 Flexible (combines various scorers)
     - ✅ Off-the-shelf (beginner-friendly); 🛠️ Can be tuned (best for advanced users)


.. _black-box-scorers:

1. Black-Box Scorers(Consistency-Based)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. image:: ./_static/images/black_box_graphic.png
   :class: only-light no-scaled-link responsive-img
   :align: center

.. image:: ./_static/images/black_box_graphic_dark.png
   :class: only-dark no-scaled-link responsive-img
   :align: center

These scorers assess uncertainty by measuring the consistency of multiple responses generated from the same prompt. They are compatible with any LLM, intuitive to use, and don't require access to internal model states or token probabilities.

  * Discrete Semantic Entropy (`Farquhar et al., 2024 <https://www.nature.com/articles/s41586-024-07421-0>`_; `Kuh et al., 2023 <https://arxiv.org/pdf/2302.09664>`_)

  * Number of Semantic Sets (`Lin et al., 2024 <https://arxiv.org/abs/2305.19187>`_; `Vashurin et al., 2025  <https://arxiv.org/abs/2406.15627>`_; `Kuhn et al., 2023 <https://arxiv.org/pdf/2302.09664>`_)

  * Non-Contradiction Probability (`Chen & Mueller, 2023 <https://arxiv.org/abs/2308.16175>`_; `Lin et al., 2025 <https://arxiv.org/abs/2305.19187>`_; `Manakul et al., 2023 <https://arxiv.org/abs/2303.08896>`_)

  * Entailment Probability (`Chen & Mueller, 2023 <https://arxiv.org/abs/2308.16175>`_; `Lin et al., 2025 <https://arxiv.org/abs/2305.19187>`_; `Manakul et al., 2023 <https://arxiv.org/abs/2303.08896>`_)

  * Exact Match (`Cole et al., 2023 <https://arxiv.org/abs/2305.14613>`_; `Chen & Mueller, 2023 <https://arxiv.org/abs/2308.16175>`_)

  * BERT-score (`Manakul et al., 2023 <https://arxiv.org/abs/2303.08896>`_; `Zheng et al., 2020 <https://arxiv.org/abs/1904.09675>`_)

  * Cosine Similarity (`Shorinwa et al., 2024 <https://arxiv.org/pdf/2412.05563>`_; `HuggingFace <https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2>`_)


.. _white-box-scorers:

2. White-Box Scorers(Token-Probability-Based)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. image:: ./_static/images/white_box_graphic.png
   :class: only-light no-scaled-link responsive-img
   :align: center

.. image:: ./_static/images/white_box_graphic_dark.png
   :class: only-dark no-scaled-link responsive-img
   :align: center

These scorers leverage token probabilities to estimate uncertainty.  They offer single-generation scoring, which is significantly faster and cheaper than black-box methods, but require access to the LLM's internal probabilities, meaning they are not necessarily compatible with all LLMs/APIs. The following single-generation scorers are available:

  * Minimum token probability (`Manakul et al., 2023 <https://arxiv.org/abs/2303.08896>`_)

  * Length-Normalized Joint Token Probability (`Malinin & Gales, 2021 <https://arxiv.org/pdf/2002.07650>`_)

  * Sequence Probability (`Vashurin et al., 2024 <https://arxiv.org/abs/2406.15627>`_)
  
  * Mean Top-K Token Negentropy (`Scalena et al., 2025 <https://arxiv.org/abs/2510.11170>`_; `Manakul et al., 2023 <https://arxiv.org/abs/2303.08896>`_)
  
  * Min Top-K Token Negentropy (`Scalena et al., 2025 <https://arxiv.org/abs/2510.11170>`_; `Manakul et al., 2023 <https://arxiv.org/abs/2303.08896>`_)
  
  * Probability Margin (`Farr et al., 2024 <https://arxiv.org/abs/2408.08217>`_)
  
UQLM also offers sampling-based white-box methods, which incur higher cost and latency, but tend have superior hallucination detection performance. The following sampling-based white-box scorers are available:

  * Monte carlo sequence probability (`Kuhn et al., 2023 <https://arxiv.org/abs/2302.09664>`_)
  
  * Consistency and Confidence (CoCoA) (`Vashurin et al., 2025 <https://arxiv.org/abs/2502.04964>`_)
  
  * Semantic Entropy (`Farquhar et al., 2024 <https://www.nature.com/articles/s41586-024-07421-0>`_)
  
  * Semantic Density (`Qiu et al., 2024 <https://arxiv.org/abs/2405.13845>`_)
  
Lastly, the P(True) scorer is offered, which is a self-reflection method that requires one additional generation per response. 

  * P(True) (`Kadavath et al., 2022 <https://arxiv.org/abs/2207.05221>`_)

.. _llm-as-a-judge-scorers:

3. LLM-as-a-Judge scorers
^^^^^^^^^^^^^^^^^^^^^^^^^

.. image:: ./_static/images/judges_graphic.png
   :class: only-light no-scaled-link responsive-img
   :align: center

.. image:: ./_static/images/judges_graphic_dark.png
   :class: only-dark no-scaled-link responsive-img
   :align: center

These scorers use one or more LLMs to evaluate the reliability of the original LLM's response. They offer high customizability through prompt engineering and the choice of judge LLM(s).

  * Categorical LLM-as-a-Judge (`Manakul et al., 2023 <https://arxiv.org/abs/2303.08896>`_; `Chen & Mueller, 2023 <https://arxiv.org/abs/2308.16175>`_; `Luo et al., 2023 <https://arxiv.org/pdf/2303.15621>`_)

  * Continuous LLM-as-a-Judge (`Xiong et al., 2024 <https://arxiv.org/pdf/2306.13063>`_)

  * Likert Scale Scoring (`Bai et al., 2023 <https://arxiv.org/pdf/2306.04181>`_)

  * Panel of LLM Judges (`Verga et al., 2024 <https://arxiv.org/abs/2404.18796>`_)


.. _ensemble-scorers:

4. Ensemble scorers
^^^^^^^^^^^^^^^^^^^

.. image:: ./_static/images/uqensemble_generate_score.png
   :class: only-light no-scaled-link responsive-img
   :align: center

.. image:: ./_static/images/uqensemble_generate_score_dark.png
   :class: only-dark no-scaled-link responsive-img
   :align: center

These scorers leverage a weighted average of multiple individual scorers to provide a more robust uncertainty/confidence estimate. They offer high flexibility and customizability, allowing you to tailor the ensemble to specific use cases.

  * BS Detector (`Chen & Mueller, 2023 <https://arxiv.org/abs/2308.16175>`_)

  * Generalized Ensemble (`Bouchard & Chauhan, 2025 <https://arxiv.org/abs/2504.19254>`_)


Contents
--------

.. toctree::
   :maxdepth: 1

   Get Started <getstarted>
   Scorer Definitions <scorer_definitions/index>
   API <api>
   /_notebooks/index
   Contributor Guide <contribute>
   FAQs <faqs>

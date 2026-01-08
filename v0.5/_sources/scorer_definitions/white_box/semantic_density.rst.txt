Semantic Density
================

.. currentmodule:: uqlm.scorers

``semantic_density``

Semantic Density (SD) approximates a probability density function (PDF) in semantic space for
estimating response correctness.

Definition
----------

Given a prompt :math:`x` with candidate response :math:`y_*`, the objective is to construct a PDF
that assigns higher density to regions in semantic space corresponding to correct responses.

**Step 1: Sample Reference Responses**

Sample :math:`M` unique reference responses :math:`y_i` (for :math:`i = 1, 2, \dots, M`) conditioned on :math:`x`.

**Step 2: Estimate Semantic Distance**

For any pair of responses :math:`y_i, y_j` with corresponding embeddings :math:`v_i, v_j`, the
semantic distance is estimated as:

.. math::

    \mathbb{E}(\|v_i - v_j\|^2) = p_c(y_i, y_j | x) + \frac{1}{2} \cdot p_n(y_i, y_j | x)

where :math:`p_c` and :math:`p_n` denote the contradiction and neutrality scores returned by an
NLI model, respectively.

**Step 3: Compute Kernel Function**

This estimated distance is incorporated in the kernel function :math:`K`:

.. math::

    K(v_*, v_i) = (1 - \mathbb{E}(\|v_* - v_i\|^2)) \cdot \mathbf{1}_{\mathbb{E}(\|v_* - v_i\|) \leq 1}

where :math:`\mathbf{1}` is the indicator function such that :math:`\mathbf{1}_{\text{condition}} = 1`
when the condition holds and :math:`0` otherwise.

**Step 4: Compute Semantic Density**

The final semantic density score is:

.. math::

    SD(y_* | x) = \frac{\sum_{i=1}^M \sqrt[L_i]{p(y_i|x)} \cdot K(v_*, v_i)}{\sum_{i=1}^M \sqrt[L_i]{p(y_i|x)}}

where :math:`L_i` denotes the length of :math:`y_i` and :math:`p(y_i|x)` is the sequence probability.

**Key Properties:**

- Combines semantic similarity with token probability weighting
- Uses NLI model to estimate distances in semantic space
- Score range: :math:`[0, 1]`

How It Works
------------

1. Generate multiple reference responses with logprobs from the same prompt
2. For each reference response, compute its length-normalized probability
3. Use an NLI model to estimate semantic distances between the original and reference responses
4. Apply a kernel function to convert distances to similarity weights
5. Compute a probability-weighted average of kernel values

Parameters
----------

When using :class:`WhiteBoxUQ`, specify ``"semantic_density"`` in the ``scorers`` list.

Example
-------

.. code-block:: python

    from uqlm import WhiteBoxUQ

    # Initialize with semantic_density scorer
    wbuq = WhiteBoxUQ(
        llm=llm,
        scorers=["semantic_density"],
        sampling_temperature=1.0,
        length_normalize=True
    )

    # Generate responses and compute scores
    results = await wbuq.generate_and_score(prompts=prompts, num_responses=5)

    # Access the semantic_density scores
    print(results.to_df()["semantic_density"])

You can also use the dedicated :class:`SemanticDensity` class:

.. code-block:: python

    from uqlm import SemanticDensity

    # Initialize SemanticDensity scorer
    sd = SemanticDensity(
        llm=llm,
        nli_model_name="microsoft/deberta-large-mnli",
        length_normalize=True
    )

    # Generate and score
    results = await sd.generate_and_score(prompts=prompts, num_responses=5)

References
----------

- Qiu, L., et al. (2024). `Semantic Density: Uncertainty Quantification for Large Language Models through Confidence Measurement in Semantic Space <https://arxiv.org/abs/2405.13845>`_. *arXiv*.

See Also
--------

- :class:`WhiteBoxUQ` - Main class for white-box uncertainty quantification
- :class:`SemanticDensity` - Dedicated class for semantic density computation
- :doc:`semantic_negentropy_whitebox` - Related semantic-based scorer


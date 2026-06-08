Functional Equivalence Scorers
==============================

.. currentmodule:: uqlm.scorers

Definition
----------

Functional equivalence scorers use an LLM to judge whether two code snippets are functionally equivalent, meaning they would produce the same outputs for valid inputs. These scorers were proposed by Bouchard et al. (2026).

``functional_equivalence_rate`` estimates the proportion of sampled responses that are functionally equivalent to the original response:

.. math::

    FER(y; \tilde{\mathbf{y}}) = \frac{1}{m} \sum_{j=1}^{m} \mathbb{I}[y \equiv \tilde{y}_j]

``functional_negentropy`` clusters the original and sampled responses by functional equivalence, computes entropy over the cluster distribution, and normalizes it to a confidence score. Let :math:`\mathcal{C}` denote the set of functional equivalence clusters, and let :math:`P(C)` denote the proportion of responses in cluster :math:`C`. Functional entropy is:

.. math::

    FE(y; \tilde{\mathbf{y}}) = -\sum_{C \in \mathcal{C}} P(C) \log P(C)

The normalized confidence score is:

.. math::

    NFN(y; \tilde{\mathbf{y}}) = 1 - \frac{FE(y; \tilde{\mathbf{y}})}{\log(m + 1)}

``functional_sets_confidence`` counts the number of functional equivalence clusters and normalizes it to :math:`[0, 1]`:

.. math::

    FSC(y; \tilde{\mathbf{y}}) = \frac{m + 1 - |\mathcal{C}|}{m}

where :math:`|\mathcal{C}|` is the number of functional equivalence clusters among the original response and :math:`m` sampled responses.

**Key Properties:**

- Directly targets functional agreement rather than textual similarity
- Requires an LLM for equivalence judgments
- Score range: :math:`[0, 1]`

Parameters
----------

When using :class:`CodeGenUQ`, specify ``"functional_equivalence_rate"``, ``"functional_negentropy"``, or ``"functional_sets_confidence"`` in the ``scorers`` list. You can set ``equivalence_llm`` to use a separate model for equivalence judgments.

Example
-------

.. code-block:: python

    from uqlm import CodeGenUQ

    code_uq = CodeGenUQ(
        llm=llm,
        equivalence_llm=equivalence_llm,
        scorers=[
            "functional_equivalence_rate",
            "functional_negentropy",
            "functional_sets_confidence",
        ],
        language="python",
    )

    results = await code_uq.generate_and_score(prompts=prompts, num_responses=5)

References
----------

- Bouchard, D., et al. (2026). `Functional Entropy: Predicting Functional Correctness in LLM-Generated Code with Uncertainty Quantification <https://arxiv.org/pdf/2605.28500>`_. *arXiv*.

See Also
--------

- :class:`CodeGenUQ` - Class for code-generation uncertainty quantification

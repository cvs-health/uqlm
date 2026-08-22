Code Similarity Scorers
=======================

.. currentmodule:: uqlm.scorers

Definition
----------

Code similarity scorers generate sampled code responses from the same prompt and compare each sampled response with the original response. Higher average similarity indicates higher confidence.

``cosine_sim`` embeds the original and sampled code responses with a code embedding model, then computes normalized average cosine similarity:

.. math::

    NCS(y; \tilde{\mathbf{y}}) = \frac{1}{2} + \frac{1}{2m} \sum_{j=1}^{m} \frac{V(y) \cdot V(\tilde{y}_j)}{\|V(y)\| \cdot \|V(\tilde{y}_j)\|}

``code_bleu`` computes average CodeBLEU similarity between the original code response and sampled responses:

.. math::

    CBC(y; \tilde{\mathbf{y}}) = \frac{1}{m} \sum_{j=1}^{m} \text{CodeBLEU}(y, \tilde{y}_j)

**Key Properties:**

- Code-adapted black-box consistency scoring
- Uses structural or embedding-based similarity rather than natural-language entailment
- Score range: :math:`[0, 1]`

Parameters
----------

When using :class:`CodeGenUQ`, specify ``"cosine_sim"`` or ``"code_bleu"`` in the ``scorers`` list. You can also set ``sentence_transformer`` for ``cosine_sim`` and ``language`` for ``code_bleu``.

Example
-------

.. code-block:: python

    from uqlm import CodeGenUQ

    code_uq = CodeGenUQ(
        llm=llm,
        scorers=["cosine_sim", "code_bleu"],
        language="python",
    )

    results = await code_uq.generate_and_score(prompts=prompts, num_responses=5)

See Also
--------

- :class:`CodeGenUQ` - Class for code-generation uncertainty quantification

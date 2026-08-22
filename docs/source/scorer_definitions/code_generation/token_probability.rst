Token-Probability Code Scorers
==============================

.. currentmodule:: uqlm.scorers

Definition
----------

Token-probability code scorers are the same methods used by :class:`WhiteBoxUQ`, applied to generated code responses through :class:`CodeGenUQ`.

Available scorers:

- ``sequence_probability``
- ``min_probability``
- ``mean_token_negentropy``
- ``min_token_negentropy``
- ``probability_margin``
- ``monte_carlo_probability``
- ``p_true``

**Key Properties:**

- Requires token probabilities from the LLM/API
- Uses the same definitions as the corresponding white-box short-form scorers
- Score range: :math:`[0, 1]`

Parameters
----------

When using :class:`CodeGenUQ`, specify one or more token-probability scorer names in the ``scorers`` list.

Example
-------

.. code-block:: python

    from uqlm import CodeGenUQ

    code_uq = CodeGenUQ(
        llm=llm,
        scorers=["sequence_probability", "min_probability"],
        language="python",
    )

    results = await code_uq.generate_and_score(prompts=prompts)

See Also
--------

- :class:`CodeGenUQ` - Class for code-generation uncertainty quantification
- :class:`WhiteBoxUQ` - Class for white-box uncertainty quantification

P(True)
=======

.. currentmodule:: uqlm.scorers

``p_true``

P(True) is a self-reflection method that presents an LLM with its own previous response and asks it
to classify the statement as "True" or "False". The confidence score is derived from the token
probability for the "True" answer.

Definition
----------

Given a prompt :math:`x` and the LLM's response :math:`y`, the P(True) scorer:

1. Constructs a self-reflection prompt asking the LLM to evaluate whether :math:`y` correctly answers :math:`x`
2. Requests the LLM to respond with "True" or "False"
3. Returns the token probability for "True" as the confidence score

.. math::

    P(True) = p(\text{"True"} | x, y, \text{reflection\_prompt})

If the model answers "False", the score is computed as:

.. math::

    P(True) = 1 - p(\text{"False"} | x, y, \text{reflection\_prompt})

**Key Properties:**

- Self-reflection approach - uses the same LLM to evaluate its own response
- Requires one additional LLM generation per response
- Score range: :math:`[0, 1]`

How It Works
------------

1. Generate an original response to the prompt
2. Construct a self-reflection prompt that presents:

   - The original question/prompt
   - The LLM's response
   - A request to classify whether the response is correct

3. Generate the classification with logprobs enabled
4. Extract the probability of "True" (or 1 - probability of "False")

This scorer leverages the model's own ability to assess the quality of its responses, providing
a form of self-consistency check.

Parameters
----------

When using :class:`WhiteBoxUQ`, specify ``"p_true"`` in the ``scorers`` list.

Example
-------

.. code-block:: python

    from uqlm import WhiteBoxUQ

    # Initialize with p_true scorer
    wbuq = WhiteBoxUQ(
        llm=llm,
        scorers=["p_true"]
    )

    # Generate responses and compute scores
    # Note: p_true generates one additional call per prompt
    results = await wbuq.generate_and_score(prompts=prompts)

    # Access the p_true scores
    print(results.to_df()["p_true"])

References
----------

- Kadavath, S., et al. (2022). `Language Models (Mostly) Know What They Know <https://arxiv.org/abs/2207.05221>`_. *arXiv*.

See Also
--------

- :class:`WhiteBoxUQ` - Main class for white-box uncertainty quantification
- :doc:`/scorer_definitions/llm_judges/index` - LLM-as-a-Judge scorers for external evaluation


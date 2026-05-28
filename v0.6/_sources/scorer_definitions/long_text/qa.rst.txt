QA-Based Uncertainty Quantification (LUQ)
=========================================

.. currentmodule:: uqlm.scorers

Definition
----------

The Claim-QA approach demonstrated here is adapted from Farquhar et al. (2024).  The original response :math:`y` is decomposed into units (claims or sentences) and LLM is used to convert each unit :math:`s` (sentence or claim) into a question for which that unit would be the answer. The method measures consistency across multiple responses to these questions, effectively applying standard black-box uncertainty quantification to those sampled responses to the unit questions. Formally, a claim-QA scorer :math:`c_g(s;\cdot)` is defined as follows:

.. math::

    c_g(s; y_0^{(s)}, \mathbf{y}^{(s)}_{\text{cand}}) = \frac{1}{m} \sum_{j=1}^m \eta(y_0^{(s)}, y_j^{(s)})

where :math:`y_0^{(s)}` is the original unit response, :math:`\mathbf{y}^{(s)}_{\text{cand}} = \{y_1^{(s)}, ..., y_m^{(s)}\}` are :math:`m` candidate responses to the unit's question, and :math:`\eta` is a consistency function such as contradiction probability, cosine similarity, or BERTScore F1. Semantic entropy, which follows a slightly different functional form, can also be used to measure consistency.

**Key Properties:**

- Claim or sententence-level scoring
- More complex (cost and latency) than LUQ-style scoring methods
- Score range: :math:`[0, 1]`

How It Works
------------

1. Generate an original response and sampled responses
2. Decompose original response into units (claims or sentences)
3. For each claim/sentence, generate one or more questions that have that claim/sentence as the answer
4. Generate multiple responses for each question generated in step 3
5. Measure consistency in the LLM responses to the claim/sentence questions to estimate claim/sentence-level confidence

Parameters
----------

When using :class:`LongTextQA`, specify ``"semantic_negentropy"`` (or alternative scoring function) in the ``scorers`` list.

Example
-------

.. code-block:: python

    from uqlm import LongTextQA

    # Initialize 
    ltqa = LongTextQA(
        llm=original_llm,
        claim_decomposition_llm=claim_decomposition_llm,
        scorers=["semantic_negentropy"],
        sampling_temperature=1.0
    )

    # Generate responses and compute scores
    results = await ltqa.generate_and_score(prompts=prompts, num_claim_qa_responses=5)

    # Access the claim-level scores
    print(results.to_df()["claims_data"])

References
----------

- Farquhar, S., et al. (2024). `Detecting hallucinations in large language models using semantic entropy <https://www.nature.com/articles/s41586-024-07421-0>`_. *Nature*.

See Also
--------

- :class:`LongTextQA` - Class for Graph-Based UQ for long-form generations

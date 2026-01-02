Normalized Cosine Similarity
============================

.. currentmodule:: uqlm.scorers

``cosine_sim``

Normalized Cosine Similarity (NCS) leverages a sentence transformer to map LLM outputs to an
embedding space and measure similarity using those sentence embeddings.

Definition
----------

Let :math:`V: \mathcal{Y} \rightarrow \mathbb{R}^d` denote the sentence transformer, where :math:`d`
is the dimension of the embedding space.

The average cosine similarity across pairings of the original response with all candidate responses
is given as follows:

.. math::

    CS(y_i; \tilde{\mathbf{y}}_i) = \frac{1}{m} \sum_{j=1}^m \frac{\mathbf{V}(y_i) \cdot \mathbf{V}(\tilde{y}_{ij})}{\|\mathbf{V}(y_i)\| \|\mathbf{V}(\tilde{y}_{ij})\|}

To ensure a standardized support of :math:`[0, 1]`, we normalize cosine similarity to obtain confidence
scores as follows:

.. math::

    NCS(y_i; \tilde{\mathbf{y}}_i) = \frac{CS(y_i; \tilde{\mathbf{y}}_i) + 1}{2}

**Key Properties:**

- Uses sentence-level embeddings rather than token-level
- Efficient computation compared to token-level methods like BERTScore
- Normalized to :math:`[0, 1]` range where 1 indicates perfect semantic similarity

How It Works
------------

1. Generate multiple candidate responses :math:`\tilde{\mathbf{y}}_i` from the same prompt
2. Encode the original response and all candidates using a sentence transformer
3. Compute cosine similarity between the original response embedding and each candidate embedding
4. Average the similarities and normalize to :math:`[0, 1]`

The default sentence transformer is ``all-MiniLM-L6-v2``, which provides a good balance between
speed and quality for semantic similarity tasks.

Parameters
----------

When using :class:`BlackBoxUQ`, specify ``"cosine_sim"`` in the ``scorers`` list.

You can also specify a custom sentence transformer using the ``sentence_transformer`` parameter.

Example
-------

.. code-block:: python

    from uqlm import BlackBoxUQ

    # Initialize with cosine_sim scorer
    bbuq = BlackBoxUQ(
        llm=llm,
        scorers=["cosine_sim"],
        sentence_transformer="all-MiniLM-L6-v2"  # Default sentence transformer
    )

    # Generate responses and compute scores
    results = await bbuq.generate_and_score(prompts=prompts, num_responses=5)

    # Access the cosine_sim scores
    print(results.to_df()["cosine_sim"])

References
----------

- Shorinwa, O., et al. (2024). `A Survey of Confidence Estimation and Calibration in Large Language Models <https://arxiv.org/abs/2412.05563>`_. *arXiv*.
- `Sentence Transformers - all-MiniLM-L6-v2 <https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2>`_. *HuggingFace*.

See Also
--------

- :class:`BlackBoxUQ` - Main class for black-box uncertainty quantification
- :doc:`bert_score` - Alternative similarity measure using BERT token embeddings


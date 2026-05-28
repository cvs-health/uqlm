BERTScore
=========

.. currentmodule:: uqlm.scorers

``bert_score``

BERTScore leverages contextualized BERT embeddings to measure the semantic similarity between
the original response and sampled candidate responses.

Definition
----------

Let a tokenized text sequence be denoted as :math:`\mathbf{t} = \{t_1,...,t_L\}` and the corresponding
contextualized word embeddings as :math:`\mathbf{E} = \{\mathbf{e}_1,...,\mathbf{e}_L\}`, where :math:`L`
is the number of tokens in the text.

The BERTScore precision, recall, and F1-scores between two tokenized texts :math:`\mathbf{t}, \mathbf{t}'`
are respectively defined as follows:

**Precision:**

.. math::

    \text{BertP}(\mathbf{t}, \mathbf{t}') = \frac{1}{|\mathbf{t}|} \sum_{t \in \mathbf{t}} \max_{t' \in \mathbf{t}'} \mathbf{e} \cdot \mathbf{e}'

**Recall:**

.. math::

    \text{BertR}(\mathbf{t}, \mathbf{t}') = \frac{1}{|\mathbf{t}'|} \sum_{t' \in \mathbf{t}'} \max_{t \in \mathbf{t}} \mathbf{e} \cdot \mathbf{e}'

**F1-Score:**

.. math::

    \text{BertF}(\mathbf{t}, \mathbf{t}') = 2\frac{\text{BertP}(\mathbf{t}, \mathbf{t}') \cdot \text{BertR}(\mathbf{t}, \mathbf{t}')}{\text{BertP}(\mathbf{t}, \mathbf{t}') + \text{BertR}(\mathbf{t}, \mathbf{t}')}

where :math:`\mathbf{e}, \mathbf{e}'` respectively correspond to :math:`t, t'`.

We compute our BERTScore-based confidence scores as:

.. math::

    \text{BertConf}(y_i; \tilde{\mathbf{y}}_i) = \frac{1}{m} \sum_{j=1}^m \text{BertF}(y_i, \tilde{y}_{ij})

i.e., the average BERTScore F1 across pairings of the original response with all candidate responses.

How It Works
------------

1. Generate multiple candidate responses :math:`\tilde{\mathbf{y}}_i` from the same prompt
2. For each pair of original response and candidate:

   - Tokenize both responses
   - Compute contextualized BERT embeddings for each token
   - Calculate pairwise token similarities using dot products
   - Compute precision, recall, and F1-score

3. Average the F1-scores across all candidates

Parameters
----------

When using :class:`BlackBoxUQ`, specify ``"bert_score"`` in the ``scorers`` list.

Example
-------

.. code-block:: python

    from uqlm import BlackBoxUQ

    # Initialize with bert_score scorer
    bbuq = BlackBoxUQ(
        llm=llm,
        scorers=["bert_score"],
        device="cuda"  # Use GPU for faster BERT inference
    )

    # Generate responses and compute scores
    results = await bbuq.generate_and_score(prompts=prompts, num_responses=5)

    # Access the bert_score scores
    print(results.to_df()["bert_score"])

References
----------

- Manakul, P., et al. (2023). `SelfCheckGPT: Zero-Resource Black-Box Hallucination Detection for Generative Large Language Models <https://arxiv.org/abs/2303.08896>`_. *arXiv*.
- Zhang, T., et al. (2020). `BERTScore: Evaluating Text Generation with BERT <https://arxiv.org/abs/1904.09675>`_. *arXiv*.

See Also
--------

- :class:`BlackBoxUQ` - Main class for black-box uncertainty quantification
- :doc:`cosine_sim` - Alternative similarity measure using sentence embeddings


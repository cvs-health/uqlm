Graph-Based Uncertainty Quantification (LUQ)
============================================

.. currentmodule:: uqlm.scorers

Definition
----------

Graph-based scorers, proposed by Jiang et al. (2024), decompose original and sampled responses into claims, obtain the union of unique claims across all responses, and compute graph centrality metrics on the bipartite graph of claim-response entailment to measure uncertainty. These scorers operate only at the claim level, as sentences typically contain multiple claims, meaning their union is not well-defined. Formally, we denote a bipartite graph :math:`G` with node set :math:`V = \mathbf{s} \cup  \mathbf{y}`, where :math:`\mathbf{y}` is a set of :math:`m` responses generated from the same prompt and :math:`\mathbf{s}` is the union of all unique claims across those decomposed responses. In particular, an edge exists between a claim-response pair :math:`(s, y) \in  \mathbf{s} \times \mathbf{y}` if and only if claim :math:`s` is entailed in response :math:`y`. We define the following graph metrics for claim :math:`s`:


* **Degree Centrality** - :math:`\frac{1}{m} \sum_{j=1}^m P(\text{entail}|y_j, s)` is the average edge weight, measured by entailment probability for claim node `s`. 

* **Betweenness Centrality** - :math:`\frac{1}{B_{\text{max}}}\sum_{u \neq v \neq s} \frac{\sigma_{uv}(s)}{\sigma_{uv}}` measures uncertainty by calculating the proportion of shortest paths between node pairs that pass through node :math:`s`, where :math:`\sigma_{uv}` represents all shortest paths between nodes :math:`u` and :math:`v`, and :math:`B_{\text{max}}` is the maximum possible value, given by :math:`B_{\text{max}}=\frac{1}{2} [m^2 (p + 1)^2 + m (p + 1)(2t - p - 1) - t (2p - t + 3)]`, :math:`p = \frac{(|\mathbf{s}| - 1)}{m}`, and :math:`t = (|\mathbf{s}| - 1) \mod m`.


* **Closeness Centrality** - :math:`\frac{m + 2(|\mathbf{s}| - 1) }{\sum_{v \neq s}dist(s, v)}` measures the inverse sum of distances to all other nodes, normalized by the minimum possible distance.

* **Harmonic Centrality** - :math:`\frac{1}{H_{\text{max}}}\sum_{v \neq s}\frac{1}{dist(s, v)}` is the sum of inverse of distances to all other nodes, normalized by the maximum possible value, where :math:`H_{\text{max}}=m + \frac{ |\mathbf{s}| - 1}{2}`.

* **Laplacian Centrality** - :math:`\frac{E_L (G)-E_L (G_{\text{-} s})}{E_L (G)}` is the proportional drop in Laplacian energy :math:`E_L (G)` resulting from dropping node :math:`s` from the graph, where :math:`G_{\text{-}s}` denotes the graph :math:`G` with node :math:`s` removed, :math:`E_L (G)  = \sum_{i} \lambda_i^2`, and :math:`\lambda_i` are the eigenvalues of :math:`G`'s Laplacian matrix.

* **PageRank** - :math:`\frac{1-d}{|V|} + d \sum_{v \in N(s)} \frac{C_{PR}(v)}{N(v)}` is the stationary distribution probability of a random walk with restart probability :math:`(1-d)`, where :math:`N(s)` denotes the set of neighboring nodes of :math:`s` and :math:`C_{PR}(v)` is PageRank of node :math:`v`.

where :math:`\mathbf{y}^{(s)}_{\text{cand}} = \{y_1^{(s)}, ..., y_m^{(s)}\}` are :math:`m` candidate responses.

**Key Properties:**

- Claim or sententence-level scoring
- More complex (cost and latency) than LUQ-style scoring methods
- Score range: :math:`[0, 1]`

How It Works
------------

1. Generate an original response and sampled responses
2. Decompose original and sampled responses into claims
3. Construct a bipartite claim-response entailment graph
4. Compute graph centrality metrics to measure claim-level confidence

Parameters
----------

When using :class:`LongTextGraph`, specify ``"closeness_centrality"`` (or alternative scoring function) in the ``scorers`` list.

Example
-------

.. code-block:: python

    from uqlm import LongTextGraph

    # Initialize 
    ltg = LongTextGraph(
        llm=original_llm,
        claim_decomposition_llm=claim_decomposition_llm,
        scorers=["closeness_centrality"],
        sampling_temperature=1.0
    )

    # Generate responses and compute scores
    results = await ltg.generate_and_score(prompts=prompts, num_responses=5)

    # Access the claim-level scores
    print(results.to_df()["claims_data"])

References
----------

- Jiang, M., et al. (2024). `Graph-based Uncertainty Metrics for Long-form Language Model Outputs <https://arxiv.org/abs/2410.20783>`_. *arXiv*.

See Also
--------

- :class:`LongTextGraph` - Class for Graph-Based UQ for long-form generations

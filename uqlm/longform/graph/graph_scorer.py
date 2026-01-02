from uqlm.longform.luq.baseclass.claims_scorer import ClaimScorer, ClaimScore
from typing import List, Optional, Any
from rich.progress import Progress
from uqlm.nli.nli import NLI
import numpy as np
import time
from scipy import sparse


class GraphScorer(ClaimScorer):
    def __init__(self, nli_model_name: Optional[str] = "microsoft/deberta-large-mnli", device: Optional[Any] = None, max_length: Optional[int] = 2000) -> None:
        try:
            import networkx as nx
        except ImportError:
            print("graph-based scoring requires `networkx` package. Please install using `pip install networkx`.")
        
        self.nli_model_name = nli_model_name
        self.device = device
        self.max_length = max_length
        self.nli = NLI(device=device, nli_model_name=nli_model_name, max_length=max_length)

    def evaluate(self, original_claim_sets: List[List[str]], master_claim_sets: List[List[str]], response_sets: List[List[str]], binary_edge_threshold: float = 0.5, progress_bar: Optional[Progress] = None) -> List[List[ClaimScore]]:
        # Step 1: Compute adjacency matrices for all response sets
        biadjacency_matrices = self._compute_adjacency_matrices(response_sets, master_claim_sets, progress_bar)

        # Step 2: Construct graphs and calculate scores for all response sets
        claim_score_lists = self._construct_graphs_and_calculate_scores(response_sets, original_claim_sets, master_claim_sets, biadjacency_matrices, binary_edge_threshold, progress_bar)

        # Small delay to ensure progress bar UI updates before function completes
        time.sleep(0.1)

        return claim_score_lists

    def _calculate_claim_node_graph_metrics(self, G: nx.Graph, num_claims: int, num_responses: int) -> dict:
        """
        Calculate claim node graph metrics using a single graph representation.

        All metrics are normalized to [0, 1] range using either NetworkX's built-in normalization
        (when reliable) or custom structural normalization based on graph topology.

        The graph has edges with "weight" attributes (actual entailment scores). Strength-based
        metrics use these weights, while path-based metrics use the unweighted graph structure.

        Args:
            G: Graph with edges weighted by entailment scores. Edges only exist where scores
               meet the binary_edge_threshold.
            num_claims: The number of claims.
            num_responses: The number of responses.

        Returns:
            A dictionary of claim node graph metrics (all normalized to [0, 1]):
            - degree_centrality: Normalized by opposite set size (structural bound)
            - betweenness_centrality: NetworkX bipartite normalization (bipartite-specific bound)
            - closeness_centrality: NetworkX bipartite normalization (clipped to [0, 1])
            - page_rank: Standard PageRank probability distribution (sums to 1)
            - laplacian_centrality: NetworkX normalization (normalized=True default)
            - harmonic_centrality: Custom normalization (theoretical max for bipartite structure)

        Notes:
            - Strength-based metrics (degree, PageRank, laplacian): Use edge weights
            - Path-based metrics (betweenness, closeness, harmonic): Use unweighted/binary approach
        """

        # Calculate weighted degree (sum of edge weights) for each node
        weighted_degrees = dict(G.degree(weight="weight"))

        # Calculate bipartite degree centrality (normalized by opposite set size)
        # We're doing the full graph for now; can do just claims for efficiency later
        claim_nodes = set(range(num_claims))
        response_nodes = set(range(num_claims, num_claims + num_responses))

        degree_centrality = {}
        for node in claim_nodes:
            degree_centrality[node] = weighted_degrees[node] / num_responses
        for node in response_nodes:
            degree_centrality[node] = weighted_degrees[node] / num_claims

        # Calculate betweenness centrality (unweighted - path-based metric)
        betweenness_centrality = nx.bipartite.betweenness_centrality(G, claim_nodes)

        # Calculate PageRank (weighted - strength-based metric)
        try:
            page_rank = nx.pagerank(G, weight="weight", max_iter=1000)
        except (nx.PowerIterationFailedConvergence, nx.NetworkXError):
            page_rank = {node: np.nan for node in G.nodes()}

        # Calculate closeness centrality (unweighted - path-based metric)
        closeness_centrality_raw = nx.bipartite.closeness_centrality(G, claim_nodes)
        closeness_centrality = {node: min(1.0, val) for node, val in closeness_centrality_raw.items()}

        # Calculate Laplacian Centrality (weighted - strength-based metric)
        laplacian_centrality = nx.laplacian_centrality(G, weight="weight", normalized=True)

        # Calculate Harmonic Centrality (unweighted - path-based metric)
        harmonic_centrality_raw = nx.harmonic_centrality(G)

        # Normalize by theoretical maximum in complete bipartite graph
        # We're doing the full graph for now; can do just claims for efficiency later
        harmonic_centrality = {}
        for node in claim_nodes:
            theoretical_max = num_responses + (num_claims - 1) * 0.5
            harmonic_centrality[node] = harmonic_centrality_raw[node] / theoretical_max
        for node in response_nodes:
            theoretical_max = num_claims + (num_responses - 1) * 0.5
            harmonic_centrality[node] = harmonic_centrality_raw[node] / theoretical_max

        return {"degree_centrality": degree_centrality, "betweenness_centrality": betweenness_centrality, "closeness_centrality": closeness_centrality, "page_rank": page_rank, "laplacian_centrality": laplacian_centrality, "harmonic_centrality": harmonic_centrality}

    def _construct_bipartite_graph(self, biadjacency_matrix: np.ndarray, num_claims: int, num_responses: int, binary_edge_threshold: float) -> nx.Graph:
        """
        Construct a bipartite graph from a biadjacency matrix.


        Args:
            biadjacency_matrix: A 2D numpy array of shape (num_claims, num_responses) with entailment scores.
            num_claims: The number of claims.
            num_responses: The number of responses.
            binary_edge_threshold: Threshold for edge existence. Only edges with scores >= this value are created.
        Returns:
            nx.Graph: Bipartite graph with edges weighted by entailment scores.
        """
        # Create filtered matrix: only keep edges at or above threshold
        # The actual entailment scores are preserved as edge weights
        filtered_matrix = np.where(biadjacency_matrix >= binary_edge_threshold, biadjacency_matrix, 0)
        biadjacency_sparse = sparse.csr_matrix(filtered_matrix)
        G = nx.bipartite.from_biadjacency_matrix(biadjacency_sparse)

        # Add node type attributes
        for node_idx in range(num_claims):
            G.nodes[node_idx]["type"] = "claim"
        for node_idx in range(num_claims, num_claims + num_responses):
            G.nodes[node_idx]["type"] = "response"

        return G

    def _compute_adjacency_matrices(self, response_sets: List[List[str]], master_claim_sets: List[List[str]], progress_bar: Optional[Progress] = None) -> List[np.ndarray]:
        """Compute adjacency matrices for response sets.

        Collects NLI tasks across response sets and executes them concurrently
        using asyncio.gather, then reconstructs the adjacency matrices.
        """
        num_response_sets = len(response_sets)

        # Create progress task if progress bar is provided
        progress_task = None
        if progress_bar:
            progress_task = progress_bar.add_task("  - Building claim-response biadjacency matrices...", total=num_response_sets)

        biadjacency_matrices = []
        for i in range(num_response_sets):
            master_claim_set = master_claim_sets[i]
            responses = response_sets[i]
            num_claims = len(master_claim_set)
            num_responses = len(responses)

            biadjacency_matrix = np.zeros((num_claims, num_responses))

            for claim_idx, claim in enumerate(master_claim_set):
                for response_idx, response in enumerate(responses):
                    entailment_probability = self.nli.predict(hypothesis=claim, premise=response)[:, -1]
                    biadjacency_matrix[claim_idx, response_idx] = entailment_probability

            biadjacency_matrices.append(biadjacency_matrix)
            if progress_bar and progress_task is not None:
                progress_bar.update(progress_task, advance=1)

        return biadjacency_matrices

    def _construct_graphs_and_calculate_scores(self, response_sets: List[List[str]], original_claim_sets: List[List[str]], master_claim_sets: List[List[str]], biadjacency_matrices: List[np.ndarray], binary_edge_threshold: float, progress_bar: Optional[Progress] = None) -> List[List[ClaimScore]]:
        """Construct bipartite graphs and calculate claim scores for all response sets."""
        num_response_sets = len(response_sets)

        # Create progress task if progress bar is provided
        progress_task = None
        if progress_bar:
            progress_task = progress_bar.add_task("  - Constructing graphs and calculating scores...", total=num_response_sets)

        claim_score_lists = []
        for i in range(num_response_sets):
            claim_scores = self._process_single_graph(i, response_sets[i], original_claim_sets[i], master_claim_sets[i], biadjacency_matrices[i], binary_edge_threshold)
            claim_score_lists.append(claim_scores)

            # Update progress
            if progress_bar and progress_task is not None:
                progress_bar.update(progress_task, advance=1)

        return claim_score_lists

    def _process_single_graph(self, index: int, responses: List[str], original_claim_set: List[str], master_claim_set: List[str], biadjacency_matrix: np.ndarray, binary_edge_threshold: float) -> List[ClaimScore]:
        """Process a single response set: construct graph and calculate claim scores."""
        num_claims = len(master_claim_set)
        num_responses = len(responses)

        # Construct bipartite graph (edges only where scores >= threshold, with actual scores as weights)
        G = self._construct_bipartite_graph(biadjacency_matrix, num_claims, num_responses, binary_edge_threshold)

        # Calculate claim node graph metrics
        gmetrics = self._calculate_claim_node_graph_metrics(G, num_claims, num_responses)

        # Gather claim scores into list of ClaimScore objects
        claim_scores = []
        for node_idx in range(num_claims):
            claim_text = master_claim_set[node_idx]
            is_original = claim_text in original_claim_set

            claim_score = ClaimScore(
                claim=claim_text,
                original_response=is_original,
                scorer_type="graphuq",
                scores={
                    "degree_centrality": round(gmetrics["degree_centrality"][node_idx], 5),
                    "betweenness_centrality": round(gmetrics["betweenness_centrality"][node_idx], 5),
                    "closeness_centrality": round(gmetrics["closeness_centrality"][node_idx], 5),
                    "harmonic_centrality": round(gmetrics["harmonic_centrality"][node_idx], 5),
                    "page_rank": round(gmetrics["page_rank"][node_idx], 5),
                    "laplacian_centrality": round(gmetrics["laplacian_centrality"][node_idx], 5),
                },
            )
            claim_scores.append(claim_score)

        return claim_scores

from uqlm.longform.black_box.baseclass.claims_scorer import ClaimScorer, ClaimScore
from typing import List, Optional, Any, Dict
from rich.progress import Progress
from langchain_core.language_models.chat_models import BaseChatModel
from uqlm.utils.nli import NLI
from uqlm.utils.prompts.claims_prompts import get_claim_dedup_prompt
from uqlm.utils.response_generator import ResponseGenerator
import re
import logging
import asyncio
import numpy as np
import networkx as nx
from scipy import sparse
import matplotlib.pyplot as plt

# Create a logger for this module
logger = logging.getLogger(__name__)


class GraphUQScorer(ClaimScorer):
    def __init__(
        self,
        judge_llm: BaseChatModel,
        nli_model_name: Optional[str] = "microsoft/deberta-large-mnli",
        nli_llm: Optional[BaseChatModel] = None,
        device: Optional[Any] = None,
        max_length: Optional[int] = 2000,
        max_calls_per_min: Optional[int] = None,
    ) -> None:
        self.nli_model_name = nli_model_name
        self.nli_llm = nli_llm
        self.device = device
        self.max_length = max_length
        
        # Initialize NLI only if a model name or LLM is provided
        # If None, NLI will not be available (limited functionality)
        if nli_model_name is None and nli_llm is None:
            self.nli = None
            logger.warning(
                "NLI model not provided. Limited functionality available. "
                "Entailment calculations will only work if complete entailment_score_sets are provided."
            )
        else:
            self.nli = NLI(
                nli_model_name=nli_model_name,
                nli_llm=nli_llm,
                device=device,
                max_length=max_length
            )
        
        self.rg = ResponseGenerator(llm=judge_llm, max_calls_per_min=max_calls_per_min)

        logger.info(f"Initialized GraphUQScorer")

    def evaluate(
        self,
        original_claim_sets: List[List[str]],
        response_sets: Optional[List[List[str]]] = None,
        sampled_claim_sets: Optional[List[List[List[str]]]] = None,
        entailment_score_sets: Optional[List[Dict[str, List[float]]]] = None,
        binary_edge_threshold: float = 0.5,
        claim_dedup_method: Optional[str] = "sequential",
        save_graph_path: Optional[str] = None,
        show_graph: bool = False,
        use_entailment_prob: bool = False,
        min_edge_weight: float = 0.001,
        progress_bar: Optional[Progress] = None,
    ) -> List[List[ClaimScore]]:
        """Evaluate the GraphUQ score and claim scores for a list of response sets.

        Args:
            original_claim_sets: The list of original claim sets (required).
            response_sets: The list of response sets. Optional if entailment_score_sets is provided.
            sampled_claim_sets: The list of sampled claim sets. Optional if entailment_score_sets is provided.
            entailment_score_sets: The list of entailment score sets. If provided, response_sets and 
                sampled_claim_sets become optional (will be inferred from scores).
            binary_edge_threshold: Threshold for binary edge existence in path-based metrics (betweenness, closeness, harmonic).
                Edges below this threshold are ignored for shortest path calculations but still contribute 
                to strength-based metrics (degree, PageRank, eigenvector, katz) with their actual weight. Default: 0.5.
            claim_dedup_method: The method to deduplicate claims. Options: "sequential", "one_shot", "exact_match" or None.
                Ignored if sampled_claim_sets is not provided.
            save_graph_path: The path to save the graph. Requires response_sets for visualization.
            show_graph: Whether to show the graph. Requires response_sets for visualization.
            use_entailment_prob: Whether to use entailment probabilities.
            min_edge_weight: Minimum weight threshold for edge existence (clips very small values to avoid sparse matrix errors).
                Edges with weight <= this value are removed entirely. Default: 0.001.
            progress_bar: The progress bar.
        Returns:
            A list of lists of ClaimScore objects, one for each response set.
        """
        return asyncio.run(
            self.a_evaluate(
                original_claim_sets,
                response_sets,
                sampled_claim_sets,
                entailment_score_sets,
                binary_edge_threshold,
                claim_dedup_method,
                save_graph_path,
                show_graph,
                use_entailment_prob,
                min_edge_weight,
                progress_bar,
            )
        )

    async def a_evaluate(
        self,
        original_claim_sets: List[List[str]],
        response_sets: Optional[List[List[str]]] = None,
        sampled_claim_sets: Optional[List[List[List[str]]]] = None,
        entailment_score_sets: Optional[List[Dict[str, List[float]]]] = None,
        binary_edge_threshold: float = 0.5,
        claim_dedup_method: Optional[str] = "sequential",
        save_graph_path: Optional[str] = None,
        show_graph: bool = False,
        use_entailment_prob: bool = False,
        min_edge_weight: float = 0.001,
        progress_bar: Optional[Progress] = None,
    ) -> List[List[ClaimScore]]:
        
        # Step 0: Validate inputs and infer missing values
        num_response_sets = len(original_claim_sets)
        # Handle None values for entailment_score_sets
        if entailment_score_sets is None:
            entailment_score_sets = [None] * num_response_sets
            # If no entailment scores provided, we must have response_sets to compute them
            if response_sets is None:
                raise ValueError("response_sets is required when entailment_score_sets is not provided")
            if sampled_claim_sets is None:
                raise ValueError("sampled_claim_sets is required when entailment_score_sets is not provided")
        else:
            # entailment_score_sets is provided
            # Infer response_sets structure if not provided
            if response_sets is None:
                # Create placeholder response_sets with correct number of responses inferred from scores
                response_sets = []
                for i, (claim_set, score_set) in enumerate(zip(original_claim_sets, entailment_score_sets)):
                    if score_set is None:
                        raise ValueError(f"entailment_score_sets[{i}] is None but response_sets is not provided. Cannot infer number of responses.")
                    if not score_set:
                        raise ValueError(f"entailment_score_sets[{i}] is empty but response_sets is not provided. Cannot infer number of responses.")
                    # Get number of responses from first claim's scores
                    num_responses = len(next(iter(score_set.values())))
                    # Create placeholder response strings
                    response_sets.append([f"Response {j}" for j in range(num_responses)])
                    logger.debug(f"[Response set {i}] Inferred {num_responses} responses from entailment scores")
            
            # Handle missing sampled_claim_sets
            if sampled_claim_sets is None:
                # Create empty sampled_claim_sets (dedup will infer claims from entailment_score_sets)
                sampled_claim_sets = [[]] * num_response_sets
                logger.debug("sampled_claim_sets not provided. Master claims will be inferred from entailment_score_sets keys.")
                # Force dedup to None since we don't have sampled claims
                if claim_dedup_method is not None:
                    logger.info("claim_dedup_method ignored because sampled_claim_sets is not provided")
                    claim_dedup_method = None
        
        logger.debug(f"Starting evaluation for {num_response_sets} response sets.")

        if num_response_sets > 10 and show_graph:
            logger.warning("More than 10 response sets and show_graph is True. This may be slow and cause memory issues.")
        
        # Warn if visualization requested but no actual response text
        if (show_graph or save_graph_path) and response_sets is not None:
            # Check if response_sets contains placeholder text
            if any("Response " in resp for resp_set in response_sets for resp in resp_set):
                logger.warning("Visualization may show placeholder response text because response_sets was inferred from entailment_score_sets")

        if claim_dedup_method not in ["sequential", "one_shot", "exact_match", None]:
            logger.warning(f"Invalid claim dedup method: {claim_dedup_method}. Skipping claim deduplication process.")
            claim_dedup_method = None

        # Step 1: Dedup claims for all response sets
        logger.debug("Step 1: Deduplicating claims for all response sets...")
        master_claim_sets = await self._dedup_claims(
            original_claim_sets,
            sampled_claim_sets,
            claim_dedup_method,
            entailment_score_sets,
            progress_bar,
        )

        # Step 2: Compute adjacency matrices for all response sets
        logger.debug("Step 2: Computing adjacency matrices for all response sets...")
        biadjacency_matrices = await self._compute_adjacency_matrices(
            response_sets,
            master_claim_sets,
            entailment_score_sets,
            use_entailment_prob,
            min_edge_weight,
            progress_bar,
        )

        # Step 3: Construct graphs and calculate scores for all response sets
        logger.debug("Step 3: Constructing graphs and calculating scores for all response sets...")
        claim_score_lists = self._construct_graphs_and_calculate_scores(
            response_sets,
            original_claim_sets,
            master_claim_sets,
            biadjacency_matrices,
            binary_edge_threshold,
            save_graph_path,
            show_graph,
            progress_bar,
        )

        # Small delay to ensure progress bar UI updates before function completes
        await asyncio.sleep(0.1)

        return claim_score_lists

    def _calculate_claim_node_graph_metrics(
        self, 
        G_weighted: nx.Graph, 
        G_binary: nx.Graph, 
        num_claims: int, 
        num_responses: int
    ) -> dict:
        """
        Calculate claim node graph metrics using two different graph representations.
        
        All metrics are normalized to [0, 1] range using either NetworkX's built-in normalization
        (when reliable) or custom structural normalization based on graph topology.
        
        Args:
            G_weighted: Graph with continuous edge weights (0-1) for strength-based metrics.
            G_binary: Graph with binary edges (0/1) for path-based metrics.
            num_claims: The number of claims.
            num_responses: The number of responses.
            
        Returns:
            A dictionary of claim node graph metrics (all normalized to [0, 1]):
            - degree_centrality: Normalized by opposite set size (structural bound)
            - betweenness_centrality: NetworkX bipartite normalization (bipartite-specific bound)
            - closeness_centrality: NetworkX normalization (normalized by n-1)
            - page_rank: Probability distribution (sums to 1)
            - laplacian_centrality: NetworkX normalization (normalized=True default)
            - harmonic_centrality: Custom normalization (theoretical max for bipartite structure)
            - katz_centrality: NetworkX normalization + clipping to [0, 1]
            
        Notes:
            - G_weighted: Used for degree, PageRank, laplacian, katz (strength-based)
            - G_binary: Used for betweenness, closeness, harmonic (path-based)
        """

        # Calculate weighted degree (sum of edge weights) for each node
        weighted_degrees = dict(G_weighted.degree(weight="weight"))
        logger.debug(f"Weighted degrees (sum of edge weights): {weighted_degrees}")

        # Calculate bipartite degree centrality (normalized by opposite set size)
        claim_nodes = set(range(num_claims))
        response_nodes = set(range(num_claims, num_claims + num_responses))
        
        degree_centrality = {}
        for node in claim_nodes:
            degree_centrality[node] = weighted_degrees[node] / num_responses if num_responses > 0 else 0.0
        for node in response_nodes:
            degree_centrality[node] = weighted_degrees[node] / num_claims if num_claims > 0 else 0.0
        logger.debug(f"Degree centrality: {degree_centrality}")

        # Calculate betweenness centrality  
        betweenness_centrality = nx.bipartite.betweenness_centrality(G_binary, claim_nodes)
        logger.debug(f"Betweenness centrality: {betweenness_centrality}")

        # Calculate PageRank 
        try:
            page_rank = nx.pagerank(G_weighted, weight="weight", max_iter=1000)
            logger.debug(f"PageRank: {page_rank}")
        except (nx.PowerIterationFailedConvergence, nx.NetworkXError) as e:
            logger.warning(f"PageRank failed to converge: {e}. Using NaN to indicate calculation failure.")
            page_rank = {node: np.nan for node in G_weighted.nodes()}

        # Calculate closeness centrality
        closeness_centrality = nx.closeness_centrality(G_binary)
        logger.debug(f"Closeness centrality: {closeness_centrality}")

        # Calculate Laplacian Centrality
        laplacian_centrality = nx.laplacian_centrality(G_weighted, weight="weight", normalized=True)
        logger.debug(f"Laplacian centrality: {laplacian_centrality}")

        # Calculate Harmonic Centrality
        harmonic_centrality_raw = nx.harmonic_centrality(G_binary)
        
        # Normalize by theoretical maximum in complete bipartite graph
        # For claim nodes: max = N_r * 1.0 + (N_c - 1) * 0.5 (distance 1 to responses, distance 2 to other claims)
        # For response nodes: max = N_c * 1.0 + (N_r - 1) * 0.5
        harmonic_centrality = {}
        for node in claim_nodes:
            theoretical_max = num_responses + (num_claims - 1) * 0.5 if num_claims > 1 else num_responses
            harmonic_centrality[node] = harmonic_centrality_raw[node] / theoretical_max if theoretical_max > 0 else 0.0
        for node in response_nodes:
            theoretical_max = num_claims + (num_responses - 1) * 0.5 if num_responses > 1 else num_claims
            harmonic_centrality[node] = harmonic_centrality_raw[node] / theoretical_max if theoretical_max > 0 else 0.0
        
        logger.debug(f"Harmonic centrality: {harmonic_centrality}")

        return {
            "degree_centrality": degree_centrality,
            "betweenness_centrality": betweenness_centrality,
            "closeness_centrality": closeness_centrality,
            "page_rank": page_rank,
            "laplacian_centrality": laplacian_centrality,
            "harmonic_centrality": harmonic_centrality,
        }

    def _construct_bipartite_graphs(
        self, 
        biadjacency_matrix: np.ndarray, 
        num_claims: int, 
        num_responses: int,
        binary_edge_threshold: float
    ) -> tuple[nx.Graph, nx.Graph]:
        """
        Construct two bipartite graphs from a biadjacency matrix.
        
        Creates two separate graphs:
        1. Weighted graph: Edges have actual entailment scores (for strength-based metrics)
        2. Binary graph: Edges are 0/1 based on binary_edge_threshold (for path-based metrics)
        
        Args:
            biadjacency_matrix: A 2D numpy array of shape (num_claims, num_responses) with entailment scores.
            num_claims: The number of claims.
            num_responses: The number of responses.
            binary_edge_threshold: Threshold for binary edge existence in path-based metrics.
        Returns:
            Tuple of (G_weighted, G_binary):
            - G_weighted: Graph with continuous edge weights (0 to 1) for strength metrics
            - G_binary: Graph with binary edges (0 or 1) for path metrics
        """
        # Graph 1: Weighted graph with actual entailment scores
        # Used by: degree centrality, PageRank, eigenvector centrality, katz centrality
        biadjacency_sparse_weighted = sparse.csr_matrix(biadjacency_matrix)
        G_weighted = nx.bipartite.from_biadjacency_matrix(biadjacency_sparse_weighted)
        
        # Graph 2: Binary graph (edges exist if weight >= threshold)
        # Used by: betweenness centrality, closeness centrality, harmonic centrality
        binary_matrix = (biadjacency_matrix >= binary_edge_threshold).astype(float)
        biadjacency_sparse_binary = sparse.csr_matrix(binary_matrix)
        G_binary = nx.bipartite.from_biadjacency_matrix(biadjacency_sparse_binary)

        # Add node type attributes to both graphs
        for G in [G_weighted, G_binary]:
            for node_idx in range(num_claims):
                G.nodes[node_idx]["type"] = "claim"
            for node_idx in range(num_claims, num_claims + num_responses):
                G.nodes[node_idx]["type"] = "response"

        return G_weighted, G_binary

    async def _dedup_claims(
        self,
        original_claim_sets: List[List[str]],
        sampled_claim_sets: List[List[List[str]]],
        claim_dedup_method: Optional[str],
        entailment_score_sets: Optional[List[Dict[str, List[float]]]] = None,
        progress_bar: Optional[Progress] = None,
    ) -> List[List[str]]:
        """Process claim deduplication for response sets.

        Leverages ResponseGenerator's ability to handle multiple prompts at once
        by collecting dedup prompts and making batch calls.
        
        If sampled_claim_sets contains only empty lists and entailment_score_sets is provided,
        infers master claims from entailment_score_sets keys. Otherwise returns original_claim_sets.
        """
        num_response_sets = len(original_claim_sets)
        
        # Check if all sampled_claim_sets are empty
        if all(not claim_set for claim_set in sampled_claim_sets):
            # If entailment scores provided, infer master claims from score keys
            if entailment_score_sets:
                logger.debug("All sampled_claim_sets are empty. Inferring master claims from entailment_score_sets keys.")
                master_claim_sets = []
                for i, score_set in enumerate(entailment_score_sets):
                    if score_set:
                        # Use the keys from entailment_score_sets as the master claims
                        master_claims = list(score_set.keys())
                        master_claim_sets.append(master_claims)
                        logger.debug(f"[Response set {i}] Inferred {len(master_claims)} claims from entailment_score_sets keys")
                    else:
                        # No scores provided, use original claims
                        master_claim_sets.append(original_claim_sets[i])
                        logger.debug(f"[Response set {i}] No entailment scores, using original {len(original_claim_sets[i])} claims")
                return master_claim_sets
            else:
                # No sampled claims and no entailment scores - just return original claims
                logger.debug("All sampled_claim_sets are empty. Returning original_claim_sets as master_claim_sets.")
                return original_claim_sets

        # Create progress task if progress bar is provided
        progress_task = None
        if progress_bar and claim_dedup_method and claim_dedup_method != "exact_match":
            progress_task = progress_bar.add_task("  - Deduplicating claims...", total=num_response_sets)

        if not claim_dedup_method:
            # No dedup, just concatenate for all response sets
            master_claim_sets = [original_claim_sets[i] + [claim for claim_set in sampled_claim_sets[i] for claim in claim_set] for i in range(num_response_sets)]

        elif claim_dedup_method == "exact_match":
            # Exact match, no LLM calls needed
            master_claim_sets = []
            for i in range(num_response_sets):
                all_sampled_claims = [claim for claim_set in sampled_claim_sets[i] for claim in claim_set]
                master_claim_set = list(set(original_claim_sets[i] + all_sampled_claims))
                logger.debug(f"[Response set {i}] Initial master claim set size: {len(original_claim_sets[i])}")
                logger.debug(f"[Response set {i}] Master claim set size after dedup: {len(master_claim_set)}")
                master_claim_sets.append(master_claim_set)

        elif claim_dedup_method == "one_shot":
            # One-shot: Batch ALL prompts across ALL response sets
            prompts = []
            prompt_metadata = []  # Track which response set each prompt belongs to

            for i in range(num_response_sets):
                all_sampled_claims = [claim for claim_set in sampled_claim_sets[i] for claim in claim_set]
                unique_sampled_claims = list(set(all_sampled_claims) - set(original_claim_sets[i]))

                logger.debug(f"[Response set {i}] Initial master claim set size: {len(original_claim_sets[i])}")
                logger.debug(f"[Response set {i}] Found {len(unique_sampled_claims)} unique sampled claims to process")

                if unique_sampled_claims:
                    prompts.append(get_claim_dedup_prompt(original_claim_sets[i], unique_sampled_claims))
                    prompt_metadata.append((i, original_claim_sets[i], all_sampled_claims))
                else:
                    prompt_metadata.append((i, original_claim_sets[i], all_sampled_claims))

            # Make single batch call to ResponseGenerator for all prompts
            if prompts:
                result = await self.rg.generate_responses(prompts=prompts)
                responses = result["data"]["response"]
            else:
                responses = []

            # Process results and build master_claim_sets
            master_claim_sets = [None] * num_response_sets
            response_idx = 0

            for i, original_set, all_sampled in prompt_metadata:
                unique_sampled = list(set(all_sampled) - set(original_set))

                if unique_sampled and response_idx < len(responses):
                    # Extract new claims from LLM response
                    response_text = responses[response_idx]
                    new_claims = re.findall(r"^\s*-\s*(.+)", response_text, re.MULTILINE)

                    if new_claims:
                        logger.debug(f"[Response set {i}] Adding {len(new_claims)} new claims to master set")
                        master_claim_set = original_set + new_claims
                    else:
                        logger.debug(f"[Response set {i}] No new claims extracted from LLM response")
                        master_claim_set = original_set

                    response_idx += 1
                else:
                    master_claim_set = original_set

                logger.debug(f"[Response set {i}] Master claim set size after dedup: {len(master_claim_set)}")
                logger.debug(f"[Response set {i}] Original claims missing: {len(set(original_set) - set(master_claim_set))}")
                logger.debug(f"[Response set {i}] Entirely new claims added: {len(set(master_claim_set) - set(original_set + all_sampled))}")

                master_claim_sets[i] = master_claim_set

                # Update progress
                if progress_bar and progress_task is not None:
                    progress_bar.update(progress_task, advance=1)

        elif claim_dedup_method == "sequential":
            # Sequential: Batch across response sets at each iteration
            master_claim_sets = [original_claim_sets[i] for i in range(num_response_sets)]
            max_iterations = max(len(sampled_set) for sampled_set in sampled_claim_sets)

            for iteration in range(max_iterations):
                prompts = []
                prompt_metadata = []  # (response_set_idx, has_prompt)

                for i in range(num_response_sets):
                    logger.debug(f"[Response set {i}] Initial master claim set size: {len(original_claim_sets[i])}")

                    if iteration < len(sampled_claim_sets[i]):
                        sampled_claims = sampled_claim_sets[i][iteration]
                        unique_sampled_claims = list(set(sampled_claims) - set(master_claim_sets[i]))

                        logger.debug(f"[Response set {i}][Iteration {iteration}] Found {len(unique_sampled_claims)} unique claims")

                        if unique_sampled_claims:
                            prompts.append(get_claim_dedup_prompt(master_claim_sets[i], unique_sampled_claims))
                            prompt_metadata.append((i, True, master_claim_sets[i], sampled_claims))
                        else:
                            prompt_metadata.append((i, False, master_claim_sets[i], sampled_claims))
                    else:
                        prompt_metadata.append((i, False, master_claim_sets[i], []))

                # Batch call for this iteration across all response sets
                if prompts:
                    result = await self.rg.generate_responses(prompts=prompts)
                    responses = result["data"]["response"]
                else:
                    responses = []

                # Update master_claim_sets with results
                response_idx = 0
                for i, has_prompt, current_master, sampled_claims in prompt_metadata:
                    if has_prompt and response_idx < len(responses):
                        response_text = responses[response_idx]
                        new_claims = re.findall(r"^\s*-\s*(.+)", response_text, re.MULTILINE)

                        if new_claims:
                            logger.debug(f"[Response set {i}][Iteration {iteration}] Adding {len(new_claims)} new claims")
                            master_claim_sets[i] = current_master + new_claims
                        else:
                            logger.debug(f"[Response set {i}][Iteration {iteration}] No new claims extracted")

                        response_idx += 1

            # Log final stats and update progress
            for i in range(num_response_sets):
                all_sampled_claims = [claim for claim_set in sampled_claim_sets[i] for claim in claim_set]
                logger.debug(f"[Response set {i}] Master claim set size after dedup: {len(master_claim_sets[i])}")
                logger.debug(f"[Response set {i}] Original claims missing: {len(set(original_claim_sets[i]) - set(master_claim_sets[i]))}")
                logger.debug(f"[Response set {i}] Entirely new claims added: {len(set(master_claim_sets[i]) - set(original_claim_sets[i] + all_sampled_claims))}")

                # Update progress
                if progress_bar and progress_task is not None:
                    progress_bar.update(progress_task, advance=1)

        else:
            raise ValueError(f"Unknown claim_dedup_method: {claim_dedup_method}")

        return master_claim_sets

    async def _compute_adjacency_matrices(
        self,
        response_sets: List[List[str]],
        master_claim_sets: List[List[str]],
        entailment_score_sets: List[Optional[Dict[str, List[float]]]],
        use_entailment_prob: bool,
        min_edge_weight: float,
        progress_bar: Optional[Progress] = None,
    ) -> List[np.ndarray]:
        """Compute adjacency matrices for response sets.

        Collects NLI tasks across response sets and executes them concurrently
        using asyncio.gather, then reconstructs the adjacency matrices.
        """
        num_response_sets = len(response_sets)

        # Create progress task if progress bar is provided
        progress_task = None
        if progress_bar:
            progress_task = progress_bar.add_task("  - Building claim-response biadjacency matrices...", total=num_response_sets)

        # Collect all NLI tasks across all response sets
        all_nli_tasks = []
        task_metadata = []  # Track which (response_set_idx, claim_idx, response_idx) each task corresponds to

        for i in range(num_response_sets):
            master_claim_set = master_claim_sets[i]
            responses = response_sets[i]
            entailment_score_set = entailment_score_sets[i]

            logger.debug(f"[Response set {i}] Computing entailment scores for master claim set...")

            # Determine which claim-response pairs need NLI computation
            if entailment_score_set is None:
                # All pairs need computation
                if self.nli is None:
                    raise ValueError(
                        f"Cannot compute entailment scores for response set {i}: "
                        "entailment_score_set is None and NLI model is not available. "
                        "Either provide entailment_score_sets or initialize GraphUQScorer with an NLI model."
                    )
                for claim_idx, claim in enumerate(master_claim_set):
                    for response_idx, response in enumerate(responses):
                        all_nli_tasks.append(self.nli.apredict(hypothesis=claim, premise=response, style="binary"))
                        task_metadata.append((i, claim_idx, response_idx))
            else:
                # Only missing pairs need computation
                for claim_idx, claim in enumerate(master_claim_set):
                    if claim in entailment_score_set:
                        scores = entailment_score_set[claim]
                        if len(scores) != len(responses):
                            # Wrong number of scores, compute all for this claim
                            if self.nli is None:
                                raise ValueError(
                                    f"Cannot compute entailment scores for response set {i}, claim '{claim[:50]}...': "
                                    f"score count ({len(scores)}) doesn't match response count ({len(responses)}) "
                                    "and NLI model is not available. "
                                    "Either provide correct entailment_score_sets or initialize GraphUQScorer with an NLI model."
                                )
                            for response_idx, response in enumerate(responses):
                                all_nli_tasks.append(self.nli.apredict(hypothesis=claim, premise=response, style="binary"))
                                task_metadata.append((i, claim_idx, response_idx))
                    else:
                        # Claim not in entailment_score_set, compute all
                        if self.nli is None:
                            raise ValueError(
                                f"Cannot compute entailment scores for response set {i}, claim '{claim[:50]}...': "
                                "claim not found in entailment_score_set and NLI model is not available. "
                                "Either provide complete entailment_score_sets or initialize GraphUQScorer with an NLI model."
                            )
                        for response_idx, response in enumerate(responses):
                            all_nli_tasks.append(self.nli.apredict(hypothesis=claim, premise=response, style="binary"))
                            task_metadata.append((i, claim_idx, response_idx))

        # Execute all NLI tasks concurrently
        logger.debug(f"Executing {len(all_nli_tasks)} NLI predictions concurrently...")
        if all_nli_tasks:
            nli_results = await asyncio.gather(*all_nli_tasks)
        else:
            nli_results = []

        # Build adjacency matrices for each response set
        biadjacency_matrices = []
        for i in range(num_response_sets):
            num_claims = len(master_claim_sets[i])
            num_responses = len(response_sets[i])
            biadjacency_matrix = np.zeros((num_claims, num_responses))

            # First, fill in provided scores if available
            if entailment_score_sets[i] is not None:
                for claim_idx, claim in enumerate(master_claim_sets[i]):
                    if claim in entailment_score_sets[i]:
                        scores = entailment_score_sets[i][claim]
                        if len(scores) == num_responses:
                            for response_idx, score in enumerate(scores):
                                if score >= 0:
                                    biadjacency_matrix[claim_idx, response_idx] = score

            biadjacency_matrices.append(biadjacency_matrix)

        # Now fill in the NLI results
        for (resp_set_idx, claim_idx, response_idx), nli_result in zip(task_metadata, nli_results):
            if use_entailment_prob:
                biadjacency_matrices[resp_set_idx][claim_idx, response_idx] = nli_result.entailment_probability
            else:
                biadjacency_matrices[resp_set_idx][claim_idx, response_idx] = 1.0 if nli_result.binary_label else 0.0

        # Apply minimum edge weight threshold to all matrices (clip very small values)
        filtered_matrices = []
        for i, biadjacency_matrix in enumerate(biadjacency_matrices):
            logger.debug(f"[Response set {i}] Biadjacency matrix shape: {biadjacency_matrix.shape}")
            biadjacency_matrix_filtered = np.where(biadjacency_matrix > min_edge_weight, biadjacency_matrix, 0)
            logger.debug(f"[Response set {i}] Filtered {np.sum(biadjacency_matrix > 0) - np.sum(biadjacency_matrix_filtered > 0)} edges below threshold {min_edge_weight}")
            filtered_matrices.append(biadjacency_matrix_filtered)

            # Update progress
            if progress_bar and progress_task is not None:
                progress_bar.update(progress_task, advance=1)

        return filtered_matrices

    def _construct_graphs_and_calculate_scores(
        self,
        response_sets: List[List[str]],
        original_claim_sets: List[List[str]],
        master_claim_sets: List[List[str]],
        biadjacency_matrices: List[np.ndarray],
        binary_edge_threshold: float,
        save_graph_path: Optional[str],
        show_graph: bool,
        progress_bar: Optional[Progress] = None,
    ) -> List[List[ClaimScore]]:
        """Construct bipartite graphs and calculate claim scores for all response sets."""
        num_response_sets = len(response_sets)

        # Create progress task if progress bar is provided
        progress_task = None
        if progress_bar:
            progress_task = progress_bar.add_task("  - Constructing graphs and calculating scores...", total=num_response_sets)

        claim_score_lists = []
        for i in range(num_response_sets):
            claim_scores = self._process_single_graph(
                i,
                response_sets[i],
                original_claim_sets[i],
                master_claim_sets[i],
                biadjacency_matrices[i],
                binary_edge_threshold,
                save_graph_path,
                show_graph,
            )
            claim_score_lists.append(claim_scores)

            # Update progress
            if progress_bar and progress_task is not None:
                progress_bar.update(progress_task, advance=1)

        return claim_score_lists

    def _process_single_graph(
        self,
        index: int,
        responses: List[str],
        original_claim_set: List[str],
        master_claim_set: List[str],
        biadjacency_matrix: np.ndarray,
        binary_edge_threshold: float,
        save_graph_path: Optional[str],
        show_graph: bool,
    ) -> List[ClaimScore]:
        """Process a single response set: construct graph and calculate claim scores."""
        num_claims = len(master_claim_set)
        num_responses = len(responses)

        # Construct bipartite graphs (weighted and binary)
        logger.debug(f"[Response set {index}] Constructing bipartite graphs...")
        G_weighted, G_binary = self._construct_bipartite_graphs(biadjacency_matrix, num_claims, num_responses, binary_edge_threshold)

        logger.debug(f"[Response set {index}] Weighted graph: {G_weighted.number_of_nodes()} nodes ({num_claims} claims, {num_responses} responses), {G_weighted.number_of_edges()} edges")
        logger.debug(f"[Response set {index}] Binary graph: {G_binary.number_of_nodes()} nodes ({num_claims} claims, {num_responses} responses), {G_binary.number_of_edges()} edges")

        # Calculate claim node graph metrics
        logger.debug(f"[Response set {index}] Calculating claim node graph metrics...")
        gmetrics = self._calculate_claim_node_graph_metrics(G_weighted, G_binary, num_claims, num_responses)

        # Gather claim scores into list of ClaimScore objects
        logger.debug(f"[Response set {index}] Gathering claim scores into list of ClaimScore objects...")
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
                    "katz_centrality": round(gmetrics["katz_centrality"][node_idx], 5),
                },
            )
            claim_scores.append(claim_score)

        # Validate that all graph metrics are in [0, 1] range
        self._validate_graph_metrics(claim_scores)

        # Optional: visualize graph (use weighted graph to show actual edge weights)
        if show_graph or save_graph_path:
            self._visualize_bipartite_graph_matplotlib(
                G_weighted,
                num_claims,
                num_responses,
                master_claim_set,
                responses,
                biadjacency_matrix,
                save_graph_path,
                show_graph,
            )

        return claim_scores

    def _visualize_bipartite_graph_matplotlib(
        self,
        G,
        num_claims,
        num_responses,
        claim_texts,
        response_texts,
        biadjacency_matrix,
        save_path=None,
        show_graph=False,
    ):
        """Create static matplotlib visualization with claim text and edge weights"""
        # Use appropriate backend
        if show_graph:
            # Use default interactive backend for inline display
            plt.switch_backend("module://matplotlib_inline.backend_inline")
        else:
            # Use non-interactive backend for file-only output
            plt.switch_backend("Agg")

        fig, ax = plt.subplots(figsize=(16, max(10, num_claims * 1.5, num_responses * 1.5)))

        # Create bipartite layout: claims on left, responses on right
        pos = {}
        claim_nodes = list(range(num_claims))
        response_nodes = list(range(num_claims, num_claims + num_responses))

        # Position claim nodes on the left
        x_claims = 0
        y_spacing_claims = 10 / max(num_claims, 1)
        for i, node in enumerate(claim_nodes):
            pos[node] = (x_claims, -i * y_spacing_claims)

        # Position response nodes on the right
        x_responses = 10
        y_spacing_responses = 10 / max(num_responses, 1)
        for i, node in enumerate(response_nodes):
            pos[node] = (x_responses, -i * y_spacing_responses)

        # Collect all non-zero weights to normalize visualization
        all_weights = []
        for claim_idx in claim_nodes:
            for response_idx in response_nodes:
                weight = biadjacency_matrix[claim_idx, response_idx - num_claims]
                if weight > 0:
                    all_weights.append(weight)

        # Calculate weight range for normalization
        if len(all_weights) > 0:
            min_weight = min(all_weights)
            max_weight = max(all_weights)
            weight_range = max_weight - min_weight
        else:
            min_weight = 0
            max_weight = 1
            weight_range = 1

        # Draw edges with weights
        for claim_idx in claim_nodes:
            for response_idx in response_nodes:
                weight = biadjacency_matrix[claim_idx, response_idx - num_claims]
                # Only draw edges that exist (filtering already done before graph construction)
                if weight > 0:
                    x0, y0 = pos[claim_idx]
                    x1, y1 = pos[response_idx]

                    # Normalize weight to [0, 1] based on actual range in this graph
                    if weight_range > 0:
                        normalized_weight = (weight - min_weight) / weight_range
                    else:
                        normalized_weight = 1.0  # All weights are the same

                    # Map normalized weight to visual properties
                    line_width = 0.5 + normalized_weight * 3.5  # 0.5 to 4
                    alpha = 0.3 + normalized_weight * 0.5  # 0.3 to 0.8

                    # Draw edge with thickness based on normalized weight
                    ax.plot(
                        [x0, x1],
                        [y0, y1],
                        linewidth=line_width,
                        alpha=alpha,
                        color="gray",
                        zorder=1,
                    )

                    # Add edge weight label showing actual weight (not normalized)
                    if weight < 1.0 and weight >= 0.01:
                        mid_x, mid_y = (x0 + x1) / 2, (y0 + y1) / 2
                        ax.text(
                            mid_x,
                            mid_y,
                            f"{weight:.3f}",
                            fontsize=7,
                            ha="center",
                            va="center",
                            bbox=dict(
                                boxstyle="round,pad=0.3",
                                facecolor="white",
                                edgecolor="gray",
                                alpha=0.7,
                            ),
                        )

        # Calculate claim node degrees for color mapping
        claim_degrees = []
        for claim_idx in claim_nodes:
            degree = sum(1 for resp_idx in response_nodes if biadjacency_matrix[claim_idx, resp_idx - num_claims] > 0)
            claim_degrees.append(degree)

        # Normalize degrees for color mapping
        max_degree = max(claim_degrees) if claim_degrees else 1
        min_degree = min(claim_degrees) if claim_degrees else 0
        degree_range = max_degree - min_degree if max_degree > min_degree else 1

        # Draw claim nodes with color based on degree
        import matplotlib.cm as cm

        cmap = cm.get_cmap("Blues")  # Light blue to dark blue

        for i, node in enumerate(claim_nodes):
            x, y = pos[node]
            # Truncate long claims for display
            claim_text = claim_texts[i]
            display_text = (claim_text[:40] + "...") if len(claim_text) > 40 else claim_text

            # Map degree to color (0.4 to 0.9 range in colormap for better visibility)
            if degree_range > 0:
                normalized_degree = (claim_degrees[i] - min_degree) / degree_range
            else:
                normalized_degree = 1.0
            color_intensity = 0.4 + normalized_degree * 0.5  # 0.4 (light) to 0.9 (dark)
            node_color = cmap(color_intensity)

            ax.scatter(
                x,
                y,
                s=600,
                c=[node_color],
                marker="s",
                edgecolors="#2171b5",
                linewidths=2,
                zorder=3,
            )
            ax.text(
                x,
                y,
                f"C{i}",
                ha="center",
                va="center",
                fontsize=10,
                fontweight="bold",
                color="white",
                zorder=4,
            )
            # Add claim text to the left with degree info
            ax.text(
                x - 0.5,
                y,
                f"{display_text} ({claim_degrees[i]})",
                ha="right",
                va="center",
                fontsize=9,
                wrap=True,
            )

        # Draw response nodes
        for i, node_idx in enumerate(response_nodes):
            x, y = pos[node_idx]
            # Truncate long responses for display
            response_text = response_texts[i]
            display_text = (response_text[:40] + "...") if len(response_text) > 40 else response_text

            ax.scatter(x, y, s=600, c="#fc8d62", marker="o", edgecolors="#e34a33", linewidths=2, zorder=3)
            ax.text(
                x,
                y,
                f"R{i}",
                ha="center",
                va="center",
                fontsize=10,
                fontweight="bold",
                color="white",
                zorder=4,
            )
            # Add response text to the right
            ax.text(x + 0.5, y, display_text, ha="left", va="center", fontsize=9, wrap=True)

        # Formatting
        ax.set_xlim(-4, 14)
        y_min = min([pos[n][1] for n in pos.keys()]) - 1
        y_max = max([pos[n][1] for n in pos.keys()]) + 1
        ax.set_ylim(y_min, y_max)
        ax.axis("off")

        # Create informative title with weight range
        if weight_range > 0:
            title = f"Bipartite Graph: Claims ↔ Responses\nClaim shading: darker = more connections | Weight range: {min_weight:.4f} - {max_weight:.4f}"
        else:
            title = "Bipartite Graph: Claims ↔ Responses\n(Claim color indicates connection count)"
        plt.title(title, fontsize=13, fontweight="bold", pad=20)

        # Add legend
        from matplotlib.lines import Line2D

        legend_elements = [
            Line2D(
                [0],
                [0],
                marker="s",
                color="w",
                markerfacecolor="#6baed6",
                markersize=10,
                label="Claims",
                markeredgecolor="#2171b5",
                markeredgewidth=2,
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="#fc8d62",
                markersize=10,
                label="Responses",
                markeredgecolor="#e34a33",
                markeredgewidth=2,
            ),
        ]
        ax.legend(handles=legend_elements, loc="upper right", fontsize=10)

        plt.tight_layout()

        # Save to file if path provided
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info(f"Static graph visualization (matplotlib) saved to: {save_path}")

        # Display inline if requested
        if show_graph:
            plt.show()
        else:
            plt.close()

    def _validate_graph_metrics(self, claim_scores: List[ClaimScore]) -> None:
        """
        Validate that all graph metrics are between 0 and 1.
        
        Logs a warning if any metric value is outside the [0, 1] range.
        
        Args:
            claim_scores: List of ClaimScore objects containing graph metrics
        """
        problematic_metrics = set()
        for claim_idx, claim_score in enumerate(claim_scores):
            for metric_name, metric_value in claim_score.scores.items():
                if metric_value < 0 or metric_value > 1:
                    problematic_metrics.add(metric_name)
                    logger.debug(
                        f"Claim {claim_idx} ('{claim_score.claim}'): Graph metric '{metric_name}' has value {metric_value:.6f} "
                    )
        if problematic_metrics:
            logger.warning(
                f"Problematic graph metrics found: {problematic_metrics}. "
                f"This may indicate an issue with the graph construction or metric calculation."
            )
            return False
        return True
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

# Optional plotly import for interactive visualizations
try:
    import plotly.graph_objects as go

    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

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
        self.nli = NLI(nli_model_name=nli_model_name, nli_llm=nli_llm, device=device, max_length=max_length)
        self.rg = ResponseGenerator(llm=judge_llm, max_calls_per_min=max_calls_per_min)

        logger.info(f"Initialized GraphUQScorer")

    def evaluate(
        self,
        response_sets: List[List[str]],
        original_claim_sets: List[List[str]],
        sampled_claim_sets: List[List[List[str]]],
        entailment_score_sets: Optional[List[Dict[str, List[float]]]] = None,
        claim_dedup_method: Optional[str] = "sequential",
        save_graph_path: Optional[str] = None,
        show_graph: bool = False,
        use_entailment_prob: bool = False,
        edge_weight_threshold: float = 0.001,
        progress_bar: Optional[Progress] = None,
    ) -> List[List[ClaimScore]]:
        """Evaluate the GraphUQ score and claim scores for a list of response sets.

        Args:
            response_sets: The list of response sets.
            original_claim_sets: The list of original claim sets.
            sampled_claim_sets: The list of sampled claim sets.
            entailment_score_sets: The list of entailment score sets.
            claim_dedup_method: The method to deduplicate claims. Options: "sequential", "one_shot", "exact_match" or None.
            save_graph_path: The path to save the graph.
            show_graph: Whether to show the graph.
            use_entailment_prob: Whether to use entailment probabilities.
            edge_weight_threshold: The threshold for filtering out very small claim-response weights.
            progress_bar: The progress bar.
        Returns:
            A list of lists of ClaimScore objects, one for each response set.
        """
        return asyncio.run(
            self.a_evaluate(
                response_sets,
                original_claim_sets,
                sampled_claim_sets,
                entailment_score_sets,
                progress_bar,
                claim_dedup_method,
                save_graph_path,
                show_graph,
                use_entailment_prob,
                edge_weight_threshold,
            )
        )

    async def a_evaluate(
        self,
        response_sets: List[List[str]],
        original_claim_sets: List[List[str]],
        sampled_claim_sets: List[List[List[str]]],
        entailment_score_sets: Optional[List[Dict[str, List[float]]]] = None,
        claim_dedup_method: Optional[str] = "sequential",
        save_graph_path: Optional[str] = None,
        show_graph: bool = False,
        use_entailment_prob: bool = False,
        edge_weight_threshold: float = 0.001,
        progress_bar: Optional[Progress] = None,
    ) -> List[List[ClaimScore]]:
        logger.debug(f"Starting evaluation for {len(response_sets)} response sets.")

        if len(response_sets) > 10 and show_graph:
            logger.warning("More than 10 response sets and show_graph is True. This will be slow and may cause memory issues.")

        if claim_dedup_method not in ["sequential", "one_shot", "exact_match", None]:
            logger.warning(f"Invalid claim dedup method: {claim_dedup_method}. Skipping claim deduplication process.")
            claim_dedup_method = None

        # Handle None values for entailment_score_sets
        if entailment_score_sets is None:
            entailment_score_sets = [None] * len(response_sets)

        num_response_sets = len(response_sets)

        # Process all response sets step-by-step (batched by step)
        # This avoids shared state issues with ResponseGenerator and NLI

        # Step 1: Dedup claims for all response sets
        logger.debug("Step 1: Deduplicating claims for all response sets...")
        master_claim_sets = await self._dedup_claims(
            original_claim_sets,
            sampled_claim_sets,
            claim_dedup_method,
            progress_bar,
        )

        # Step 2: Compute adjacency matrices for all response sets
        logger.debug("Step 2: Computing adjacency matrices for all response sets...")
        biadjacency_matrices = await self._compute_adjacency_matrices(
            response_sets,
            master_claim_sets,
            entailment_score_sets,
            use_entailment_prob,
            edge_weight_threshold,
            progress_bar,
        )

        # Step 3: Construct graphs and calculate scores for all response sets
        logger.debug("Step 3: Constructing graphs and calculating scores for all response sets...")
        claim_score_lists = self._construct_graphs_and_calculate_scores(
            response_sets,
            original_claim_sets,
            master_claim_sets,
            biadjacency_matrices,
            save_graph_path,
            show_graph,
            progress_bar,
        )

        # Small delay to ensure progress bar UI updates before function completes
        await asyncio.sleep(0.1)

        return claim_score_lists

    def _calculate_claim_node_graph_metrics(self, G: nx.Graph, num_claims: int, num_responses: int) -> dict:
        """
        Calculate claim node graph metrics.
        Args:
            G: A NetworkX graph.
            num_claims: The number of claims.
            num_responses: The number of responses.
        Returns:
            A dictionary of claim node graph metrics.
        """

        # Calculate raw degree (number of connections) for each node
        raw_degrees = dict(G.degree())
        logger.debug(f"Raw degrees (number of connections): {raw_degrees}")

        # Calculate bipartite degree centrality (normalized by the size of the opposite node set)
        # For claim nodes, this is degree / num_responses
        # For response nodes, this is degree / num_claims
        claim_nodes = set(range(num_claims))
        degree_centrality = nx.bipartite.degree_centrality(G, claim_nodes)
        logger.debug(f"Bipartite degree centrality (normalized) for all nodes: {degree_centrality}")

        # Calculate betweenness centrality (bipartite-aware)
        # In bipartite graphs, shortest paths alternate between node sets
        betweenness_centrality = nx.bipartite.betweenness_centrality(G, claim_nodes)
        logger.debug(f"Bipartite betweenness centrality for all nodes: {betweenness_centrality}")

        # Calculate PageRank (standard version works for bipartite)
        # Random walks naturally respect bipartite structure
        page_rank = nx.pagerank(G)
        logger.debug(f"PageRank for all nodes: {page_rank}")

        # Calculate closeness centrality (bipartite-aware)
        # In bipartite graphs, nodes in same set have min distance = 2
        closeness_centrality = nx.bipartite.closeness_centrality(G, claim_nodes)
        logger.debug(f"Bipartite closeness centrality for all nodes: {closeness_centrality}")

        # Note: Eigenvector centrality removed - not well-defined for bipartite graphs
        # The principal eigenvector alternates in sign between the two node sets

        # # Calculate harmonic centrality for claim nodes
        # harmonic_centrality = nx.harmonic_centrality(G)
        # logger.debug(f"Harmonic centrality for claim nodes: {harmonic_centrality}")

        # # Calculate load centrality for claim nodes
        # load_centrality = nx.load_centrality(G)
        # logger.debug(f"Load centrality for claim nodes: {load_centrality}")

        # # Calculate communicability centrality for claim nodes
        # communicability_centrality = nx.communicability_centrality(G)
        # logger.debug(f"Communicability centrality for claim nodes: {communicability_centrality}")

        # # Calculate current flow betweenness centrality for claim nodes
        # current_flow_betweenness_centrality = nx.current_flow_betweenness_centrality(G)
        # logger.debug(f"Current flow betweenness centrality for claim nodes: {current_flow_betweenness_centrality}")

        return {
            "raw_degrees": raw_degrees,
            "degree_centrality": degree_centrality,
            "betweenness_centrality": betweenness_centrality,
            "closeness_centrality": closeness_centrality,
            "page_rank": page_rank,
        }

    def _construct_bipartite_graph(self, biadjacency_matrix: np.ndarray, num_claims: int, num_responses: int) -> nx.Graph:
        """
        Construct a bipartite graph from a biadjacency matrix.
        Args:
            biadjacency_matrix: A 2D numpy array of shape (num_claims, num_responses) representing the biadjacency matrix.
            num_claims: The number of claims.
            num_responses: The number of responses.
            weight_threshold: The threshold for filtering out very small weights.
        Returns:
            A NetworkX graph.
        """
        # Create bipartite graph from biadjacency matrix
        # Rows (claims) become top node set, columns (responses) become bottom node set
        biadjacency_sparse = sparse.csr_matrix(biadjacency_matrix)
        G = nx.bipartite.from_biadjacency_matrix(biadjacency_sparse)

        # Add claim text as node attribute
        for node_idx in range(num_claims):
            G.nodes[node_idx]["type"] = "claim"

        # Add response text as node attribute
        for node_idx in range(num_claims, num_claims + num_responses):
            G.nodes[node_idx]["type"] = "response"

        return G

    async def _dedup_claims(
        self,
        original_claim_sets: List[List[str]],
        sampled_claim_sets: List[List[List[str]]],
        claim_dedup_method: Optional[str],
        progress_bar: Optional[Progress] = None,
    ) -> List[List[str]]:
        """Process claim deduplication for response sets.

        Leverages ResponseGenerator's ability to handle multiple prompts at once
        by collecting dedup prompts and making batch calls.
        """
        num_response_sets = len(original_claim_sets)

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
        edge_weight_threshold: float,
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
                            for response_idx, response in enumerate(responses):
                                all_nli_tasks.append(self.nli.apredict(hypothesis=claim, premise=response, style="binary"))
                                task_metadata.append((i, claim_idx, response_idx))
                    else:
                        # Claim not in entailment_score_set, compute all
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

        # Apply edge weight threshold to all matrices
        filtered_matrices = []
        for i, biadjacency_matrix in enumerate(biadjacency_matrices):
            logger.debug(f"[Response set {i}] Biadjacency matrix shape: {biadjacency_matrix.shape}")
            biadjacency_matrix_filtered = np.where(biadjacency_matrix > edge_weight_threshold, biadjacency_matrix, 0)
            logger.debug(f"[Response set {i}] Filtered {np.sum(biadjacency_matrix > 0) - np.sum(biadjacency_matrix_filtered > 0)} edges below threshold {edge_weight_threshold}")
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
        save_graph_path: Optional[str],
        show_graph: bool,
    ) -> List[ClaimScore]:
        """Process a single response set: construct graph and calculate claim scores."""
        num_claims = len(master_claim_set)
        num_responses = len(responses)

        # Construct bipartite graph
        logger.debug(f"[Response set {index}] Constructing bipartite graph...")
        G = self._construct_bipartite_graph(biadjacency_matrix, num_claims, num_responses)

        logger.debug(f"[Response set {index}] Bipartite graph constructed: {G.number_of_nodes()} nodes ({num_claims} claims, {num_responses} responses), {G.number_of_edges()} edges")

        # Calculate claim node graph metrics
        logger.debug(f"[Response set {index}] Calculating claim node graph metrics...")
        gmetrics = self._calculate_claim_node_graph_metrics(G, num_claims, num_responses)

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
                    "raw_degree": gmetrics["raw_degrees"][node_idx],
                    "degree_centrality": gmetrics["degree_centrality"][node_idx],
                    "betweenness_centrality": gmetrics["betweenness_centrality"][node_idx],
                    "closeness_centrality": gmetrics["closeness_centrality"][node_idx],
                    "page_rank": gmetrics["page_rank"][node_idx],
                },
            )
            claim_scores.append(claim_score)

        # Optional: visualize graph
        if show_graph or save_graph_path:
            if not PLOTLY_AVAILABLE:
                logger.warning("Plotly is not installed. Falling back to matplotlib. To use interactive HTML visualizations, install plotly: pip install plotly")
                self._visualize_bipartite_graph_matplotlib(
                    G,
                    num_claims,
                    num_responses,
                    master_claim_set,
                    responses,
                    biadjacency_matrix,
                    save_graph_path,
                    show_graph,
                )
            else:
                self._visualize_bipartite_graph_plotly(
                    G,
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

    def _visualize_bipartite_graph_plotly(
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
        """Create interactive Plotly visualization of the bipartite graph"""

        if not PLOTLY_AVAILABLE:
            raise ImportError("Plotly is required for interactive HTML visualizations. Install it with: pip install plotly")

        # Create bipartite layout: claims on left, responses on right
        claim_nodes = list(range(num_claims))
        response_nodes = list(range(num_claims, num_claims + num_responses))

        # Position nodes
        pos = {}
        spacing_y_claims = 1.0 if num_claims == 1 else 1.0 / (num_claims - 1)
        spacing_y_responses = 1.0 if num_responses == 1 else 1.0 / (num_responses - 1)

        for i, node in enumerate(claim_nodes):
            pos[node] = (0, 1.0 - i * spacing_y_claims)

        for i, node in enumerate(response_nodes):
            pos[node] = (1, 1.0 - i * spacing_y_responses)

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

        # Create edge traces with hover info
        edge_traces = []
        for claim_idx in claim_nodes:
            for response_idx in response_nodes:
                weight = biadjacency_matrix[claim_idx, response_idx - num_claims]
                # Only draw edges that exist (filtering already done before graph construction)
                if weight > 0:
                    x0, y0 = pos[claim_idx]
                    x1, y1 = pos[response_idx]

                    # Truncate texts for hover
                    claim_text_short = (claim_texts[claim_idx][:60] + "...") if len(claim_texts[claim_idx]) > 60 else claim_texts[claim_idx]
                    response_text_short = (response_texts[response_idx - num_claims][:60] + "...") if len(response_texts[response_idx - num_claims]) > 60 else response_texts[response_idx - num_claims]

                    # Normalize weight to [0, 1] based on actual range in this graph
                    if weight_range > 0:
                        normalized_weight = (weight - min_weight) / weight_range
                    else:
                        normalized_weight = 1.0  # All weights are the same

                    # Map normalized weight to visual properties
                    # Width: 1 (thin) to 6 (thick) based on normalized weight
                    line_width = 1 + normalized_weight * 5
                    # Opacity: 0.3 (faint) to 0.9 (solid) based on normalized weight
                    opacity = 0.3 + normalized_weight * 0.6

                    # Create edge trace
                    edge_trace = go.Scatter(
                        x=[x0, x1, None],
                        y=[y0, y1, None],
                        mode="lines",
                        line=dict(width=line_width, color=f"rgba(100, 100, 100, {opacity:.6f})"),
                        hovertemplate=f"<b>Weight: {weight:.4f}</b><br>"
                        + f"Normalized: {normalized_weight:.2f}<br>"
                        + f"Claim: {claim_text_short}<br>"
                        + f"Response: {response_text_short}<extra></extra>",
                        hoverinfo="text",
                        showlegend=False,
                    )
                    edge_traces.append(edge_trace)

                    # Add invisible scatter point at edge midpoint for better hover detection
                    mid_x, mid_y = (x0 + x1) / 2, (y0 + y1) / 2
                    edge_midpoint = go.Scatter(
                        x=[mid_x],
                        y=[mid_y],
                        mode="markers",
                        marker=dict(size=12, color="rgba(150, 150, 150, 0.01)"),  # Nearly invisible but clickable
                        hovertemplate=f"<b>Weight: {weight:.4f}</b><br>"
                        + f"Normalized: {normalized_weight:.2f}<br>"
                        + f"C{claim_idx} → R{response_idx - num_claims}<br>"
                        + f"Claim: {claim_text_short}<br>"
                        + f"Response: {response_text_short}<extra></extra>",
                        showlegend=False,
                    )
                    edge_traces.append(edge_midpoint)

        # Calculate claim node degrees for color mapping
        claim_degrees = []
        for claim_idx in claim_nodes:
            degree = sum(1 for resp_idx in response_nodes if biadjacency_matrix[claim_idx, resp_idx - num_claims] > 0)
            claim_degrees.append(degree)

        # Normalize degrees for color mapping
        max_degree = max(claim_degrees) if claim_degrees else 1
        min_degree = min(claim_degrees) if claim_degrees else 0
        degree_range = max_degree - min_degree if max_degree > min_degree else 1

        # Create claim node trace with color based on degree
        claim_x = [pos[node][0] for node in claim_nodes]
        claim_y = [pos[node][1] for node in claim_nodes]
        claim_text = [f"<b>C{i}</b><br>Degree: {claim_degrees[i]}<br>{claim_texts[i]}" for i in claim_nodes]

        # Generate color scale (light blue to dark blue)
        claim_colors = []
        for i in claim_nodes:
            if degree_range > 0:
                normalized_degree = (claim_degrees[i] - min_degree) / degree_range
            else:
                normalized_degree = 1.0
            # Map to RGB: light blue (173, 216, 230) to dark blue (25, 25, 112)
            r = int(173 - normalized_degree * 148)  # 173 to 25
            g = int(216 - normalized_degree * 191)  # 216 to 25
            b = int(230 - normalized_degree * 118)  # 230 to 112
            claim_colors.append(f"rgb({r}, {g}, {b})")

        claim_trace = go.Scatter(
            x=claim_x,
            y=claim_y,
            mode="markers+text",
            marker=dict(size=20, color=claim_colors, line=dict(width=2, color="#2171b5"), symbol="square"),
            text=[f"C{i}" for i in claim_nodes],
            textposition="middle center",
            textfont=dict(size=10, color="white", family="Arial Black"),
            hovertemplate="%{customdata}<extra></extra>",
            customdata=claim_text,
            name="Claims",
            showlegend=True,
        )

        # Create response node trace
        response_x = [pos[node][0] for node in response_nodes]
        response_y = [pos[node][1] for node in response_nodes]
        response_text = [f"<b>R{i}</b><br>{response_texts[i]}" for i in range(num_responses)]

        response_trace = go.Scatter(
            x=response_x,
            y=response_y,
            mode="markers+text",
            marker=dict(size=20, color="#fc8d62", line=dict(width=2, color="#e34a33"), symbol="circle"),
            text=[f"R{i}" for i in range(num_responses)],
            textposition="middle center",
            textfont=dict(size=10, color="white", family="Arial Black"),
            hovertemplate="%{customdata}<extra></extra>",
            customdata=response_text,
            name="Responses",
            showlegend=True,
        )

        # Create figure
        fig = go.Figure(data=edge_traces + [claim_trace, response_trace])

        # Create informative title with weight range info
        if weight_range > 0:
            title_text = f"<b>Bipartite Graph: Claims ↔ Responses</b><br><sub>Claim color: darker = more connections • Edge weight range: {min_weight:.4f} - {max_weight:.4f}</sub>"
        else:
            title_text = "<b>Bipartite Graph: Claims ↔ Responses</b><br><sub>Claim color indicates connection count</sub>"

        fig.update_layout(
            title=dict(text=title_text, x=0.5, xanchor="center", font=dict(size=20)),
            showlegend=True,
            hovermode="closest",
            margin=dict(b=20, l=100, r=100, t=80),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor="white",
            width=1200,
            height=max(600, num_claims * 80, num_responses * 80),
        )

        # Save to file if path provided
        if save_path:
            fig.write_html(save_path)
            logger.info(f"Interactive graph visualization (plotly) saved to: {save_path}")

        # Display inline if requested
        if show_graph:
            fig.show()

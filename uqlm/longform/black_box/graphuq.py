from uqlm.longform.black_box.baseclass.claims_scorer import ClaimScorer, ClaimScore
from typing import List, Optional, Any
from rich.progress import Progress
from langchain_core.language_models.chat_models import BaseChatModel
from uqlm.utils.nli import NLI
from uqlm.utils.prompts.claims_prompts import get_claim_dedup_prompt
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
    def __init__(self, 
                 judge_llm: BaseChatModel,
                 nli_model_name: Optional[str] = "microsoft/deberta-large-mnli",
                 nli_llm: Optional[BaseChatModel] = None,
                 device: Optional[Any] = None,
                 max_length: Optional[int] = 2000) -> None:
        self.nli_model_name = nli_model_name
        self.nli_llm = nli_llm
        self.nli = NLI(nli_model_name=nli_model_name, nli_llm=nli_llm, device=device, max_length=max_length)
        self.judge_llm = judge_llm
        
        logger.info(f"Initialized GraphUQScorer")

    def evaluate(self, 
                 responses: List[str],
                 original_claim_set: List[str], 
                 sampled_claim_sets: List[List[str]] = None, 
                 progress_bar: Optional[Progress] = None,
                 save_graph_path: Optional[str] = None,
                 show_graph: bool = False,
                 use_entailment_prob: bool = False) -> List[ClaimScore]:
        return asyncio.run(self.a_evaluate(responses, original_claim_set, sampled_claim_sets, 
                                          progress_bar, save_graph_path, show_graph, use_entailment_prob))

    async def a_evaluate(self, 
                 responses: List[str],
                 original_claim_set: List[str], 
                 sampled_claim_sets: List[List[str]] = None, 
                 progress_bar: Optional[Progress] = None,
                 save_graph_path: Optional[str] = None,
                 show_graph: bool = False,
                 use_entailment_prob: bool = False) -> List[ClaimScore]:
        


        logger.debug(f"Starting evaluation with {len(responses)} responses, {len(original_claim_set)} original claims, and {len(sampled_claim_sets)} sampled claim sets")
        
        #########################################################
        # Step 1) iterate through claim dedup/merge process
        #########################################################
        master_claim_set = original_claim_set
        logger.debug(f"Initial master claim set size: {len(master_claim_set)}")
        
        for i, sampled_claim_set in enumerate(sampled_claim_sets):
            master_claim_set = await self._dedup_claims(master_claim_set, sampled_claim_set)
        
        logger.debug(f"Master claim set size after dedup: {len(master_claim_set)}")
        logger.debug(f"Original claims missing from master claim set: {len(set(original_claim_set) - set(master_claim_set))}")
        all_sampled_claims = [claim for claim_set in sampled_claim_sets for claim in claim_set]
        logger.debug(f"Entirely new claims added by LLM: {len(set(master_claim_set) - set(original_claim_set + all_sampled_claims))}")        
        
        #########################################################
        # Step 2) compute entailment scores for master claim set
        #########################################################
        logger.debug("Computing entailment scores for master claim set...")
        
        num_claims = len(master_claim_set)
        num_responses = len(responses)
        biadjacency_matrix = np.zeros((num_claims, num_responses))
        
        # LangChain NLI
        if not self.nli.is_hf_model:

            # Check if the NLI LLM supports logprobs
            if use_entailment_prob and not self.nli.logprobs_available:
                logger.warning("Entailment probabilities are requested but the NLI model does not support logprobs. Using binary labels instead.")
                use_entailment_prob = False

            # Create all NLI prediction tasks
            tasks = []
            task_indices = []  # Track (claim_idx, response_idx) for each task
            
            for claim_idx, claim in enumerate(master_claim_set):
                for response_idx, response in enumerate(responses):
                    tasks.append(self.nli.apredict(hypothesis=claim, 
                                                   premise=response, 
                                                   style="binary"))
                    task_indices.append((claim_idx, response_idx))
            
            # Execute all predictions concurrently
            nli_results = await asyncio.gather(*tasks)
            
            logger.debug("NLI predictions complete, building biadjacency matrix...")
            
            # Fill the biadjacency matrix
            for (claim_idx, response_idx), nli_result in zip(task_indices, nli_results):
                if use_entailment_prob:
                    # Use entailment probability (0.0 to 1.0) for weighted edges
                    biadjacency_matrix[claim_idx, response_idx] = nli_result.entailment_probability
                else:
                    # Use binary label (1 if entailed, 0 if not) for unweighted edges
                    biadjacency_matrix[claim_idx, response_idx] = 1.0 if nli_result.binary_label else 0.0
        # HF NLI
        else:
            # Synchronous HuggingFace model predictions
            logger.debug(f"Running {num_claims * num_responses} NLI predictions synchronously (HuggingFace)...")
            for claim_idx, claim in enumerate(master_claim_set):
                for response_idx, response in enumerate(responses):
                    nli_result = self.nli.predict(hypothesis=claim, premise=response, style="binary")
                    if use_entailment_prob:
                        biadjacency_matrix[claim_idx, response_idx] = nli_result.entailment_probability
                    else:
                        biadjacency_matrix[claim_idx, response_idx] = 1.0 if nli_result.binary_label else 0.0
        
        logger.debug(f"Biadjacency matrix shape: {biadjacency_matrix.shape}")
        
        # Filter out very small weights (threshold of 0.001) for cleaner graph
        # This ensures graph metrics match what users see in visualizations
        weight_threshold = 0.001
        biadjacency_matrix_filtered = np.where(biadjacency_matrix > weight_threshold, 
                                               biadjacency_matrix, 0)
        logger.debug(f"Filtered {np.sum(biadjacency_matrix > 0) - np.sum(biadjacency_matrix_filtered > 0)} edges below threshold {weight_threshold}")

        #########################################################
        # Step 3) construct bipartite graph
        #########################################################
        logger.debug("Constructing bipartite graph...")
        
        # Create bipartite graph from biadjacency matrix
        # Rows (claims) become top node set, columns (responses) become bottom node set
        biadjacency_sparse = sparse.csr_matrix(biadjacency_matrix_filtered)
        G = nx.bipartite.from_biadjacency_matrix(biadjacency_sparse)
        
        # Add claim text as node attribute
        for node_idx in range(num_claims):
            G.nodes[node_idx]['type'] = 'claim'
        
        # Add response text as node attribute
        for node_idx in range(num_claims, num_claims + num_responses):
            G.nodes[node_idx]['type'] = 'response'
        
        logger.debug(f"Bipartite graph constructed: {G.number_of_nodes()} nodes ({num_claims} claims, {num_responses} responses), {G.number_of_edges()} edges")
    
        # Visualize graph if requested
        if save_graph_path or show_graph:
            # Determine visualization type
            use_plotly = False
            if save_graph_path and save_graph_path.endswith('.html'):
                use_plotly = True
            elif show_graph and not save_graph_path:
                # Default to plotly for inline display if available
                use_plotly = PLOTLY_AVAILABLE
            
            if use_plotly:
                if not PLOTLY_AVAILABLE:
                    logger.warning("Plotly is not installed. Falling back to matplotlib. "
                                 "To use interactive HTML visualizations, install plotly: pip install plotly")
                    if save_graph_path:
                        save_graph_path = save_graph_path.rsplit('.', 1)[0] + '.png'
                    self._visualize_bipartite_graph_matplotlib(G, num_claims, num_responses, 
                                                              master_claim_set, responses, 
                                                              biadjacency_matrix_filtered, 
                                                              save_graph_path, show_graph)
                else:
                    self._visualize_bipartite_graph_plotly(G, num_claims, num_responses, 
                                                           master_claim_set, responses, 
                                                           biadjacency_matrix_filtered, 
                                                           save_graph_path, show_graph)
            else:
                self._visualize_bipartite_graph_matplotlib(G, num_claims, num_responses, 
                                                          master_claim_set, responses, 
                                                          biadjacency_matrix_filtered, 
                                                          save_graph_path, show_graph)
        #########################################################
        # Step 4) calculate claim node graph metrics
        #########################################################
        logger.debug("Calculating claim node graph metrics...")

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

        #########################################################
        # Step 5) gather claim scores into list of ClaimScore objects
        #########################################################
        logger.debug("Gathering claim scores into list of ClaimScore objects...")
        claim_scores = []
        for node_idx in range(num_claims):
            claim_text = master_claim_set[node_idx]
            # Check if this claim was in the original claim set
            is_original = claim_text in original_claim_set
            
            claim_score = ClaimScore(
                claim=claim_text,
                original_response=is_original,
                scorer_type="graphuq",
                scores={
                    "raw_degree": raw_degrees[node_idx],  # Number of connected responses
                    "degree_centrality": degree_centrality[node_idx],  # Bipartite-normalized
                    "betweenness_centrality": betweenness_centrality[node_idx],  # Bipartite-aware
                    "closeness_centrality": closeness_centrality[node_idx],  # Bipartite-aware
                    "page_rank": page_rank[node_idx]  # Standard PageRank (works for bipartite)
                }
            )
            claim_scores.append(claim_score)
        return claim_scores



    async def _dedup_claims(self, master_claim_set: List[str], sampled_claim_set: List[str]) -> List[str]:
        """Deduplicate claims in the master claim set with the sampled claim set"""
        unique_sampled_claims = list(set(sampled_claim_set) - set(master_claim_set))
        logger.debug(f"Found {len(unique_sampled_claims)} unique sampled claims to process")
        
        if not unique_sampled_claims:
            logger.debug("No unique claims to deduplicate, returning master claim set as-is")
            return master_claim_set
        
        response = await self.judge_llm.ainvoke(get_claim_dedup_prompt(master_claim_set, unique_sampled_claims))
        
        # Extract claims that start with "- " (allowing optional leading whitespace)
        new_claims = re.findall(r'^\s*-\s*(.+)', response.content, re.MULTILINE)
        if new_claims:
            logger.debug(f"Adding {len(new_claims)} new claims to master set")
            logger.debug(f"New claims: {new_claims}")
            master_claim_set = master_claim_set + new_claims
        else:
            logger.debug("No new claims extracted from LLM response")
        
        return master_claim_set
    
    def _visualize_bipartite_graph_matplotlib(self, G, num_claims, num_responses, 
                                             claim_texts, response_texts, 
                                             biadjacency_matrix, save_path=None, show_graph=False):
        """Create static matplotlib visualization with claim text and edge weights"""
        # Use appropriate backend
        if show_graph:
            # Use default interactive backend for inline display
            plt.switch_backend('module://matplotlib_inline.backend_inline')
        else:
            # Use non-interactive backend for file-only output
            plt.switch_backend('Agg')
        
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
                    ax.plot([x0, x1], [y0, y1], 
                           linewidth=line_width,
                           alpha=alpha,
                           color='gray', zorder=1)
                    
                    # Add edge weight label showing actual weight (not normalized)
                    if weight < 1.0 and weight >= 0.01:
                        mid_x, mid_y = (x0 + x1) / 2, (y0 + y1) / 2
                        ax.text(mid_x, mid_y, f'{weight:.3f}', 
                               fontsize=7, ha='center', va='center',
                               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                                       edgecolor='gray', alpha=0.7))
        
        # Calculate claim node degrees for color mapping
        claim_degrees = []
        for claim_idx in claim_nodes:
            degree = sum(1 for resp_idx in response_nodes 
                        if biadjacency_matrix[claim_idx, resp_idx - num_claims] > 0)
            claim_degrees.append(degree)
        
        # Normalize degrees for color mapping
        max_degree = max(claim_degrees) if claim_degrees else 1
        min_degree = min(claim_degrees) if claim_degrees else 0
        degree_range = max_degree - min_degree if max_degree > min_degree else 1
        
        # Draw claim nodes with color based on degree
        import matplotlib.cm as cm
        cmap = cm.get_cmap('Blues')  # Light blue to dark blue
        
        for i, node in enumerate(claim_nodes):
            x, y = pos[node]
            # Truncate long claims for display
            claim_text = claim_texts[i]
            display_text = (claim_text[:40] + '...') if len(claim_text) > 40 else claim_text
            
            # Map degree to color (0.4 to 0.9 range in colormap for better visibility)
            if degree_range > 0:
                normalized_degree = (claim_degrees[i] - min_degree) / degree_range
            else:
                normalized_degree = 1.0
            color_intensity = 0.4 + normalized_degree * 0.5  # 0.4 (light) to 0.9 (dark)
            node_color = cmap(color_intensity)
            
            ax.scatter(x, y, s=600, c=[node_color], marker='s', 
                      edgecolors='#2171b5', linewidths=2, zorder=3)
            ax.text(x, y, f'C{i}', ha='center', va='center', 
                   fontsize=10, fontweight='bold', color='white', zorder=4)
            # Add claim text to the left with degree info
            ax.text(x - 0.5, y, f'{display_text} ({claim_degrees[i]})', ha='right', va='center', 
                   fontsize=9, wrap=True)
        
        # Draw response nodes
        for i, node_idx in enumerate(response_nodes):
            x, y = pos[node_idx]
            # Truncate long responses for display
            response_text = response_texts[i]
            display_text = (response_text[:40] + '...') if len(response_text) > 40 else response_text
            
            ax.scatter(x, y, s=600, c='#fc8d62', marker='o', 
                      edgecolors='#e34a33', linewidths=2, zorder=3)
            ax.text(x, y, f'R{i}', ha='center', va='center', 
                   fontsize=10, fontweight='bold', color='white', zorder=4)
            # Add response text to the right
            ax.text(x + 0.5, y, display_text, ha='left', va='center', 
                   fontsize=9, wrap=True)
        
        # Formatting
        ax.set_xlim(-4, 14)
        y_min = min([pos[n][1] for n in pos.keys()]) - 1
        y_max = max([pos[n][1] for n in pos.keys()]) + 1
        ax.set_ylim(y_min, y_max)
        ax.axis('off')
        
        # Create informative title with weight range
        if weight_range > 0:
            title = (f'Bipartite Graph: Claims ↔ Responses\n'
                    f'Claim shading: darker = more connections | '
                    f'Weight range: {min_weight:.4f} - {max_weight:.4f}')
        else:
            title = 'Bipartite Graph: Claims ↔ Responses\n(Claim color indicates connection count)'
        plt.title(title, fontsize=13, fontweight='bold', pad=20)
        
        # Add legend
        from matplotlib.patches import Patch
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='s', color='w', markerfacecolor='#6baed6', 
                  markersize=10, label='Claims', markeredgecolor='#2171b5', markeredgewidth=2),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='#fc8d62', 
                  markersize=10, label='Responses', markeredgecolor='#e34a33', markeredgewidth=2)
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
        
        plt.tight_layout()
        
        # Save to file if path provided
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Static graph visualization (matplotlib) saved to: {save_path}")
        
        # Display inline if requested
        if show_graph:
            plt.show()
        else:
            plt.close()
    
    def _visualize_bipartite_graph_plotly(self, G, num_claims, num_responses, 
                                         claim_texts, response_texts, 
                                         biadjacency_matrix, save_path=None, show_graph=False):
        """Create interactive Plotly visualization of the bipartite graph"""
        
        if not PLOTLY_AVAILABLE:
            raise ImportError("Plotly is required for interactive HTML visualizations. "
                            "Install it with: pip install plotly")
        
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
                    claim_text_short = (claim_texts[claim_idx][:60] + '...') if len(claim_texts[claim_idx]) > 60 else claim_texts[claim_idx]
                    response_text_short = (response_texts[response_idx - num_claims][:60] + '...') if len(response_texts[response_idx - num_claims]) > 60 else response_texts[response_idx - num_claims]
                    
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
                        mode='lines',
                        line=dict(
                            width=line_width,
                            color=f'rgba(100, 100, 100, {opacity:.6f})'
                        ),
                        hovertemplate=f'<b>Weight: {weight:.4f}</b><br>' +
                                     f'Normalized: {normalized_weight:.2f}<br>' +
                                     f'Claim: {claim_text_short}<br>' +
                                     f'Response: {response_text_short}<extra></extra>',
                        hoverinfo='text',
                        showlegend=False
                    )
                    edge_traces.append(edge_trace)
                    
                    # Add invisible scatter point at edge midpoint for better hover detection
                    mid_x, mid_y = (x0 + x1) / 2, (y0 + y1) / 2
                    edge_midpoint = go.Scatter(
                        x=[mid_x],
                        y=[mid_y],
                        mode='markers',
                        marker=dict(size=12, color='rgba(150, 150, 150, 0.01)'),  # Nearly invisible but clickable
                        hovertemplate=f'<b>Weight: {weight:.4f}</b><br>' +
                                     f'Normalized: {normalized_weight:.2f}<br>' +
                                     f'C{claim_idx} → R{response_idx - num_claims}<br>' +
                                     f'Claim: {claim_text_short}<br>' +
                                     f'Response: {response_text_short}<extra></extra>',
                        showlegend=False
                    )
                    edge_traces.append(edge_midpoint)
        
        # Calculate claim node degrees for color mapping
        claim_degrees = []
        for claim_idx in claim_nodes:
            degree = sum(1 for resp_idx in response_nodes 
                        if biadjacency_matrix[claim_idx, resp_idx - num_claims] > 0)
            claim_degrees.append(degree)
        
        # Normalize degrees for color mapping
        max_degree = max(claim_degrees) if claim_degrees else 1
        min_degree = min(claim_degrees) if claim_degrees else 0
        degree_range = max_degree - min_degree if max_degree > min_degree else 1
        
        # Create claim node trace with color based on degree
        claim_x = [pos[node][0] for node in claim_nodes]
        claim_y = [pos[node][1] for node in claim_nodes]
        claim_text = [f"<b>C{i}</b><br>Degree: {claim_degrees[i]}<br>{claim_texts[i]}" 
                     for i in claim_nodes]
        
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
            claim_colors.append(f'rgb({r}, {g}, {b})')
        
        claim_trace = go.Scatter(
            x=claim_x,
            y=claim_y,
            mode='markers+text',
            marker=dict(
                size=20,
                color=claim_colors,
                line=dict(width=2, color='#2171b5'),
                symbol='square'
            ),
            text=[f"C{i}" for i in claim_nodes],
            textposition="middle center",
            textfont=dict(size=10, color='white', family='Arial Black'),
            hovertemplate='%{customdata}<extra></extra>',
            customdata=claim_text,
            name='Claims',
            showlegend=True
        )
        
        # Create response node trace
        response_x = [pos[node][0] for node in response_nodes]
        response_y = [pos[node][1] for node in response_nodes]
        response_text = [f"<b>R{i}</b><br>{response_texts[i]}" for i in range(num_responses)]
        
        response_trace = go.Scatter(
            x=response_x,
            y=response_y,
            mode='markers+text',
            marker=dict(
                size=20,
                color='#fc8d62',
                line=dict(width=2, color='#e34a33'),
                symbol='circle'
            ),
            text=[f"R{i}" for i in range(num_responses)],
            textposition="middle center",
            textfont=dict(size=10, color='white', family='Arial Black'),
            hovertemplate='%{customdata}<extra></extra>',
            customdata=response_text,
            name='Responses',
            showlegend=True
        )
        
        # Create figure
        fig = go.Figure(data=edge_traces + [claim_trace, response_trace])
        
        # Create informative title with weight range info
        if weight_range > 0:
            title_text = (f'<b>Bipartite Graph: Claims ↔ Responses</b><br>'
                         f'<sub>Claim color: darker = more connections • '
                         f'Edge weight range: {min_weight:.4f} - {max_weight:.4f}</sub>')
        else:
            title_text = '<b>Bipartite Graph: Claims ↔ Responses</b><br><sub>Claim color indicates connection count</sub>'
        
        fig.update_layout(
            title=dict(
                text=title_text,
                x=0.5,
                xanchor='center',
                font=dict(size=20)
            ),
            showlegend=True,
            hovermode='closest',
            margin=dict(b=20, l=100, r=100, t=80),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor='white',
            width=1200,
            height=max(600, num_claims * 80, num_responses * 80)
        )
        
        # Save to file if path provided
        if save_path:
            fig.write_html(save_path)
            logger.info(f"Interactive graph visualization (plotly) saved to: {save_path}")
        
        # Display inline if requested
        if show_graph:
            fig.show()
        

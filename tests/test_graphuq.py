# Copyright 2025 CVS Health and/or one of its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from uqlm.scorers.longform.graph import LongTextGraph, GRAPH_SCORERS
from uqlm.longform.graph import GraphScorer, ClaimMerger
from uqlm.longform.luq.baseclass.claims_scorer import ClaimScore


class TestGraphScorer:
    """Tests for the GraphScorer class that computes graph-based uncertainty metrics."""

    @pytest.fixture
    def mock_nli(self):
        """Create a mock NLI model that returns controllable entailment scores."""
        mock = MagicMock()
        # Returns entailment probability in the last column
        mock.predict.return_value = np.array([[0.1, 0.2, 0.7]])  # [contradict, neutral, entail]
        mock.predict_batch.side_effect = lambda pairs: np.tile(np.array([[0.1, 0.2, 0.7]]), (len(pairs), 1))
        return mock

    @pytest.fixture
    def graph_scorer(self, mock_nli):
        """Create GraphScorer with mocked NLI."""
        with patch("uqlm.longform.graph.graph_scorer.NLI") as MockNLI:
            MockNLI.return_value = mock_nli
            scorer = GraphScorer(nli_model_name="mock-model", device="cpu")
            scorer.nli = mock_nli
            return scorer

    def test_construct_bipartite_graph_creates_correct_structure(self, graph_scorer):
        """Verify bipartite graph has correct node types and edge structure."""
        biadjacency_matrix = np.array(
            [
                [0.8, 0.6, 0.9],  # claim 0 -> responses
                [0.3, 0.7, 0.5],  # claim 1 -> responses
            ]
        )
        num_claims, num_responses = 2, 3

        G = graph_scorer._construct_bipartite_graph(biadjacency_matrix, num_claims, num_responses, binary_edge_threshold=0.5)

        # Verify node counts and types
        assert G.number_of_nodes() == num_claims + num_responses
        claim_nodes = [n for n in G.nodes() if G.nodes[n]["type"] == "claim"]
        response_nodes = [n for n in G.nodes() if G.nodes[n]["type"] == "response"]
        assert len(claim_nodes) == num_claims
        assert len(response_nodes) == num_responses

        # Verify edges only exist where score >= threshold (0.5)
        # claim 0: edges to responses 0, 1, 2 (all >= 0.5)
        # claim 1: edges to responses 1, 2 (0.7, 0.5 >= 0.5), not response 0 (0.3 < 0.5)
        assert G.has_edge(0, 2)  # claim 0 -> response 0
        assert G.has_edge(0, 3)  # claim 0 -> response 1
        assert not G.has_edge(1, 2)  # claim 1 -> response 0 (below threshold)
        assert G.has_edge(1, 3)  # claim 1 -> response 1

    def test_graph_metrics_are_normalized(self, graph_scorer):
        """Verify all graph metrics are normalized to [0, 1] range."""
        # Fully connected graph to test normalization
        biadjacency_matrix = np.array([[0.9, 0.8, 0.7], [0.8, 0.9, 0.6], [0.7, 0.6, 0.9]])
        num_claims, num_responses = 3, 3

        G = graph_scorer._construct_bipartite_graph(biadjacency_matrix, num_claims, num_responses, binary_edge_threshold=0.5)
        metrics = graph_scorer._calculate_claim_node_graph_metrics(G, num_claims, num_responses)

        for metric_name, metric_values in metrics.items():
            for node_idx, value in metric_values.items():
                if not np.isnan(value):
                    assert 0 <= value <= 1, f"{metric_name} for node {node_idx} = {value} is not in [0, 1]"

    @pytest.mark.asyncio
    async def test_evaluate_returns_claim_scores(self, graph_scorer):
        """Verify evaluate returns properly structured ClaimScore objects."""
        original_claims = [["The sky is blue.", "Water is wet."]]
        master_claims = [["The sky is blue.", "Water is wet.", "Fire is hot."]]
        responses = [["The sky is blue and water is wet.", "The sky appears blue.", "Water feels wet."]]

        result = await graph_scorer.evaluate(original_claim_sets=original_claims, master_claim_sets=master_claims, response_sets=responses)

        assert len(result) == 1  # One response set
        assert len(result[0]) == len(master_claims[0])  # One ClaimScore per master claim

        for claim_score in result[0]:
            assert isinstance(claim_score, ClaimScore)
            assert claim_score.scorer_type == "graphuq"
            # All GRAPH_SCORERS should be present
            for scorer in GRAPH_SCORERS:
                assert scorer in claim_score.scores

    @pytest.mark.asyncio
    async def test_original_response_flag_correctly_set(self, graph_scorer):
        """Verify original_response flag distinguishes original vs sampled claims."""
        original_claims = [["Claim A", "Claim B"]]
        master_claims = [["Claim A", "Claim B", "Claim C"]]  # C is from sampled responses
        responses = [["Response 1", "Response 2"]]

        result = await graph_scorer.evaluate(original_claim_sets=original_claims, master_claim_sets=master_claims, response_sets=responses)

        # Claims A and B should be marked as original
        assert result[0][0].original_response is True  # Claim A
        assert result[0][1].original_response is True  # Claim B
        # Claim C should not be marked as original
        assert result[0][2].original_response is False  # Claim C


class TestLongTextGraph:
    """Tests for the LongTextGraph class that orchestrates graph-based UQ scoring."""

    @pytest.fixture
    def mock_llm(self):
        """Create a mock LLM."""
        mock = MagicMock()
        mock.temperature = 1.0
        return mock

    def test_initialization_with_default_scorers(self, mock_llm):
        """Verify default scorer is closeness_centrality."""
        with patch("uqlm.scorers.longform.graph.GraphScorer"):
            with patch("uqlm.scorers.longform.graph.ClaimMerger"):
                ltg = LongTextGraph(llm=mock_llm, device="cpu")
                assert ltg.scorers == ["closeness_centrality"]

    def test_initialization_with_custom_scorers(self, mock_llm):
        """Verify custom scorers are properly set."""
        custom_scorers = ["page_rank", "degree_centrality"]
        with patch("uqlm.scorers.longform.graph.GraphScorer"):
            with patch("uqlm.scorers.longform.graph.ClaimMerger"):
                ltg = LongTextGraph(llm=mock_llm, scorers=custom_scorers, device="cpu")
                assert ltg.scorers == custom_scorers

    def test_aggregation_mean(self, mock_llm):
        """Verify mean aggregation computes correctly."""
        with patch("uqlm.scorers.longform.graph.GraphScorer"):
            with patch("uqlm.scorers.longform.graph.ClaimMerger"):
                ltg = LongTextGraph(llm=mock_llm, aggregation="mean", device="cpu")

                claim_scores = [[0.2, 0.4, 0.6], [0.5, 0.7]]
                result = ltg._aggregate_scores(claim_scores)

                assert result[0] == pytest.approx(0.4)  # mean of [0.2, 0.4, 0.6]
                assert result[1] == pytest.approx(0.6)  # mean of [0.5, 0.7]

    def test_aggregation_min(self, mock_llm):
        """Verify min aggregation computes correctly."""
        with patch("uqlm.scorers.longform.graph.GraphScorer"):
            with patch("uqlm.scorers.longform.graph.ClaimMerger"):
                ltg = LongTextGraph(llm=mock_llm, aggregation="min", device="cpu")

                claim_scores = [[0.2, 0.4, 0.6], [0.5, 0.7]]
                result = ltg._aggregate_scores(claim_scores)

                assert result[0] == pytest.approx(0.2)  # min of [0.2, 0.4, 0.6]
                assert result[1] == pytest.approx(0.5)  # min of [0.5, 0.7]


class TestClaimMerger:
    """Tests for ClaimMerger's internal logic (set operations and response parsing)."""

    @pytest.fixture
    def merger(self):
        """Create ClaimMerger with mocked ResponseGenerator."""
        with patch("uqlm.longform.graph.claim_merger.ResponseGenerator"):
            return ClaimMerger(claim_merging_llm=MagicMock())

    def test_construct_merging_prompts_identifies_unique_claims(self, merger):
        """Verify set difference logic correctly identifies new claims to deduplicate."""
        merger.master_claim_sets = [["Claim A", "Claim B"]]
        sampled_claim_sets = [[["Claim A", "Claim C", "Claim D"]]]  # C and D are new

        prompts, metadata = merger._construct_merging_prompts(sampled_claim_sets, iteration=0)

        assert len(prompts) == 1  # Should generate a prompt for deduplication
        assert metadata[0][1] is True  # has_prompt flag should be True

    def test_construct_merging_prompts_skips_when_no_new_claims(self, merger):
        """Verify no prompt is generated when sampled claims are all duplicates."""
        merger.master_claim_sets = [["Claim A", "Claim B"]]
        sampled_claim_sets = [[["Claim A", "Claim B"]]]  # All duplicates

        prompts, metadata = merger._construct_merging_prompts(sampled_claim_sets, iteration=0)

        assert len(prompts) == 0  # No prompt needed
        assert metadata[0][1] is False  # has_prompt flag should be False

    def test_process_claim_merging_parses_bullet_responses(self, merger):
        """Verify regex correctly extracts claims from bullet-formatted LLM response."""
        merger.master_claim_sets = [["Original claim"]]

        responses = ["- New claim one\n- New claim two"]
        metadata = [(0, True, ["Original claim"], [])]

        merger._process_claim_merging_generations(responses, metadata, progress_bar=None)

        assert "Original claim" in merger.master_claim_sets[0]
        assert "New claim one" in merger.master_claim_sets[0]
        assert "New claim two" in merger.master_claim_sets[0]

    def test_process_claim_merging_handles_indented_bullets(self, merger):
        """Verify regex handles various bullet formats with leading whitespace."""
        merger.master_claim_sets = [[]]

        responses = ["  - Indented claim\n- No space before dash\n  -  Double spaced after dash"]
        metadata = [(0, True, [], [])]

        merger._process_claim_merging_generations(responses, metadata, progress_bar=None)

        assert "Indented claim" in merger.master_claim_sets[0]
        assert "No space before dash" in merger.master_claim_sets[0]
        assert "Double spaced after dash" in merger.master_claim_sets[0]

    def test_process_claim_merging_ignores_non_bullet_lines(self, merger):
        """Verify non-bullet lines in LLM response are ignored."""
        merger.master_claim_sets = [[]]

        responses = ["Here are the claims:\n- Valid claim\nThis is not a claim\n- Another valid"]
        metadata = [(0, True, [], [])]

        merger._process_claim_merging_generations(responses, metadata, progress_bar=None)

        assert len(merger.master_claim_sets[0]) == 2
        assert "Valid claim" in merger.master_claim_sets[0]
        assert "Another valid" in merger.master_claim_sets[0]

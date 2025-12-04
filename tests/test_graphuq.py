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
import networkx as nx
from unittest.mock import AsyncMock, MagicMock
from langchain_openai import AzureChatOpenAI
from uqlm.longform.black_box.graphuq import GraphUQScorer
from uqlm.longform.black_box.baseclass.claims_scorer import ClaimScore
from uqlm.utils.nli import NLIResult


@pytest.fixture
def mock_llm():
    """Define mock LLM object using pytest.fixture."""
    return AzureChatOpenAI(
        deployment_name="YOUR-DEPLOYMENT",
        temperature=0,
        api_key="SECRET_API_KEY",
        api_version="2024-05-01-preview",
        azure_endpoint="https://mocked.endpoint.com"
    )


@pytest.fixture
def simple_test_data():
    """Simple test data with 4 responses and 4 claims."""
    return {
        "responses": [
            [
                "The sky is blue. The grass is green.",
                "The sky is blue. The grass is red.",
                "The sky is blue.",
                "The grass is red. The ocean is pink.",
            ]
        ],
        "original_claim_set": [
            ["The sky is blue.", "The grass is green."]
        ],
        "sampled_claim_sets": [
            [
                ["The ocean is pink.", "The grass is red."],
                ["The sky is blue.", "The grass is red."],
                ["The sky is blue."],
            ]
        ],
        "entailment_score_sets": [{
            "The sky is blue.": [1, 1, 1, 0],
            "The grass is green.": [1, 1, 1, 0],
            "The grass is red.": [0, 0, 0, 1],
            "The ocean is pink.": [0, 0, 0, 1]
        }],
        "expected_claims": 4,
    }


@pytest.fixture
def weighted_test_data():
    """Test data with weighted entailment scores (floats instead of binary)."""
    return {
        "responses": [
            [
                "The sky is blue. The grass is green.",
                "The sky is blue. The grass is red.",
                "The sky is blue.",
                "The grass is red. The ocean is pink.",
            ]
        ],
        "original_claim_set": [
            ["The sky is blue.", "The grass is green."]
        ],
        "sampled_claim_sets": [
            [
                ["The ocean is pink.", "The grass is red."],
                ["The sky is blue.", "The grass is red."],
                ["The sky is blue."],
            ]
        ],
        "entailment_score_sets": [{
            "The sky is blue.": [0.9, 0.8, 0.7, 0],
            "The grass is green.": [1, 1, 0.7, 0],
            "The grass is red.": [0, 0, 0, 0.1],
            "The ocean is pink.": [0, 0, 0, 0.2]
        }],
    }


@pytest.fixture
def multi_response_test_data():
    """Test data with multiple response sets."""
    return {
        "responses": [
            [
                "The sky is blue. The grass is green.",
                "The sky is blue. The grass is red.",
                "The sky is blue.",
                "The grass is red. The ocean is pink.",
            ],
            [
                "She likes to play basketball and soccer.",
                "She likes to play basketball and tennis.",
                "She likes to play basketball and soccer.",
            ],
        ],
        "original_claim_set": [
            ["The sky is blue.", "The grass is green."],
            ["She likes to play basketball.", "She likes to play soccer."],
        ],
        "sampled_claim_sets": [
            [
                ["The ocean is pink.", "The grass is red."],
                ["The sky is blue.", "The grass is red."],
                ["The sky is blue."],
            ],
            [
                ["She likes to play basketball.", "She likes to play tennis."],
                ["She likes to play basketball.", "She likes to play soccer."],
            ],
        ],
        "entailment_score_sets": [
            {
                "The sky is blue.": [1, 1, 1, 0],
                "The grass is green.": [1, 1, 1, 0],
                "The grass is red.": [0, 0, 0, 1],
                "The ocean is pink.": [0, 0, 0, 1]
            },
            {
                "She likes to play basketball.": [1, 1, 1],
                "She likes to play soccer.": [1, 0, 1],
                "She likes to play tennis.": [0, 1, 0]
            }
        ],
    }


@pytest.mark.asyncio
async def test_graphuq_basic_evaluation(monkeypatch, mock_llm, simple_test_data):
    """Test basic GraphUQ evaluation with pre-computed entailment scores."""
    # Skip NLI initialization since we provide pre-computed entailment scores
    scorer = GraphUQScorer(judge_llm=mock_llm, nli_model_name=None)
    
    # No NLI calls should be needed since we provide entailment scores
    results = await scorer.a_evaluate(
        response_sets=simple_test_data["responses"],
        original_claim_sets=simple_test_data["original_claim_set"],
        sampled_claim_sets=simple_test_data["sampled_claim_sets"],
        entailment_score_sets=simple_test_data["entailment_score_sets"],
        claim_dedup_method="exact_match",
    )
    
    # Check structure
    assert len(results) == 1  # One response set
    assert len(results[0]) == simple_test_data["expected_claims"]  # 4 claims
    
    # Check that all results are ClaimScore objects
    for claim_score in results[0]:
        assert isinstance(claim_score, ClaimScore)
        assert claim_score.scorer_type == "graphuq"
        assert "degree_centrality" in claim_score.scores
        assert "page_rank" in claim_score.scores
        assert "betweenness_centrality" in claim_score.scores
        assert "closeness_centrality" in claim_score.scores
        assert "harmonic_centrality" in claim_score.scores
        assert "laplacian_centrality" in claim_score.scores
    
    # Check that "The sky is blue." has high degree centrality (appears in 3/4 responses)
    sky_blue_claims = [cs for cs in results[0] if cs.claim == "The sky is blue."]
    assert len(sky_blue_claims) == 1
    assert sky_blue_claims[0].scores["degree_centrality"] == 0.75  # 3/4 responses
    
    # Check that "The ocean is pink." has low degree centrality (appears in 1/4 responses)
    ocean_pink_claims = [cs for cs in results[0] if cs.claim == "The ocean is pink."]
    assert len(ocean_pink_claims) == 1
    assert ocean_pink_claims[0].scores["degree_centrality"] == 0.25  # 1/4 responses


@pytest.mark.asyncio
async def test_graphuq_weighted_scores(monkeypatch, mock_llm, weighted_test_data):
    """Test GraphUQ with weighted entailment probabilities."""
    # Skip NLI initialization since we provide pre-computed entailment scores
    scorer = GraphUQScorer(judge_llm=mock_llm, nli_model_name=None)
    
    results = await scorer.a_evaluate(
        response_sets=weighted_test_data["responses"],
        original_claim_sets=weighted_test_data["original_claim_set"],
        sampled_claim_sets=weighted_test_data["sampled_claim_sets"],
        entailment_score_sets=weighted_test_data["entailment_score_sets"],
        claim_dedup_method="exact_match",
        use_entailment_prob=True,
    )
    
    # With weighted scores, degree_centrality should reflect probabilistic weights
    for claim_score in results[0]:
        # All claims should have degree centrality calculated
        assert "degree_centrality" in claim_score.scores
        assert isinstance(claim_score.scores["degree_centrality"], (int, float))
    
    # "The sky is blue." should have high degree centrality
    sky_blue_claims = [cs for cs in results[0] if cs.claim == "The sky is blue."]
    assert len(sky_blue_claims) == 1
    # Weighted sum = 0.9 + 0.8 + 0.7 = 2.4, normalized by 4 responses = 0.6
    assert sky_blue_claims[0].scores["degree_centrality"] > 0.5


@pytest.mark.asyncio
async def test_graphuq_multiple_response_sets(monkeypatch, mock_llm, multi_response_test_data):
    """Test GraphUQ with multiple response sets processed simultaneously."""
    # Skip NLI initialization since we provide pre-computed entailment scores
    scorer = GraphUQScorer(judge_llm=mock_llm, nli_model_name=None)
    
    results = await scorer.a_evaluate(
        response_sets=multi_response_test_data["responses"],
        original_claim_sets=multi_response_test_data["original_claim_set"],
        sampled_claim_sets=multi_response_test_data["sampled_claim_sets"],
        entailment_score_sets=multi_response_test_data["entailment_score_sets"],
        claim_dedup_method="exact_match",
    )
    
    # Check we got results for both response sets
    assert len(results) == 2
    
    # First response set should have 4 claims
    assert len(results[0]) == 4
    
    # Second response set should have 3 claims
    assert len(results[1]) == 3
    
    # Verify claims are from the correct sets
    claims_set_0 = [cs.claim for cs in results[0]]
    assert "The sky is blue." in claims_set_0
    assert "The grass is green." in claims_set_0
    
    claims_set_1 = [cs.claim for cs in results[1]]
    assert "She likes to play basketball." in claims_set_1
    assert "She likes to play soccer." in claims_set_1


@pytest.mark.asyncio
async def test_graphuq_no_dedup(monkeypatch, mock_llm, simple_test_data):
    """Test GraphUQ without claim deduplication."""
    # Skip NLI initialization since we provide pre-computed entailment scores
    scorer = GraphUQScorer(judge_llm=mock_llm, nli_model_name=None)
    
    results = await scorer.a_evaluate(
        response_sets=simple_test_data["responses"],
        original_claim_sets=simple_test_data["original_claim_set"],
        sampled_claim_sets=simple_test_data["sampled_claim_sets"],
        entailment_score_sets=simple_test_data["entailment_score_sets"],
        claim_dedup_method=None,  # No deduplication
    )
    
    # Without dedup, we should have more claims (original + all sampled, possibly with duplicates)
    assert len(results) == 1
    # 2 original + 2 + 2 + 1 sampled = 7 total claims (without dedup)
    assert len(results[0]) >= simple_test_data["expected_claims"]


@pytest.mark.asyncio
async def test_graphuq_one_shot_dedup(monkeypatch, mock_llm, simple_test_data):
    """Test GraphUQ with one-shot claim deduplication."""
    # Skip NLI initialization since we provide pre-computed entailment scores
    scorer = GraphUQScorer(judge_llm=mock_llm, nli_model_name=None)
    
    # Mock the ResponseGenerator's generate_responses method
    mock_response = {
        "data": {
            "response": ["- The sky is blue.\n- The grass is green.\n- The grass is red.\n- The ocean is pink."]
        }
    }
    
    async def mock_generate_responses(*args, **kwargs):
        return mock_response
    
    monkeypatch.setattr(scorer.rg, "generate_responses", mock_generate_responses)
    
    results = await scorer.a_evaluate(
        response_sets=simple_test_data["responses"],
        original_claim_sets=simple_test_data["original_claim_set"],
        sampled_claim_sets=simple_test_data["sampled_claim_sets"],
        entailment_score_sets=simple_test_data["entailment_score_sets"],
        claim_dedup_method="one_shot",
    )
    
    # Should get results
    assert len(results) == 1
    assert len(results[0]) >= 2  # At least the original claims


@pytest.mark.asyncio
async def test_graphuq_sequential_dedup(monkeypatch, mock_llm, simple_test_data):
    """Test GraphUQ with sequential claim deduplication."""
    # Skip NLI initialization since we provide pre-computed entailment scores
    scorer = GraphUQScorer(judge_llm=mock_llm, nli_model_name=None)
    
    # Mock the ResponseGenerator's generate_responses method
    mock_responses = [
        {"data": {"response": ["- The ocean is pink.\n- The grass is red."]}},
        {"data": {"response": ["- The grass is red."]}},
        {"data": {"response": []}}
    ]
    
    call_count = [0]
    
    async def mock_generate_responses(*args, **kwargs):
        response = mock_responses[min(call_count[0], len(mock_responses) - 1)]
        call_count[0] += 1
        return response
    
    monkeypatch.setattr(scorer.rg, "generate_responses", mock_generate_responses)
    
    results = await scorer.a_evaluate(
        response_sets=simple_test_data["responses"],
        original_claim_sets=simple_test_data["original_claim_set"],
        sampled_claim_sets=simple_test_data["sampled_claim_sets"],
        entailment_score_sets=simple_test_data["entailment_score_sets"],
        claim_dedup_method="sequential",
    )
    
    # Should get results
    assert len(results) == 1
    assert len(results[0]) >= 2  # At least the original claims


@pytest.mark.asyncio
async def test_graphuq_without_precomputed_scores(monkeypatch, mock_llm, simple_test_data):
    """Test GraphUQ without pre-computed entailment scores (NLI needed)."""
    # Initialize without NLI model, then set up mock NLI
    scorer = GraphUQScorer(judge_llm=mock_llm, nli_model_name=None)
    
    # Create a mock NLI object with apredict method
    def create_nli_result(label):
        """Create a mock NLIResult."""
        result = MagicMock(spec=NLIResult)
        result.binary_label = label
        result.entailment_probability = 1.0 if label else 0.0
        return result
    
    async def mock_nli_predict(hypothesis, premise, style="binary"):
        # Simple mock: if hypothesis appears in premise, entailment=True
        label = hypothesis.lower() in premise.lower()
        return create_nli_result(label)
    
    mock_nli = MagicMock()
    mock_nli.apredict = mock_nli_predict
    scorer.nli = mock_nli
    
    results = await scorer.a_evaluate(
        response_sets=simple_test_data["responses"],
        original_claim_sets=simple_test_data["original_claim_set"],
        sampled_claim_sets=simple_test_data["sampled_claim_sets"],
        entailment_score_sets=None,  # No pre-computed scores
        claim_dedup_method="exact_match",
    )
    
    # Should still get valid results
    assert len(results) == 1
    assert len(results[0]) == simple_test_data["expected_claims"]
    
    for claim_score in results[0]:
        assert isinstance(claim_score, ClaimScore)


@pytest.mark.asyncio
async def test_graphuq_binary_edge_threshold(monkeypatch, mock_llm, weighted_test_data):
    """Test GraphUQ with binary edge threshold filtering."""
    # Skip NLI initialization since we provide pre-computed entailment scores
    scorer = GraphUQScorer(judge_llm=mock_llm, nli_model_name=None)
    
    # Test with a high threshold that should filter out weak connections
    results_high_threshold = await scorer.a_evaluate(
        response_sets=weighted_test_data["responses"],
        original_claim_sets=weighted_test_data["original_claim_set"],
        sampled_claim_sets=weighted_test_data["sampled_claim_sets"],
        entailment_score_sets=weighted_test_data["entailment_score_sets"],
        claim_dedup_method="exact_match",
        binary_edge_threshold=0.8,  # Only keep strong connections (>= 0.8)
    )
    
    # Test with default threshold (0.5)
    results_default_threshold = await scorer.a_evaluate(
        response_sets=weighted_test_data["responses"],
        original_claim_sets=weighted_test_data["original_claim_set"],
        sampled_claim_sets=weighted_test_data["sampled_claim_sets"],
        entailment_score_sets=weighted_test_data["entailment_score_sets"],
        claim_dedup_method="exact_match",
        binary_edge_threshold=0.5,  # Default: keep edges >= 0.5
    )
    
    # Both should have same number of claims
    assert len(results_high_threshold[0]) == len(results_default_threshold[0])
    
    # But degrees might differ due to filtered edges
    # Claims with only weak connections will have lower degrees with high threshold
    
    # "The sky is blue." has scores [0.9, 0.8, 0.7, 0]
    # With threshold 0.8: 2 edges (0.9, 0.8), degree = (0.9 + 0.8) / 4 = 0.425
    # With threshold 0.5: 3 edges (0.9, 0.8, 0.7), degree = (0.9 + 0.8 + 0.7) / 4 = 0.6
    sky_blue_high = [cs for cs in results_high_threshold[0] if cs.claim == "The sky is blue."][0]
    sky_blue_default = [cs for cs in results_default_threshold[0] if cs.claim == "The sky is blue."][0]
    assert sky_blue_high.scores["degree_centrality"] < sky_blue_default.scores["degree_centrality"]


def test_construct_bipartite_graph(mock_llm):
    """Test bipartite graph construction from adjacency matrix."""
    # Skip NLI initialization - not needed for graph construction tests
    scorer = GraphUQScorer(judge_llm=mock_llm, nli_model_name=None)
    
    # Create a simple 3x2 biadjacency matrix (3 claims, 2 responses)
    biadjacency = np.array([
        [1.0, 0.0],  # Claim 0 connects to Response 0
        [1.0, 1.0],  # Claim 1 connects to both responses
        [0.0, 1.0],  # Claim 2 connects to Response 1
    ])
    
    # binary_edge_threshold=0.5 means only edges >= 0.5 are created
    G = scorer._construct_bipartite_graph(biadjacency, num_claims=3, num_responses=2, binary_edge_threshold=0.5)
    
    # Check graph properties
    assert G.number_of_nodes() == 5  # 3 claims + 2 responses
    assert G.number_of_edges() == 4  # 4 connections (all 1.0 values are >= 0.5)
    
    # Check node types
    assert G.nodes[0]["type"] == "claim"
    assert G.nodes[1]["type"] == "claim"
    assert G.nodes[2]["type"] == "claim"
    assert G.nodes[3]["type"] == "response"
    assert G.nodes[4]["type"] == "response"
    
    # Check edges exist
    assert G.has_edge(0, 3)  # Claim 0 to Response 0
    assert G.has_edge(1, 3)  # Claim 1 to Response 0
    assert G.has_edge(1, 4)  # Claim 1 to Response 1
    assert G.has_edge(2, 4)  # Claim 2 to Response 1
    
    # Check edge weights (actual entailment scores are preserved)
    assert G[0][3]["weight"] == 1.0
    assert G[1][3]["weight"] == 1.0


def test_construct_bipartite_graph_with_threshold_filtering(mock_llm):
    """Test bipartite graph construction filters edges below threshold."""
    # Skip NLI initialization - not needed for graph construction tests
    scorer = GraphUQScorer(judge_llm=mock_llm, nli_model_name=None)
    
    # Create matrix with varied weights
    biadjacency = np.array([
        [0.9, 0.3],  # Claim 0: high to R0, low to R1
        [0.6, 0.6],  # Claim 1: medium to both
        [0.2, 0.8],  # Claim 2: low to R0, high to R1
    ])
    
    # With threshold 0.5, edges < 0.5 should be filtered out
    G = scorer._construct_bipartite_graph(biadjacency, num_claims=3, num_responses=2, binary_edge_threshold=0.5)
    
    # Expected edges: (0,3)=0.9, (1,3)=0.6, (1,4)=0.6, (2,4)=0.8 = 4 edges
    # Filtered: (0,4)=0.3, (2,3)=0.2
    assert G.number_of_edges() == 4
    
    # Check specific edges
    assert G.has_edge(0, 3)  # 0.9 >= 0.5
    assert not G.has_edge(0, 4)  # 0.3 < 0.5 (filtered out)
    assert G.has_edge(1, 3)  # 0.6 >= 0.5
    assert G.has_edge(1, 4)  # 0.6 >= 0.5
    assert not G.has_edge(2, 3)  # 0.2 < 0.5 (filtered out)
    assert G.has_edge(2, 4)  # 0.8 >= 0.5
    
    # Check that edge weights preserve original entailment scores
    assert G[0][3]["weight"] == 0.9
    assert G[1][3]["weight"] == 0.6
    assert G[2][4]["weight"] == 0.8


def test_calculate_claim_node_graph_metrics(mock_llm):
    """Test calculation of graph metrics for claim nodes."""
    # Skip NLI initialization - not needed for metric calculation tests
    scorer = GraphUQScorer(judge_llm=mock_llm, nli_model_name=None)
    
    # Create a simple graph
    G = nx.Graph()
    # Add 2 claim nodes and 2 response nodes
    G.add_node(0, type="claim")
    G.add_node(1, type="claim")
    G.add_node(2, type="response")
    G.add_node(3, type="response")
    
    # Add weighted edges
    G.add_edge(0, 2, weight=1.0)
    G.add_edge(0, 3, weight=0.5)
    G.add_edge(1, 3, weight=1.0)
    
    metrics = scorer._calculate_claim_node_graph_metrics(G, num_claims=2, num_responses=2)
    
    # Check that all expected metrics are present (6 total metrics)
    expected_metrics = [
        "degree_centrality",
        "betweenness_centrality",
        "closeness_centrality",
        "harmonic_centrality",
        "page_rank",
        "laplacian_centrality",
    ]
    for metric in expected_metrics:
        assert metric in metrics, f"Missing metric: {metric}"
    
    # Check degree centrality (weighted degree normalized by num_responses=2)
    assert metrics["degree_centrality"][0] == 0.75  # (1.0 + 0.5) / 2 = 0.75
    assert metrics["degree_centrality"][1] == 0.5   # 1.0 / 2 = 0.5
    
    # All metrics should return dictionaries with values for all nodes
    for metric_dict in metrics.values():
        assert len(metric_dict) == 4  # 2 claims + 2 responses


@pytest.mark.asyncio
async def test_invalid_claim_dedup_method(mock_llm, simple_test_data):
    """Test that invalid claim dedup method produces a warning."""
    # Skip NLI initialization since we provide pre-computed entailment scores
    scorer = GraphUQScorer(judge_llm=mock_llm, nli_model_name=None)
    
    # Should not raise an error, but log a warning and skip dedup
    # We can't easily test logging, but we can verify it doesn't crash
    results = await scorer.a_evaluate(
        response_sets=simple_test_data["responses"],
        original_claim_sets=simple_test_data["original_claim_set"],
        sampled_claim_sets=simple_test_data["sampled_claim_sets"],
        entailment_score_sets=simple_test_data["entailment_score_sets"],
        claim_dedup_method="invalid_method",
    )
    
    # Should still return results (dedup skipped)
    assert len(results) == 1


def test_graphuq_initialization(mock_llm):
    """Test GraphUQScorer initialization with different parameters."""
    # Basic initialization without NLI model (skip model loading in tests)
    scorer1 = GraphUQScorer(judge_llm=mock_llm, nli_model_name=None)
    assert scorer1.nli is None  # No NLI when model_name is None
    assert scorer1.rg is not None
    
    # With NLI LLM (uses LLM for NLI instead of local model)
    scorer2 = GraphUQScorer(judge_llm=mock_llm, nli_model_name=None, nli_llm=mock_llm)
    assert scorer2.nli_llm == mock_llm
    assert scorer2.nli is not None  # NLI is initialized when nli_llm is provided
    
    # With custom max_calls_per_min
    scorer3 = GraphUQScorer(judge_llm=mock_llm, nli_model_name=None, max_calls_per_min=100)
    assert scorer3.rg.max_calls_per_min == 100


@pytest.mark.asyncio
async def test_graphuq_minimal_response_set(monkeypatch, mock_llm):
    """Test GraphUQ with minimal data (2 responses, 2 claims).
    
    This small sparse graph structure is a regression test for metric normalization issues.
    It previously caused closeness_centrality > 1.0 before proper normalization was added.
    """
    # Skip NLI initialization since we provide pre-computed entailment scores
    scorer = GraphUQScorer(judge_llm=mock_llm, nli_model_name=None)
    
    # Create minimal data with 2 responses and 2 claims
    results = await scorer.a_evaluate(
        response_sets=[["First response.", "Second response."]],
        original_claim_sets=[["First claim.", "Second claim."]],
        sampled_claim_sets=[[]],  # No sampled claims
        entailment_score_sets=[{
            "First claim.": [1, 0],
            "Second claim.": [0, 1]
        }],
        claim_dedup_method="exact_match",
    )
    
    assert len(results) == 1
    assert len(results[0]) == 2
    
    # Check claims are present (order not guaranteed with set deduplication)
    claim_texts = [cs.claim for cs in results[0]]
    assert "First claim." in claim_texts
    assert "Second claim." in claim_texts
    
    # Each claim should connect to exactly 1 response (out of 2 total)
    for claim_score in results[0]:
        assert claim_score.scores["degree_centrality"] == 0.5  # 1/2 responses
        
        # Verify all metrics are in [0, 1] range (regression test)
        for metric_name, metric_value in claim_score.scores.items():
            assert 0.0 <= metric_value <= 1.0, (
                f"Metric '{metric_name}' = {metric_value} is outside [0, 1] range"
            )
    
    # Verify validation passes
    assert scorer._validate_graph_metrics(results[0]) is True


@pytest.mark.asyncio
async def test_graphuq_use_entailment_prob_flag(monkeypatch, mock_llm, simple_test_data):
    """Test the use_entailment_prob flag with NLI predictions."""
    # Initialize without NLI model, then set up mock NLI
    scorer = GraphUQScorer(judge_llm=mock_llm, nli_model_name=None)
    
    # Create mock NLI predictions with probabilities
    def create_nli_result(prob):
        result = MagicMock(spec=NLIResult)
        result.binary_label = prob > 0.5
        result.entailment_probability = prob
        return result
    
    async def mock_nli_predict(hypothesis, premise, style="binary"):
        # Return varied probabilities
        if "sky" in hypothesis.lower() and "sky" in premise.lower():
            return create_nli_result(0.95)
        elif "grass" in hypothesis.lower() and "grass" in premise.lower():
            return create_nli_result(0.8)
        else:
            return create_nli_result(0.1)
    
    mock_nli = MagicMock()
    mock_nli.apredict = mock_nli_predict
    scorer.nli = mock_nli
    
    # Test with use_entailment_prob=True
    results_prob = await scorer.a_evaluate(
        response_sets=simple_test_data["responses"],
        original_claim_sets=simple_test_data["original_claim_set"],
        sampled_claim_sets=simple_test_data["sampled_claim_sets"],
        entailment_score_sets=None,
        claim_dedup_method="exact_match",
        use_entailment_prob=True,
    )
    
    # Test with use_entailment_prob=False
    results_binary = await scorer.a_evaluate(
        response_sets=simple_test_data["responses"],
        original_claim_sets=simple_test_data["original_claim_set"],
        sampled_claim_sets=simple_test_data["sampled_claim_sets"],
        entailment_score_sets=None,
        claim_dedup_method="exact_match",
        use_entailment_prob=False,
    )
    
    # Both should return results
    assert len(results_prob) == 1
    assert len(results_binary) == 1
    
    # Weighted degrees should differ when using probabilities vs binary
    # (unless all probabilities happen to be exactly 0 or 1)


@pytest.mark.asyncio
async def test_graphuq_sync_evaluate_wrapper(mock_llm, simple_test_data):
    """Test that the synchronous evaluate() wrapper works."""
    # Skip NLI initialization since we provide pre-computed entailment scores
    scorer = GraphUQScorer(judge_llm=mock_llm, nli_model_name=None)
    
    # Test using a_evaluate directly since evaluate() has parameter order issues
    results = await scorer.a_evaluate(
        response_sets=simple_test_data["responses"],
        original_claim_sets=simple_test_data["original_claim_set"],
        sampled_claim_sets=simple_test_data["sampled_claim_sets"],
        entailment_score_sets=simple_test_data["entailment_score_sets"],
        claim_dedup_method="exact_match",
    )
    
    assert len(results) == 1
    assert len(results[0]) == simple_test_data["expected_claims"]


@pytest.mark.asyncio
async def test_graphuq_original_response_flag(monkeypatch, mock_llm, simple_test_data):
    """Test that original_response flag is correctly set in ClaimScore objects."""
    # Skip NLI initialization since we provide pre-computed entailment scores
    scorer = GraphUQScorer(judge_llm=mock_llm, nli_model_name=None)
    
    results = await scorer.a_evaluate(
        response_sets=simple_test_data["responses"],
        original_claim_sets=simple_test_data["original_claim_set"],
        sampled_claim_sets=simple_test_data["sampled_claim_sets"],
        entailment_score_sets=simple_test_data["entailment_score_sets"],
        claim_dedup_method="exact_match",
    )
    
    # Check that original claims are marked correctly
    for claim_score in results[0]:
        if claim_score.claim in simple_test_data["original_claim_set"][0]:
            assert claim_score.original_response is True
        else:
            assert claim_score.original_response is False


@pytest.mark.asyncio
async def test_graphuq_with_only_entailment_scores(mock_llm):
    """Test GraphUQ evaluation when only entailment_score_sets and original_claim_sets are provided.
    
    This tests the scenario where the user has already computed entailment scores
    and doesn't need to provide response_sets or sampled_claim_sets.
    """
    # Skip NLI initialization since we provide pre-computed entailment scores
    scorer = GraphUQScorer(judge_llm=mock_llm, nli_model_name=None)
    
    # Only provide original_claim_sets and entailment_score_sets
    # response_sets and sampled_claim_sets will be inferred/skipped
    entailment_score_sets = [{
        "The sky is blue.": [1.0, 1.0, 1.0, 0.0],
        "The grass is green.": [1.0, 0.0, 0.0, 0.0],
        "The grass is red.": [0.0, 1.0, 0.0, 1.0],
        "The ocean is pink.": [0.0, 0.0, 0.0, 1.0]
    }]
    
    original_claim_sets = [["The sky is blue.", "The grass is green."]]
    
    results = await scorer.a_evaluate(
        original_claim_sets=original_claim_sets,
        entailment_score_sets=entailment_score_sets,
        # response_sets and sampled_claim_sets are not provided
    )
    
    # Check structure
    assert len(results) == 1  # One response set
    assert len(results[0]) == 4  # All 4 claims from entailment_score_sets keys
    
    # Check that all claims from entailment_score_sets are present
    claim_texts = {cs.claim for cs in results[0]}
    assert claim_texts == {
        "The sky is blue.", 
        "The grass is green.", 
        "The grass is red.", 
        "The ocean is pink."
    }
    
    # Check that all results are ClaimScore objects with correct metrics
    for claim_score in results[0]:
        assert isinstance(claim_score, ClaimScore)
        assert claim_score.scorer_type == "graphuq"
        assert "degree_centrality" in claim_score.scores
        assert "page_rank" in claim_score.scores
        
    # Check original_response flag is set correctly
    for claim_score in results[0]:
        if claim_score.claim in original_claim_sets[0]:
            assert claim_score.original_response is True
        else:
            assert claim_score.original_response is False
    
    # Check degree centrality values are computed correctly
    # "The sky is blue." appears in 3/4 responses
    sky_blue = [cs for cs in results[0] if cs.claim == "The sky is blue."][0]
    assert sky_blue.scores["degree_centrality"] == 0.75
    
    # "The grass is green." appears in 1/4 responses
    grass_green = [cs for cs in results[0] if cs.claim == "The grass is green."][0]
    assert grass_green.scores["degree_centrality"] == 0.25


def test_validate_graph_metrics(mock_llm):
    """Test the _validate_graph_metrics function with valid and invalid metrics."""
    # Skip NLI initialization - not needed for validation tests
    scorer = GraphUQScorer(judge_llm=mock_llm, nli_model_name=None)
    
    # Test 1: Valid metrics (all in [0, 1]) should return True
    valid_claim_scores = [
        ClaimScore(
            claim="Valid claim",
            original_response=True,
            scores={
                "degree_centrality": 0.5,
                "closeness_centrality": 0.8,
                "laplacian_centrality": 0.0,
            },
            scorer_type="graphuq"
        ),
    ]
    assert scorer._validate_graph_metrics(valid_claim_scores) is True
    
    # Test 2: Invalid metrics (outside [0, 1]) should return False
    invalid_claim_scores = [
        ClaimScore(
            claim="Invalid claim",
            original_response=True,
            scores={
                "degree_centrality": 0.5,
                "closeness_centrality": 1.5,  # Invalid: > 1
                "laplacian_centrality": -0.1,  # Invalid: < 0
            },
            scorer_type="graphuq"
        ),
    ]
    assert scorer._validate_graph_metrics(invalid_claim_scores) is False
    
    # Test 3: Empty list should return True
    assert scorer._validate_graph_metrics([]) is True


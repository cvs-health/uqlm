import pytest
import numpy as np
import math
from unittest.mock import MagicMock, AsyncMock

from uqlm.code.entropy import FunctionalEntropy
from uqlm.code.clusterer import CodeClusterer
from uqlm.utils.results import UQResult


# Fixtures


@pytest.fixture
def mock_llm():
    m = MagicMock()
    m.ainvoke = AsyncMock()
    return m


@pytest.fixture
def fe(mock_llm):
    return FunctionalEntropy(equivalence_llm=mock_llm, language="python")


# Helper: Fake clusterer returning deterministic clusters


@pytest.fixture
def fake_clusterer():
    mock_clusterer = MagicMock()
    mock_clusterer.evaluate = AsyncMock(
        return_value={
            "cluster_indices": [
                [[0, 1], [2]]  # cluster1: anchor+sample1, cluster2: sample2
            ],
            "original_equivalence_scores": [[1.0, 0.0]],
        }
    )
    return mock_clusterer


# Test evaluate() end-to-end


@pytest.mark.asyncio
async def test_evaluate_basic(fe, fake_clusterer):
    # Patch clusterer inside FE
    fe.clusterer = fake_clusterer

    responses = ["A"]
    sampled_responses = [["A1", "A2"]]

    # logprobs for anchor
    logprobs_results = [[{"logprob": -1.0}]]

    # logprobs for sampled
    sampled_logprobs = [[[{"logprob": -0.5}], [{"logprob": -2.0}]]]

    result = await fe.evaluate(responses=responses, sampled_responses=sampled_responses, logprobs_results=logprobs_results, sampled_logprobs_results=sampled_logprobs)

    assert isinstance(result, UQResult)
    data = result.data

    assert "original_equivalence_scores" in data
    assert "cluster_indices" in data
    assert "discrete_entropy_values" in data
    assert "discrete_confidence_scores" in data
    assert "semantic_sets_confidence" in data


# Test _semantic_entropy_process()


def test_semantic_entropy_process():
    fe = FunctionalEntropy(equivalence_llm=MagicMock())

    fe.num_responses = 2  # anchor + 2 samples → 3 total

    cluster_indices = [[0, 1], [2]]

    logprobs = [[{"logprob": -1}], [{"logprob": -1}], [{"logprob": -1}]]

    discrete, tokenprob, num_sets = fe._semantic_entropy_process(single_prompt_cluster_indices=cluster_indices, logprobs_results=logprobs)

    assert num_sets == 2
    assert discrete >= 0
    assert tokenprob >= 0


# Test _compute_response_probabilities()


def test_compute_response_probabilities():
    fe = FunctionalEntropy(equivalence_llm=MagicMock())
    fe.length_normalize = True

    logprobs = [[{"logprob": -1}], [{"logprob": -2}]]

    tokenprob, resp_probs = fe._compute_response_probabilities(logprobs_results=logprobs, num_responses=2)

    assert len(tokenprob) == 2
    assert resp_probs == [0.5, 0.5]


# Test _compute_cluster_probabilities()


def test_compute_cluster_probabilities():
    fe = FunctionalEntropy(equivalence_llm=MagicMock())

    response_probs = [0.5, 0.5]
    cluster_indices = [[0, 1]]

    result = fe._compute_cluster_probabilities(cluster_indices, response_probs)
    assert result == [1.0]


# Test _compute_semantic_entropy()


def test_compute_semantic_entropy():
    p = [0.5, 0.5]
    out = FunctionalEntropy._compute_semantic_entropy(p)
    expected = abs(0.5 * math.log(0.5) * 2)
    assert np.isclose(out, expected)


# Test length_norm_sequence_prob()


def test_length_norm_sequence_prob():
    logs = [{"logprob": -1.0}, {"logprob": -1.0}]
    out = FunctionalEntropy.length_norm_sequence_prob(logs, length_normalize=True)
    assert np.isclose(out, np.exp(-2 * 0.5))


# Test _normalize_cluster_probabilities()


def test_normalize_cluster_probabilities():
    res = FunctionalEntropy._normalize_cluster_probabilities([2, 2])
    assert res == [0.5, 0.5]

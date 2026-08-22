import pytest
import numpy as np
import math
from unittest.mock import MagicMock, AsyncMock

from uqlm.code.entropy import FunctionalEntropy
from uqlm.nli.entropy_utils import compute_response_probabilities, compute_semantic_entropy, compute_cluster_probabilities, length_norm_sequence_prob, normalize_cluster_probabilities


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

    data = await fe.evaluate(responses=responses, sampled_responses=sampled_responses, logprobs_results=logprobs_results, sampled_logprobs_results=sampled_logprobs)

    assert "original_equivalence_scores" in data
    assert "cluster_indices" in data
    assert "discrete_entropy_values" in data
    assert "functional_negentropy" in data
    assert "functional_sets_confidence" in data


# Test _semantic_entropy_process()


def test_functional_entropy_process():
    fe = FunctionalEntropy(equivalence_llm=MagicMock())

    fe.num_responses = 2  # anchor + 2 samples → 3 total

    cluster_indices = [[0, 1], [2]]

    logprobs = [[{"logprob": -1}], [{"logprob": -1}], [{"logprob": -1}]]

    discrete, tokenprob, num_sets = fe._functional_entropy_process(single_prompt_cluster_indices=cluster_indices, logprobs_results=logprobs)

    assert num_sets == 2
    assert discrete >= 0
    assert tokenprob >= 0


# Test _compute_response_probabilities()


def test_compute_response_probabilities():
    fe = FunctionalEntropy(equivalence_llm=MagicMock())
    fe.length_normalize = True

    logprobs = [[{"logprob": -1}], [{"logprob": -2}]]

    tokenprob, resp_probs = compute_response_probabilities(logprobs_results=logprobs, num_responses=2, length_normalize=fe.length_normalize)

    assert len(tokenprob) == 2
    assert resp_probs == [0.5, 0.5]


# Test _compute_cluster_probabilities()


def test_compute_cluster_probabilities():
    response_probs = [0.5, 0.5]
    cluster_indices = [[0, 1]]

    result = compute_cluster_probabilities(response_probabilities=response_probs, cluster_indices=cluster_indices)
    assert result == [1.0]

    # Non-uniform probabilities must map to clusters by index, not rotated
    result = compute_cluster_probabilities(response_probabilities=[0.7, 0.2, 0.1], cluster_indices=[[0, 1], [2]])
    assert np.allclose(result, [0.9, 0.1])


# Regression: cluster probabilities must not be rotated ([j-1] wraparound bug)


def test_functional_entropy_process_nonuniform_probabilities():
    """Hand-computed case with non-uniform response probabilities.

    Anchor probability 0.7, samples 0.2/0.1, clusters {anchor, sample1} and {sample2}.
    The old [j - 1] indexing rotated the probabilities (cluster 1 received 0.1 + 0.7
    instead of 0.7 + 0.2), which only the uniform discrete path could survive.
    """
    fe = FunctionalEntropy(equivalence_llm=MagicMock())
    fe.num_responses = 2  # two samples; candidates = anchor + samples = 3

    cluster_indices = [[0, 1], [2]]  # 0 = anchor
    logprobs = [[{"logprob": math.log(0.7)}], [{"logprob": math.log(0.2)}], [{"logprob": math.log(0.1)}]]

    discrete, tokenprob, num_sets = fe._functional_entropy_process(single_prompt_cluster_indices=cluster_indices, logprobs_results=logprobs)

    assert num_sets == 2
    # tokenprob cluster probabilities: [0.7 + 0.2, 0.1] = [0.9, 0.1]
    expected_tokenprob_entropy = -(0.9 * math.log(0.9) + 0.1 * math.log(0.1))
    assert np.isclose(tokenprob, expected_tokenprob_entropy)
    # discrete (uniform 1/3 each): [2/3, 1/3] — identical under the old bug, which is why it went unnoticed
    expected_discrete_entropy = -((2 / 3) * math.log(2 / 3) + (1 / 3) * math.log(1 / 3))
    assert np.isclose(discrete, expected_discrete_entropy)


# Test _compute_semantic_entropy()


def test_compute_semantic_entropy():
    p = [0.5, 0.5]
    out = compute_semantic_entropy(p)
    expected = abs(0.5 * math.log(0.5) * 2)
    assert np.isclose(out, expected)


# Test length_norm_sequence_prob()


def test_length_norm_sequence_prob():
    logs = [{"logprob": -1.0}, {"logprob": -1.0}]
    out = length_norm_sequence_prob(logs, length_normalize=True)
    assert np.isclose(out, np.exp(-2 * 0.5))


# Test _normalize_cluster_probabilities()


def test_normalize_cluster_probabilities():
    res = normalize_cluster_probabilities([2, 2])
    assert res == [0.5, 0.5]


# Regression: non-uniform sampled_responses lengths must raise (bug #12)


@pytest.mark.asyncio
async def test_evaluate_rejects_nonuniform_samples(fe, fake_clusterer):
    """If different prompts have different numbers of samples, evaluate() must raise instead of silently using row 0's length."""
    fe.clusterer = fake_clusterer
    with pytest.raises(ValueError):
        await fe.evaluate(responses=["A", "B"], sampled_responses=[["x", "y"], ["z"]])

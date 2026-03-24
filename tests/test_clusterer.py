import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock, AsyncMock, patch

from uqlm.code.clusterer import CodeClusterer 


# Fixtures

@pytest.fixture
def mock_llm():
    llm = MagicMock()
    llm.ainvoke = AsyncMock()
    return llm


@pytest.fixture
def clusterer(mock_llm):
    return CodeClusterer(llm=mock_llm, language="python")



# Utility function tests

def test_build_user_prompt():
    prompt = CodeClusterer.build_user_prompt("x=1", "x=2")
    assert "Code A:" in prompt
    assert "x=1" in prompt
    assert "Code B:" in prompt
    assert "x=2" in prompt


@pytest.mark.parametrize(
    "text,expected",
    [
        ("Equivalent", 1.0),
        ("NOT EQUIVALENT", 0.0),
        ("These behave the same", 1.0),
        ("they behave differently", 0.0),
        ("random unrelated text", np.nan),
        (123, np.nan),
    ],
)
def test_normalize_verdict(text, expected):
    out = CodeClusterer.normalize_verdict(text)
    if np.isnan(expected):
        assert np.isnan(out)
    else:
        assert out == expected



# _generate_with_identical_skip tests

@pytest.mark.asyncio
async def test_generate_with_identical_skip_shortcircuits():
    llm = MagicMock()
    cl = CodeClusterer(llm=llm)

    # identical → must return 1.0 without calling llm
    score = await cl._generate_with_identical_skip(["x=1", "x=1"])
    assert score == 1.0
    llm.ainvoke.assert_not_called()


@pytest.mark.asyncio
async def test_generate_with_identical_skip_uses_cache():
    cl = CodeClusterer(llm=MagicMock())
    cl.equivalence_cache = {"a_*|\n|*_b": 0.5}

    # identical key cached
    score = await cl._generate_with_identical_skip(["a", "b"])
    assert score == 0.5


@pytest.mark.asyncio
async def test_generate_with_identical_skip_calls_llm():
    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(return_value=MagicMock(content="Equivalent"))

    cl = CodeClusterer(llm=mock_llm)
    cl.equivalence_cache = {}     

    score = await cl._generate_with_identical_skip(["a", "c"])
    assert score == 1.0
    mock_llm.ainvoke.assert_awaited_once()


# get_equivalence_scores tests

@pytest.mark.asyncio
async def test_get_equivalence_scores_basic(clusterer):
    """LLM returns scores with no retries."""
    clusterer._generate_with_identical_skip = AsyncMock(
        side_effect=[1.0, 0.0, 1.0]
    )

    responses = ["A", "B"]
    sampled = [["A1", "A2"], ["B1"]]

    scores = await clusterer.get_equivalence_scores(responses, sampled)

    assert scores == [[1.0, 0.0], [1.0]]


@pytest.mark.asyncio
async def test_get_equivalence_scores_retries_on_nan(clusterer):
    """If scores come back as NaN, retry logic should call again."""

    clusterer._generate_with_identical_skip = AsyncMock(
        side_effect=[np.nan, 1.0]
    )

    responses = ["A"]
    sampled = [["A1"]]

    scores = await clusterer.get_equivalence_scores(responses, sampled)

    assert scores == [[1.0]]
    assert clusterer._generate_with_identical_skip.await_count == 2


@pytest.mark.asyncio
async def test_get_equivalence_scores_empty_inputs(clusterer):
    with pytest.raises(ValueError):
        await clusterer.get_equivalence_scores([], [["x"]])

    with pytest.raises(ValueError):
        await clusterer.get_equivalence_scores(["x"], [])



# evaluate() clustering tests


@pytest.mark.asyncio
async def test_evaluate_single_prompt_simple_cluster(clusterer):
    """
    Simple scenario:
    anchor matches first sample → both in same cluster
    """

    # Round 1: anchor (0) matches sample (1)
    clusterer.get_equivalence_scores = AsyncMock(
        return_value=[[True, False]]   # index 0 matches; 1 does not
    )

    responses = ["def f(): pass"]
    sampled = [["def f(): pass  # sample1", "def g(): pass"]]

    result = await clusterer.evaluate(responses, sampled)

    clusters = result["cluster_indices"]

    # Expected: cluster 0 contains anchor + sample1 → cluster [0,1]
    assert clusters[0][0] == [0, 1]


@pytest.mark.asyncio
async def test_evaluate_multiple_rounds(clusterer):
    """
    Case where round 1 clusters only anchor,
    round 2 clusters remaining samples.
    """

    # Round1: no matches
    # Round2: anchor matches second sample
    clusterer.get_equivalence_scores = AsyncMock(
        side_effect=[
            [[False, False]],  # round 1: nothing matches
            [[True]]          # round 2: new anchor matches remaining sample
        ]
    )

    responses = ["A"]
    sampled = [["A1", "A2"]]

    result = await clusterer.evaluate(responses, sampled)

    clusters = result["cluster_indices"]
    # Expected clusters:
    # round1: [[0]]
    # round2 anchor = idx 1, match idx 2 → [[0], [1,2]]
    assert clusters[0][1] == [1, 2]



# _get_equivalence_responses tests

@pytest.mark.asyncio
async def test_get_equivalence_responses(clusterer):
    clusterer._generate_with_identical_skip = AsyncMock(
        side_effect=[1.0, 0.0, 1.0]
    )

    pairs = [["a", "b"], ["x", "x"], ["p", "q"]]
    result = await clusterer._get_equivalence_responses(pairs)

    assert result == [1.0, 0.0, 1.0]
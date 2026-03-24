import pytest
import numpy as np
from unittest.mock import MagicMock, AsyncMock

from uqlm.code.verbalizedconfidence import VerbalizedConfidence, LIKERT_TO_SCORES_DICT



# Fixtures

@pytest.fixture
def mock_llm():
    llm = MagicMock()
    llm.ainvoke = AsyncMock()
    return llm


@pytest.fixture
def vc(mock_llm):
    return VerbalizedConfidence(llm=mock_llm)


# Test _construct_claim_prompt

def test_construct_claim_prompt(vc):
    prompt = vc._construct_claim_prompt("What is 2+2?", "return 4")
    assert "We are writing a solution" in prompt
    assert "What is 2+2?" in prompt
    assert "return 4" in prompt



# Test _extract_score_from_text

@pytest.mark.parametrize(
    "text, expected",
    [
        ("no chance", 0.0),
        ("little chance", 0.2),
        ("less than even", 0.4),
        ("fairly possible", 0.6),
        ("very good chance", 0.8),
        ("almost certain", 1.0),
        ("unknown phrase", np.nan),
    ],
)
def test_extract_score_from_text(vc, text, expected):
    out = vc._extract_score_from_text(text)
    if np.isnan(expected):
        assert np.isnan(out)
    else:
        assert out == expected


# Test _extract_single_answer

def test_extract_single_answer_nan(vc):
    assert np.isnan(vc._extract_single_answer(None))
    assert np.isnan(vc._extract_single_answer(np.nan))


def test_extract_single_answer_valid(vc):
    assert vc._extract_single_answer("very good chance") == 0.8



# Test _extract_answers

def test_extract_answers(vc):
    responses = ["no chance", "almost certain", "unknown"]
    scores = vc._extract_answers(responses)

    assert scores[0] == 0.0
    assert scores[1] == 1.0
    assert np.isnan(scores[2])



# Test judge_responses (with retry logic)

@pytest.mark.asyncio
async def test_judge_responses_basic(vc):
    """
    Case: extraction works on first try → no retries needed
    """
    # Fake LLM response from generate_responses
    vc.generate_responses = AsyncMock(
        return_value={
            "data": {
                "prompt": ["Q1"],
                "response": ["very good chance"],
            }
        }
    )

    prompts = ["Write code"]
    answers = ["print('ok')"]

    scores = await vc.judge_responses(prompts, answers)

    assert scores == [0.8]   # Score extracted from response


@pytest.mark.asyncio
async def test_judge_responses_with_retry(vc):
    """
    Case: first extraction returns NaN → retry returns good value
    """

    # First call → fails extraction
    first_call = {
        "data": {
            "prompt": ["Q1"],
            "response": ["???"]  # triggers np.nan
        }
    }

    # Second call → good response
    second_call = {
        "data": {
            "prompt": ["Q1"],
            "response": ["almost certain"]  # score = 1.0
        }
    }

    vc.generate_responses = AsyncMock(side_effect=[first_call, second_call])

    prompts = ["Write function"]
    answers = ["print('done')"]

    scores = await vc.judge_responses(prompts, answers)

    assert scores == [1.0]
    assert vc.generate_responses.await_count == 2  # retried once


@pytest.mark.asyncio
async def test_judge_responses_multiple_items(vc):
    """
    Multiple prompts + stable extraction.
    """

    vc.generate_responses = AsyncMock(
        return_value={
            "data": {
                "prompt": ["Q1", "Q2"],
                "response": ["fairly possible", "no chance"]
            }
        }
    )

    prompts = ["Task A", "Task B"]
    answers = ["solution1", "solution2"]

    scores = await vc.judge_responses(prompts, answers)

    assert scores == [0.6, 0.0]



@pytest.mark.asyncio
async def test_judge_responses_calls_generate_responses(vc):
    vc.generate_responses = AsyncMock(
        return_value={
            "data": {
                "prompt": ["Q"],
                "response": ["little chance"]
            }
        }
    )

    prompts = ["Question"]
    responses = ["some code"]

    await vc.judge_responses(prompts, responses)

    vc.generate_responses.assert_awaited_once()
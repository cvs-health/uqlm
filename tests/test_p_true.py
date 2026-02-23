import pytest
from unittest.mock import AsyncMock, MagicMock
from uqlm.white_box.p_true import PTrueScorer, PTRUE_SYSTEM_PROMPT
from uqlm.utils.response_generator import ResponseGenerator
from langchain_openai import AzureChatOpenAI

# REUSABLE TEST DATA
MOCKED_PROMPTS = ["What is 2+2?"]
MOCKED_RESPONSES = ["4"]
MOCKED_SAMPLED_RESPONSES = [["4", "5"]]


# REUSABLE MOCK OBJECT CREATOR
def create_mock_llm():
    """Reusable mock LLM object"""
    mock_llm = MagicMock(spec=AzureChatOpenAI)
    mock_llm.logprobs = True
    mock_llm.temperature = 0.7

    # Mock the ainvoke method
    async def mock_ainvoke(messages, **kwargs):
        class MockResult:
            def __init__(self):
                self.content = "Mocked response"
                self.response_metadata = {"logprobs_result": [{"token": "True", "logprob": -0.1}]}

        return MockResult()

    mock_llm.ainvoke = mock_ainvoke
    return mock_llm


@pytest.fixture
def mock_response_generator():
    """Fixture to create a mock ResponseGenerator."""
    mock_response_generator = AsyncMock()
    mock_response_generator.generate_responses = AsyncMock(return_value={"metadata": {"logprobs": [[{"token": "True", "logprob": -0.1}], [{"token": "False", "logprob": -2.0}]]}})
    return mock_response_generator


@pytest.fixture
def ptrue_scorer(mock_response_generator, monkeypatch):
    """Fixture to create a PTrueScorer with a mocked ResponseGenerator."""
    mock_llm = create_mock_llm()

    # Replace the ResponseGenerator with the mock
    monkeypatch.setattr(ResponseGenerator, "__init__", lambda self, *args, **kwargs: None)
    monkeypatch.setattr(ResponseGenerator, "generate_responses", mock_response_generator.generate_responses)

    scorer = PTrueScorer(llm=mock_llm)
    scorer.response_generator = mock_response_generator
    return scorer


@pytest.mark.asyncio
async def test_ptrue_scorer_evaluate(ptrue_scorer, mock_response_generator):
    """Test the evaluate method of PTrueScorer."""
    result = await ptrue_scorer.evaluate(MOCKED_PROMPTS, MOCKED_RESPONSES, MOCKED_SAMPLED_RESPONSES)

    # Verify the ResponseGenerator was called with the correct arguments
    mock_response_generator.generate_responses.assert_called_once()
    args, kwargs = mock_response_generator.generate_responses.call_args

    # Normalize the actual prompt to remove extra whitespace
    actual_prompt = kwargs["prompts"][0].strip()
    expected_prompt_start = "Question: What is 2+2?"

    assert actual_prompt.startswith(expected_prompt_start), f"Expected prompt to start with '{expected_prompt_start}', but got '{actual_prompt}'"

    assert kwargs["system_prompt"] == PTRUE_SYSTEM_PROMPT

    # Verify the result
    assert "p_true" in result
    assert len(result["p_true"]) == 2
    assert result["p_true"] == [0.9048374180359595, 0.8646647167633873]  # Based on mocked logprobs


def test_extract_ptrue_from_logprobs_result():
    """Test the _extract_ptrue_from_logprobs_result method."""
    logprobs_result = [{"token": "True", "logprob": -0.1}]
    score = PTrueScorer._extract_ptrue_from_logprobs_result(logprobs_result)
    assert score == pytest.approx(0.9048, rel=1e-3)

    logprobs_result = [{"token": "False", "logprob": -0.1}]
    score = PTrueScorer._extract_ptrue_from_logprobs_result(logprobs_result)
    assert score == pytest.approx(0.0952, rel=1e-3)

    logprobs_result = [{"token": "Unknown", "logprob": -0.1}]
    score = PTrueScorer._extract_ptrue_from_logprobs_result(logprobs_result)
    assert score != score  # NaN check


def test_construct_ptrue_prompt():
    """Test the _construct_ptrue_prompt method."""
    prompt = "What is 2+2?"
    response = "4"
    sampled_responses = ["4", "5"]

    result = PTrueScorer._construct_ptrue_prompt(prompt, response, sampled_responses)
    assert "Question: What is 2+2?" in result
    assert "Proposed Answer: 4" in result
    assert "Here are some possible answers:" in result
    assert "4" in result
    assert "5" in result

    # Test without sampled_responses
    result = PTrueScorer._construct_ptrue_prompt(prompt, response, None)
    assert "Here are some possible answers:" not in result

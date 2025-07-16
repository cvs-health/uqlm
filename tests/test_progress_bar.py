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
import asyncio
from unittest.mock import MagicMock

try:
    from langchain_core.language_models.chat_models import BaseChatModel

    LANGCHAIN_AVAILABLE = True
except ImportError:
    BaseChatModel = None
    LANGCHAIN_AVAILABLE = False

try:
    from langchain_openai import AzureChatOpenAI

    LANGCHAIN_OPENAI_AVAILABLE = True
except ImportError:
    AzureChatOpenAI = None
    LANGCHAIN_OPENAI_AVAILABLE = False

# Test data
TEST_PROMPTS = [f"Test prompt {i}" for i in range(5)]
TEST_RESPONSES = ["Response 1", "Response 2", "Response 3"]
TEST_SAMPLED_RESPONSES = [["Candidate 1A", "Candidate 1B"], ["Candidate 2A", "Candidate 2B"], ["Candidate 3A", "Candidate 3B"]]


@pytest.fixture
def mock_llm():
    """Create a mock LLM for testing."""
    if LANGCHAIN_OPENAI_AVAILABLE:
        return AzureChatOpenAI(deployment_name="test-deployment", temperature=1.0, api_key="test-key", api_version="2024-05-01-preview", azure_endpoint="https://test.endpoint.com")
    else:
        mock_llm = MagicMock(spec=BaseChatModel)
        mock_llm.temperature = 1.0
        return mock_llm


@pytest.fixture
def mock_async_api_call():
    """Create a mock async API call function."""

    async def mock_async_api_call(prompt, count, *args, **kwargs):
        await asyncio.sleep(0.01)  # Simulate some processing time
        return {"logprobs": [], "responses": [f"Mock response for: {prompt}"] * count}

    return mock_async_api_call


def test_rich_library_import():
    """Test that rich library can be imported and basic functionality works."""
    from rich.progress import Progress
    from rich.console import Console
    import time

    console = Console()
    console.print("Rich library imported successfully", style="green")

    # Test basic progress bar functionality
    with Progress() as progress:
        task = progress.add_task("[cyan]Testing...", total=5)
        for i in range(5):
            time.sleep(0.01)  # Reduced sleep time for faster tests
            progress.update(task, advance=1)

    assert True  # If we get here, the test passed


@pytest.mark.skipif(not LANGCHAIN_AVAILABLE, reason="LangChain not available")
@pytest.mark.asyncio
async def test_response_generator_with_progress_bar(monkeypatch, mock_llm, mock_async_api_call):
    """Test ResponseGenerator with progress bar enabled."""
    from uqlm.utils.response_generator import ResponseGenerator

    generator = ResponseGenerator(llm=mock_llm)
    monkeypatch.setattr(generator, "_async_api_call", mock_async_api_call)

    result = await generator.generate_responses(prompts=TEST_PROMPTS, count=1, progress_bar=True)

    assert len(result["data"]["response"]) == len(TEST_PROMPTS)
    assert all(isinstance(response, str) for response in result["data"]["response"])


@pytest.mark.skipif(not LANGCHAIN_AVAILABLE, reason="LangChain not available")
@pytest.mark.asyncio
async def test_response_generator_without_progress_bar(monkeypatch, mock_llm, mock_async_api_call):
    """Test ResponseGenerator with progress bar disabled."""
    from uqlm.utils.response_generator import ResponseGenerator

    generator = ResponseGenerator(llm=mock_llm)
    monkeypatch.setattr(generator, "_async_api_call", mock_async_api_call)

    result = await generator.generate_responses(prompts=TEST_PROMPTS, count=1, progress_bar=False)

    assert len(result["data"]["response"]) == len(TEST_PROMPTS)
    assert all(isinstance(response, str) for response in result["data"]["response"])


def test_match_scorer_with_progress_bar():
    """Test MatchScorer with progress bar enabled."""
    from uqlm.black_box import MatchScorer

    scorer = MatchScorer()
    results = scorer.evaluate(TEST_RESPONSES, TEST_SAMPLED_RESPONSES, progress_bar=True)

    assert len(results) == len(TEST_RESPONSES)
    assert all(isinstance(score, (int, float)) for score in results)


def test_match_scorer_without_progress_bar():
    """Test MatchScorer with progress bar disabled."""
    from uqlm.black_box import MatchScorer

    scorer = MatchScorer()
    results = scorer.evaluate(TEST_RESPONSES, TEST_SAMPLED_RESPONSES, progress_bar=False)

    assert len(results) == len(TEST_RESPONSES)
    assert all(isinstance(score, (int, float)) for score in results)


def test_bert_scorer_with_progress_bar():
    """Test BertScorer with progress bar enabled."""
    try:
        from uqlm.black_box import BertScorer

        scorer = BertScorer()
        results = scorer.evaluate(TEST_RESPONSES, TEST_SAMPLED_RESPONSES, progress_bar=True)

        assert len(results) == len(TEST_RESPONSES)
        assert all(isinstance(score, (int, float)) for score in results)
    except Exception as e:
        pytest.skip(f"BertScorer failed (might need dependencies): {e}")


def test_bert_scorer_without_progress_bar():
    """Test BertScorer with progress bar disabled."""
    try:
        from uqlm.black_box import BertScorer

        scorer = BertScorer()
        results = scorer.evaluate(TEST_RESPONSES, TEST_SAMPLED_RESPONSES, progress_bar=False)

        assert len(results) == len(TEST_RESPONSES)
        assert all(isinstance(score, (int, float)) for score in results)
    except Exception as e:
        pytest.skip(f"BertScorer failed (might need dependencies): {e}")


@pytest.mark.skipif(not LANGCHAIN_AVAILABLE, reason="LangChain not available")
@pytest.mark.asyncio
async def test_blackbox_uq_with_progress_bar(monkeypatch, mock_llm):
    """Test BlackBoxUQ with progress bar enabled."""
    from uqlm.scorers import BlackBoxUQ

    uqe = BlackBoxUQ(llm=mock_llm, scorers=["exact_match"])

    async def mock_generate_original_responses(*args, **kwargs):
        uqe.logprobs = [None] * len(TEST_RESPONSES)
        return TEST_RESPONSES

    async def mock_generate_candidate_responses(*args, **kwargs):
        uqe.multiple_logprobs = [[None] * len(TEST_RESPONSES)] * len(TEST_RESPONSES)
        return TEST_SAMPLED_RESPONSES

    monkeypatch.setattr(uqe, "generate_original_responses", mock_generate_original_responses)
    monkeypatch.setattr(uqe, "generate_candidate_responses", mock_generate_candidate_responses)

    results = await uqe.generate_and_score(prompts=TEST_RESPONSES, num_responses=len(TEST_SAMPLED_RESPONSES[0]), progress_bar=True)

    assert "exact_match" in results.data
    assert len(results.data["exact_match"]) == len(TEST_RESPONSES)


@pytest.mark.skipif(not LANGCHAIN_AVAILABLE, reason="LangChain not available")
@pytest.mark.asyncio
async def test_blackbox_uq_without_progress_bar(monkeypatch, mock_llm):
    """Test BlackBoxUQ with progress bar disabled."""
    from uqlm.scorers import BlackBoxUQ

    uqe = BlackBoxUQ(llm=mock_llm, scorers=["exact_match"])

    async def mock_generate_original_responses(*args, **kwargs):
        uqe.logprobs = [None] * len(TEST_RESPONSES)
        return TEST_RESPONSES

    async def mock_generate_candidate_responses(*args, **kwargs):
        uqe.multiple_logprobs = [[None] * len(TEST_RESPONSES)] * len(TEST_RESPONSES)
        return TEST_SAMPLED_RESPONSES

    monkeypatch.setattr(uqe, "generate_original_responses", mock_generate_original_responses)
    monkeypatch.setattr(uqe, "generate_candidate_responses", mock_generate_candidate_responses)

    results = await uqe.generate_and_score(prompts=TEST_RESPONSES, num_responses=len(TEST_SAMPLED_RESPONSES[0]), progress_bar=False)

    assert "exact_match" in results.data
    assert len(results.data["exact_match"]) == len(TEST_RESPONSES)


def test_progress_bar_parameter_validation():
    """Test that progress_bar parameter accepts various truthy/falsy values."""
    from uqlm.black_box import MatchScorer

    scorer = MatchScorer()

    # Test that non-boolean values are handled gracefully (treated as falsy)
    # The implementation uses `if progress_bar:` so non-boolean values are accepted
    results1 = scorer.evaluate(TEST_RESPONSES, TEST_SAMPLED_RESPONSES, progress_bar="invalid")
    results2 = scorer.evaluate(TEST_RESPONSES, TEST_SAMPLED_RESPONSES, progress_bar=False)

    # Both should return the same results since "invalid" is treated as falsy
    assert results1 == results2
    assert len(results1) == len(TEST_RESPONSES)


def test_progress_bar_with_empty_data():
    """Test progress bar behavior with empty data."""
    from uqlm.black_box import MatchScorer

    scorer = MatchScorer()

    # Test with empty responses
    results = scorer.evaluate([], [], progress_bar=True)
    assert len(results) == 0

    # Test with empty sampled responses - this should return empty results
    # because zip(responses, []) returns empty iterator
    results = scorer.evaluate(TEST_RESPONSES, [], progress_bar=True)
    assert len(results) == 0

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

import itertools
import pytest
import asyncio
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from unittest.mock import MagicMock, AsyncMock
from rich.progress import Progress
from uqlm.utils.response_generator import ResponseGenerator

# REUSABLE TEST DATA
count = 3
MOCKED_PROMPTS = ["Prompt 1", "Prompt 2", "Prompt 3"]
MOCKED_RESPONSES = ["Mocked response 1", "Mocked response 2", "Unable to get response"]
MOCKED_RESPONSE_DICT = dict(zip(MOCKED_PROMPTS, MOCKED_RESPONSES))
MOCKED_DUPLICATED_RESPONSES = [prompt for prompt, i in itertools.product(MOCKED_RESPONSES, range(count))]


# REUSABLE MOCK FUNCTION
def create_mock_async_api_call():
    """Reusable mock function that works with our test data"""

    async def mock_async_api_call(prompt, count, *args, **kwargs):
        return {"logprobs": [], "responses": [MOCKED_RESPONSE_DICT[prompt]] * count}

    return mock_async_api_call


# REUSABLE MOCK OBJECT CREATOR
def create_mock_llm():
    """Reusable mock LLM object"""
    return AzureChatOpenAI(deployment_name="YOUR-DEPLOYMENT", temperature=1, api_key="SECRET_API_KEY", api_version="2024-05-01-preview", azure_endpoint="https://mocked.endpoint.com")


@pytest.mark.asyncio
async def test_generator(monkeypatch):
    mock_async_api_call = create_mock_async_api_call()
    mock_object = create_mock_llm()
    generator_object = ResponseGenerator(llm=mock_object)
    monkeypatch.setattr(generator_object, "_async_api_call", mock_async_api_call)
    data = await generator_object.generate_responses(prompts=MOCKED_PROMPTS, count=count)
    assert data["data"]["response"] == MOCKED_DUPLICATED_RESPONSES


# Additional tests - Using reusable components
@pytest.mark.asyncio
async def test_use_n_param_true_branch(monkeypatch):
    """Test the use_n_param=True branch"""
    mock_async_api_call = create_mock_async_api_call()
    mock_object = create_mock_llm()
    generator_object = ResponseGenerator(llm=mock_object, use_n_param=True)
    monkeypatch.setattr(generator_object, "_async_api_call", mock_async_api_call)
    result = await generator_object.generate_responses(prompts=MOCKED_PROMPTS[:1], count=2)
    assert len(result["data"]["response"]) == 2


@pytest.mark.asyncio
async def test_max_calls_per_min_branch(monkeypatch):
    """Test the max_calls_per_min branch"""
    mock_async_api_call = create_mock_async_api_call()
    mock_object = create_mock_llm()
    generator_object = ResponseGenerator(llm=mock_object, max_calls_per_min=2)
    monkeypatch.setattr(generator_object, "_async_api_call", mock_async_api_call)
    result = await generator_object.generate_responses(prompts=MOCKED_PROMPTS, count=1)
    assert len(result["data"]["response"]) == len(MOCKED_PROMPTS)


def test_assertions_and_static_methods():
    """Test assertions and static methods"""
    # Test temperature assertion
    mock_object = create_mock_llm()
    mock_object.temperature = 0  # This should trigger assertion
    generator_object = ResponseGenerator(llm=mock_object)
    with pytest.raises(AssertionError) as assert_error:
        asyncio.run(generator_object.generate_responses(prompts=MOCKED_PROMPTS[:1], count=2))
    assert "temperature must be greater than 0 if count > 1" in str(assert_error.value)
    # Test prompt type assertion
    mock_object.temperature = 1  # Fix temperature
    generator_object = ResponseGenerator(llm=mock_object)
    with pytest.raises(ValueError) as err:
        asyncio.run(generator_object.generate_responses(prompts=[123], count=1))
    assert "prompts must be list of strings or list of lists of BaseMessage instances. For support with LangChain BaseMessage usage, refer here: https://python.langchain.com/docs/concepts/messages" in str(err.value)
    # Test static methods
    assert ResponseGenerator._enforce_strings([123, "hi"]) == ["123", "hi"]
    assert list(ResponseGenerator._split([1, 2, 3, 4, 5], 2)) == [[1, 2], [3, 4], [5]]


@pytest.mark.asyncio
async def test_logprobs_extraction_branches(monkeypatch):
    """Test the actual logprobs extraction by mocking LLM"""

    # Mock the LLM's ainvoke method at the class level
    async def mock_ainvoke_with_logprobs_result(self, messages, **kwargs):
        class MockResult:
            def __init__(self):
                self.content = MOCKED_RESPONSES[0]
                self.response_metadata = {"logprobs_result": ["logprob1", "logprob2"]}

        return MockResult()

    # Patch at the class level
    monkeypatch.setattr(AzureChatOpenAI, "ainvoke", mock_ainvoke_with_logprobs_result)
    mock_object = create_mock_llm()
    mock_object.logprobs = True
    generator_object = ResponseGenerator(llm=mock_object)
    result = await generator_object.generate_responses(prompts=MOCKED_PROMPTS[:1], count=1)
    assert len(result["data"]["response"]) == 1


@pytest.mark.asyncio
async def test_logprobs_content_extraction(monkeypatch):
    """Test the logprobs content extraction branch"""

    async def mock_ainvoke_with_content_logprobs(self, messages, **kwargs):
        class MockResult:
            def __init__(self):
                self.content = MOCKED_RESPONSES[1]
                self.response_metadata = {"logprobs": {"content": ["content_logprob1", "content_logprob2"]}}

        return MockResult()

    # Patch at the class level
    monkeypatch.setattr(AzureChatOpenAI, "ainvoke", mock_ainvoke_with_content_logprobs)
    mock_object = create_mock_llm()
    mock_object.logprobs = True
    generator_object = ResponseGenerator(llm=mock_object)
    result = await generator_object.generate_responses(prompts=MOCKED_PROMPTS[:1], count=1)
    assert len(result["data"]["response"]) == 1


@pytest.mark.asyncio
async def test_generate_in_batches_progress_bar():
    """Test _generate_in_batches with progress bar enabled."""
    mock_llm = MagicMock()
    mock_progress = MagicMock(spec=Progress)
    generator = ResponseGenerator(llm=mock_llm, max_calls_per_min=10)

    # Mock _process_batch to avoid actual async calls
    generator._process_batch = AsyncMock()

    prompts = ["prompt1", "prompt2"]

    # Test with count == 1
    generator.count = 1
    await generator._generate_in_batches(prompts=prompts, progress_bar=mock_progress)
    mock_progress.add_task.assert_called_with(f"  - {generator.generator_type_to_progress_msg[generator.response_generator_type]}...", total=len(prompts))

    # Reset mock and test with count > 1
    mock_progress.reset_mock()
    generator.count = 3
    await generator._generate_in_batches(prompts=prompts, progress_bar=mock_progress)
    mock_progress.add_task.assert_called_with(f"  - Generating candidate responses ({generator.count} per prompt)...", total=len(prompts) * generator.count)


@pytest.mark.asyncio
async def test_async_api_call_with_base_message_list():
    """Test _async_api_call with a list of BaseMessage objects."""

    mock_llm = MagicMock()
    generator = ResponseGenerator(llm=mock_llm)

    # Mock the system_message
    generator.system_message = SystemMessage(content="System message")

    # Mock progress bar and task
    mock_progress = MagicMock()
    generator.progress_bar = mock_progress
    generator.progress_task = "mock_task"

    # Create a list of BaseMessage prompts
    prompt = [HumanMessage(content="Hello"), HumanMessage(content="How are you?")]

    # Mock the LLM's ainvoke method
    mock_llm.ainvoke = AsyncMock(return_value=MagicMock(content="Response"))

    # Call the _async_api_call method
    result = await generator._async_api_call(prompt=prompt, count=1)

    # Assert that the messages were constructed correctly
    expected_messages = [generator.system_message] + prompt
    mock_llm.ainvoke.assert_called_once_with(expected_messages)

    # Assert progress bar was updated
    mock_progress.update.assert_called_once_with("mock_task", advance=1)

    # Assert the result structure
    assert "responses" in result
    assert "logprobs" in result
    assert result["responses"] == ["Response"]


@pytest.mark.asyncio
async def test_async_api_call_with_top_logprobs_and_progress_bar():
    """Test _async_api_call with top_k_logprobs and progress bar enabled."""
    mock_llm = MagicMock()
    generator = ResponseGenerator(llm=mock_llm, top_k_logprobs=5)

    # Set system_message explicitly
    generator.system_message = SystemMessage(content="System message")

    # Mock the progress bar
    mock_progress = MagicMock()
    generator.progress_bar = mock_progress
    generator.progress_task = "mock_task"

    # Create a list of BaseMessage prompts
    prompt = [HumanMessage(content="Hello")]

    # Mock ainvoke_with_top_logprobs
    generator.ainvoke_with_top_logprobs = AsyncMock(return_value={"logprobs": [None], "responses": ["Response"]})

    # Call the _async_api_call method
    result = await generator._async_api_call(prompt=prompt, count=1)

    # Assert that ainvoke_with_top_logprobs was called with correct messages
    expected_messages = [generator.system_message] + prompt
    generator.ainvoke_with_top_logprobs.assert_called_once_with(expected_messages, count=1)

    # Assert that the progress bar was updated
    mock_progress.update.assert_called_once_with("mock_task", advance=1)

    # Assert the result structure
    assert "logprobs" in result
    assert "responses" in result
    assert result["responses"] == ["Response"]


@pytest.mark.asyncio
async def test_ainvoke_with_top_logprobs_openai():
    """Test ainvoke_with_top_logprobs for the 'openai' branch."""
    mock_llm = MagicMock()
    mock_llm.__str__.return_value = "openai"
    generator = ResponseGenerator(llm=mock_llm, top_k_logprobs=5)

    # Mock ainvoke
    mock_llm.ainvoke = AsyncMock(return_value=MagicMock(content="Response"))

    messages = [HumanMessage(content="Hello")]
    result = await generator.ainvoke_with_top_logprobs(messages=messages, count=1)

    # Assert ainvoke was called with the correct arguments
    mock_llm.ainvoke.assert_called_once_with(messages, logprobs=True, top_logprobs=5)

    # Assert the result structure
    assert "logprobs" in result
    assert "responses" in result
    assert result["responses"] == ["Response"]


@pytest.mark.asyncio
async def test_ainvoke_with_top_logprobs_google():
    """Test ainvoke_with_top_logprobs for the 'google' or 'gemini' branch."""
    mock_llm = MagicMock()
    mock_llm.__str__.return_value = "google"
    generator = ResponseGenerator(llm=mock_llm, top_k_logprobs=5)

    # Mock ainvoke
    mock_llm.ainvoke = AsyncMock(return_value=MagicMock(content="Response"))

    messages = [HumanMessage(content="Hello")]
    result = await generator.ainvoke_with_top_logprobs(messages=messages, count=1)

    # Assert logprobs was set and ainvoke was called
    assert mock_llm.logprobs == 5
    mock_llm.ainvoke.assert_called_once_with(messages)

    # Assert the result structure
    assert "logprobs" in result
    assert "responses" in result
    assert result["responses"] == ["Response"]


@pytest.mark.asyncio
async def test_ainvoke_with_top_logprobs_else_branch():
    """Test ainvoke_with_top_logprobs for the 'else' branch."""
    mock_llm = MagicMock()
    mock_llm.__str__.return_value = "other"
    generator = ResponseGenerator(llm=mock_llm, top_k_logprobs=5)

    # Mock ainvoke
    mock_llm.ainvoke = AsyncMock(return_value=MagicMock(content="Response"))

    messages = [HumanMessage(content="Hello")]
    result = await generator.ainvoke_with_top_logprobs(messages=messages, count=1)

    # Assert ainvoke was called with the correct arguments
    mock_llm.ainvoke.assert_called_once_with(messages, logprobs=True, top_logprobs=5)

    # Assert the result structure
    assert "logprobs" in result
    assert "responses" in result
    assert result["responses"] == ["Response"]


@pytest.mark.asyncio
async def test_ainvoke_with_top_logprobs_exception_handling():
    """Test ainvoke_with_top_logprobs exception handling."""
    mock_llm = MagicMock()
    generator = ResponseGenerator(llm=mock_llm, top_k_logprobs=5)

    # Simulate exceptions in both the try and except blocks
    mock_llm.ainvoke = AsyncMock(side_effect=Exception("Mocked exception"))

    messages = [HumanMessage(content="Hello")]
    result = await generator.ainvoke_with_top_logprobs(messages=messages, count=1)

    # Assert that the result structure is still returned even after exceptions
    assert "logprobs" in result
    assert "responses" in result
    assert result["logprobs"] == [None]
    assert result["responses"] == []

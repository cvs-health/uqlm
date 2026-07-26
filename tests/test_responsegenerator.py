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
from langchain_core.language_models.chat_models import BaseChatModel
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
async def test_structured_outputs():
    """Test that structured_response and output_extractor parameters are validated and work together."""

    mock_llm = MagicMock(spec=BaseChatModel)
    mock_llm.temperature = 1
    generator = ResponseGenerator(llm=mock_llm, structured_response={"field": "value"}, output_extractor=lambda x: x["field"])
    # Mock the LLM's ainvoke method to return a structured response
    structured_llm = MagicMock()
    structured_llm.ainvoke = AsyncMock(return_value={"field": "extracted_value"})
    mock_llm.with_structured_output = MagicMock(return_value=structured_llm)
    result = await generator.generate_responses(prompts=MOCKED_PROMPTS[:1], count=1)

    # Assert that the output_extractor was called on the structured response
    assert result["data"]["response"] == ["extracted_value"]

    with pytest.raises(ValueError):
        ResponseGenerator(llm=mock_llm, structured_response={"field": "value"})
    with pytest.raises(ValueError):
        ResponseGenerator(llm=mock_llm, output_extractor=lambda x: x["field"])


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
    """Test that ainvoke_with_top_logprobs raises the provider error when all attempts fail.

    Regression guard for issue #416: the previous behavior swallowed the exception and
    returned an empty (shorter-than-count) response list, which silently misaligned all
    subsequent prompt/response pairs in the batch.
    """
    mock_llm = MagicMock()
    generator = ResponseGenerator(llm=mock_llm, top_k_logprobs=5)

    # Simulate exceptions in both the try and except blocks
    mock_llm.ainvoke = AsyncMock(side_effect=Exception("Mocked exception"))

    messages = [HumanMessage(content="Hello")]
    with pytest.raises(Exception, match="Mocked exception"):
        await generator.ainvoke_with_top_logprobs(messages=messages, count=1)


# ---------------------------------------------------------------------------
# Regression tests for issue #416
# ---------------------------------------------------------------------------


def create_flaky_llm(failing_substring="Prompt 2", error=None):
    """Mock BaseChatModel whose ainvoke always fails for prompts containing `failing_substring`."""
    mock_llm = MagicMock(spec=BaseChatModel)
    mock_llm.__str__ = lambda self: "other-provider"
    mock_llm.temperature = 1

    class FakeResult:
        def __init__(self, content):
            self.content = content
            self.response_metadata = {"logprobs": {"content": [{"token": "x", "logprob": -0.1}]}}

    async def flaky_ainvoke(messages, **kwargs):
        text = messages[-1].content
        if failing_substring in text:
            raise error or RuntimeError("429 rate limited")
        return FakeResult(f"Answer to: {text}")

    mock_llm.ainvoke = flaky_ainvoke
    return mock_llm


@pytest.mark.asyncio
async def test_all_logprob_attempts_fail_keeps_alignment():
    """Regression test for issue #416: a prompt whose generation fails after all retries
    must yield a correctly-sized placeholder instead of silently shortening the batch."""
    generator = ResponseGenerator(llm=create_flaky_llm(), top_k_logprobs=15, max_retries=1)
    generator._retry_base_delay = 0.001
    prompts = ["Prompt 1", "Prompt 2", "Prompt 3"]
    with pytest.warns(UserWarning, match="alignment is preserved"):
        results = await generator.generate_responses(prompts=prompts, count=1)
    data = results["data"]
    assert len(data["response"]) == len(prompts)
    assert data["response"][0] == "Answer to: Prompt 1"
    assert data["response"][1] == "Unable to get response"
    assert data["response"][2] == "Answer to: Prompt 3"


@pytest.mark.asyncio
async def test_failed_generation_reported_in_metadata():
    """Failures exhausted of retries must appear in metadata['failures'] with index/prompt/error."""
    generator = ResponseGenerator(llm=create_flaky_llm(), top_k_logprobs=15, max_retries=0)
    with pytest.warns(UserWarning):
        results = await generator.generate_responses(prompts=["Prompt 1", "Prompt 2"], count=1)
    failures = results["metadata"]["failures"]
    assert len(failures) == 1
    assert failures[0]["prompt_index"] == 1
    assert failures[0]["prompt"] == "Prompt 2"
    assert "429 rate limited" in failures[0]["error"]


@pytest.mark.asyncio
async def test_retry_succeeds_after_transient_failure():
    """A transiently failing call must be retried and succeed without placeholders."""
    mock_llm = MagicMock(spec=BaseChatModel)
    mock_llm.temperature = 1
    attempts = {"n": 0}

    async def transient_ainvoke(messages, **kwargs):
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise RuntimeError("transient 500")
        return MagicMock(content="Recovered", response_metadata={})

    mock_llm.ainvoke = transient_ainvoke
    generator = ResponseGenerator(llm=mock_llm, max_retries=2)
    generator._retry_base_delay = 0.001
    results = await generator.generate_responses(prompts=["Prompt 1"], count=1)
    assert results["data"]["response"] == ["Recovered"]
    assert results["metadata"]["failures"] == []
    assert attempts["n"] == 2


@pytest.mark.asyncio
async def test_concurrency_bounded_by_semaphore():
    """No more than max_concurrency requests may be in flight simultaneously."""
    mock_llm = MagicMock(spec=BaseChatModel)
    mock_llm.temperature = 1
    in_flight = {"now": 0, "peak": 0}

    async def tracking_ainvoke(messages, **kwargs):
        in_flight["now"] += 1
        in_flight["peak"] = max(in_flight["peak"], in_flight["now"])
        await asyncio.sleep(0.01)
        in_flight["now"] -= 1
        return MagicMock(content="ok", response_metadata={})

    mock_llm.ainvoke = tracking_ainvoke
    generator = ResponseGenerator(llm=mock_llm, max_concurrency=2)
    prompts = [f"Prompt {i}" for i in range(8)]
    await generator.generate_responses(prompts=prompts, count=1)
    assert in_flight["peak"] <= 2


@pytest.mark.asyncio
async def test_no_blocking_sleep_in_async_paths(monkeypatch):
    """time.sleep must not be called inside the async generation paths (issue #416)."""
    import uqlm.utils.response_generator as rg_module

    def fail_on_blocking_sleep(*args, **kwargs):
        raise AssertionError("blocking time.sleep called inside async generation path")

    monkeypatch.setattr(rg_module.time, "sleep", fail_on_blocking_sleep)
    mock_async_api_call = create_mock_async_api_call()
    generator = ResponseGenerator(llm=create_mock_llm(), max_calls_per_min=1000)
    monkeypatch.setattr(generator, "_async_api_call", mock_async_api_call)
    result = await generator.generate_responses(prompts=MOCKED_PROMPTS, count=1)
    assert len(result["data"]["response"]) == len(MOCKED_PROMPTS)


def test_empty_prompts_raises_value_error():
    """An empty prompt list must raise a clear ValueError instead of a confusing range() error."""
    generator = ResponseGenerator(llm=create_mock_llm())
    with pytest.raises(ValueError, match="non-empty"):
        asyncio.run(generator.generate_responses(prompts=[], count=1))


def test_mixed_message_list_raises_value_error():
    """A message list containing a non-BaseMessage item must raise ValueError, not UnboundLocalError."""
    generator = ResponseGenerator(llm=create_mock_llm())
    with pytest.raises(ValueError, match="BaseMessage"):
        asyncio.run(generator.generate_responses(prompts=[[HumanMessage(content="hi"), "not a message"]], count=1))


@pytest.mark.asyncio
async def test_temperature_missing_attribute_ok(monkeypatch):
    """Custom models without a `temperature` attribute must not crash generate_responses."""
    mock_llm = MagicMock(spec=BaseChatModel)
    # spec=BaseChatModel does not define `temperature`; ensure attribute access is guarded
    del mock_llm.temperature
    mock_llm.ainvoke = AsyncMock(return_value=MagicMock(content="ok", response_metadata={}))
    generator = ResponseGenerator(llm=mock_llm)
    result = await generator.generate_responses(prompts=["Prompt 1"], count=1)
    assert result["data"]["response"] == ["ok"]
    assert result["metadata"]["temperature"] is None

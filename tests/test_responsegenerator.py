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
    with pytest.raises(AssertionError) as assert_error:
        asyncio.run(generator_object.generate_responses(prompts=[123], count=1))
    assert "If using custom prompts, please ensure `prompts` is of type list[str]" in str(assert_error.value)
    # Test static methods
    assert ResponseGenerator._enforce_strings([123, "hi"]) == ["123", "hi"]
    assert list(ResponseGenerator._split([1, 2, 3, 4, 5], 2)) == [[1, 2], [3, 4], [5]]


@pytest.mark.asyncio
async def test_logprobs_extraction_branches(monkeypatch):
    """Test the actual logprobs extraction by mocking LLM"""

    # Mock the LLM's agenerate method at the class level
    async def mock_agenerate_with_logprobs_result(self, messages, **kwargs):
        class MockGeneration:
            def __init__(self):
                self.text = MOCKED_RESPONSES[0]
                self.generation_info = {"logprobs_result": ["logprob1", "logprob2"]}

        class MockResult:
            def __init__(self):
                self.generations = [[MockGeneration()]]

        return MockResult()

    # Patch at the class level
    monkeypatch.setattr(AzureChatOpenAI, "agenerate", mock_agenerate_with_logprobs_result)
    mock_object = create_mock_llm()
    mock_object.logprobs = True
    generator_object = ResponseGenerator(llm=mock_object)
    result = await generator_object.generate_responses(prompts=MOCKED_PROMPTS[:1], count=1)
    assert len(result["data"]["response"]) == 1


@pytest.mark.asyncio
async def test_logprobs_content_extraction(monkeypatch):
    """Test the logprobs content extraction branch"""

    async def mock_agenerate_with_content_logprobs(self, messages, **kwargs):
        class MockGeneration:
            def __init__(self):
                self.text = MOCKED_RESPONSES[1]
                self.generation_info = {"logprobs": {"content": ["content_logprob1", "content_logprob2"]}}

        class MockResult:
            def __init__(self):
                self.generations = [[MockGeneration()]]

        return MockResult()

    # Patch at the class level
    monkeypatch.setattr(AzureChatOpenAI, "agenerate", mock_agenerate_with_content_logprobs)
    mock_object = create_mock_llm()
    mock_object.logprobs = True
    generator_object = ResponseGenerator(llm=mock_object)
    result = await generator_object.generate_responses(prompts=MOCKED_PROMPTS[:1], count=1)
    assert len(result["data"]["response"]) == 1

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
from langchain_openai import AzureChatOpenAI

from uqlm.utils.response_generator import ResponseGenerator


@pytest.mark.asyncio
async def test_generator(monkeypatch):
    count = 3
    MOCKED_PROMPTS = ["Prompt 1", "Prompt 2", "Prompt 3"]
    MOCKED_RESPONSES = ["Mocked response 1", "Mocked response 2", "Unable to get response"]
    MOCKED_DUPLICATED_RESPONSES = [prompt for prompt, i in itertools.product(MOCKED_RESPONSES, range(count))]
    MOCKED_RESPONSE_DICT = dict(zip(MOCKED_PROMPTS, MOCKED_RESPONSES))

    async def mock_async_api_call(prompt, count, *args, **kwargs):
        return {"logprobs": [], "responses": [MOCKED_RESPONSE_DICT[prompt]] * count}

    mock_object = AzureChatOpenAI(deployment_name="YOUR-DEPLOYMENT", temperature=1, api_key="SECRET_API_KEY", api_version="2024-05-01-preview", azure_endpoint="https://mocked.endpoint.com")

    generator_object = ResponseGenerator(llm=mock_object)

    monkeypatch.setattr(generator_object, "_async_api_call", mock_async_api_call)
    data = await generator_object.generate_responses(prompts=MOCKED_PROMPTS, count=count)

    assert data["data"]["response"] == MOCKED_DUPLICATED_RESPONSES

    # assert generator_object._valid_exceptions(exceptions=BaseException) is True
    # assert generator_object._valid_exceptions(exceptions=(BaseException)) is True
    # assert generator_object._valid_exceptions(exceptions='Exception') is False

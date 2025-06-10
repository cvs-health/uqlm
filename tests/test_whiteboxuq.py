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
import json
from uqlm.scorers import WhiteBoxUQ
from langchain_openai import AzureChatOpenAI

datafile_path = "tests/data/scorers/whitebox_results_file.json"
with open(datafile_path, "r") as f:
    expected_result = json.load(f)

data = expected_result["data"]
metadata = expected_result["metadata"]

PROMPTS = data["prompts"]
MOCKED_RESPONSES = data["responses"]
MOCKED_LOGPROBS = data["logprobs"]

mock_object = AzureChatOpenAI(deployment_name="YOUR-DEPLOYMENT", temperature=1.0, api_key="SECRET_API_KEY", api_version="2024-05-01-preview", azure_endpoint="https://mocked.endpoint.com")


@pytest.mark.asyncio
async def test_whiteboxuq(monkeypatch):
    wbuq = WhiteBoxUQ(llm=mock_object)

    async def mock_generate_original_responses(*args, **kwargs):
        wbuq.logprobs = MOCKED_LOGPROBS
        return MOCKED_RESPONSES

    monkeypatch.setattr(wbuq, "generate_original_responses", mock_generate_original_responses)
    results = await wbuq.generate_and_score(prompts=PROMPTS)

    assert all([results.data["normalized_probability"][i] == pytest.approx(data["normalized_probability"][i]) for i in range(len(PROMPTS))])

    assert all([results.data["min_probability"][i] == pytest.approx(data["min_probability"][i]) for i in range(len(PROMPTS))])

    assert results.metadata == metadata

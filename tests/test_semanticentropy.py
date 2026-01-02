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
from uqlm.scorers.shortform.entropy import SemanticEntropy
from langchain_openai import AzureChatOpenAI

datafile_path = "tests/data/scorers/semanticentropy_results_file.json"
with open(datafile_path, "r") as f:
    expected_result = json.load(f)

data = expected_result["data"]
metadata = expected_result["metadata"]

mock_object = AzureChatOpenAI(deployment_name="YOUR-DEPLOYMENT", temperature=1, api_key="SECRET_API_KEY", api_version="2024-05-01-preview", azure_endpoint="https://mocked.endpoint.com")


@pytest.mark.flaky(reruns=3)
@pytest.mark.asyncio
async def test_semanticentropy(monkeypatch):
    PROMPTS = data["prompts"]
    MOCKED_RESPONSES = data["responses"]
    MOCKED_SAMPLED_RESPONSES = data["sampled_responses"]

    # Initiate SemanticEntropy class object
    se_object = SemanticEntropy(llm=mock_object, use_best=False, device="cpu")

    async def mock_generate_original_responses(*args, **kwargs):
        se_object.logprobs = [None] * 5
        return MOCKED_RESPONSES

    async def mock_generate_candidate_responses(*args, **kwargs):
        se_object.multiple_logprobs = [[None] * 5] * 5
        return MOCKED_SAMPLED_RESPONSES

    monkeypatch.setattr(se_object, "generate_original_responses", mock_generate_original_responses)
    monkeypatch.setattr(se_object, "generate_candidate_responses", mock_generate_candidate_responses)

    for show_progress_bars in [False, True]:
        se_results = await se_object.generate_and_score(prompts=PROMPTS, show_progress_bars=show_progress_bars)
        se_object.logprobs = None
        se_results = se_object.score(responses=MOCKED_RESPONSES, sampled_responses=MOCKED_SAMPLED_RESPONSES)
        assert se_results.data["responses"] == data["responses"]
        assert se_results.data["sampled_responses"] == data["sampled_responses"]
        assert se_results.data["prompts"] == data["prompts"]
        assert all([abs(se_results.data["discrete_entropy_values"][i] - data["entropy_values"][i]) < 1e-5 for i in range(len(PROMPTS))])
        assert all([abs(se_results.data["discrete_confidence_scores"][i] - data["confidence_scores"][i]) < 1e-5 for i in range(len(PROMPTS))])
        assert se_results.metadata == metadata

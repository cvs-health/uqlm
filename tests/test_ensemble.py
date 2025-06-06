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
from uqlm.scorers import UQEnsemble
from uqlm.scorers.baseclass.uncertainty import UQResult
from langchain_openai import AzureChatOpenAI

datafile_path = "tests/data/scorers/ensemble_results_file.json"
with open(datafile_path, "r") as f:
    expected_result = json.load(f)

data = expected_result["data"]
metadata = expected_result["metadata"]

PROMPTS = data["prompts"]
MOCKED_RESPONSES = data["responses"]
MOCKED_SAMPLED_RESPONSES = data["sampled_responses"]
MOCKED_JUDGE_SCORES = data["judge_1"]
MOCKED_LOGPROBS = metadata["logprobs"]

mock_object = AzureChatOpenAI(deployment_name="YOUR-DEPLOYMENT", temperature=1, api_key="SECRET_API_KEY", api_version="2024-05-01-preview", azure_endpoint="https://mocked.endpoint.com")


@pytest.mark.asyncio
async def test_ensemble(monkeypatch):
    uqe = UQEnsemble(llm=mock_object, scorers=["exact_match", "noncontradiction", "min_probability", mock_object])

    async def mock_generate_original_responses(*args, **kwargs):
        uqe.logprobs = MOCKED_LOGPROBS
        return MOCKED_RESPONSES

    async def mock_generate_candidate_responses(*args, **kwargs):
        uqe.multiple_logprobs = [MOCKED_LOGPROBS] * 5
        return MOCKED_SAMPLED_RESPONSES

    async def mock_judge_scores(*args, **kwargs):
        return UQResult({"data": {"judge_1": MOCKED_JUDGE_SCORES}})

    monkeypatch.setattr(uqe, "generate_original_responses", mock_generate_original_responses)
    monkeypatch.setattr(uqe, "generate_candidate_responses", mock_generate_candidate_responses)
    monkeypatch.setattr(uqe.judges_object, "score", mock_judge_scores)

    results = await uqe.generate_and_score(prompts=PROMPTS, num_responses=5)

    assert all([results.data["ensemble_scores"][i] == pytest.approx(data["ensemble_scores"][i]) for i in range(len(PROMPTS))])

    assert all([results.data["min_probability"][i] == pytest.approx(data["min_probability"][i]) for i in range(len(PROMPTS))])

    assert all([results.data["exact_match"][i] == pytest.approx(data["exact_match"][i]) for i in range(len(PROMPTS))])

    assert all([results.data["noncontradiction"][i] == pytest.approx(data["noncontradiction"][i]) for i in range(len(PROMPTS))])

    assert all([abs(results.data["judge_1"][i] - data["judge_1"][i]) < 1e-5 for i in range(len(PROMPTS))])

    assert results.metadata == metadata

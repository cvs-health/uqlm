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
from uqlm.scorers.baseclass.uncertainty import UncertaintyQuantifier
from uqlm.judges.judge import LLMJudge
from langchain_openai import AzureChatOpenAI

datafile_path = "tests/data/scorers/bsdetector_results_file.json"
with open(datafile_path, "r") as f:
    data = json.load(f)

mock_object = AzureChatOpenAI(deployment_name="YOUR-DEPLOYMENT", temperature=1, api_key="SECRET_API_KEY", api_version="2024-05-01-preview", azure_endpoint="https://mocked.endpoint.com")


@pytest.mark.asyncio
async def test_uncertainty(monkeypatch):
    PROMPTS = data["prompts"]
    MOCKED_RESPONSES = data["responses"]
    MOCKED_sampled_responses = data["sampled_responses"]

    temperature = 1
    sampling_temperature = 2
    num_responses = 5

    def postprocesser(x):
        return x.lower() if isinstance(x, str) else x

    uq_object = UncertaintyQuantifier(llm=mock_object, postprocessor=postprocesser)

    async def mock_generate_responses(prompts, **args):
        return {"logprobs": [None], "responses": MOCKED_RESPONSES}

    monkeypatch.setattr(uq_object, "_generate_responses", mock_generate_responses)
    uq_object.prompts = PROMPTS
    uq_object.llm.temperature = temperature
    uq_object.sampling_temperature = sampling_temperature
    uq_object.num_responses = num_responses

    responses = await uq_object.generate_original_responses(prompts=PROMPTS)
    tmp = [postprocesser(r) for r in MOCKED_RESPONSES]
    assert responses == tmp

    async def mock_generate_responses(prompts, **args):
        return {"logprobs": [None], "responses": sum(MOCKED_sampled_responses, [])}

    monkeypatch.setattr(uq_object, "_generate_responses", mock_generate_responses)
    sampled_responses = await uq_object.generate_candidate_responses(prompts=PROMPTS)
    tmp = [[postprocesser(r) for r in m] for m in MOCKED_sampled_responses]
    assert sampled_responses == tmp
    assert uq_object.llm.temperature == temperature

    tmp = uq_object._construct_judge()
    assert isinstance(tmp, LLMJudge)

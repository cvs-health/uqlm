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

data = expected_result["ensemble"]["data"]
metadata = expected_result["ensemble"]["metadata"]
    
PROMPTS = data["prompts"]
MOCKED_RESPONSES = data["responses"]
MOCKED_SAMPLED_RESPONSES = data["sampled_responses"]
MOCKED_JUDGE_SCORES = data['judge_1']
MOCKED_LOGPROBS = metadata["logprobs"]
    
@pytest.fixture
def mock_llm():
    """Extract judge object using pytest.fixture."""
    return AzureChatOpenAI(
        deployment_name="YOUR-DEPLOYMENT",
        temperature=1,
        api_key="SECRET_API_KEY",
        api_version="2024-05-01-preview",
        azure_endpoint="https://mocked.endpoint.com",
    )

def test_validate_grader(mock_llm):
    uqe = UQEnsemble(
        llm=mock_llm,
        scorers=["exact_match"]
    )
    uqe._validate_grader(None)

    with pytest.raises(ValueError) as value_error:
        uqe._validate_grader(lambda res, ans: res==ans)
    assert "grader_function must have 'resposne' and 'answer' parameters" == str(value_error.value)

    with pytest.raises(ValueError) as value_error:
        uqe._validate_grader(lambda response, answer: len(response)+len(answer))
    assert "grader_function must return boolean" == str(value_error.value)

def test_wrong_components(mock_llm):
    with pytest.raises(ValueError) as value_error:
        UQEnsemble(
            llm=mock_llm,
            scorers=["eaxct_match"]
        )
    assert "Components must be an instance of LLMJudge, BaseChatModel" in str(value_error.value)

def test_wrong_weights(mock_llm):
    with pytest.raises(ValueError) as value_error:
        UQEnsemble(llm=mock_llm, scorers=["exact_match"], weights=[0.5, 0.5])
    assert "Must have same number of weights as components" in str(value_error.value)

def test_bsdetector_weights(mock_llm):
    uqe = UQEnsemble(llm=mock_llm)
    assert (uqe.weights == [0.7 * 0.8, 0.7 * 0.2, 0.3]).all()

@pytest.mark.asyncio
async def test_ensemble(monkeypatch, mock_llm):
    uqe = UQEnsemble(
        llm=mock_llm,
        scorers=[
            "exact_match",  
            "noncontradiction",  
            "min_probability",  
            mock_llm,  
        ]
    )
    
    async def mock_generate_original_responses(*args, **kwargs):
        uqe.logprobs = MOCKED_LOGPROBS
        return MOCKED_RESPONSES
    
    async def mock_generate_candidate_responses(*args, **kwargs):
        uqe.multiple_logprobs = [MOCKED_LOGPROBS] * 5
        return MOCKED_SAMPLED_RESPONSES
    
    async def mock_judge_scores(*args, **kwargs):
        return UQResult({'data': {'judge_1': MOCKED_JUDGE_SCORES}})
    
    monkeypatch.setattr(uqe, "generate_original_responses", mock_generate_original_responses)
    monkeypatch.setattr(uqe, "generate_candidate_responses", mock_generate_candidate_responses)
    monkeypatch.setattr(uqe.judges_object, "score", mock_judge_scores)
    
    results = await uqe.generate_and_score(
        prompts=PROMPTS,
        num_responses=5,
    )
    
    assert all(
        [
            results.data["ensemble_scores"][i]
            == pytest.approx(data["ensemble_scores"][i])
            for i in range(len(PROMPTS))
        ]
    )

    assert all(
        [
            results.data["min_probability"][i]
            == pytest.approx(data["min_probability"][i])
            for i in range(len(PROMPTS))
        ]
    )

    assert all(
        [
            results.data["exact_match"][i] == pytest.approx(data["exact_match"][i])
            for i in range(len(PROMPTS))
        ]
    )

    assert all(
        [
            results.data["noncontradiction"][i]
            == pytest.approx(data["noncontradiction"][i])
            for i in range(len(PROMPTS))
        ]
    )

    assert all(
        [
            abs(results.data["judge_1"][i] - data["judge_1"][i]) < 1e-5
            for i in range(len(PROMPTS))
        ]
    )

    assert results.metadata == metadata
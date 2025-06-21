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

data = expected_result["ensemble1"]["data"]
metadata = expected_result["ensemble1"]["metadata"]

PROMPTS = data["prompts"]
MOCKED_RESPONSES = data["responses"]
MOCKED_SAMPLED_RESPONSES = data["sampled_responses"]
MOCKED_JUDGE_SCORES = data["judge_1"]
MOCKED_LOGPROBS = metadata["logprobs"]


@pytest.fixture
def mock_llm():
    """Extract judge object using pytest.fixture."""
    return AzureChatOpenAI(deployment_name="YOUR-DEPLOYMENT", temperature=1, api_key="SECRET_API_KEY", api_version="2024-05-01-preview", azure_endpoint="https://mocked.endpoint.com")


def test_validate_grader(mock_llm):
    uqe = UQEnsemble(llm=mock_llm, scorers=["exact_match"])
    uqe._validate_grader(None)

    with pytest.raises(ValueError) as value_error:
        uqe._validate_grader(lambda res, ans: res == ans)
    assert "grader_function must have 'response' and 'answer' parameters" == str(value_error.value)

    with pytest.raises(ValueError) as value_error:
        uqe._validate_grader(lambda response, answer: len(response) + len(answer))
    assert "grader_function must return boolean" == str(value_error.value)


def test_wrong_components(mock_llm):
    with pytest.raises(ValueError) as value_error:
        UQEnsemble(llm=mock_llm, scorers=["eaxct_match"])
    assert "Components must be an instance of LLMJudge, BaseChatModel" in str(value_error.value)


@pytest.mark.asyncio
async def test_error_sampled_response(mock_llm):
    with pytest.raises(ValueError) as value_error:
        uqe = UQEnsemble(llm=mock_llm, scorers=["exact_match"])
        await uqe.score(prompts=PROMPTS, responses=MOCKED_RESPONSES)
    assert "sampled_responses must be provided if using black-box scorers" == str(value_error.value)


@pytest.mark.asyncio
async def test_error_logprobs_results(mock_llm):
    with pytest.raises(ValueError) as value_error:
        uqe = UQEnsemble(llm=mock_llm, scorers=["min_probability"])
        await uqe.score(prompts=PROMPTS, responses=MOCKED_RESPONSES)
    assert "logprobs_results must be provided if using white-box scorers" == str(value_error.value)


def test_wrong_weights(mock_llm):
    with pytest.raises(ValueError) as value_error:
        UQEnsemble(llm=mock_llm, scorers=["exact_match"], weights=[0.5, 0.5])
    assert "Must have same number of weights as components" in str(value_error.value)


def test_bsdetector_weights(mock_llm):
    uqe = UQEnsemble(llm=mock_llm)
    assert uqe.weights == [0.7 * 0.8, 0.7 * 0.2, 0.3]


@pytest.mark.asyncio
async def test_ensemble(monkeypatch, mock_llm):
    uqe = UQEnsemble(llm=mock_llm, scorers=["exact_match", "noncontradiction", "min_probability", mock_llm])

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

    assert all([results.data["judge_1"][i] == pytest.approx(data["judge_1"][i], abs=1e-5) for i in range(len(PROMPTS))])

    assert results.metadata == metadata

    tune_results = {"weights": [0.5, 0.2, 0.3], "thresh": 0.75}

    def mock_tune_params(*args, **kwargs):
        return tune_results

    uqe = UQEnsemble(llm=mock_llm, scorers=["exact_match", "noncontradiction", mock_llm])

    monkeypatch.setattr(uqe.tuner, "tune_params", mock_tune_params)
    monkeypatch.setattr(uqe, "generate_original_responses", mock_generate_original_responses)
    monkeypatch.setattr(uqe, "generate_candidate_responses", mock_generate_candidate_responses)
    monkeypatch.setattr(uqe.judges_object, "score", mock_judge_scores)

    result = await uqe.tune(prompts=PROMPTS, ground_truth_answers=PROMPTS)
    assert result.metadata["weights"] == tune_results["weights"]
    assert result.metadata["thresh"] == tune_results["thresh"]

    result = await uqe.tune(prompts=PROMPTS, ground_truth_answers=[PROMPTS[0]] + [" "] * len(PROMPTS[:-1]), grader_function=lambda response, answer: response == answer)
    assert result.metadata["thresh"] == tune_results["thresh"]


@pytest.mark.asyncio
async def test_ensemble2(monkeypatch, mock_llm):
    data = expected_result["ensemble2"]["data"]
    metadata = expected_result["ensemble2"]["metadata"]

    PROMPTS = data["prompts"]
    MOCKED_RESPONSES = data["responses"]
    MOCKED_JUDGE_SCORES = data["judge_1"]
    MOCKED_LOGPROBS = metadata["logprobs"]
    uqe = UQEnsemble(llm=mock_llm, scorers=["min_probability", mock_llm])

    async def mock_generate_original_responses(*args, **kwargs):
        uqe.logprobs = MOCKED_LOGPROBS
        return MOCKED_RESPONSES

    async def mock_judge_scores(*args, **kwargs):
        return UQResult({"data": {"judge_1": MOCKED_JUDGE_SCORES}})

    monkeypatch.setattr(uqe, "generate_original_responses", mock_generate_original_responses)
    monkeypatch.setattr(uqe.judges_object, "score", mock_judge_scores)

    results = await uqe.generate_and_score(prompts=PROMPTS)

    assert all([results.data["min_probability"][i] == pytest.approx(data["min_probability"][i], abs=1e-5) for i in range(len(PROMPTS))])

    assert all([results.data["judge_1"][i] == pytest.approx(data["judge_1"][i], abs=1e-5) for i in range(len(PROMPTS))])

    assert results.metadata == metadata


@pytest.mark.asyncio
async def test_default_logprob(monkeypatch, mock_llm):
    async def mock_judge_scores(*args, **kwargs):
        return UQResult({"data": {"judge_1": MOCKED_JUDGE_SCORES}})

    uqe = UQEnsemble(llm=mock_llm, scorers=[mock_llm])
    monkeypatch.setattr(uqe.judges_object, "score", mock_judge_scores)
    await uqe.score(prompts=PROMPTS, responses=MOCKED_RESPONSES, logprobs_results=None)
    assert list(set(uqe.logprobs)) == [None]
    assert list(set(sum(uqe.multiple_logprobs, []))) == [None]

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
from uqlm.scorers.shortform.baseclass.uncertainty import ShortFormUQ
from uqlm.utils.results import UQResult
from uqlm.judges import LLMJudge
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

    uq_object = ShortFormUQ(llm=mock_object, postprocessor=postprocesser)

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


@pytest.mark.asyncio
async def test_edge_cases(monkeypatch):
    # Test generate_original_responses without postprocessor
    uq_no_post = ShortFormUQ(llm=mock_object)
    PROMPTS = data["prompts"]
    MOCKED_RESPONSES = data["responses"]

    async def mock_generate_responses(prompts, **args):
        return {"logprobs": [None], "responses": MOCKED_RESPONSES}

    monkeypatch.setattr(uq_no_post, "_generate_responses", mock_generate_responses)
    responses = await uq_no_post.generate_original_responses(prompts=PROMPTS)
    assert responses == MOCKED_RESPONSES  # No postprocessing
    #  Test _construct_judge
    uq_with_llm = ShortFormUQ(llm=mock_object)
    # Test case 1: Use self.llm as judge (llm parameter is None - default)
    judge1 = uq_with_llm._construct_judge()
    assert isinstance(judge1, LLMJudge)
    # Test case 2: Use provided llm as judge
    judge2 = uq_with_llm._construct_judge(llm=mock_object)
    assert isinstance(judge2, LLMJudge)
    #  Test _update_best
    uq_update = ShortFormUQ(llm=mock_object)
    uq_update.responses = ["r1", "r1"]
    uq_update.logprobs = [0.11, 0.21]
    uq_update.sampled_responses = [["r2", "s2"], ["r3", "s3"]]
    uq_update.multiple_logprobs = [[0.12, 0.22], [0.13, 0.23]]

    new_responses, new_logprobs = ["r2", "s3"], [0.12, 0.23]
    uq_update._update_best(new_responses)
    assert uq_update.responses == new_responses
    assert uq_update.logprobs == new_logprobs


@pytest.mark.asyncio
async def test_generate_responses():
    """Test the _generate_responses ValueError"""
    uq_no_llm = ShortFormUQ(llm=None)
    with pytest.raises(ValueError, match="llm must be provided to generate responses"):
        await uq_no_llm._generate_responses(prompts=["test"], count=1)


def test_uq_result():
    """Test UQResult class"""
    # Test with sampled_responses present (the ELSE branch)
    result_dict_with_sampled = {"data": {"responses": ["r1", "r2"], "confidence_scores": [0.8, 0.9], "sampled_responses": ["s1", "s2"]}, "metadata": {"model": "test"}, "parameters": {"temp": 1.0}}
    uq_result = UQResult(result_dict_with_sampled)
    # Test all properties including sampled_responses
    assert isinstance(uq_result, UQResult)
    assert uq_result.data is not None
    assert uq_result.to_dict() == result_dict_with_sampled
    df = uq_result.to_df()
    assert len(df) == 2


@pytest.mark.asyncio
async def test_multiple_logprobs_and_temperature(monkeypatch):
    """Test multiple_logprobs.append in generate_candidate_responses and Temperature handling in _generate_responses"""
    # Test multiple_logprobs.append in generate_candidate_responses
    uq_for_logprobs = ShortFormUQ(llm=mock_object)
    PROMPTS = data["prompts"]

    # Mock to return logprobs
    async def mock_generate_with_logprobs(prompts, **args):
        # Return multiple logprobs to trigger the append operation
        return {"logprobs": [0.1, 0.2, 0.3, 0.4], "responses": ["resp1", "resp2", "resp3", "resp4"]}

    monkeypatch.setattr(uq_for_logprobs, "_generate_responses", mock_generate_with_logprobs)
    uq_for_logprobs.num_responses = 2
    uq_for_logprobs.sampling_temperature = 2
    await uq_for_logprobs.generate_candidate_responses(prompts=PROMPTS)

    # Test the temperature handling in _generate_responses
    uq_temp = ShortFormUQ(llm=mock_object)

    # Mock the LLM's agenerate method directly
    async def mock_llm_agenerate(*args, **kwargs):
        class MockGeneration:
            def __init__(self):
                self.text = "test response"
                self.generation_info = {"logprobs_result": [0.1, 0.2]}

        class MockResult:
            def __init__(self):
                self.generations = [[MockGeneration()]]

        return MockResult()

    # Mock at the LLM class level
    monkeypatch.setattr("langchain_openai.AzureChatOpenAI.agenerate", mock_llm_agenerate)
    result = await uq_temp._generate_responses(prompts=["test"], count=1, temperature=0.5)
    assert result is not None
    assert "responses" in result
    assert "logprobs" in result


@pytest.mark.asyncio
async def test_generate_responses_exception(monkeypatch):
    uq = ShortFormUQ(llm=mock_object)

    class MockProgress:
        def stop(self):
            setattr(self, "stopped", True)

    uq.progress_bar = MockProgress()

    async def mock_generate(*args, **kwargs):
        raise RuntimeError("Simulated failure")

    monkeypatch.setattr("uqlm.utils.response_generator.ResponseGenerator.generate_responses", mock_generate)

    with pytest.raises(RuntimeError, match="Simulated failure"):
        await uq._generate_responses(prompts=["test"], count=1, progress_bar=uq.progress_bar)
    assert hasattr(uq.progress_bar, "stopped")


def test_update_best_with_raw(monkeypatch):
    uq = ShortFormUQ(llm=mock_object, postprocessor=lambda x: x)
    uq.responses = ["r1"]
    uq.sampled_responses = [["r2", "r3"]]
    uq.logprobs = [0.1]
    uq.multiple_logprobs = [[0.2, 0.3]]
    uq.raw_responses = ["raw_r1"]
    uq.raw_sampled_responses = [["raw_r2", "raw_r3"]]

    uq._update_best(["r2"])
    assert uq.raw_responses == ["raw_r2"]
    assert uq.raw_sampled_responses == [["raw_r1", "raw_r3"]]


def test_construct_black_box_return_data_variants():
    uq = ShortFormUQ(llm=mock_object, postprocessor=lambda x: x)
    uq.responses = ["r1"]
    uq.sampled_responses = [["r2", "r3"]]
    uq.raw_responses = ["raw_r1"]
    uq.raw_sampled_responses = [["raw_r2", "raw_r3"]]
    uq.prompts = ["prompt1"]

    # Case: return_responses = "all"
    uq.return_responses = "all"
    result = uq._construct_black_box_return_data()
    assert "raw_responses" in result
    assert "prompts" in result

    # Case: return_responses = "raw"
    uq.return_responses = "raw"
    result = uq._construct_black_box_return_data()
    assert result["responses"] == uq.raw_responses
    assert result["sampled_responses"] == uq.raw_sampled_responses


def test_progress_bar_exceptions(monkeypatch):
    uq = ShortFormUQ(llm=mock_object)

    class FaultyProgress:
        def start(self):
            raise AttributeError("start failed")

        def stop(self):
            raise RuntimeError("stop failed")

        def add_task(self, *args, **kwargs):
            raise OSError("add_task failed")

        live = None

    # Inject faulty progress bar
    uq.progress_bar = FaultyProgress()

    # Test _construct_progress_bar with faulty start
    try:
        uq._construct_progress_bar(show_progress_bars=True, _existing_progress_bar=uq.progress_bar)
    except AttributeError:
        pass  # Expected

    # Test display headers with faulty add_task
    try:
        uq._display_generation_header(show_progress_bars=True)
        uq._display_scoring_header(show_progress_bars=True)
        uq._display_optimization_header(show_progress_bars=True)
    except OSError:
        pass

    # Test stop progress bar with faulty stop
    try:
        uq._stop_progress_bar()
    except RuntimeError:
        pass

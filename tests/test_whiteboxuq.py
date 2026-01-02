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
from uqlm.scorers.shortform.white_box import WhiteBoxUQ
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
async def test_whiteboxuq_basic(monkeypatch):
    wbuq = WhiteBoxUQ(llm=mock_object, scorers=["normalized_probability", "min_probability"])

    async def mock_generate_original_responses(*args, **kwargs):
        wbuq.logprobs = MOCKED_LOGPROBS
        return MOCKED_RESPONSES

    monkeypatch.setattr(wbuq, "generate_original_responses", mock_generate_original_responses)

    for show_progress_bars in [False, True]:
        results = await wbuq.generate_and_score(prompts=PROMPTS, show_progress_bars=show_progress_bars)

        for i in range(len(PROMPTS)):
            assert results.data["normalized_probability"][i] == pytest.approx(data["normalized_probability"][i])
            assert results.data["min_probability"][i] == pytest.approx(data["min_probability"][i])

        assert results.metadata == metadata


@pytest.mark.asyncio
async def test_whiteboxuq_top_logprobs(monkeypatch):
    wbuq = WhiteBoxUQ(llm=mock_object, scorers=["sequence_probability"])

    async def mock_generate_original_responses(*args, **kwargs):
        wbuq.logprobs = MOCKED_LOGPROBS
        return MOCKED_RESPONSES

    monkeypatch.setattr(wbuq, "generate_original_responses", mock_generate_original_responses)

    results = await wbuq.generate_and_score(prompts=PROMPTS, show_progress_bars=False)
    assert "sequence_probability" in results.data


@pytest.mark.asyncio
async def test_whiteboxuq_sampled_logprobs(monkeypatch):
    wbuq = WhiteBoxUQ(llm=mock_object, scorers=["monte_carlo_probability"])

    async def mock_generate_original_responses(*args, **kwargs):
        wbuq.logprobs = MOCKED_LOGPROBS
        return MOCKED_RESPONSES

    async def mock_generate_candidate_responses(*args, **kwargs):
        wbuq.multiple_logprobs = [[[{"token": "Hello", "logprob": -0.1}]]] * len(PROMPTS)
        return [["Hello world"] * 5] * len(PROMPTS)

    monkeypatch.setattr(wbuq, "generate_original_responses", mock_generate_original_responses)
    monkeypatch.setattr(wbuq, "generate_candidate_responses", mock_generate_candidate_responses)

    results = await wbuq.generate_and_score(prompts=PROMPTS, show_progress_bars=False)
    assert "monte_carlo_probability" in results.data


@pytest.mark.asyncio
async def test_whiteboxuq_p_true(monkeypatch):
    wbuq = WhiteBoxUQ(llm=mock_object, scorers=["p_true"])

    async def mock_generate_original_responses(*args, **kwargs):
        wbuq.logprobs = MOCKED_LOGPROBS
        return MOCKED_RESPONSES

    async def mock_p_true_evaluate(*args, **kwargs):
        return {"p_true": [0.9] * len(PROMPTS)}

    monkeypatch.setattr(wbuq, "generate_original_responses", mock_generate_original_responses)
    monkeypatch.setattr(wbuq.p_true_scorer, "evaluate", mock_p_true_evaluate)

    results = await wbuq.generate_and_score(prompts=PROMPTS, show_progress_bars=False)
    assert "p_true" in results.data


def test_whiteboxuq_invalid_scorer():
    with pytest.raises(ValueError, match="Invalid scorer provided: invalid_scorer"):
        WhiteBoxUQ(llm=mock_object, scorers=["invalid_scorer"])


@pytest.mark.asyncio
async def test_whiteboxuq_top_logprobs_full(monkeypatch):
    wbuq = WhiteBoxUQ(llm=mock_object, scorers=["mean_token_negentropy"], top_k_logprobs=10)

    async def mock_generate_original_responses(*args, **kwargs):
        wbuq.logprobs = MOCKED_LOGPROBS
        return MOCKED_RESPONSES

    monkeypatch.setattr(wbuq, "generate_original_responses", mock_generate_original_responses)

    # Optional: monkeypatch the scorer to ensure evaluate is called and returns something
    wbuq.top_logprobs_scorer.evaluate = lambda logprobs_results: {"mean_token_negentropy": [0.8] * len(PROMPTS)}

    results = await wbuq.generate_and_score(prompts=PROMPTS, show_progress_bars=False)
    assert "mean_token_negentropy" in results.data

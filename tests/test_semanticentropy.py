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


@pytest.mark.asyncio
async def test_semanticentropy_no_logprobs_warns():
    """generate_and_score warns when LLM lacks logprobs attribute (line 151)."""
    import warnings
    from unittest.mock import MagicMock, patch, AsyncMock
    from langchain_core.language_models.chat_models import BaseChatModel

    mock_no_logprobs = MagicMock(spec=BaseChatModel)
    mock_no_logprobs.temperature = 0.7

    with patch("uqlm.scorers.baseclass.uncertainty.NLI") as MockNLI, \
         patch("uqlm.scorers.shortform.entropy.SemanticClusterer"):
        MockNLI.return_value = MagicMock()
        se = SemanticEntropy(llm=mock_no_logprobs, use_best=False)

    se.generate_original_responses = AsyncMock(return_value=["response"])
    se.generate_candidate_responses = AsyncMock(return_value=[["sample1", "sample2"]])
    se.score = MagicMock(return_value=MagicMock())

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        await se.generate_and_score(prompts=["test prompt"], show_progress_bars=False)

    assert any("does not support logprobs" in str(warning.message) for warning in w)


def test_score_with_use_best():
    """Cover entropy.py line 226: _update_best is called when use_best=True."""
    from unittest.mock import MagicMock, patch

    with patch("uqlm.scorers.baseclass.uncertainty.NLI") as MockNLI, \
         patch("uqlm.scorers.shortform.entropy.SemanticClusterer"):
        MockNLI.return_value = MagicMock()
        se = SemanticEntropy(use_best=True)

    se._semantic_entropy_process = MagicMock(return_value=("best_r", 0.5, None, 2))
    se._construct_progress_bar = MagicMock()
    se._display_scoring_header = MagicMock()
    se._stop_progress_bar = MagicMock()
    se._construct_black_box_return_data = MagicMock(return_value={})
    se._update_best = MagicMock()
    se.progress_bar = None

    responses = ["r1"]
    sampled_responses = [["s1", "s2"]]

    se.score(responses=responses, sampled_responses=sampled_responses)
    se._update_best.assert_called_once()  # line 226

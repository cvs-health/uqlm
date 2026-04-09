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
from uqlm.scorers.shortform.density import SemanticDensity
from unittest.mock import AsyncMock, MagicMock
from uqlm.utils.results import UQResult
from langchain_openai import AzureChatOpenAI

datafile_path = "tests/data/scorers/semanticdensity_results_file.json"
with open(datafile_path, "r") as f:
    expected_result = json.load(f)

data = expected_result["data"]
metadata = expected_result["metadata"]

mock_object = AzureChatOpenAI(deployment_name="YOUR-DEPLOYMENT", temperature=1, api_key="SECRET_API_KEY", api_version="2024-05-01-preview", azure_endpoint="https://mocked.endpoint.com")


@pytest.mark.flaky(reruns=3)
@pytest.mark.asyncio
async def test_semanticdensity(monkeypatch):
    PROMPTS = data["prompts"]
    MOCKED_RESPONSES = data["responses"]
    MOCKED_SAMPLED_RESPONSES = data["sampled_responses"]

    # Initiate SemanticDensity class object
    sd_object = SemanticDensity(llm=mock_object, device="cpu")

    async def mock_generate_original_responses(*args, **kwargs):
        sd_object.logprobs = [None] * 5
        return MOCKED_RESPONSES

    async def mock_generate_candidate_responses(*args, **kwargs):
        sd_object.multiple_logprobs = data["multiple_logprobs"]
        return MOCKED_SAMPLED_RESPONSES

    monkeypatch.setattr(sd_object, "generate_original_responses", mock_generate_original_responses)
    monkeypatch.setattr(sd_object, "generate_candidate_responses", mock_generate_candidate_responses)

    for show_progress_bars in [True, False]:
        se_results = await sd_object.generate_and_score(prompts=PROMPTS, show_progress_bars=show_progress_bars)
        sd_object.logprobs = None
        sd_results = sd_object.score(responses=MOCKED_RESPONSES, sampled_responses=MOCKED_SAMPLED_RESPONSES)
        assert sd_results.data["responses"] == data["responses"]
        assert sd_results.data["sampled_responses"] == data["sampled_responses"]
        assert sd_results.data["prompts"] == data["prompts"]
        assert all([abs(sd_results.data["semantic_density_values"][i] - data["semantic_density_values"][i]) < 1e-5 for i in range(len(PROMPTS))])
        assert se_results.metadata == metadata


@pytest.mark.asyncio
async def test_generate_and_score_mocked():
    mock_llm = MagicMock()
    mock_llm.logprobs = True

    semantic_density = SemanticDensity(llm=mock_llm)
    semantic_density._setup_nli = MagicMock()
    semantic_density._construct_progress_bar = MagicMock()
    semantic_density._display_generation_header = MagicMock()
    semantic_density.generate_original_responses = AsyncMock(return_value=["response1", "response2"])
    semantic_density.generate_candidate_responses = AsyncMock(return_value=[["sample1", "sample2"], ["sample3", "sample4"]])
    semantic_density.score = MagicMock(return_value=UQResult({"data": {}, "metadata": {}}))

    prompts = ["prompt1", "prompt2"]

    # Manually set prompts since score is mocked
    semantic_density.prompts = prompts

    result = await semantic_density.generate_and_score(prompts, num_responses=2)

    assert isinstance(result, UQResult)
    assert semantic_density.prompts == prompts
    assert semantic_density.num_responses == 2
    semantic_density.generate_original_responses.assert_called_once_with(prompts, progress_bar=semantic_density.progress_bar)
    semantic_density.generate_candidate_responses.assert_called_once_with(prompts, num_responses=2, progress_bar=semantic_density.progress_bar)
    semantic_density.score.assert_called_once()


def test_score_mocked():
    semantic_density = SemanticDensity()
    semantic_density._semantic_density_process = MagicMock(return_value=("density_value", None))
    semantic_density._construct_progress_bar = MagicMock()
    semantic_density._display_scoring_header = MagicMock()
    semantic_density._stop_progress_bar = MagicMock()
    semantic_density._construct_black_box_return_data = MagicMock(return_value={})
    semantic_density.progress_bar = MagicMock()
    semantic_density.progress_bar.add_task = MagicMock(return_value="task_id")
    semantic_density.progress_bar.update = MagicMock()

    # Required attributes
    responses = ["response1", "response2"]
    sampled_responses = [["sample1", "sample2"], ["sample3", "sample4"]]
    prompts = ["prompt1", "prompt2"]
    sampled_logprobs_results = [["logprob1", "logprob2"], ["logprob3", "logprob4"]]
    logprobs_results = [None, None]

    result = semantic_density.score(prompts=prompts, responses=responses, sampled_responses=sampled_responses, logprobs_results=logprobs_results, sampled_logprobs_results=sampled_logprobs_results)

    assert "semantic_density_values" in result.data
    assert "multiple_logprobs" in result.data
    semantic_density._semantic_density_process.assert_called()


def test_semantic_density_process_verbose_and_cached_nli():
    """Cover density.py lines 202 (verbose print) and 215 (cached NLI probabilities)."""
    import numpy as np
    from unittest.mock import patch, MagicMock

    with patch("uqlm.scorers.baseclass.uncertainty.NLI"), \
         patch("uqlm.scorers.shortform.density.SemanticClusterer"):
        sd = SemanticDensity(verbose=True)

    mock_nli = MagicMock()
    mock_nli.label_mapping = ["contradiction", "neutral", "entailment"]
    # Pre-populate probabilities cache so the else-branch (line 215) is taken
    cached_score = np.array([[0.1, 0.1, 0.8]])
    prompt_response_key = "prompt\nresponse_prompt\ncandidate"
    mock_nli.probabilities = {prompt_response_key: cached_score}
    sd.nli = mock_nli
    sd.length_normalize = True

    logprobs = [[{"token": "x", "logprob": -0.1}]]
    # i=0 triggers the verbose print (line 202); cached key triggers line 215
    result = sd._semantic_density_process("prompt", "response", ["candidate"], i=0, logprobs_results=logprobs)
    assert result is not None  # returns (semantic_density, nli_scores)
    # nli.predict should NOT have been called (used cache instead)
    mock_nli.predict.assert_not_called()


def test_semantic_density_process_zero_probabilities():
    """Cover density.py line 230: semantic_density = np.nan when weights sum to zero."""
    import numpy as np
    from unittest.mock import patch, MagicMock

    with patch("uqlm.scorers.baseclass.uncertainty.NLI"), \
         patch("uqlm.scorers.shortform.density.SemanticClusterer"):
        sd = SemanticDensity()

    mock_nli = MagicMock()
    mock_nli.label_mapping = ["contradiction", "neutral", "entailment"]
    mock_nli.probabilities = {}
    mock_nli.predict.return_value = np.array([[0.1, 0.1, 0.8]])
    sd.nli = mock_nli
    sd.length_normalize = True

    # logprob of -1000 underflows to exp(-1000)=0.0, so weights sum to zero
    zero_logprobs = [[{"token": "x", "logprob": -1000}]]
    result_val, nli_scores = sd._semantic_density_process("prompt", "response", ["candidate"], i=None, logprobs_results=zero_logprobs)
    # With zero weights, semantic_density should be np.nan (line 230)
    import math
    assert math.isnan(result_val)


@pytest.mark.asyncio
async def test_semanticdensity_no_logprobs_raises():
    """generate_and_score raises ValueError when LLM lacks logprobs attribute (line 115)."""
    from unittest.mock import MagicMock, patch
    from langchain_core.language_models.chat_models import BaseChatModel

    mock_no_logprobs = MagicMock(spec=BaseChatModel)
    mock_no_logprobs.temperature = 0.7

    with patch("uqlm.scorers.baseclass.uncertainty.NLI") as MockNLI, \
         patch("uqlm.scorers.shortform.density.SemanticClusterer"):
        MockNLI.return_value = MagicMock()
        sd = SemanticDensity(llm=mock_no_logprobs)

    with pytest.raises(ValueError, match="does not support logprobs"):
        await sd.generate_and_score(prompts=["test prompt"])

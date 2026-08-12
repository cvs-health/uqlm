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
import json
import pytest
from uqlm.scorers import BlackBoxUQ
from uqlm.scorers.shortform.baseclass.uncertainty import DEFAULT_BLACK_BOX_SCORERS
from langchain_openai import AzureChatOpenAI

datafile_path = "tests/data/scorers/blackbox_results_file.json"
with open(datafile_path, "r") as f:
    expected_result = json.load(f)

data = expected_result["data"]
metadata = expected_result["metadata"]

PROMPTS = data["prompts"]
MOCKED_RESPONSES = data["responses"]
MOCKED_SAMPLED_RESPONSES = data["sampled_responses"]


@pytest.fixture
def mock_llm():
    """Define mock LLM object using pytest.fixture."""
    return AzureChatOpenAI(deployment_name="YOUR-DEPLOYMENT", temperature=1, api_key="SECRET_API_KEY", api_version="2024-05-01-preview", azure_endpoint="https://mocked.endpoint.com")


@pytest.mark.flaky(reruns=3)
@pytest.mark.asyncio
async def test_bbuq(monkeypatch, mock_llm):
    uqe = BlackBoxUQ(llm=mock_llm, scorers=["noncontradiction", "exact_match", "semantic_negentropy"], device="cpu")

    async def mock_generate_original_responses(*args, **kwargs):
        uqe.logprobs = [None] * 5
        return MOCKED_RESPONSES

    async def mock_generate_candidate_responses(*args, **kwargs):
        uqe.multiple_logprobs = [[None] * 5] * 5
        return MOCKED_SAMPLED_RESPONSES

    monkeypatch.setattr(uqe, "generate_original_responses", mock_generate_original_responses)
    monkeypatch.setattr(uqe, "generate_candidate_responses", mock_generate_candidate_responses)
    for show_progress_bars in [False, True]:
        results = await uqe.generate_and_score(prompts=PROMPTS, num_responses=5, show_progress_bars=show_progress_bars)

        assert all([results.data["exact_match"][i] == pytest.approx(data["exact_match"][i]) for i in range(len(PROMPTS))])

        assert all([results.data["noncontradiction"][i] == pytest.approx(data["noncontradiction"][i]) for i in range(len(PROMPTS))])

        assert all([results.data["semantic_negentropy"][i] == pytest.approx(data["semantic_negentropy"][i]) for i in range(len(PROMPTS))])

        assert results.metadata == metadata

    # Test invalid scorer
    with pytest.raises(ValueError):
        BlackBoxUQ(llm=mock_llm, scorers=["invalid_scorer"], device="cpu")

    # Test default scorers
    uqe_default = BlackBoxUQ(llm=mock_llm, scorers=None, device="cpu")
    assert len(uqe_default.scorers) == len(DEFAULT_BLACK_BOX_SCORERS)

    BlackBoxUQ(llm=mock_llm, scorers=["bert_score"], device="cpu")


@pytest.mark.flaky(reruns=3)
def test_bbuq_generate_and_score_sync(monkeypatch, mock_llm):
    """generate_and_score_sync should be usable with no `await`/event loop management from the
    caller, and must produce the same scores as the async `generate_and_score` method it wraps."""
    uqe = BlackBoxUQ(llm=mock_llm, scorers=["noncontradiction", "exact_match", "semantic_negentropy"], device="cpu")

    async def mock_generate_original_responses(*args, **kwargs):
        uqe.logprobs = [None] * 5
        return MOCKED_RESPONSES

    async def mock_generate_candidate_responses(*args, **kwargs):
        uqe.multiple_logprobs = [[None] * 5] * 5
        return MOCKED_SAMPLED_RESPONSES

    monkeypatch.setattr(uqe, "generate_original_responses", mock_generate_original_responses)
    monkeypatch.setattr(uqe, "generate_candidate_responses", mock_generate_candidate_responses)

    # Note: this test function is intentionally NOT `async def` and is NOT awaited by pytest-asyncio.
    # There is no event loop running in this thread, exercising the `asyncio.run` code path in `run_sync`.
    results = uqe.generate_and_score_sync(prompts=PROMPTS, num_responses=5, show_progress_bars=False)

    assert all([results.data["exact_match"][i] == pytest.approx(data["exact_match"][i]) for i in range(len(PROMPTS))])
    assert all([results.data["noncontradiction"][i] == pytest.approx(data["noncontradiction"][i]) for i in range(len(PROMPTS))])
    assert all([results.data["semantic_negentropy"][i] == pytest.approx(data["semantic_negentropy"][i]) for i in range(len(PROMPTS))])
    assert results.metadata == metadata


@pytest.mark.flaky(reruns=3)
@pytest.mark.asyncio
async def test_bbuq_generate_and_score_sync_inside_running_loop(monkeypatch, mock_llm):
    """generate_and_score_sync must also work when called from a thread that already has a running
    event loop (e.g. a Jupyter/IPython kernel), where a naive `asyncio.run(...)` wrapper would raise
    `RuntimeError: asyncio.run() cannot be called from a running event loop`."""
    uqe = BlackBoxUQ(llm=mock_llm, scorers=["noncontradiction", "exact_match", "semantic_negentropy"], device="cpu")

    async def mock_generate_original_responses(*args, **kwargs):
        uqe.logprobs = [None] * 5
        return MOCKED_RESPONSES

    async def mock_generate_candidate_responses(*args, **kwargs):
        uqe.multiple_logprobs = [[None] * 5] * 5
        return MOCKED_SAMPLED_RESPONSES

    monkeypatch.setattr(uqe, "generate_original_responses", mock_generate_original_responses)
    monkeypatch.setattr(uqe, "generate_candidate_responses", mock_generate_candidate_responses)

    # This test itself is `async def`, so pytest-asyncio drives it with a running event loop. Calling
    # the *synchronous* generate_and_score_sync from here (with no `await`) is exactly the scenario
    # that requires the thread-offload fallback inside `run_sync`.
    results = uqe.generate_and_score_sync(prompts=PROMPTS, num_responses=5, show_progress_bars=False)

    assert all([results.data["exact_match"][i] == pytest.approx(data["exact_match"][i]) for i in range(len(PROMPTS))])
    assert all([results.data["noncontradiction"][i] == pytest.approx(data["noncontradiction"][i]) for i in range(len(PROMPTS))])
    assert all([results.data["semantic_negentropy"][i] == pytest.approx(data["semantic_negentropy"][i]) for i in range(len(PROMPTS))])
    assert results.metadata == metadata
def test_single_nli_model_instance():
    """Regression test: default BlackBoxUQ loaded the NLI model once per scorer (2x, ~1.4 GB each)."""
    from unittest.mock import MagicMock, patch

    mock_tokenizer = MagicMock()
    mock_tokenizer.model_max_length = 512
    mock_model = MagicMock()
    mock_model.to.return_value = mock_model
    mock_model.eval.return_value = mock_model

    with patch("uqlm.nli.nli.AutoModelForSequenceClassification.from_pretrained", return_value=mock_model) as mock_model_loader, patch("uqlm.nli.nli.AutoTokenizer.from_pretrained", return_value=mock_tokenizer), patch("sentence_transformers.SentenceTransformer", return_value=MagicMock()):
        uqe = BlackBoxUQ(device="cpu")

    assert mock_model_loader.call_count == 1
    assert uqe.scorer_objects["consistency"].nli is uqe.scorer_objects["semantic_negentropy"].nli

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

"""Tests for uqlm/longform/graph/claim_merger.py"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from uqlm.longform.graph.claim_merger import ClaimMerger


@pytest.fixture
def mock_llm():
    return MagicMock()


def _make_merger(mock_llm):
    with patch("uqlm.longform.graph.claim_merger.ResponseGenerator") as MockRG:
        rg_instance = MagicMock()
        MockRG.return_value = rg_instance
        merger = ClaimMerger(claim_merging_llm=mock_llm)
        merger.rg = rg_instance
        return merger, rg_instance


# ---------------------------------------------------------------------------
# merge_claims — happy path (lines 34-54)
# ---------------------------------------------------------------------------


def test_merge_claims_no_new_claims(mock_llm):
    """When sampled claims are all already in the master set, no LLM calls needed."""
    merger, rg_instance = _make_merger(mock_llm)
    original_claim_sets = [["claim A", "claim B"]]
    # sampled_claim_sets: 1 response set × 2 iterations, each iteration has same claims
    sampled_claim_sets = [[["claim A"], ["claim B"]]]

    result = asyncio.run(merger.merge_claims(original_claim_sets=original_claim_sets, sampled_claim_sets=sampled_claim_sets))
    assert result == [["claim A", "claim B"]]
    rg_instance.generate_responses.assert_not_called()


def test_merge_claims_adds_new_claims(mock_llm):
    """New claims in sampled sets should be added to master after LLM dedup call."""
    merger, rg_instance = _make_merger(mock_llm)

    # LLM returns a markdown list with one new claim
    rg_instance.generate_responses = AsyncMock(return_value={"data": {"response": ["- new claim C"]}})

    original_claim_sets = [["claim A"]]
    sampled_claim_sets = [[["new claim C"]]]  # 1 response set, 1 iteration, one unique claim

    result = asyncio.run(merger.merge_claims(original_claim_sets=original_claim_sets, sampled_claim_sets=sampled_claim_sets))
    assert "new claim C" in result[0]


def test_merge_claims_with_progress_bar(mock_llm):
    """Progress bar task should be created when provided (line 39)."""
    merger, rg_instance = _make_merger(mock_llm)
    rg_instance.generate_responses = AsyncMock(return_value={"data": {"response": []}})

    progress_bar = MagicMock()
    progress_bar.add_task.return_value = MagicMock()

    original_claim_sets = [["claim A"]]
    sampled_claim_sets = [[["claim A"]]]

    asyncio.run(merger.merge_claims(original_claim_sets=original_claim_sets, sampled_claim_sets=sampled_claim_sets, progress_bar=progress_bar))
    progress_bar.add_task.assert_called_once()


def test_merge_claims_multiple_response_sets(mock_llm):
    """Multiple response sets are each handled independently."""
    merger, rg_instance = _make_merger(mock_llm)
    rg_instance.generate_responses = AsyncMock(return_value={"data": {"response": ["- extra D", "- extra E"]}})

    original_claim_sets = [["claim A"], ["claim B"]]
    sampled_claim_sets = [[["extra D"]], [["extra E"]]]

    result = asyncio.run(merger.merge_claims(original_claim_sets=original_claim_sets, sampled_claim_sets=sampled_claim_sets))
    assert len(result) == 2


# ---------------------------------------------------------------------------
# _process_claim_merging_generations — progress_bar branch (line 70)
# ---------------------------------------------------------------------------


def test_process_claim_merging_generations_with_progress_bar(mock_llm):
    merger, _ = _make_merger(mock_llm)
    merger.master_claim_sets = [["existing"]]
    merger.progress_task = MagicMock()

    progress_bar = MagicMock()
    responses = ["- brand new claim"]
    prompt_metadata = [(0, True, ["existing"], ["brand new claim"])]

    merger._process_claim_merging_generations(responses, prompt_metadata, progress_bar)
    progress_bar.update.assert_called_once()


# ---------------------------------------------------------------------------
# _construct_merging_prompts — else branch (line 87)
# ---------------------------------------------------------------------------


def test_construct_merging_prompts_iteration_beyond_samples(mock_llm):
    """When iteration >= len(sampled_claim_sets[i]), append (i, False, ...) (line 87)."""
    merger, _ = _make_merger(mock_llm)
    merger.master_claim_sets = [["claim A"]]
    sampled_claim_sets = [[["claim A"]]]  # only 1 iteration worth of samples

    prompts, metadata = merger._construct_merging_prompts(sampled_claim_sets=sampled_claim_sets, iteration=5)
    assert prompts == []
    assert metadata[0][1] is False  # has_prompt is False

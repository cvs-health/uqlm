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

"""Tests for uqlm/nli/entailment.py"""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from uqlm.nli.entailment import EntailmentClassifier


@pytest.fixture
def mock_llm():
    llm = MagicMock()
    llm.ainvoke = AsyncMock()
    return llm


@pytest.fixture
def classifier(mock_llm):
    return EntailmentClassifier(nli_llm=mock_llm)


# ---------------------------------------------------------------------------
# _evaluate_claim_response_pair  (lines 143-153)
# ---------------------------------------------------------------------------


def test_evaluate_claim_response_pair_returns_content(mock_llm):
    mock_llm.ainvoke.return_value = MagicMock(content="yes")
    ec = EntailmentClassifier(nli_llm=mock_llm)
    result = asyncio.run(ec._evaluate_claim_response_pair("Is the sky blue?"))
    assert result == "yes"


def test_evaluate_claim_response_pair_with_progress_bar(mock_llm):
    mock_llm.ainvoke.return_value = MagicMock(content="no")
    ec = EntailmentClassifier(nli_llm=mock_llm)
    ec.num_responses = None

    progress_bar = MagicMock()
    progress_task = MagicMock()
    ec.progress_task = progress_task
    ec.num_prompts = 3

    result = asyncio.run(ec._evaluate_claim_response_pair("some prompt", progress_bar=progress_bar))
    assert result == "no"
    progress_bar.update.assert_called_once()


def test_evaluate_claim_response_pair_with_progress_bar_num_responses(mock_llm):
    mock_llm.ainvoke.return_value = MagicMock(content="yes")
    ec = EntailmentClassifier(nli_llm=mock_llm)
    ec.num_responses = 4
    ec.completed = 0
    ec.num_prompts = 8

    progress_bar = MagicMock()
    ec.progress_task = MagicMock()

    asyncio.run(ec._evaluate_claim_response_pair("prompt", progress_bar=progress_bar))
    assert ec.completed == 1
    progress_bar.update.assert_called_once()


# ---------------------------------------------------------------------------
# judge_entailment  (lines 72-106)
# ---------------------------------------------------------------------------


def test_judge_entailment_basic(mock_llm):
    mock_llm.ainvoke.return_value = MagicMock(content="yes")
    ec = EntailmentClassifier(nli_llm=mock_llm)
    premises = ["The sky is blue.", "Water is wet."]
    hypotheses = ["The sky has color.", "Water is liquid."]
    result = asyncio.run(ec.judge_entailment(premises=premises, hypotheses=hypotheses))
    assert "scores" in result
    assert "judge_prompts" in result
    assert "judge_responses" in result
    assert len(result["scores"]) == 2


def test_judge_entailment_with_progress_bar(mock_llm):
    mock_llm.ainvoke.return_value = MagicMock(content="no")
    ec = EntailmentClassifier(nli_llm=mock_llm)
    progress_bar = MagicMock()
    progress_bar.add_task.return_value = MagicMock()

    result = asyncio.run(ec.judge_entailment(premises=["p1"], hypotheses=["h1"], progress_bar=progress_bar))
    progress_bar.add_task.assert_called_once()
    assert len(result["scores"]) == 1


def test_judge_entailment_with_num_responses(mock_llm):
    mock_llm.ainvoke.return_value = MagicMock(content="yes")
    ec = EntailmentClassifier(nli_llm=mock_llm)
    ec.num_responses = 2

    progress_bar = MagicMock()
    progress_bar.add_task.return_value = MagicMock()

    result = asyncio.run(ec.judge_entailment(premises=["p1", "p2"], hypotheses=["h1", "h2"], progress_bar=progress_bar))
    assert len(result["scores"]) == 2


# ---------------------------------------------------------------------------
# evaluate_claim_entailment  (lines 131-135)
# ---------------------------------------------------------------------------


def test_evaluate_claim_entailment_returns_list(mock_llm):
    mock_llm.ainvoke.return_value = MagicMock(content="yes")
    ec = EntailmentClassifier(nli_llm=mock_llm)
    response_sets = [["resp A", "resp B"], ["resp C"]]
    claim_sets = [["claim 1"], ["claim 2"]]
    result = asyncio.run(ec.evaluate_claim_entailment(response_sets=response_sets, claim_sets=claim_sets))
    assert isinstance(result, list)
    assert len(result) == 2
    assert isinstance(result[0], np.ndarray)


def test_evaluate_claim_entailment_shape(mock_llm):
    mock_llm.ainvoke.return_value = MagicMock(content="yes")
    ec = EntailmentClassifier(nli_llm=mock_llm)
    response_sets = [["r1", "r2", "r3"]]
    claim_sets = [["c1", "c2"]]
    result = asyncio.run(ec.evaluate_claim_entailment(response_sets=response_sets, claim_sets=claim_sets))
    # shape should be (num_claims, num_responses)
    assert result[0].shape == (2, 3)

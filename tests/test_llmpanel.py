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
import os
import json
from unittest.mock import MagicMock
from uqlm.judges.judge import LLMJudge
from uqlm.scorers.shortform.panel import LLMPanel
from langchain_core.language_models.chat_models import BaseChatModel


datafile_path = os.path.join(os.path.dirname(__file__), "data/scorers/test_data_panelquantifier.json")
with open(datafile_path, "r") as f:
    data = json.load(f)


@pytest.fixture
def mock_judges():
    judge1 = MagicMock(spec=LLMJudge)
    judge2 = MagicMock(spec=LLMJudge)
    return [judge1, judge2]


@pytest.fixture
def mock_judge_single():
    judge = MagicMock(spec=LLMJudge)
    return judge


@pytest.fixture
def mock_llm():
    mock_llm = MagicMock(spec=BaseChatModel)
    mock_llm.temperature = 0.7
    return mock_llm


@pytest.fixture
def quantifier(mock_judges, mock_llm):
    return LLMPanel(judges=mock_judges, llm=mock_llm)


@pytest.mark.asyncio
async def test_llmpanel(monkeypatch, quantifier):
    PROMPTS = data["prompts"]
    MOCKED_RESPONSES = data["responses"]
    SCORES = data["scores"]
    METADATA = data["metadata"]

    # Mock methods
    async def mock_generate_original_responses(*args, **kwargs):
        return MOCKED_RESPONSES

    monkeypatch.setattr(quantifier, "generate_original_responses", mock_generate_original_responses)

    async def mock_judge_responses(*args, **kwargs):
        return {"scores": [0.8, 0.9]}

    for judge in quantifier.judges:
        monkeypatch.setattr(judge, "judge_responses", mock_judge_responses)

    # Call generate_and_score method to compute scores
    for show_progress_bars in [False, True]:
        result = await quantifier.generate_and_score(prompts=PROMPTS, show_progress_bars=show_progress_bars)

        expected_data = {"prompts": PROMPTS, "responses": MOCKED_RESPONSES, "judge_1": SCORES["judge_1"], "judge_2": SCORES["judge_2"], "avg": SCORES["avg"], "max": SCORES["max"], "min": SCORES["min"], "median": SCORES["median"]}

        expected_result = {"data": expected_data, "metadata": METADATA}

        assert result.data == expected_result["data"]
        assert result.metadata == expected_result["metadata"]


def test_scoring_templates_length_validation(mock_judges, mock_llm):
    """Test ValueError when scoring_templates length != judges length"""
    with pytest.raises(ValueError) as value_error:
        LLMPanel(judges=mock_judges, llm=mock_llm, scoring_templates=["template1"])
    assert "Length of scoring_templates list must be equal to length of judges list" == str(value_error.value)


def test_custom_scoring_templates(mock_judge_single, mock_llm):
    """Test the else branch when custom scoring_templates provided"""
    panel = LLMPanel(judges=[mock_judge_single], llm=mock_llm, scoring_templates=["custom_template"])
    assert panel.scoring_templates == ["custom_template"]


def test_invalid_judge_type(mock_llm):
    """Test ValueError for invalid judge types"""
    with pytest.raises(ValueError) as value_error:
        LLMPanel(judges=["invalid_judge"], llm=mock_llm)
    assert "judges must be a list containing instances of either LLMJudge or BaseChatModel" == str(value_error.value)


def test_basechatmodel_judge_conversion(monkeypatch, mock_llm):
    """Test BaseChatModel judges get converted to LLMJudge"""
    mock_judge = MagicMock(spec=BaseChatModel)
    mock_llm_judge = MagicMock(spec=LLMJudge)
    # Mock LLMJudge constructor
    monkeypatch.setattr("uqlm.judges.judge.LLMJudge", lambda **kwargs: mock_llm_judge)
    panel = LLMPanel(judges=[mock_judge], llm=mock_llm)
    assert len(panel.judges) == 1


def test_explanations_parameter_default(mock_judge_single, mock_llm):
    """Test that explanations parameter defaults to False"""
    panel = LLMPanel(judges=[mock_judge_single], llm=mock_llm)
    assert not panel.explanations


def test_explanations_parameter_enabled(mock_judge_single, mock_llm):
    """Test that explanations parameter can be set to True"""
    panel = LLMPanel(judges=[mock_judge_single], llm=mock_llm, explanations=True)
    assert panel.explanations


@pytest.mark.asyncio
async def test_llmpanel_with_explanations(monkeypatch, mock_judges, mock_llm):
    """Test LLMPanel with explanations enabled"""
    PROMPTS = data["prompts"]
    MOCKED_RESPONSES = data["responses"]
    SCORES = data["scores"]
    EXPLANATIONS = ["Explanation 1", "Explanation 2"]
    METADATA = data["metadata"]

    panel = LLMPanel(judges=mock_judges, llm=mock_llm, explanations=True)

    # Mock methods
    async def mock_generate_original_responses(*args, **kwargs):
        return MOCKED_RESPONSES

    monkeypatch.setattr(panel, "generate_original_responses", mock_generate_original_responses)

    async def mock_judge_responses(*args, **kwargs):
        return {"scores": [0.8, 0.9], "explanations": EXPLANATIONS}

    for judge in panel.judges:
        monkeypatch.setattr(judge, "judge_responses", mock_judge_responses)

    # Call generate_and_score method
    result = await panel.generate_and_score(prompts=PROMPTS, show_progress_bars=False)

    # Check that explanation columns are included
    expected_data = {"prompts": PROMPTS, "responses": MOCKED_RESPONSES, "judge_1": SCORES["judge_1"], "judge_1_explanation": EXPLANATIONS, "judge_2": SCORES["judge_2"], "judge_2_explanation": EXPLANATIONS, "avg": SCORES["avg"], "max": SCORES["max"], "min": SCORES["min"], "median": SCORES["median"]}

    assert result.data == expected_data
    assert result.metadata == METADATA


@pytest.mark.asyncio
async def test_llmpanel_without_explanations(monkeypatch, mock_judges, mock_llm):
    """Test LLMPanel without explanations (backward compatibility)"""
    PROMPTS = data["prompts"]
    MOCKED_RESPONSES = data["responses"]
    SCORES = data["scores"]
    METADATA = data["metadata"]

    panel = LLMPanel(judges=mock_judges, llm=mock_llm, explanations=False)

    # Mock methods
    async def mock_generate_original_responses(*args, **kwargs):
        return MOCKED_RESPONSES

    monkeypatch.setattr(panel, "generate_original_responses", mock_generate_original_responses)

    async def mock_judge_responses(*args, **kwargs):
        return {"scores": [0.8, 0.9]}

    for judge in panel.judges:
        monkeypatch.setattr(judge, "judge_responses", mock_judge_responses)

    # Call generate_and_score method
    result = await panel.generate_and_score(prompts=PROMPTS, show_progress_bars=False)

    # Check that explanation columns are NOT included
    expected_data = {"prompts": PROMPTS, "responses": MOCKED_RESPONSES, "judge_1": SCORES["judge_1"], "judge_2": SCORES["judge_2"], "avg": SCORES["avg"], "max": SCORES["max"], "min": SCORES["min"], "median": SCORES["median"]}

    assert result.data == expected_data
    assert result.metadata == METADATA
    # Ensure no explanation columns are present
    assert "judge_1_explanation" not in result.data
    assert "judge_2_explanation" not in result.data

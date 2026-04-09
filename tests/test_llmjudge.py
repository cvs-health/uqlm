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

import numpy as np
import pytest
import json
from uqlm.judges.judge import LLMJudge
from langchain_openai import AzureChatOpenAI
import warnings
from unittest.mock import AsyncMock


@pytest.fixture
def mock_llm():
    """Extract judge object using pytest.fixture."""
    return AzureChatOpenAI(deployment_name="YOUR-DEPLOYMENT", temperature=1, api_key="SECRET_API_KEY", api_version="2024-05-01-preview", azure_endpoint="https://mocked.endpoint.com")


@pytest.fixture
def test_data():
    """Load test data for all templates."""
    datafile_path = "tests/data/scorers/llmjudge_results_file.json"
    with open(datafile_path, "r") as f:
        return json.load(f)


@pytest.mark.asyncio
async def test_judge_responses(monkeypatch, mock_llm, test_data):
    likert_data = test_data["templates"]["likert"]
    tmp = {"data": {"prompt": likert_data["judge_result"]["judge_prompts"], "response": likert_data["judge_result"]["judge_responses"].copy()}}
    tmp["data"]["response"][2] = np.nan
    tmp1 = [tmp, {"data": {"response": [likert_data["judge_result"]["judge_responses"][2]]}}]

    async def mock_generate_responses(*args, **kwargs):
        return tmp1.pop(0)

    judge = LLMJudge(llm=mock_llm, scoring_template="likert")
    monkeypatch.setattr(judge, "generate_responses", mock_generate_responses)
    data = await judge.judge_responses(prompts=test_data["prompts"], responses=test_data["responses"])
    assert data["scores"] == likert_data["judge_result"]["scores"]


def test_extract_single_answer_likert(mock_llm, test_data):
    """Test Likert score extraction"""
    judge = LLMJudge(llm=mock_llm, scoring_template="likert")
    # Access Likert-specific data
    likert_data = test_data["templates"]["likert"]
    judge_responses = likert_data["judge_result"]["judge_responses"]
    expected_scores = likert_data["extract_answer"]
    for i, (response, expected) in enumerate(zip(judge_responses, expected_scores)):
        extracted_score = judge._extract_single_answer(response)
        assert extracted_score == expected, f"Failed for response {i}: {response}"
    # Test basic Likert extraction
    assert judge._extract_single_answer("5") == 1.0
    assert judge._extract_single_answer("4") == 0.75
    assert judge._extract_single_answer("3") == 0.5
    assert judge._extract_single_answer("2") == 0.25
    assert judge._extract_single_answer("1") == 0.0
    assert judge._extract_single_answer("partially correct") == 0.5


def test_extract_single_answer_continuous(mock_llm, test_data):
    """Test continuous score extraction"""
    judge = LLMJudge(llm=mock_llm, scoring_template="continuous")
    # Access continuous-specific data
    continuous_data = test_data["templates"]["continuous"]
    judge_responses = continuous_data["judge_result"]["judge_responses"]
    expected_scores = continuous_data["extract_answer"]
    for i, (response, expected) in enumerate(zip(judge_responses, expected_scores)):
        extracted_score = judge._extract_single_answer(response)
        assert extracted_score == expected, f"Failed for response {i}: {response}"
    # Test basic continuous extraction
    assert judge._extract_single_answer("95") == 0.95
    assert judge._extract_single_answer("50") == 0.5
    assert judge._extract_single_answer("0") == 0.0


def test_extract_single_answer_true_false(mock_llm, test_data):
    """Test true/false score extraction"""
    judge = LLMJudge(llm=mock_llm, scoring_template="true_false")
    # Access true_false-specific data
    true_false_data = test_data["templates"]["true_false"]
    judge_responses = true_false_data["judge_result"]["judge_responses"]
    expected_scores = true_false_data["extract_answer"]
    for i, (response, expected) in enumerate(zip(judge_responses, expected_scores)):
        extracted_score = judge._extract_single_answer(response)
        assert extracted_score == expected, f"Failed for response {i}: {response}"
    # Test basic true/false extraction
    assert judge._extract_single_answer("correct") == 1.0
    assert judge._extract_single_answer("incorrect") == 0.0
    # Should not have uncertain option
    assert 0.5 not in judge.keywords_to_scores_dict.keys()


def test_extract_single_answer_true_false_uncertain(mock_llm, test_data):
    """Test true/false/uncertain score extraction"""
    judge = LLMJudge(llm=mock_llm, scoring_template="true_false_uncertain")
    # Access true_false_uncertain-specific data
    tfu_data = test_data["templates"]["true_false_uncertain"]
    judge_responses = tfu_data["judge_result"]["judge_responses"]
    expected_scores = tfu_data["extract_answer"]
    for i, (response, expected) in enumerate(zip(judge_responses, expected_scores)):
        extracted_score = judge._extract_single_answer(response)
        assert extracted_score == expected, f"Failed for response {i}: {response}"
    # Test basic true/false/uncertain extraction
    assert judge._extract_single_answer("correct") == 1.0
    assert judge._extract_single_answer("uncertain") == 0.5
    assert judge._extract_single_answer("incorrect") == 0.0


def test_extract_answers_batch(mock_llm, test_data):
    """Test batch extraction using  data for all templates"""
    templates = ["true_false_uncertain", "true_false", "continuous", "likert"]
    for template_name in templates:
        print(f"\nTesting batch extraction for {template_name}")
        judge = LLMJudge(llm=mock_llm, scoring_template=template_name)
        # Get template-specific data
        template_data = test_data["templates"][template_name]
        judge_responses = template_data["judge_result"]["judge_responses"]
        expected_scores = template_data["extract_answer"]
        # Test batch extraction
        extracted_scores = judge._extract_answers(responses=judge_responses)
        assert len(extracted_scores) == len(expected_scores)
        for i, (actual, expected) in enumerate(zip(extracted_scores, expected_scores)):
            assert actual == expected, f"Batch extraction failed for {template_name} item {i}: Expected {expected}, got {actual}"


def test_custom_validate_inputs3(mock_llm):
    with pytest.raises(ValueError) as value_error:
        LLMJudge(llm=mock_llm, scoring_template="wrong")
    assert "If provided, scoring_template must be one of 'true_false_uncertain', 'true_false', 'continuous', 'likert'" == str(value_error.value)


def test_parse_structured_response_malformed(mock_llm):
    """Test parsing of malformed structured responses"""
    judge = LLMJudge(llm=mock_llm, scoring_template="true_false_uncertain")

    # Test missing Score line
    response = "Explanation: This is just an explanation without a score."
    score, explanation = judge._parse_structured_response(response)
    assert np.isnan(score)
    assert explanation == "No explanation provided"

    # Test missing Explanation line
    response = "Score: correct"
    score, explanation = judge._parse_structured_response(response)
    assert score == 1.0
    assert explanation == "No explanation provided"


@pytest.mark.asyncio
async def test_judge_responses_with_explanations(monkeypatch, mock_llm):
    """Test judge_responses with explanations enabled"""
    judge = LLMJudge(llm=mock_llm, scoring_template="true_false_uncertain")

    prompts = ["Question 1", "Question 2"]
    responses = ["Answer 1", "Answer 2"]

    # Mock the generate_responses method
    mock_responses = {"data": {"prompt": ["Question 1", "Question 2"], "response": ["Score: correct\nExplanation: This is correct.", "Score: incorrect\nExplanation: This is incorrect."]}}

    async def mock_generate_responses(*args, **kwargs):
        return mock_responses

    monkeypatch.setattr(judge, "generate_responses", mock_generate_responses)

    result = await judge.judge_responses(prompts=prompts, responses=responses, explanations=True)

    assert "scores" in result
    assert "explanations" in result
    assert result["scores"] == [1.0, 0.0]
    assert result["explanations"] == ["This is correct.", "This is incorrect."]


@pytest.mark.asyncio
async def test_judge_responses_without_explanations(monkeypatch, mock_llm):
    """Test judge_responses without explanations (backward compatibility)"""
    judge = LLMJudge(llm=mock_llm, scoring_template="true_false_uncertain")

    prompts = ["Question 1", "Question 2"]
    responses = ["Answer 1", "Answer 2"]

    # Mock the generate_responses method
    mock_responses = {"data": {"prompt": ["Question 1", "Question 2"], "response": ["correct", "incorrect"]}}

    async def mock_generate_responses(*args, **kwargs):
        return mock_responses

    monkeypatch.setattr(judge, "generate_responses", mock_generate_responses)

    result = await judge.judge_responses(prompts=prompts, responses=responses, explanations=False)

    assert "scores" in result
    assert "explanations" not in result
    assert result["scores"] == [1.0, 0.0]


@pytest.mark.asyncio
async def test_judge_responses_retry_with_explanations(monkeypatch):
    llm_judge = LLMJudge(llm="fake_llm", scoring_template="true_false_uncertain")

    prompts = ["What is 2+2?"]
    responses = ["It is four."]

    first_mock_response = {"data": {"prompt": ["Q1"], "response": ["invalid"]}}
    second_mock_response = {"data": {"prompt": ["Q1"], "response": ["Score: 1.0\nExplanation: Correct"]}}

    # Mock generate_responses for initial + retry calls
    mock_generate = AsyncMock(side_effect=[first_mock_response, second_mock_response, second_mock_response])
    monkeypatch.setattr(llm_judge, "generate_responses", mock_generate)

    # Mock _extract_answers to simulate retry success
    def extract_answers_mock(responses, explanations=False):
        if responses == ["invalid"]:
            return ([np.nan], ["Parsing failed"]) if explanations else [np.nan]
        return ([1.0], ["Correct"]) if explanations else [1.0]

    monkeypatch.setattr(llm_judge, "_extract_answers", extract_answers_mock)

    result = await llm_judge.judge_responses(prompts, responses, retries=1, explanations=True)

    # Assert
    assert result["scores"][0] == 1.0
    assert result["explanations"][0] == "Correct"


def test_extract_score_likert_nan():
    judge = LLMJudge(llm=None, scoring_template="likert")
    score = judge._extract_score_from_text("random text")
    assert np.isnan(score)


def test_extract_score_continuous_nan():
    judge = LLMJudge(llm=None, scoring_template="continuous")
    score = judge._extract_score_from_text("no numbers here")
    assert np.isnan(score)


def test_extract_score_true_false_nan():
    judge = LLMJudge(llm=None, scoring_template="true_false_uncertain")
    score = judge._extract_score_from_text("completely unrelated")
    assert np.isnan(score)


def test_invalid_scoring_template_raises():
    with pytest.raises(ValueError):
        LLMJudge(llm=None, scoring_template="unknown")


def test_parse_structured_response_exception(monkeypatch):
    judge = LLMJudge(llm=None, scoring_template="true_false_uncertain")

    # Force _extract_score_from_text to raise an exception
    monkeypatch.setattr(judge, "_extract_score_from_text", lambda x: (_ for _ in ()).throw(ValueError("bad parse")))

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        score, explanation = judge._parse_structured_response("Score: something Explanation: text")

        # Assertions
        assert np.isnan(score)
        assert explanation == "Parsing failed - using NaN"
        assert any("Failed to parse judge response" in str(warn.message) for warn in w)


def test_extract_score_from_text_unknown_template_returns_nan():
    """Bypass __init__ validation: unknown scoring_template → fallback return np.nan (line 237)."""
    judge = LLMJudge(llm=None, scoring_template="true_false")
    # Override after construction to bypass _validate_inputs
    judge.scoring_template = "_unknown_template_xyz_"
    result = judge._extract_score_from_text("yes")
    assert np.isnan(result)

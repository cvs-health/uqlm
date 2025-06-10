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
from uqlm.judges import LLMJudge
from langchain_openai import AzureChatOpenAI


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


def test_custom_validate_inputs1(mock_llm):
    with pytest.raises(ValueError) as value_error:
        LLMJudge(llm=mock_llm, template_ques_ans="template{}{}", keywords_to_scores_dict={"1": ["1"]})
    assert "keys in keywords_to_scores_dict must be floats" == str(value_error.value)


def test_custom_validate_inputs2(mock_llm):
    with pytest.raises(ValueError) as value_error:
        LLMJudge(llm=mock_llm, template_ques_ans="template{}{}", keywords_to_scores_dict={1.0: 1})
    assert "values in keywords_to_scores_dict must be lists of strings" == str(value_error.value)


def test_custom_validate_inputs3(mock_llm):
    with pytest.raises(ValueError) as value_error:
        LLMJudge(llm=mock_llm, scoring_template="wrong")
    assert "If provided, scoring_template must be one of 'true_false_uncertain', 'true_false', 'continuous', 'likert'. Otherwise, valid template_ques_ans and keywords_to_scores_dict must be provided" == str(value_error.value)

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
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from typing import List, Dict, Any
from rich.progress import Progress
from langchain_core.language_models.chat_models import BaseChatModel

from uqlm.longform.qa.question_generator import QuestionGenerator
from uqlm.utils.response_generator import ResponseGenerator


class TestQuestionGenerator:
    @pytest.fixture
    def mock_llm(self):
        """Create a mock LLM for testing."""
        mock = MagicMock(spec=BaseChatModel)
        return mock

    @pytest.fixture
    def mock_response_generator(self):
        """Create a mock ResponseGenerator."""
        mock = MagicMock(spec=ResponseGenerator)
        mock.generate_responses = AsyncMock(return_value={"data": {"response": ["Mock question"]}})
        return mock

    @pytest.fixture
    def question_generator(self, mock_llm, mock_response_generator):
        """Create a QuestionGenerator instance with mocked components."""
        with patch("uqlm.longform.qa.question_generator.ResponseGenerator", return_value=mock_response_generator):
            generator = QuestionGenerator(question_generator_llm=mock_llm)
            return generator

    def test_initialization(self, mock_llm):
        """Test proper initialization of the question generator."""
        # We need to patch the ResponseGenerator to avoid the assertion error
        with patch("uqlm.longform.qa.question_generator.ResponseGenerator") as mock_rg_class:
            mock_rg_instance = MagicMock()
            mock_rg_class.return_value = mock_rg_instance

            # Test with default max_calls_per_min
            generator = QuestionGenerator(question_generator_llm=mock_llm)
            mock_rg_class.assert_called_once_with(llm=mock_llm, max_calls_per_min=None)
            assert generator.rg == mock_rg_instance
            assert mock_rg_instance.response_generator_type == "question_generator"

            # Reset mock for second test
            mock_rg_class.reset_mock()

            # Test with custom max_calls_per_min
            generator = QuestionGenerator(question_generator_llm=mock_llm, max_calls_per_min=10)
            mock_rg_class.assert_called_once_with(llm=mock_llm, max_calls_per_min=10)

    def test_construct_question_generation_prompts_single_question(self):
        """Test constructing prompts for single question generation."""
        claim_sets = [["Claim 1", "Claim 2"], ["Claim 3"]]

        with patch("uqlm.longform.qa.question_generator.get_question_template") as mock_template:
            mock_template.side_effect = lambda claim: f"Single question for: {claim}"

            result = QuestionGenerator._construct_question_generation_prompts(claim_sets=claim_sets, num_questions=1)

            assert len(result) == 3
            assert result[0] == "Single question for: Claim 1"
            assert result[1] == "Single question for: Claim 2"
            assert result[2] == "Single question for: Claim 3"
            assert mock_template.call_count == 3

    def test_construct_question_generation_prompts_multiple_questions(self):
        """Test constructing prompts for multiple question generation."""
        claim_sets = [["Claim 1"], ["Claim 2"]]
        responses = ["Response 1", "Response 2"]

        with patch("uqlm.longform.qa.question_generator.get_multiple_question_template") as mock_template:
            mock_template.side_effect = lambda claim, num, response: f"Multiple questions for: {claim}, count: {num}, response: {response}"

            result = QuestionGenerator._construct_question_generation_prompts(claim_sets=claim_sets, num_questions=3, responses=responses)

            assert len(result) == 2
            assert result[0] == "Multiple questions for: Claim 1, count: 3, response: Response 1"
            assert result[1] == "Multiple questions for: Claim 2, count: 3, response: Response 2"
            assert mock_template.call_count == 2

    def test_construct_question_generation_prompts_no_responses(self):
        """Test constructing prompts without providing responses."""
        claim_sets = [["Claim 1"]]

        with patch("uqlm.longform.qa.question_generator.get_multiple_question_template") as mock_template:
            mock_template.side_effect = lambda claim, num, response: f"Multiple questions for: {claim}, count: {num}, response: {response}"

            result = QuestionGenerator._construct_question_generation_prompts(claim_sets=claim_sets, num_questions=2)

            # Verify None was passed as response
            mock_template.assert_called_once_with("Claim 1", 2, response=None)

    def test_extract_questions_from_generations_single_question(self):
        """Test extracting questions when num_questions=1."""
        question_generations = {"data": {"response": ["Question 1", "Question 2", "Question 3"]}}

        result = QuestionGenerator._extract_questions_from_generations(question_generations=question_generations, num_questions=1)

        assert result == ["Question 1", "Question 2", "Question 3"]

    def test_extract_questions_from_generations_multiple_questions(self):
        """Test extracting questions when num_questions>1."""
        question_generations = {"data": {"response": ["###Question 1###Question 2", "###Question 3###Question 4"]}}

        result = QuestionGenerator._extract_questions_from_generations(question_generations=question_generations, num_questions=2)

        assert result == ["Question 1", "Question 2", "Question 3", "Question 4"]

    def test_extract_questions_from_generations_empty_segments(self):
        """Test extracting questions with empty segments."""
        question_generations = {"data": {"response": ["###Question 1######Question 2"]}}

        result = QuestionGenerator._extract_questions_from_generations(question_generations=question_generations, num_questions=3)

        # Empty segments should be filtered out
        assert result == ["Question 1", "Question 2"]

    @pytest.mark.asyncio
    async def test_generate_questions_single_question(self, question_generator):
        """Test generating a single question per claim."""
        claim_sets = [["Claim 1", "Claim 2"], ["Claim 3"]]

        # Mock the response generator's return value
        question_generator.rg.generate_responses.return_value = {"data": {"response": ["Question for Claim 1", "Question for Claim 2", "Question for Claim 3"]}}

        # Mock the helper methods
        with patch.object(QuestionGenerator, "_construct_question_generation_prompts") as mock_construct:
            mock_construct.return_value = ["Prompt 1", "Prompt 2", "Prompt 3"]

            result = await question_generator.generate_questions(claim_sets=claim_sets, num_questions=1)

            # Verify the response generator was called with the right prompts
            question_generator.rg.generate_responses.assert_called_once_with(prompts=["Prompt 1", "Prompt 2", "Prompt 3"], progress_bar=None)

            # Verify the result
            assert result == ["Question for Claim 1", "Question for Claim 2", "Question for Claim 3"]

            # Verify num_questions was set
            assert question_generator.num_questions == 1

    @pytest.mark.asyncio
    async def test_generate_questions_multiple_questions(self, question_generator):
        """Test generating multiple questions per claim."""
        claim_sets = [["Claim 1"], ["Claim 2"]]

        # Mock the response generator's return value
        question_generator.rg.generate_responses.return_value = {"data": {"response": ["###Question 1.1###Question 1.2", "###Question 2.1###Question 2.2"]}}

        # Mock the helper methods
        with patch.object(QuestionGenerator, "_construct_question_generation_prompts") as mock_construct:
            mock_construct.return_value = ["Prompt 1", "Prompt 2"]

            result = await question_generator.generate_questions(claim_sets=claim_sets, num_questions=2)

            # Verify the result
            assert result == ["Question 1.1", "Question 1.2", "Question 2.1", "Question 2.2"]

            # Verify num_questions was set
            assert question_generator.num_questions == 2

    @pytest.mark.asyncio
    async def test_generate_questions_with_responses(self, question_generator):
        """Test generating questions with original responses provided."""
        claim_sets = [["Claim 1"]]
        responses = ["Original response"]

        # Mock the response generator's return value
        question_generator.rg.generate_responses.return_value = {"data": {"response": ["Question for Claim 1"]}}

        # Mock the helper methods
        with patch.object(QuestionGenerator, "_construct_question_generation_prompts") as mock_construct:
            mock_construct.return_value = ["Prompt 1"]

            await question_generator.generate_questions(claim_sets=claim_sets, responses=responses, num_questions=1)

            # Verify responses were passed to the construct method
            mock_construct.assert_called_once_with(claim_sets=claim_sets, responses=responses, num_questions=1)

    @pytest.mark.asyncio
    async def test_generate_questions_with_progress_bar(self, question_generator):
        """Test generating questions with a progress bar."""
        claim_sets = [["Claim 1"]]
        progress_bar = MagicMock(spec=Progress)

        # Mock the response generator's return value
        question_generator.rg.generate_responses.return_value = {"data": {"response": ["Question for Claim 1"]}}

        # Mock the helper methods
        with patch.object(QuestionGenerator, "_construct_question_generation_prompts") as mock_construct:
            mock_construct.return_value = ["Prompt 1"]

            await question_generator.generate_questions(claim_sets=claim_sets, progress_bar=progress_bar)

            # Verify progress bar was passed to the response generator
            question_generator.rg.generate_responses.assert_called_once_with(prompts=["Prompt 1"], progress_bar=progress_bar)

    @pytest.mark.asyncio
    async def test_generate_questions_empty_claim_sets(self, question_generator):
        """Test generating questions with empty claim sets."""
        claim_sets = []

        # Mock the response generator's return value for empty prompts
        question_generator.rg.generate_responses.return_value = {"data": {"response": []}}

        # Mock the helper methods
        with patch.object(QuestionGenerator, "_construct_question_generation_prompts") as mock_construct:
            mock_construct.return_value = []

            result = await question_generator.generate_questions(claim_sets=claim_sets)

            # Verify the response generator was called with empty prompts
            question_generator.rg.generate_responses.assert_called_once_with(prompts=[], progress_bar=None)

            # Verify construct was called with empty claim sets
            mock_construct.assert_called_once_with(claim_sets=[], responses=None, num_questions=1)

            # Verify result is an empty list
            assert result == []

    @pytest.mark.asyncio
    async def test_generate_questions_empty_claims(self, question_generator):
        """Test generating questions with empty claims in claim sets."""
        claim_sets = [[], []]

        # Mock the response generator's return value for empty prompts
        question_generator.rg.generate_responses.return_value = {"data": {"response": []}}

        # Mock the helper methods
        with patch.object(QuestionGenerator, "_construct_question_generation_prompts") as mock_construct:
            mock_construct.return_value = []

            result = await question_generator.generate_questions(claim_sets=claim_sets)

            # Verify construct was called with empty claims
            mock_construct.assert_called_once_with(claim_sets=[[], []], responses=None, num_questions=1)

            # Verify result is an empty list
            assert result == []

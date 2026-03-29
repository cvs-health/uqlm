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
from unittest.mock import AsyncMock, MagicMock, patch
from rich.progress import Progress

from uqlm.longform.benchmark.factscore_grader import FactScoreGrader
from uqlm.utils.prompts.factscore_prompts import FACTSCORE_SYSTEM_PROMPT, SUBJECTIVE_SYSTEM_PROMPT


class TestFactScoreGrader:
    @pytest.fixture
    def mock_llm(self):
        """Create a mock LLM for testing."""
        return MagicMock()

    @pytest.fixture
    def mock_response_generator(self):
        """Create a mock ResponseGenerator with AsyncMock for generate_responses."""
        mock = MagicMock()
        mock.generate_responses = AsyncMock()
        return mock

    @pytest.fixture
    def grader(self, mock_llm, mock_response_generator):
        """Create a FactScoreGrader with mocked ResponseGenerator."""
        with patch("uqlm.longform.benchmark.factscore_grader.ResponseGenerator", return_value=mock_response_generator):
            return FactScoreGrader(llm=mock_llm)

    def test_initialization(self, mock_llm):
        """Test initialization of FactScoreGrader."""
        with patch("uqlm.longform.benchmark.factscore_grader.ResponseGenerator") as mock_rg:
            mock_rg_instance = MagicMock()
            mock_rg.return_value = mock_rg_instance

            grader = FactScoreGrader(llm=mock_llm)

            # Check ResponseGenerator was initialized correctly
            mock_rg.assert_called_once_with(mock_llm, max_calls_per_min=None)

            # Check attributes were set correctly
            assert grader.grader_system_prompt == FACTSCORE_SYSTEM_PROMPT
            assert grader.subjective_system_prompt == SUBJECTIVE_SYSTEM_PROMPT
            assert grader.rg == mock_rg_instance
            assert grader.rg.response_generator_type == "factscore_grader"

    def test_construct_entailment_prompt(self, grader):
        """Test construction of entailment prompt."""
        claim = "The Earth is round."
        answer = "The Earth is an oblate spheroid."

        prompt = grader.construct_entailment_prompt(claim, answer)

        assert claim in prompt
        assert answer in prompt
        assert "Is the claim supported by the context above?" in prompt
        assert "Answer only Yes or No:" in prompt

    def test_construct_subjective_prompt(self, grader):
        """Test construction of subjective prompt."""
        claim = "The Earth is beautiful."

        prompt = grader.construct_subjective_prompt(claim)

        assert claim in prompt
        assert "Is the input subjective or objective?" in prompt
        assert "Answer only subjective or objective:" in prompt

    def test_str_to_bool_yes_no(self, grader):
        """Test _str_to_bool with yes/no strings."""
        assert grader._str_to_bool("Yes") is True
        assert grader._str_to_bool("yes") is True
        assert grader._str_to_bool("YES") is True
        assert grader._str_to_bool("The answer is yes.") is True

        assert grader._str_to_bool("No") is False
        assert grader._str_to_bool("no") is False
        assert grader._str_to_bool("NO") is False
        assert grader._str_to_bool("The answer is no.") is False

        assert grader._str_to_bool("Maybe") is False
        assert grader._str_to_bool("") is False

    def test_str_to_bool_objective_subjective(self, grader):
        """Test _str_to_bool with objective/subjective strings."""
        assert grader._str_to_bool("Objective", strings_to_check=["objective", "subjective"]) is True
        assert grader._str_to_bool("objective", strings_to_check=["objective", "subjective"]) is True
        assert grader._str_to_bool("The claim is objective", strings_to_check=["objective", "subjective"]) is True

        assert grader._str_to_bool("Subjective", strings_to_check=["objective", "subjective"]) is False
        assert grader._str_to_bool("subjective", strings_to_check=["objective", "subjective"]) is False
        assert grader._str_to_bool("The claim is subjective", strings_to_check=["objective", "subjective"]) is False

        assert grader._str_to_bool("Neither", strings_to_check=["objective", "subjective"]) is False

    def test_format_outputs(self, grader):
        """Test _format_outputs method."""
        flat_grades = ["Yes", "No", "Yes", "No"]
        reference_structure = [["Claim 1", "Claim 2"], ["Claim 3", "Claim 4"]]

        result = grader._format_outputs(flat_grades, reference_structure)

        assert len(result) == 2
        assert result[0] == [True, False]
        assert result[1] == [True, False]

    def test_format_outputs_custom_strings(self, grader):
        """Test _format_outputs with custom strings."""
        flat_grades = ["objective", "subjective"]
        reference_structure = [["Claim 1"], ["Claim 2"]]

        result = grader._format_outputs(flat_grades, reference_structure, strings_to_check=["objective", "subjective"])

        assert len(result) == 2
        assert result[0] == [True]
        assert result[1] == [False]

    @pytest.mark.asyncio
    async def test_grade_claims(self, grader, mock_response_generator):
        """Test grade_claims method."""
        # Setup test data
        claim_sets = [["Claim 1", "Claim 2"], ["Claim 3"]]
        answers = ["Answer 1", "Answer 2"]

        # Setup mock response
        mock_response_generator.generate_responses.return_value = {"data": {"response": ["Yes", "No", "Yes"]}}

        # Call the method
        result = await grader.grade_claims(claim_sets, answers)

        # Check the response generator was called correctly
        mock_response_generator.generate_responses.assert_called_once()
        call_args = mock_response_generator.generate_responses.call_args[1]
        assert len(call_args["prompts"]) == 3
        assert call_args["system_prompt"] == FACTSCORE_SYSTEM_PROMPT

        # Check the result
        assert len(result) == 2
        assert result[0] == [True, False]
        assert result[1] == [True]

    @pytest.mark.asyncio
    async def test_grade_claims_with_progress_bar(self, grader, mock_response_generator):
        """Test grade_claims with progress bar."""
        # Setup test data
        claim_sets = [["Claim 1"]]
        answers = ["Answer 1"]
        progress_bar = MagicMock(spec=Progress)

        # Setup mock response
        mock_response_generator.generate_responses.return_value = {"data": {"response": ["Yes"]}}

        # Call the method
        await grader.grade_claims(claim_sets, answers, progress_bar)

        # Check progress bar was passed
        assert mock_response_generator.generate_responses.call_args[1]["progress_bar"] == progress_bar

    @pytest.mark.asyncio
    async def test_evaluate_claim_objectivity(self, grader, mock_response_generator):
        """Test evaluate_claim_objectivity method."""
        # Setup test data
        claim_sets = [["Claim 1", "Claim 2"], ["Claim 3"]]

        # Setup mock response
        mock_response_generator.generate_responses.return_value = {"data": {"response": ["objective", "subjective", "objective"]}}

        # Call the method
        result = await grader.evaluate_claim_objectivity(claim_sets)

        # Check the response generator was called correctly
        mock_response_generator.generate_responses.assert_called_once()
        call_args = mock_response_generator.generate_responses.call_args[1]
        assert len(call_args["prompts"]) == 3
        assert call_args["system_prompt"] == SUBJECTIVE_SYSTEM_PROMPT

        # Check the result
        assert len(result) == 2
        assert result[0] == [True, False]  # objective, subjective
        assert result[1] == [True]  # objective

    @pytest.mark.asyncio
    async def test_evaluate_claim_objectivity_with_progress_bar(self, grader, mock_response_generator):
        """Test evaluate_claim_objectivity with progress bar."""
        # Setup test data
        claim_sets = [["Claim 1"]]
        progress_bar = MagicMock(spec=Progress)

        # Setup mock response
        mock_response_generator.generate_responses.return_value = {"data": {"response": ["objective"]}}

        # Call the method
        await grader.evaluate_claim_objectivity(claim_sets, progress_bar)

        # Check progress bar was passed
        assert mock_response_generator.generate_responses.call_args[1]["progress_bar"] == progress_bar

    @pytest.mark.asyncio
    async def test_grade_claims_empty_sets(self, grader, mock_response_generator):
        """Test grade_claims with empty claim sets."""
        # Setup test data
        claim_sets = [[], []]
        answers = ["Answer 1", "Answer 2"]

        # Setup mock response
        mock_response_generator.generate_responses.return_value = {"data": {"response": []}}

        # Call the method
        result = await grader.grade_claims(claim_sets, answers)

        # Check the result
        assert len(result) == 2
        assert result[0] == []
        assert result[1] == []

    @pytest.mark.asyncio
    async def test_evaluate_claim_objectivity_empty_sets(self, grader, mock_response_generator):
        """Test evaluate_claim_objectivity with empty claim sets."""
        # Setup test data
        claim_sets = [[], []]

        # Setup mock response
        mock_response_generator.generate_responses.return_value = {"data": {"response": []}}

        # Call the method
        result = await grader.evaluate_claim_objectivity(claim_sets)

        # Check the result
        assert len(result) == 2
        assert result[0] == []
        assert result[1] == []

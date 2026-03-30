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
from langchain_openai import AzureChatOpenAI
from unittest.mock import MagicMock, patch, AsyncMock
import numpy as np
from langchain_core.language_models.chat_models import BaseChatModel
from rich.progress import Progress
from uqlm.utils.results import UQResult
from uqlm.scorers.longform.baseclass.uncertainty import LongFormUQ
from uqlm.longform.qa.question_generator import QuestionGenerator
from uqlm.scorers import BlackBoxUQ

# Import the class to test
from uqlm.scorers.longform.qa import LongTextQA


class TestLongTextQA:
    @pytest.fixture
    def mock_llm(self):
        """Create a mock LLM."""
        mock = MagicMock(spec=BaseChatModel)
        mock.temperature = 0.7
        return mock

    @pytest.fixture
    def mock_bb_result(self):
        """Create a mock BlackBoxUQ result."""
        mock_result = MagicMock()
        mock_result.to_dict.return_value = {"data": {"responses": ["Response 1", "Response 2", "Response 3", "Response 4"], "sampled_responses": [["Sample 1.1", "Sample 1.2"], ["Sample 2.1", "Sample 2.2"], ["Sample 3.1", "Sample 3.2"], ["Sample 4.1", "Sample 4.2"]], "semantic_negentropy": [0.8, 0.7, 0.9, 0.6]}}
        return mock_result

    def test_initialization(self, mock_llm):
        """Test initialization with default parameters."""
        with patch("uqlm.scorers.longform.qa.QuestionGenerator") as mock_qg, patch("uqlm.scorers.longform.qa.BlackBoxUQ") as mock_bb:
            qa = LongTextQA(llm=mock_llm)

            # Check that QuestionGenerator was initialized correctly
            mock_qg.assert_called_once()

            # Check that BlackBoxUQ was initialized correctly
            mock_bb.assert_called_once_with(llm=mock_llm, scorers=["semantic_negentropy"], device=None, max_calls_per_min=None, sampling_temperature=1.0, max_length=1000)

            # Check that attributes were set correctly
            assert qa.scorers == ["semantic_negentropy"]
            assert qa.llm == mock_llm
            assert qa.granularity == "claim"
            assert qa.aggregation == "mean"
            assert qa.response_refinement is False
            assert qa.uad_result == {}

    def test_initialization_custom(self, mock_llm):
        """Test initialization with custom parameters."""
        custom_decomposition_llm = MagicMock(spec=BaseChatModel)
        custom_question_generator_llm = MagicMock(spec=BaseChatModel)

        with patch("uqlm.scorers.longform.qa.QuestionGenerator") as mock_qg, patch("uqlm.scorers.longform.qa.BlackBoxUQ") as mock_bb:
            qa = LongTextQA(
                llm=mock_llm,
                scorers=["semantic_negentropy", "entailment"],
                granularity="claim",
                aggregation="min",
                response_refinement=True,
                claim_filtering_scorer="entailment",
                system_prompt="Custom prompt",
                claim_decomposition_llm=custom_decomposition_llm,
                question_generator_llm=custom_question_generator_llm,
                sampling_temperature=0.5,
                max_calls_per_min=10,
                questioner_max_calls_per_min=5,
                max_length=500,
                device="cuda",
                use_n_param=True,
            )

            # Check that QuestionGenerator was initialized correctly
            mock_qg.assert_called_once_with(question_generator_llm=custom_question_generator_llm, max_calls_per_min=5)

            # Check that BlackBoxUQ was initialized correctly
            mock_bb.assert_called_once_with(llm=mock_llm, scorers=["semantic_negentropy", "entailment"], device="cuda", max_calls_per_min=10, sampling_temperature=0.5, max_length=500)

            # Check that attributes were set correctly
            assert qa.scorers == ["semantic_negentropy", "entailment"]
            assert qa.llm == mock_llm
            assert qa.granularity == "claim"
            assert qa.aggregation == "min"
            assert qa.response_refinement is True
            assert qa.claim_filtering_scorer == "entailment"
            assert qa.system_prompt == "Custom prompt"

    @pytest.mark.asyncio
    async def test_generate_and_score(self, mock_llm):
        """Test generate_and_score method."""
        prompts = ["What is AI?", "Explain quantum computing"]

        qa = LongTextQA(llm=mock_llm)
        qa._construct_progress_bar = MagicMock()
        qa._display_generation_header = MagicMock()
        qa.generate_original_responses = AsyncMock(return_value=["AI is...", "Quantum computing is..."])
        qa.score = AsyncMock(return_value=MagicMock(spec=UQResult))

        result = await qa.generate_and_score(prompts=prompts, num_questions=2, num_claim_qa_responses=3, response_refinement_threshold=0.4, show_progress_bars=True)

        # Check that methods were called with the right arguments
        qa._construct_progress_bar.assert_called_once_with(True)
        qa._display_generation_header.assert_called_once_with(True)
        qa.generate_original_responses.assert_called_once_with(prompts=prompts, progress_bar=qa.progress_bar)
        qa.score.assert_called_once_with(prompts=prompts, responses=["AI is...", "Quantum computing is..."], num_questions=2, num_claim_qa_responses=3, response_refinement_threshold=0.4, show_progress_bars=True)

        assert result is not None

    @pytest.mark.asyncio
    async def test_score(self, mock_llm):
        """Test score method."""
        prompts = ["What is AI?"]
        responses = ["AI is a field of computer science."]

        qa = LongTextQA(llm=mock_llm)
        qa._construct_progress_bar = MagicMock()
        qa._decompose_responses = AsyncMock()
        # Mock the _decompose_responses to set claim_sets
        qa._decompose_responses.side_effect = lambda x: setattr(qa, "claim_sets", [["AI is a field."]])
        qa._score_from_decomposed = AsyncMock(return_value=MagicMock(spec=UQResult))

        result = await qa.score(prompts=prompts, responses=responses, num_questions=2, num_claim_qa_responses=3, response_refinement_threshold=0.4, show_progress_bars=True)

        # Check that attributes were set correctly
        assert qa.prompts == prompts
        assert qa.responses == responses

        # Check that methods were called with the right arguments
        qa._construct_progress_bar.assert_called_once_with(True)
        qa._decompose_responses.assert_called_once_with(True)
        qa._score_from_decomposed.assert_called_once()

        assert result is not None

    @pytest.mark.asyncio
    async def test_score_from_decomposed(self, mock_llm, mock_bb_result):
        """Test _score_from_decomposed method."""
        claim_sets = [["AI is a field of computer science.", "AI involves machine learning."], ["Quantum computing uses qubits."]]
        prompts = ["What is AI?", "Explain quantum computing"]

        qa = LongTextQA(llm=mock_llm)
        qa.question_generator = MagicMock(spec=QuestionGenerator)
        qa.question_generator.generate_questions = AsyncMock(return_value=["Question 1", "Question 2", "Question 3", "Question 4"])

        qa.bb_object = MagicMock(spec=BlackBoxUQ)
        qa.bb_object.generate_and_score = AsyncMock(return_value=mock_bb_result)
        qa.bb_object.scorers = ["semantic_negentropy"]

        # Set up claim_scores before testing response_refinement
        qa.claim_scores = {"semantic_negentropy": [[0.8, 0.7], [0.9]]}
        qa.uad_scorer = "semantic_negentropy"  # Add this line

        qa._process_bb_result = MagicMock(return_value={"semantic_negentropy": 0.75})
        qa._extract_claim_data = MagicMock(return_value=[[{"claim": "AI is a field of computer science.", "semantic_negentropy": 0.8}, {"claim": "AI involves machine learning.", "semantic_negentropy": 0.7}], [{"claim": "Quantum computing uses qubits.", "semantic_negentropy": 0.9}]])
        qa._stop_progress_bar = MagicMock()
        qa._construct_result = MagicMock(return_value=MagicMock(spec=UQResult))

        # For response refinement test
        qa.response_refinement = False
        qa.uad_result = {}

        result = await qa._score_from_decomposed(claim_sets=claim_sets, prompts=prompts, num_questions=2, num_claim_qa_responses=3, response_refinement_threshold=0.4)

        # Check that attributes were set correctly
        assert qa.num_questions == 2
        assert qa.num_claim_qa_responses == 3
        assert qa.claim_sets == claim_sets
        assert qa.prompts == prompts

        # Check that methods were called with the right arguments
        qa.question_generator.generate_questions.assert_called_once_with(claim_sets=claim_sets, num_questions=2, progress_bar=None)

        qa.bb_object.generate_and_score.assert_called_once()
        qa._process_bb_result.assert_called_once()
        qa._extract_claim_data.assert_called_once()
        qa._stop_progress_bar.assert_called_once()
        qa._construct_result.assert_called_once()

        assert result is not None

        # Test with response_refinement=True
        qa.response_refinement = True
        qa.uncertainty_aware_decode = AsyncMock(return_value={"refined_responses": ["Refined response"]})

        await qa._score_from_decomposed(claim_sets=claim_sets, prompts=prompts, num_questions=2, num_claim_qa_responses=3, response_refinement_threshold=0.4)

        qa.uncertainty_aware_decode.assert_called_once()

    def test_process_bb_result(self, mock_llm, mock_bb_result):
        """Test _process_bb_result method."""
        formatted_claim_questions = ["Question 1", "Question 2", "Question 3", "Question 4"]
        num_claims = [2, 1]  # 2 claims for first response, 1 for second

        qa = LongTextQA(llm=mock_llm)
        qa.claim_sets = [["Claim 1.1", "Claim 1.2"], ["Claim 2.1"]]
        qa.bb_object = MagicMock()
        qa.bb_object.scorers = ["semantic_negentropy"]
        qa._aggregate_scores = MagicMock(return_value=0.75)
        qa.num_questions = 1

        result = qa._process_bb_result(bb_result=mock_bb_result, formatted_claim_questions=formatted_claim_questions, num_claims=num_claims)

        # Check that claim_scores were set correctly
        assert "semantic_negentropy" in qa.claim_scores
        assert len(qa.claim_scores["semantic_negentropy"]) == 2
        assert len(qa.claim_scores["semantic_negentropy"][0]) == 2  # 2 claims for first response
        assert len(qa.claim_scores["semantic_negentropy"][1]) == 1  # 1 claim for second response

        # Check that response_fact_questions were set correctly
        assert len(qa.response_fact_questions) == 2
        assert len(qa.response_fact_questions[0]) == 2
        assert len(qa.response_fact_questions[1]) == 1

        # Check that _aggregate_scores was called
        qa._aggregate_scores.assert_called_once()

        # Check the result
        assert result == {"semantic_negentropy": 0.75}

        # Test with num_questions > 1
        qa.num_questions = 2
        qa._process_bb_result(bb_result=mock_bb_result, formatted_claim_questions=formatted_claim_questions, num_claims=num_claims)

        # Should call np.mean for each claim's questions
        assert qa._aggregate_scores.call_count == 2

    def test_extract_claim_data(self, mock_llm):
        """Test _extract_claim_data method."""
        qa = LongTextQA(llm=mock_llm)
        qa.granularity = "claim"
        qa.claim_sets = [["Claim 1.1", "Claim 1.2"], ["Claim 2.1"]]
        qa.response_fact_questions = [[["Q1.1"], ["Q1.2"]], [["Q2.1"]]]
        qa.response_fact_questions_responses = [[["R1.1"], ["R1.2"]], [["R2.1"]]]
        qa.response_fact_questions_sampled_responses = [[["SR1.1"], ["SR1.2"]], [["SR2.1"]]]
        qa.claim_scores = {"semantic_negentropy": [[0.8, 0.7], [0.9]]}
        qa.bb_object = MagicMock()
        qa.bb_object.scorers = ["semantic_negentropy"]

        # Test without response refinement
        qa.uad_result = {}
        qa.response_refinement = False

        result = qa._extract_claim_data()

        assert len(result) == 2
        assert len(result[0]) == 2
        assert len(result[1]) == 1

        assert result[0][0]["claim"] == "Claim 1.1"
        assert result[0][0]["removed"] is False
        assert result[0][0]["claim_questions"] == ["Q1.1"]
        assert result[0][0]["claim_qa_responses"] == ["R1.1"]
        assert result[0][0]["claim_qa_sampled_responses"] == ["SR1.1"]
        assert result[0][0]["semantic_negentropy"] == 0.8

        # Test with response refinement
        qa.uad_result = {"removed": [[True, False], [False]]}
        qa.response_refinement = True

        result = qa._extract_claim_data()

        assert result[0][0]["removed"] is True
        assert result[0][1]["removed"] is False
        assert result[1][0]["removed"] is False

    def test_construct_result(self, mock_llm):
        """Test _construct_result method."""
        qa = LongTextQA(llm=mock_llm)
        qa.prompts = ["What is AI?", "Explain quantum computing"]
        qa.responses = ["AI is...", "Quantum computing is..."]
        qa.scores_dict = {"semantic_negentropy": 0.75}
        qa.uad_result = {"refined_responses": ["Refined AI...", "Refined quantum..."]}
        qa.granularity = "claim"
        qa.aggregation = "mean"
        qa.bb_object = MagicMock()
        qa.bb_object.sampling_temperature = 1.0
        qa.num_claim_qa_responses = 3
        qa.response_refinement_threshold = 0.4

        # Just verify we get a result without errors
        result = qa._construct_result()
        assert isinstance(result, UQResult)

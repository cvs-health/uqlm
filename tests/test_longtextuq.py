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
import numpy as np
from unittest.mock import MagicMock, patch, AsyncMock
from rich.progress import Progress

from uqlm.scorers.longform.baseclass.uncertainty import LongFormUQ
from uqlm.utils.results import UQResult
from uqlm.longform.luq.matched_unit import MatchedUnitScorer
from uqlm.longform.luq.unit_response import UnitResponseScorer
from uqlm.scorers.longform.longtext import LongTextUQ


class TestLongTextUQ:
    @pytest.fixture
    def mock_llm(self):
        """Create a mock LLM."""
        mock = MagicMock()
        mock.temperature = 0.7
        return mock

    @pytest.fixture
    def mock_unit_response_scorer(self):
        """Create a mock UnitResponseScorer."""
        mock = MagicMock(spec=UnitResponseScorer)
        mock.evaluate.return_value.to_dict.return_value = {"entailment": [[0.8, 0.7], [0.9]], "noncontradiction": [[0.9, 0.8], [0.95]], "contrasted_entailment": [[0.85, 0.75], [0.92]]}
        return mock

    @pytest.fixture
    def mock_matched_unit_scorer(self):
        """Create a mock MatchedUnitScorer."""
        mock = MagicMock(spec=MatchedUnitScorer)
        mock.evaluate.return_value.to_dict.return_value = {"entailment": [[0.8, 0.7], [0.9]], "noncontradiction": [[0.9, 0.8], [0.95]], "contrasted_entailment": [[0.85, 0.75], [0.92]], "bert_score": [[0.75, 0.65], [0.85]], "cosine_sim": [[0.7, 0.6], [0.8]]}
        return mock

    @pytest.fixture
    def uq_unit_response(self, mock_llm, mock_unit_response_scorer):
        """Create a LongTextUQ instance with unit_response mode."""
        with patch("uqlm.scorers.longform.longtext.UnitResponseScorer", return_value=mock_unit_response_scorer):
            return LongTextUQ(llm=mock_llm, device="cpu", mode="unit_response", scorers=["entailment", "noncontradiction", "contrasted_entailment"])

    @pytest.fixture
    def uq_matched_unit(self, mock_llm, mock_matched_unit_scorer):
        """Create a LongTextUQ instance with matched_unit mode."""
        with patch("uqlm.scorers.longform.longtext.MatchedUnitScorer", return_value=mock_matched_unit_scorer):
            return LongTextUQ(llm=mock_llm, device="cpu", mode="matched_unit", scorers=["entailment", "noncontradiction", "contrasted_entailment", "bert_score", "cosine_sim"])

    def test_initialization_default(self, mock_llm):
        """Test initialization with default parameters."""
        with patch("uqlm.scorers.longform.longtext.UnitResponseScorer") as mock_unit_response_class:
            uq = LongTextUQ(llm=mock_llm, device="cpu")

            # Check that UnitResponseScorer was initialized with the correct parameters
            mock_unit_response_class.assert_called_once_with(nli_model_name="microsoft/deberta-large-mnli", device="cpu", max_length=2000, nli_llm=None)

            # Check that attributes were set correctly
            assert uq.llm == mock_llm
            assert uq.granularity == "claim"
            assert uq.mode == "unit_response"
            assert uq.scorers == ["entailment"]
            assert uq.aggregation == "mean"
            assert uq.response_refinement is False
            assert uq.nli_model_name == "microsoft/deberta-large-mnli"
            assert uq.max_length == 2000
            assert uq.sampling_temperature == 1.0
            assert uq.unit_response_scorer is not None
            assert uq.matched_unit_scorer is None

    def test_initialization_unit_response_custom(self, mock_llm):
        """Test initialization with custom parameters for unit_response mode."""
        with patch("uqlm.scorers.longform.longtext.UnitResponseScorer") as mock_unit_response_class:
            uq = LongTextUQ(llm=mock_llm, granularity="sentence", mode="unit_response", scorers=["entailment", "noncontradiction"], aggregation="min", response_refinement=False, nli_model_name="custom/model", device="cpu", sampling_temperature=0.5, max_length=1000)

            # Check that UnitResponseScorer was initialized with the correct parameters
            mock_unit_response_class.assert_called_once_with(nli_model_name="custom/model", device="cpu", max_length=1000, nli_llm=None)

            # Check that attributes were set correctly
            assert uq.llm == mock_llm
            assert uq.granularity == "sentence"
            assert uq.mode == "unit_response"
            assert uq.scorers == ["entailment", "noncontradiction"]
            assert uq.aggregation == "min"
            assert uq.response_refinement is False
            assert uq.nli_model_name == "custom/model"
            assert uq.max_length == 1000
            assert uq.sampling_temperature == 0.5
            assert uq.unit_response_scorer is not None
            assert uq.matched_unit_scorer is None

    def test_initialization_matched_unit_custom(self, mock_llm):
        """Test initialization with custom parameters for matched_unit mode."""
        with patch("uqlm.scorers.longform.longtext.MatchedUnitScorer") as mock_matched_unit_class:
            uq = LongTextUQ(llm=mock_llm, granularity="sentence", mode="matched_unit", scorers=["entailment", "bert_score", "cosine_sim"], aggregation="min", response_refinement=False, device="cpu", nli_model_name="custom/model", sampling_temperature=0.5, max_length=1000)

            # Check that MatchedUnitScorer was initialized with the correct parameters
            mock_matched_unit_class.assert_called_once_with(nli_model_name="custom/model", device="cpu", max_length=1000)

            # Check that attributes were set correctly
            assert uq.llm == mock_llm
            assert uq.granularity == "sentence"
            assert uq.mode == "matched_unit"
            assert uq.scorers == ["entailment", "bert_score", "cosine_sim"]
            assert uq.aggregation == "min"
            assert uq.response_refinement is False
            assert uq.nli_model_name == "custom/model"
            assert uq.max_length == 1000
            assert uq.sampling_temperature == 0.5
            assert uq.unit_response_scorer is None
            assert uq.matched_unit_scorer is not None

    def test_initialization_invalid_mode(self, mock_llm):
        """Test initialization with invalid mode."""
        with pytest.raises(ValueError) as excinfo:
            LongTextUQ(llm=mock_llm, mode="invalid_mode", device="cpu")

        assert "Invalid mode" in str(excinfo.value)

    def test_initialization_invalid_scorers_unit_response(self, mock_llm):
        """Test initialization with invalid scorers for unit_response mode."""
        with pytest.raises(ValueError) as excinfo:
            LongTextUQ(llm=mock_llm, mode="unit_response", scorers=["entailment", "bert_score"], device="cpu")

        assert "Invalid scorers" in str(excinfo.value)

    def test_initialization_invalid_scorers_matched_unit(self, mock_llm):
        """Test initialization with invalid scorers for matched_unit mode."""
        with pytest.raises(ValueError) as excinfo:
            LongTextUQ(llm=mock_llm, mode="matched_unit", scorers=["entailment", "invalid_scorer"], device="cpu")

        assert "Invalid scorers" in str(excinfo.value)

    @pytest.mark.asyncio
    async def test_generate_and_score_unit_response(self, uq_unit_response):
        """Test generate_and_score method with unit_response mode."""
        prompts = ["Prompt 1", "Prompt 2"]
        responses = ["Response 1", "Response 2"]
        sampled_responses = [["Sample 1.1", "Sample 1.2"], ["Sample 2.1", "Sample 2.2"]]
        claim_sets = [["Claim 1.1", "Claim 1.2"], ["Claim 2.1"]]

        # Patch the necessary methods
        with (
            patch.object(LongFormUQ, "generate_original_responses", new_callable=AsyncMock) as mock_gen_orig,
            patch.object(LongFormUQ, "generate_candidate_responses", new_callable=AsyncMock) as mock_gen_cand,
            patch.object(LongTextUQ, "score", new_callable=AsyncMock) as mock_score,
            patch.object(LongTextUQ, "_construct_progress_bar") as mock_construct_progress,
            patch.object(LongTextUQ, "_display_generation_header") as mock_display_header,
        ):
            # Set up return values
            mock_gen_orig.return_value = responses
            mock_gen_cand.return_value = sampled_responses
            mock_score.return_value = UQResult({"data": {"responses": responses, "scores": {"entailment": [0.8, 0.9]}}})

            result = await uq_unit_response.generate_and_score(prompts=prompts, num_responses=2)

            # Check that methods were called with the right arguments
            mock_construct_progress.assert_called_once_with(True)
            mock_display_header.assert_called_once_with(True)
            mock_gen_orig.assert_called_once_with(prompts=prompts, progress_bar=uq_unit_response.progress_bar)
            mock_gen_cand.assert_called_once_with(prompts=prompts, progress_bar=uq_unit_response.progress_bar, num_responses=2)
            mock_score.assert_called_once_with(responses=responses, sampled_responses=sampled_responses, response_refinement_threshold=1 / 3, show_progress_bars=True)

            # Check that attributes were set correctly
            assert uq_unit_response.prompts == prompts
            assert uq_unit_response.num_responses == 2

            # Check the result
            assert isinstance(result, UQResult)
            assert result.data["responses"] == responses
            assert result.data["scores"]["entailment"] == [0.8, 0.9]

    @pytest.mark.asyncio
    async def test_score_unit_response(self, uq_unit_response):
        """Test score method with unit_response mode."""
        responses = ["Response 1", "Response 2"]
        sampled_responses = [["Sample 1.1", "Sample 1.2"], ["Sample 2.1", "Sample 2.2"]]
        claim_sets = [["Claim 1.1", "Claim 1.2"], ["Claim 2.1"]]

        # Patch the necessary methods
        with (
            patch.object(LongTextUQ, "_decompose_responses", new_callable=AsyncMock) as mock_decompose,
            patch.object(LongTextUQ, "_score_from_decomposed") as mock_score_from_decomposed,
            patch.object(LongTextUQ, "_construct_progress_bar") as mock_construct_progress,
            patch.object(LongTextUQ, "_display_scoring_header") as mock_display_header,
            patch.object(LongTextUQ, "_stop_progress_bar") as mock_stop_progress,
            patch.object(LongTextUQ, "_construct_result") as mock_construct_result,
        ):
            # Set up attributes and return values
            uq_unit_response.claim_sets = claim_sets
            uq_unit_response.uad_result = {}  # Empty dict for no refinement

            # Set up claim_scores - this is what was missing
            uq_unit_response.claim_scores = {"entailment": [[0.8, 0.7], [0.9]], "noncontradiction": [[0.9, 0.8], [0.95]], "contrasted_entailment": [[0.85, 0.75], [0.92]]}

            mock_score_from_decomposed.return_value = {"entailment": [0.8, 0.9], "noncontradiction": [0.9, 0.95], "contrasted_entailment": [0.85, 0.92]}
            mock_construct_result.return_value = UQResult({"data": {"responses": responses, "scores": {"entailment": [0.8, 0.9]}}})

            result = await uq_unit_response.score(responses=responses, sampled_responses=sampled_responses)

            # Check that methods were called with the right arguments
            mock_construct_progress.assert_called_once_with(True)
            mock_decompose.assert_called_once_with(True)
            mock_display_header.assert_called_once_with(True)
            mock_score_from_decomposed.assert_called_once_with(claim_sets=claim_sets, sampled_responses=sampled_responses, sampled_claim_sets=None, progress_bar=uq_unit_response.progress_bar)
            mock_stop_progress.assert_called_once()
            mock_construct_result.assert_called_once()

            # Check that attributes were set correctly
            assert uq_unit_response.responses == responses
            assert uq_unit_response.sampled_responses == sampled_responses
            assert uq_unit_response.num_responses == 2
            assert uq_unit_response.scores_dict == mock_score_from_decomposed.return_value

            # Check the result
            assert isinstance(result, UQResult)
            assert result.data["responses"] == responses
            assert result.data["scores"]["entailment"] == [0.8, 0.9]

    @pytest.mark.asyncio
    async def test_score_matched_unit(self, uq_matched_unit):
        """Test score method with matched_unit mode."""
        responses = ["Response 1", "Response 2"]
        sampled_responses = [["Sample 1.1", "Sample 1.2"], ["Sample 2.1", "Sample 2.2"]]
        claim_sets = [["Claim 1.1", "Claim 1.2"], ["Claim 2.1"]]
        sampled_claim_sets = [[["Claim 1.1.1", "Claim 1.1.2"], ["Claim 1.2.1"]], [["Claim 2.1.1"], ["Claim 2.2.1"]]]

        # Patch the necessary methods
        with (
            patch.object(LongTextUQ, "_decompose_responses", new_callable=AsyncMock) as mock_decompose_resp,
            patch.object(LongTextUQ, "_decompose_candidate_responses", new_callable=AsyncMock) as mock_decompose_cand,
            patch.object(LongTextUQ, "_score_from_decomposed") as mock_score_from_decomposed,
            patch.object(LongTextUQ, "_construct_progress_bar") as mock_construct_progress,
            patch.object(LongTextUQ, "_display_scoring_header") as mock_display_header,
            patch.object(LongTextUQ, "_stop_progress_bar") as mock_stop_progress,
            patch.object(LongTextUQ, "_construct_result") as mock_construct_result,
        ):
            # Set up attributes and return values
            uq_matched_unit.claim_sets = claim_sets
            uq_matched_unit.sampled_claim_sets = sampled_claim_sets
            uq_matched_unit.uad_result = {}  # Empty dict for no refinement

            # Set up claim_scores - this is what was missing
            uq_matched_unit.claim_scores = {"entailment": [[0.8, 0.7], [0.9]], "noncontradiction": [[0.9, 0.8], [0.95]], "contrasted_entailment": [[0.85, 0.75], [0.92]], "bert_score": [[0.75, 0.65], [0.85]], "cosine_sim": [[0.7, 0.6], [0.8]]}

            mock_score_from_decomposed.return_value = {"entailment": [0.8, 0.9], "noncontradiction": [0.9, 0.95], "contrasted_entailment": [0.85, 0.92], "bert_score": [0.75, 0.85], "cosine_sim": [0.7, 0.8]}
            mock_construct_result.return_value = UQResult({"data": {"responses": responses, "scores": {"entailment": [0.8, 0.9]}}})

            result = await uq_matched_unit.score(responses=responses, sampled_responses=sampled_responses)

            # Check that methods were called with the right arguments
            mock_construct_progress.assert_called_once_with(True)
            mock_decompose_resp.assert_called_once_with(True)
            mock_decompose_cand.assert_called_once_with(True)
            mock_display_header.assert_called_once_with(True)
            mock_score_from_decomposed.assert_called_once_with(claim_sets=claim_sets, sampled_responses=sampled_responses, sampled_claim_sets=sampled_claim_sets, progress_bar=uq_matched_unit.progress_bar)
            mock_stop_progress.assert_called_once()
            mock_construct_result.assert_called_once()

            # Check that attributes were set correctly
            assert uq_matched_unit.responses == responses
            assert uq_matched_unit.sampled_responses == sampled_responses
            assert uq_matched_unit.num_responses == 2
            assert uq_matched_unit.scores_dict == mock_score_from_decomposed.return_value

            # Check the result
            assert isinstance(result, UQResult)
            assert result.data["responses"] == responses
            assert result.data["scores"]["entailment"] == [0.8, 0.9]

    @pytest.mark.asyncio
    async def test_score_with_response_refinement(self, uq_unit_response):
        """Test score method with response_refinement enabled."""
        responses = ["Response 1", "Response 2"]
        sampled_responses = [["Sample 1.1", "Sample 1.2"], ["Sample 2.1", "Sample 2.2"]]
        claim_sets = [["Claim 1.1", "Claim 1.2"], ["Claim 2.1"]]

        # Enable response refinement
        uq_unit_response.response_refinement = True
        uq_unit_response.uad_scorer = "entailment"

        # Patch the necessary methods
        with (
            patch.object(LongTextUQ, "_decompose_responses", new_callable=AsyncMock) as mock_decompose,
            patch.object(LongTextUQ, "_score_from_decomposed") as mock_score_from_decomposed,
            patch.object(LongTextUQ, "uncertainty_aware_decode", new_callable=AsyncMock) as mock_uad,
            patch.object(LongTextUQ, "_construct_progress_bar") as mock_construct_progress,
            patch.object(LongTextUQ, "_display_scoring_header") as mock_display_header,
            patch.object(LongTextUQ, "_stop_progress_bar") as mock_stop_progress,
            patch.object(LongTextUQ, "_construct_result") as mock_construct_result,
        ):
            # Set up attributes and return values
            uq_unit_response.claim_sets = claim_sets
            uq_unit_response.claim_scores = {"entailment": [[0.8, 0.7], [0.9]], "noncontradiction": [[0.9, 0.8], [0.95]], "contrasted_entailment": [[0.85, 0.75], [0.92]]}
            mock_score_from_decomposed.return_value = {"entailment": [0.8, 0.9], "noncontradiction": [0.9, 0.95], "contrasted_entailment": [0.85, 0.92]}
            mock_uad.return_value = {"refined_responses": ["Refined 1", "Refined 2"], "removed": [[False, True], [False]]}
            mock_construct_result.return_value = UQResult({"data": {"responses": responses, "refined_responses": ["Refined 1", "Refined 2"]}})

            result = await uq_unit_response.score(responses=responses, sampled_responses=sampled_responses, response_refinement_threshold=0.5)

            # Check that uncertainty_aware_decode was called with the right arguments
            mock_uad.assert_called_once_with(claim_sets=claim_sets, claim_scores=uq_unit_response.claim_scores["entailment"], response_refinement_threshold=0.5, show_progress_bars=True)

            # Check that the UAD result was processed correctly
            assert uq_unit_response.uad_result == {"refined_responses": ["Refined 1", "Refined 2"]}
            assert "removed" not in uq_unit_response.uad_result

            # Check that claims_data was constructed correctly
            assert "claims_data" in uq_unit_response.scores_dict
            assert len(uq_unit_response.scores_dict["claims_data"]) == 2
            assert len(uq_unit_response.scores_dict["claims_data"][0]) == 2
            assert len(uq_unit_response.scores_dict["claims_data"][1]) == 1

            # Check the first claim's data
            assert uq_unit_response.scores_dict["claims_data"][0][0]["claim"] == "Claim 1.1"
            assert uq_unit_response.scores_dict["claims_data"][0][0]["removed"] is False
            assert uq_unit_response.scores_dict["claims_data"][0][0]["entailment"] == 0.8
            assert uq_unit_response.scores_dict["claims_data"][0][0]["noncontradiction"] == 0.9
            assert uq_unit_response.scores_dict["claims_data"][0][0]["contrasted_entailment"] == 0.85

            # Check the second claim's data
            assert uq_unit_response.scores_dict["claims_data"][0][1]["claim"] == "Claim 1.2"
            assert uq_unit_response.scores_dict["claims_data"][0][1]["removed"] is True

            # Check the result
            assert isinstance(result, UQResult)
            assert result.data["responses"] == responses
            assert result.data["refined_responses"] == ["Refined 1", "Refined 2"]

    @pytest.mark.asyncio
    async def test_score_from_decomposed_unit_response(self, uq_unit_response):
        """Test _score_from_decomposed method with unit_response mode."""
        claim_sets = [["Claim 1.1", "Claim 1.2"], ["Claim 2.1"]]
        sampled_responses = [["Sample 1.1", "Sample 1.2"], ["Sample 2.1", "Sample 2.2"]]
        progress_bar = MagicMock(spec=Progress)

        # Create a mock for the UnitResponseScorer's evaluate method
        mock_evaluate = MagicMock()
        mock_evaluate.return_value.to_dict.return_value = {"entailment": [[0.8, 0.7], [0.9]], "noncontradiction": [[0.9, 0.8], [0.95]], "contrasted_entailment": [[0.85, 0.75], [0.92]]}

        # Patch the UnitResponseScorer's evaluate method
        with patch.object(UnitResponseScorer, "evaluate", mock_evaluate), patch.object(LongFormUQ, "_aggregate_scores") as mock_aggregate:
            # Set up return values for each scorer
            mock_aggregate.side_effect = [0.8, 0.9, 0.85]  # For entailment, noncontradiction, contrasted_entailment

            # Call the method
            result = await uq_unit_response._score_from_decomposed(claim_sets=claim_sets, sampled_responses=sampled_responses, progress_bar=progress_bar)

            # Check that _aggregate_scores was called for each scorer
            assert mock_aggregate.call_count == 3
            mock_aggregate.assert_any_call([[0.8, 0.7], [0.9]])  # entailment
            mock_aggregate.assert_any_call([[0.9, 0.8], [0.95]])  # noncontradiction
            mock_aggregate.assert_any_call([[0.85, 0.75], [0.92]])  # contrasted_entailment

            # Check the result
            assert result == {"entailment": 0.8, "noncontradiction": 0.9, "contrasted_entailment": 0.85}

    @pytest.mark.asyncio
    async def test_score_from_decomposed_matched_unit(self, uq_matched_unit):
        """Test _score_from_decomposed method with matched_unit mode."""
        claim_sets = [["Claim 1.1", "Claim 1.2"], ["Claim 2.1"]]
        sampled_claim_sets = [[["Claim 1.1.1", "Claim 1.1.2"], ["Claim 1.2.1"]], [["Claim 2.1.1"], ["Claim 2.2.1"]]]
        progress_bar = MagicMock(spec=Progress)

        # Create a mock for the MatchedUnitScorer's evaluate method
        mock_evaluate = MagicMock()
        mock_evaluate.return_value.to_dict.return_value = {"entailment": [[0.8, 0.7], [0.9]], "noncontradiction": [[0.9, 0.8], [0.95]], "contrasted_entailment": [[0.85, 0.75], [0.92]], "bert_score": [[0.75, 0.65], [0.85]], "cosine_sim": [[0.7, 0.6], [0.8]]}

        # Patch the MatchedUnitScorer's evaluate method
        with patch.object(MatchedUnitScorer, "evaluate", mock_evaluate), patch.object(LongFormUQ, "_aggregate_scores") as mock_aggregate:
            # Set up return values for each scorer
            mock_aggregate.side_effect = [0.8, 0.9, 0.85, 0.75, 0.7]  # For all five scorers

            # Call the method
            result = await uq_matched_unit._score_from_decomposed(claim_sets=claim_sets, sampled_claim_sets=sampled_claim_sets, progress_bar=progress_bar)

            # Check that _aggregate_scores was called for each scorer
            assert mock_aggregate.call_count == 5
            mock_aggregate.assert_any_call([[0.8, 0.7], [0.9]])  # entailment
            mock_aggregate.assert_any_call([[0.9, 0.8], [0.95]])  # noncontradiction
            mock_aggregate.assert_any_call([[0.85, 0.75], [0.92]])  # contrasted_entailment
            mock_aggregate.assert_any_call([[0.75, 0.65], [0.85]])  # bert_score
            mock_aggregate.assert_any_call([[0.7, 0.6], [0.8]])  # cosine_sim

            # Check the result
            assert result == {"entailment": 0.8, "noncontradiction": 0.9, "contrasted_entailment": 0.85, "bert_score": 0.75, "cosine_sim": 0.7}

    def test_construct_result(self, uq_unit_response):
        """Test _construct_result method."""
        # Set up attributes
        uq_unit_response.responses = ["Response 1", "Response 2"]
        uq_unit_response.sampled_responses = [["Sample 1.1", "Sample 1.2"], ["Sample 2.1", "Sample 2.2"]]
        uq_unit_response.prompts = ["Prompt 1", "Prompt 2"]
        uq_unit_response.claim_sets = [["Claim 1.1", "Claim 1.2"], ["Claim 2.1"]]
        uq_unit_response.num_responses = 2
        uq_unit_response.scores_dict = {"entailment": [0.8, 0.9], "noncontradiction": [0.9, 0.95], "contrasted_entailment": [0.85, 0.92], "claims_data": [[{"claim": "Claim 1.1", "removed": False, "entailment": 0.8}, {"claim": "Claim 1.2", "removed": True, "entailment": 0.7}], [{"claim": "Claim 2.1", "removed": False, "entailment": 0.9}]]}
        uq_unit_response.uad_result = {"refined_responses": ["Refined 1", "Refined 2"]}
        uq_unit_response.response_refinement_threshold = 0.5

        result = uq_unit_response._construct_result()

        # Check that the result is a UQResult
        assert isinstance(result, UQResult)

        # Check the data
        assert result.data["prompts"] == ["Prompt 1", "Prompt 2"]
        assert result.data["responses"] == ["Response 1", "Response 2"]
        assert result.data["sampled_responses"] == [["Sample 1.1", "Sample 1.2"], ["Sample 2.1", "Sample 2.2"]]
        assert result.data["entailment"] == [0.8, 0.9]
        assert result.data["noncontradiction"] == [0.9, 0.95]
        assert result.data["contrasted_entailment"] == [0.85, 0.92]
        assert result.data["claims_data"] == [[{"claim": "Claim 1.1", "removed": False, "entailment": 0.8}, {"claim": "Claim 1.2", "removed": True, "entailment": 0.7}], [{"claim": "Claim 2.1", "removed": False, "entailment": 0.9}]]
        assert result.data["refined_responses"] == ["Refined 1", "Refined 2"]

        # Check the metadata
        assert result.metadata["mode"] == "unit_response"
        assert result.metadata["granularity"] == "claim"
        assert result.metadata["aggregation"] == "mean"
        assert result.metadata["temperature"] == 0.7
        assert result.metadata["sampling_temperature"] == 1.0
        assert result.metadata["num_responses"] == 2
        assert result.metadata["response_refinement_threshold"] == 0.5

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
from unittest.mock import MagicMock, patch, call
from rich.progress import Progress

from uqlm.longform.luq.baseclass.claims_scorer import ClaimScores
from uqlm.longform.luq.matched_unit import MatchedUnitScorer


class TestMatchedUnitScorer:
    @pytest.fixture
    def mock_nli(self):
        """Create a mock NLI instance."""
        mock = MagicMock()
        mock.predict.return_value = np.array([[0.1, 0.1, 0.8]])  # [contradiction, neutral, entailment]
        return mock

    @pytest.fixture
    def mock_cosine_scorer(self):
        """Create a mock CosineScorer instance."""
        mock = MagicMock()
        mock._compute_score.return_value = 0.75
        return mock

    @pytest.fixture
    def mock_bert_scorer(self):
        """Create a mock BertScorer instance."""
        mock = MagicMock()
        mock._compute_score.return_value = 0.85
        return mock

    @pytest.fixture
    def scorer_all_functions(self, mock_nli, mock_cosine_scorer, mock_bert_scorer):
        """Create a MatchedUnitScorer with all scoring functions."""
        with patch("uqlm.longform.luq.matched_unit.NLI", return_value=mock_nli), patch("uqlm.longform.luq.matched_unit.CosineScorer", return_value=mock_cosine_scorer), patch("uqlm.longform.luq.matched_unit.BertScorer", return_value=mock_bert_scorer):
            return MatchedUnitScorer(consistency_functions=["nli", "bert_score", "cosine_sim"])

    @pytest.fixture
    def scorer_nli_only(self, mock_nli):
        """Create a MatchedUnitScorer with only NLI scoring."""
        with patch("uqlm.longform.luq.matched_unit.NLI", return_value=mock_nli):
            return MatchedUnitScorer(consistency_functions=["nli"])

    @pytest.fixture
    def scorer_cosine_only(self, mock_cosine_scorer):
        """Create a MatchedUnitScorer with only cosine similarity scoring."""
        with patch("uqlm.longform.luq.matched_unit.CosineScorer", return_value=mock_cosine_scorer):
            return MatchedUnitScorer(consistency_functions=["cosine_sim"])

    @pytest.fixture
    def scorer_bert_only(self, mock_bert_scorer):
        """Create a MatchedUnitScorer with only BERT scoring."""
        with patch("uqlm.longform.luq.matched_unit.BertScorer", return_value=mock_bert_scorer):
            return MatchedUnitScorer(consistency_functions=["bert_score"])

    def test_initialization_default(self):
        """Test initialization with default parameters."""
        with patch("uqlm.longform.luq.matched_unit.NLI") as mock_nli_class, patch("uqlm.longform.luq.matched_unit.CosineScorer") as mock_cosine_class, patch("uqlm.longform.luq.matched_unit.BertScorer") as mock_bert_class:
            scorer = MatchedUnitScorer()

            # Check that models were initialized with the correct parameters
            mock_nli_class.assert_called_once_with(device=None, nli_model_name="microsoft/deberta-large-mnli", max_length=2000)
            mock_cosine_class.assert_called_once_with(transformer="all-MiniLM-L6-v2")
            mock_bert_class.assert_called_once_with(device=None)

            # Check that attributes were set correctly
            assert scorer.nli_model_name == "microsoft/deberta-large-mnli"
            assert scorer.consistency_functions == ["nli", "bert_score", "cosine_sim"]
            assert scorer.matched_claim is True
            assert scorer.progress_bar is None

    def test_initialization_custom(self):
        """Test initialization with custom parameters."""
        with patch("uqlm.longform.luq.matched_unit.NLI") as mock_nli_class, patch("uqlm.longform.luq.matched_unit.CosineScorer") as mock_cosine_class:
            # Only use NLI and cosine similarity with custom parameters
            scorer = MatchedUnitScorer(consistency_functions=["nli", "cosine_sim"], device="cuda", transformer="custom-transformer", nli_model_name="custom/model", max_length=1000)

            # Check that models were initialized with the correct parameters
            mock_nli_class.assert_called_once_with(device="cuda", nli_model_name="custom/model", max_length=1000)
            mock_cosine_class.assert_called_once_with(transformer="custom-transformer")

            # Check that attributes were set correctly
            assert scorer.nli_model_name == "custom/model"
            assert scorer.consistency_functions == ["nli", "cosine_sim"]
            assert scorer.matched_claim is True
            assert scorer.progress_bar is None
            assert scorer.bert_scorer is None  # Should be None since bert_score not in consistency_functions

    def test_initialization_invalid_function(self):
        """Test initialization with invalid consistency function."""
        with pytest.raises(ValueError) as excinfo:
            MatchedUnitScorer(consistency_functions=["invalid_function"])

        assert "consistency_functions must be subset of" in str(excinfo.value)

    def test_evaluate_mismatched_lengths(self, scorer_all_functions):
        """Test evaluate with mismatched claim_sets and sampled_claim_sets lengths."""
        claim_sets = [["Claim 1"], ["Claim 2"]]
        sampled_claim_sets = [[["Response 1"]]]  # Only one response set

        with pytest.raises(ValueError) as excinfo:
            scorer_all_functions.evaluate(claim_sets, sampled_claim_sets)

        assert "claim_sets and sampled_claim_sets must be of equal length" in str(excinfo.value)

    def test_evaluate_all_functions(self, scorer_all_functions):
        """Test evaluate method with all scoring functions."""
        claim_sets = [["Claim 1"], ["Claim 2"]]
        sampled_claim_sets = [[["Response 1"]], [["Response 2"]]]

        # Patch the compute methods
        with patch.object(MatchedUnitScorer, "_compute_response_level_nli_score_lists") as mock_nli_compute, patch.object(MatchedUnitScorer, "_compute_response_level_cosine_score_lists") as mock_cosine_compute, patch.object(MatchedUnitScorer, "_compute_response_level_bert_score_lists") as mock_bert_compute:
            # Set up return values
            mock_nli_compute.return_value = (
                [np.array([[0.8]])],  # entail_scores
                [np.array([[0.9]])],  # noncontradict_scores
                [np.array([[0.89]])],  # contrast_entail_scores
            )
            mock_cosine_compute.return_value = [np.array([[0.75]])]
            mock_bert_compute.return_value = [np.array([[0.85]])]

            result = scorer_all_functions.evaluate(claim_sets, sampled_claim_sets)

            # Check that all compute methods were called
            mock_nli_compute.assert_called_once_with(claim_sets=claim_sets, sampled_claim_sets=sampled_claim_sets)
            mock_cosine_compute.assert_called_once_with(claim_sets=claim_sets, sampled_claim_sets=sampled_claim_sets)
            mock_bert_compute.assert_called_once_with(claim_sets=claim_sets, sampled_claim_sets=sampled_claim_sets)

            # Check that the result is a ClaimScores object with all score types
            assert isinstance(result, ClaimScores)
            assert result.entailment_score_lists is not None
            assert result.noncontradict_score_lists is not None
            assert result.contrasted_entailment_score_lists is not None
            assert result.cosine_similarity_lists is not None
            assert result.bert_score_lists is not None

    def test_evaluate_nli_only(self, scorer_nli_only):
        """Test evaluate method with only NLI scoring."""
        claim_sets = [["Claim 1"], ["Claim 2"]]
        sampled_claim_sets = [[["Response 1"]], [["Response 2"]]]

        # Patch the compute methods
        with patch.object(MatchedUnitScorer, "_compute_response_level_nli_score_lists") as mock_nli_compute:
            # Set up return values
            mock_nli_compute.return_value = (
                [np.array([[0.8]])],  # entail_scores
                [np.array([[0.9]])],  # noncontradict_scores
                [np.array([[0.89]])],  # contrast_entail_scores
            )

            result = scorer_nli_only.evaluate(claim_sets, sampled_claim_sets)

            # Check that only NLI compute method was called
            mock_nli_compute.assert_called_once_with(claim_sets=claim_sets, sampled_claim_sets=sampled_claim_sets)

            # Check that the result is a ClaimScores object with only NLI scores
            assert isinstance(result, ClaimScores)
            assert result.entailment_score_lists is not None
            assert result.noncontradict_score_lists is not None
            assert result.contrasted_entailment_score_lists is not None
            assert result.cosine_similarity_lists is None
            assert result.bert_score_lists is None

    def test_evaluate_cosine_only(self, scorer_cosine_only):
        """Test evaluate method with only cosine similarity scoring."""
        claim_sets = [["Claim 1"], ["Claim 2"]]
        sampled_claim_sets = [[["Response 1"]], [["Response 2"]]]

        # Patch the compute methods
        with patch.object(MatchedUnitScorer, "_compute_response_level_cosine_score_lists") as mock_cosine_compute:
            # Set up return values
            mock_cosine_compute.return_value = [np.array([[0.75]])]

            result = scorer_cosine_only.evaluate(claim_sets, sampled_claim_sets)

            # Check that only cosine compute method was called
            mock_cosine_compute.assert_called_once_with(claim_sets=claim_sets, sampled_claim_sets=sampled_claim_sets)

            # Check that the result is a ClaimScores object with only cosine scores
            assert isinstance(result, ClaimScores)
            assert result.entailment_score_lists is None
            assert result.noncontradict_score_lists is None
            assert result.contrasted_entailment_score_lists is None
            assert result.cosine_similarity_lists is not None
            assert result.bert_score_lists is None

    def test_evaluate_bert_only(self, scorer_bert_only):
        """Test evaluate method with only BERT scoring."""
        claim_sets = [["Claim 1"], ["Claim 2"]]
        sampled_claim_sets = [[["Response 1"]], [["Response 2"]]]

        # Patch the compute methods
        with patch.object(MatchedUnitScorer, "_compute_response_level_bert_score_lists") as mock_bert_compute:
            # Set up return values
            mock_bert_compute.return_value = [np.array([[0.85]])]

            result = scorer_bert_only.evaluate(claim_sets, sampled_claim_sets)

            # Check that only BERT compute method was called
            mock_bert_compute.assert_called_once_with(claim_sets=claim_sets, sampled_claim_sets=sampled_claim_sets)

            # Check that the result is a ClaimScores object with only BERT scores
            assert isinstance(result, ClaimScores)
            assert result.entailment_score_lists is None
            assert result.noncontradict_score_lists is None
            assert result.contrasted_entailment_score_lists is None
            assert result.cosine_similarity_lists is None
            assert result.bert_score_lists is not None

    def test_compute_response_level_cosine_score_lists(self, scorer_cosine_only):
        """Test _compute_response_level_cosine_score_lists method."""
        claim_sets = [["Claim 1", "Claim 2"], ["Claim 3"]]
        sampled_claim_sets = [[["Response 1"], ["Response 2"]], [["Response 3"]]]

        # Patch the _compute_claim_level_cosine_scores method
        with patch.object(MatchedUnitScorer, "_compute_claim_level_cosine_scores") as mock_compute:
            # Set up return values for each call
            mock_compute.side_effect = [
                np.ones((2, 2)) * 0.75,  # First claim set
                np.ones((1, 1)) * 0.75,  # Second claim set
            ]

            result = scorer_cosine_only._compute_response_level_cosine_score_lists(claim_sets=claim_sets, sampled_claim_sets=sampled_claim_sets)

            # Check that _compute_claim_level_cosine_scores was called with the right arguments
            assert mock_compute.call_count == 2
            mock_compute.assert_has_calls([call(claims=claim_sets[0], candidates=sampled_claim_sets[0]), call(claims=claim_sets[1], candidates=sampled_claim_sets[1])])

            # Check the structure and values of the result
            assert len(result) == 2
            assert np.all(result[0] == 0.75)
            assert np.all(result[1] == 0.75)

    def test_compute_claim_level_cosine_scores(self, scorer_cosine_only):
        """Test _compute_claim_level_cosine_scores method."""
        claims = ["Claim 1", "Claim 2"]
        candidates = [["Response 1"], ["Response 2"]]

        # Patch the _compute_matched_cosine_scores method
        with patch.object(MatchedUnitScorer, "_compute_matched_cosine_scores") as mock_compute:
            # Set up return value
            mock_compute.return_value = 0.75

            result = scorer_cosine_only._compute_claim_level_cosine_scores(claims, candidates)

            # Check that _compute_matched_cosine_scores was called for each claim-candidate pair
            assert mock_compute.call_count == 4  # 2 claims * 2 candidates
            mock_compute.assert_has_calls([call(claims[0], candidates[0]), call(claims[0], candidates[1]), call(claims[1], candidates[0]), call(claims[1], candidates[1])])

            # Check the shape and values of the result
            assert result.shape == (2, 2)
            assert np.all(result == 0.75)

    def test_compute_matched_cosine_scores(self, scorer_cosine_only, mock_cosine_scorer):
        """Test _compute_matched_cosine_scores method."""
        claim = "Claim 1"
        candidate_claims = ["Response 1", "Response 2"]

        # Set up the mock to return different values for different inputs
        mock_cosine_scorer._compute_score.side_effect = [0.7, 0.8]

        result = scorer_cosine_only._compute_matched_cosine_scores(claim, candidate_claims)

        # Check that _compute_score was called for each candidate
        assert mock_cosine_scorer._compute_score.call_count == 2
        mock_cosine_scorer._compute_score.assert_has_calls([call(claim, [candidate_claims[0]]), call(claim, [candidate_claims[1]])])

        # Check that the maximum score was returned
        assert result == 0.8

    def test_compute_response_level_bert_score_lists(self, scorer_bert_only):
        """Test _compute_response_level_bert_score_lists method."""
        claim_sets = [["Claim 1", "Claim 2"], ["Claim 3"]]
        sampled_claim_sets = [[["Response 1"], ["Response 2"]], [["Response 3"]]]

        # Patch the _compute_claim_level_bert_scores method
        with patch.object(MatchedUnitScorer, "_compute_claim_level_bert_scores") as mock_compute:
            # Set up return values for each call
            mock_compute.side_effect = [
                np.ones((2, 2)) * 0.85,  # First claim set
                np.ones((1, 1)) * 0.85,  # Second claim set
            ]

            result = scorer_bert_only._compute_response_level_bert_score_lists(claim_sets=claim_sets, sampled_claim_sets=sampled_claim_sets)

            # Check that _compute_claim_level_bert_scores was called with the right arguments
            assert mock_compute.call_count == 2
            mock_compute.assert_has_calls([call(claims=claim_sets[0], candidates=sampled_claim_sets[0]), call(claims=claim_sets[1], candidates=sampled_claim_sets[1])])

            # Check the structure and values of the result
            assert len(result) == 2
            assert np.all(result[0] == 0.85)
            assert np.all(result[1] == 0.85)

    def test_compute_claim_level_bert_scores(self, scorer_bert_only):
        """Test _compute_claim_level_bert_scores method."""
        claims = ["Claim 1", "Claim 2"]
        candidates = [["Response 1"], ["Response 2"]]

        # Patch the _compute_matched_bert_scores method
        with patch.object(MatchedUnitScorer, "_compute_matched_bert_scores") as mock_compute:
            # Set up return value
            mock_compute.return_value = 0.85

            result = scorer_bert_only._compute_claim_level_bert_scores(claims, candidates)

            # Check that _compute_matched_bert_scores was called for each claim-candidate pair
            assert mock_compute.call_count == 4  # 2 claims * 2 candidates
            mock_compute.assert_has_calls([call(claims[0], candidates[0]), call(claims[0], candidates[1]), call(claims[1], candidates[0]), call(claims[1], candidates[1])])

            # Check the shape and values of the result
            assert result.shape == (2, 2)
            assert np.all(result == 0.85)

    def test_compute_matched_bert_scores(self, scorer_bert_only, mock_bert_scorer):
        """Test _compute_matched_bert_scores method."""
        claim = "Claim 1"
        candidate_claims = ["Response 1", "Response 2"]

        # Set up the mock to return different values for different inputs
        mock_bert_scorer._compute_score.side_effect = [0.8, 0.9]

        result = scorer_bert_only._compute_matched_bert_scores(claim, candidate_claims)

        # Check that _compute_score was called for each candidate
        assert mock_bert_scorer._compute_score.call_count == 2
        mock_bert_scorer._compute_score.assert_has_calls([call(claim, [candidate_claims[0]]), call(claim, [candidate_claims[1]])])

        # Check that the maximum score was returned
        assert result == 0.9

    def test_with_progress_bar(self, scorer_all_functions):
        """Test that progress bar is used correctly."""
        claim_sets = [["Claim 1"], ["Claim 2"]]
        sampled_claim_sets = [[["Response 1"]], [["Response 2"]]]

        # Create a mock progress bar
        mock_progress_bar = MagicMock(spec=Progress)
        mock_progress_bar.add_task.return_value = "task_id"

        # Patch the compute methods to avoid actual computation
        with patch.object(MatchedUnitScorer, "_compute_response_level_nli_score_lists") as mock_nli_compute, patch.object(MatchedUnitScorer, "_compute_response_level_cosine_score_lists") as mock_cosine_compute, patch.object(MatchedUnitScorer, "_compute_response_level_bert_score_lists") as mock_bert_compute:
            # Set up return values
            mock_nli_compute.return_value = ([], [], [])
            mock_cosine_compute.return_value = []
            mock_bert_compute.return_value = []

            scorer_all_functions.evaluate(claim_sets, sampled_claim_sets, progress_bar=mock_progress_bar)

            # Check that progress_bar was set
            assert scorer_all_functions.progress_bar is mock_progress_bar

            # Check that compute methods were called with the progress bar set
            mock_nli_compute.assert_called_once()
            mock_cosine_compute.assert_called_once()
            mock_bert_compute.assert_called_once()

    def test_compute_response_level_cosine_score_lists_with_progress_bar(self, scorer_cosine_only):
        """progress_bar branches in _compute_response_level_cosine_score_lists (lines 96, 102)."""
        claim_sets = [["Claim 1", "Claim 2"], ["Claim 3"]]
        sampled_claim_sets = [[["R1"], ["R2"]], [["R3"]]]

        mock_pb = MagicMock(spec=Progress)
        mock_pb.add_task.return_value = "task_id"
        scorer_cosine_only.progress_bar = mock_pb

        with patch.object(MatchedUnitScorer, "_compute_claim_level_cosine_scores") as mock_compute:
            mock_compute.side_effect = [np.ones((2, 2)) * 0.75, np.ones((1, 1)) * 0.75]
            scorer_cosine_only._compute_response_level_cosine_score_lists(claim_sets=claim_sets, sampled_claim_sets=sampled_claim_sets)

        mock_pb.add_task.assert_called_once()
        assert mock_pb.update.call_count == 2

    def test_compute_response_level_bert_score_lists_with_progress_bar(self, scorer_bert_only):
        """progress_bar branches in _compute_response_level_bert_score_lists (lines 126, 132)."""
        claim_sets = [["Claim A"], ["Claim B"]]
        sampled_claim_sets = [[["R1"]], [["R2"]]]

        mock_pb = MagicMock(spec=Progress)
        mock_pb.add_task.return_value = "task_id"
        scorer_bert_only.progress_bar = mock_pb

        with patch.object(MatchedUnitScorer, "_compute_claim_level_bert_scores") as mock_compute:
            mock_compute.side_effect = [np.ones((1, 1)) * 0.85, np.ones((1, 1)) * 0.85]
            scorer_bert_only._compute_response_level_bert_score_lists(claim_sets=claim_sets, sampled_claim_sets=sampled_claim_sets)

        mock_pb.add_task.assert_called_once()
        assert mock_pb.update.call_count == 2

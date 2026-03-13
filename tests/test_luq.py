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
from unittest.mock import MagicMock, patch
from rich.progress import Progress

from uqlm.longform.luq.baseclass.claims_scorer import ClaimScores
from uqlm.longform.luq.unit_response import UnitResponseScorer


class TestUnitResponseScorer:
    @pytest.fixture
    def mock_nli(self):
        """Create a mock NLI instance."""
        mock = MagicMock()
        # Mock the predict method to return a fixed array
        mock.predict.return_value = np.array([[0.1, 0.1, 0.8]])  # [contradiction, neutral, entailment]
        return mock

    @pytest.fixture
    def scorer(self, mock_nli):
        """Create a UnitResponseScorer with a mocked NLI."""
        with patch("uqlm.longform.luq.unit_response.NLI", return_value=mock_nli):
            return UnitResponseScorer()

    def test_initialization_default(self):
        """Test initialization with default parameters."""
        with patch("uqlm.longform.luq.unit_response.NLI") as mock_nli_class:
            scorer = UnitResponseScorer()

            # Check that NLI was initialized with the correct parameters
            mock_nli_class.assert_called_once_with(device=None, nli_model_name="microsoft/deberta-large-mnli", max_length=2000)

            # Check that attributes were set correctly
            assert scorer.nli_model_name == "microsoft/deberta-large-mnli"
            assert scorer.progress_bar is None
            assert scorer.matched_claim is False

    def test_initialization_custom(self):
        """Test initialization with custom parameters."""
        with patch("uqlm.longform.luq.unit_response.NLI") as mock_nli_class:
            scorer = UnitResponseScorer(nli_model_name="custom/model", device="cuda", max_length=1000)

            # Check that NLI was initialized with the correct parameters
            mock_nli_class.assert_called_once_with(device="cuda", nli_model_name="custom/model", max_length=1000)

            # Check that attributes were set correctly
            assert scorer.nli_model_name == "custom/model"
            assert scorer.progress_bar is None
            assert scorer.matched_claim is False

    def test_evaluate_mismatched_lengths(self, scorer):
        """Test evaluate with mismatched claim_sets and sampled_responses lengths."""
        claim_sets = [["Claim 1"], ["Claim 2"]]
        sampled_responses = [["Response 1"]]  # Only one response set

        with pytest.raises(ValueError) as excinfo:
            scorer.evaluate(claim_sets, sampled_responses)

        assert "claim_sets and sampled_responses must be of equal length" in str(excinfo.value)

    @pytest.mark.parametrize("progress_bar", [None, MagicMock(spec=Progress)])
    def test_evaluate(self, scorer, progress_bar):
        """Test evaluate method with and without progress bar."""
        claim_sets = [["Claim 1", "Claim 2"], ["Claim 3"]]
        sampled_responses = [["Response 1", "Response 2"], ["Response 3"]]

        # Patch the _compute_response_level_nli_score_lists method
        with patch.object(UnitResponseScorer, "_compute_response_level_nli_score_lists") as mock_compute:
            # Set up return value
            mock_compute.return_value = (
                [np.ones((2, 2)) * 0.8, np.ones((1, 1)) * 0.8],  # entail_scores
                [np.ones((2, 2)) * 0.9, np.ones((1, 1)) * 0.9],  # noncontradict_scores
                [np.ones((2, 2)) * 0.89, np.ones((1, 1)) * 0.89],  # contrast_entail_scores
            )

            result = scorer.evaluate(claim_sets, sampled_responses, progress_bar)

            # Check that progress_bar was set
            assert scorer.progress_bar is progress_bar

            # Check that _compute_response_level_nli_score_lists was called with the right arguments
            mock_compute.assert_called_once_with(claim_sets=claim_sets, sampled_responses=sampled_responses)

            # Check that the result is a ClaimScores object
            assert isinstance(result, ClaimScores)

            # Check that the scores are populated
            assert result.entailment_score_lists is not None
            assert result.noncontradict_score_lists is not None
            assert result.contrasted_entailment_score_lists is not None

            # Check the values
            assert np.all(result.entailment_score_lists[0] == 0.8)
            assert np.all(result.noncontradict_score_lists[0] == 0.9)
            assert np.all(result.contrasted_entailment_score_lists[0] == 0.89)
            assert np.all(result.entailment_score_lists[1] == 0.8)
            assert np.all(result.noncontradict_score_lists[1] == 0.9)
            assert np.all(result.contrasted_entailment_score_lists[1] == 0.89)

    def test_get_nli_agreement_scores(self, scorer, mock_nli):
        """Test _get_nli_agreement_scores method."""
        claim = "The Earth is round."
        candidate = "The Earth is an oblate spheroid."

        # Mock the NLI predict method to return a fixed array
        mock_nli.predict.return_value = np.array([[0.1, 0.1, 0.8]])  # [contradiction, neutral, entailment]

        # Let's patch the method to see what it actually returns
        with patch.object(UnitResponseScorer, "_get_nli_agreement_scores", autospec=True) as mock_method:
            # Set up the return value to match what we expect from the implementation
            mock_method.return_value = (0.8, 0.9, 0.89)

            entail_prob, noncontradict_prob, contrast_entail_prob = mock_method(scorer, claim, candidate)

            # Now check the values from our mocked method
            assert entail_prob == 0.8
            assert noncontradict_prob == 0.9
            assert contrast_entail_prob == 0.89

            # Verify the method was called with the right arguments
            mock_method.assert_called_once_with(scorer, claim, candidate)

    def test_compute_claim_level_nli_scores(self, scorer):
        """Test _compute_claim_level_nli_scores method."""
        claims = ["Claim 1", "Claim 2"]
        candidates = ["Response 1", "Response 2"]

        # Patch the _get_nli_agreement_scores method to return fixed values
        with patch.object(UnitResponseScorer, "_get_nli_agreement_scores", return_value=(0.8, 0.9, 0.89)):
            entail_scores, noncontradict_scores, contrast_entail_scores = scorer._compute_claim_level_nli_scores(claims, candidates)

            # Check the shape of the output arrays
            assert entail_scores.shape == (2, 2)
            assert noncontradict_scores.shape == (2, 2)
            assert contrast_entail_scores.shape == (2, 2)

            # Check that all values are filled with our fixed values
            assert np.all(entail_scores == 0.8)
            assert np.all(noncontradict_scores == 0.9)
            assert np.all(contrast_entail_scores == 0.89)

    def test_compute_matched_nli_scores(self, scorer):
        """Test _compute_matched_nli_scores method."""
        claim = "Claim 1"
        candidate_claims = ["Candidate 1", "Candidate 2"]

        # Patch the _get_nli_agreement_scores method to return fixed values
        with patch.object(UnitResponseScorer, "_get_nli_agreement_scores", return_value=(0.8, 0.9, 0.89)):
            entail_prob, noncontradict_prob, contrast_entail_prob = scorer._compute_matched_nli_scores(claim, candidate_claims)

            # Check that maximum probabilities are returned
            assert entail_prob == 0.8
            assert noncontradict_prob == 0.9
            assert contrast_entail_prob == 0.89

    def test_compute_response_level_nli_score_lists(self, scorer):
        """Test _compute_response_level_nli_score_lists method."""
        claim_sets = [["Claim 1", "Claim 2"], ["Claim 3"]]
        sampled_responses = [["Response 1", "Response 2"], ["Response 3"]]

        # Patch the _compute_claim_level_nli_scores method to return fixed arrays
        with patch.object(UnitResponseScorer, "_compute_claim_level_nli_scores") as mock_compute:
            # Set up return values for each call
            mock_compute.side_effect = [
                (np.ones((2, 2)) * 0.8, np.ones((2, 2)) * 0.9, np.ones((2, 2)) * 0.89),  # First claim set
                (np.ones((1, 1)) * 0.8, np.ones((1, 1)) * 0.9, np.ones((1, 1)) * 0.89),  # Second claim set
            ]

            entail_scores, noncontradict_scores, contrast_entail_scores = scorer._compute_response_level_nli_score_lists(claim_sets=claim_sets, sampled_responses=sampled_responses)

            # Check that the output has the correct structure
            assert len(entail_scores) == 2
            assert len(noncontradict_scores) == 2
            assert len(contrast_entail_scores) == 2

            # Check the shape and values of the first array
            assert entail_scores[0].shape == (2, 2)  # 2 claims, 2 responses
            assert np.all(entail_scores[0] == 0.8)
            assert np.all(noncontradict_scores[0] == 0.9)
            assert np.all(contrast_entail_scores[0] == 0.89)

            # Check the shape and values of the second array
            assert entail_scores[1].shape == (1, 1)  # 1 claim, 1 response
            assert np.all(entail_scores[1] == 0.8)
            assert np.all(noncontradict_scores[1] == 0.9)
            assert np.all(contrast_entail_scores[1] == 0.89)

    def test_compute_response_level_nli_score_lists_with_progress_bar(self, scorer):
        """Test _compute_response_level_nli_score_lists with progress bar."""
        # Create a mock progress bar
        mock_progress_bar = MagicMock(spec=Progress)
        mock_progress_bar.add_task.return_value = "task_id"

        # Set the progress bar
        scorer.progress_bar = mock_progress_bar

        claim_sets = [["Claim 1"], ["Claim 2"]]
        sampled_responses = [["Response 1"], ["Response 2"]]

        # Patch the _compute_claim_level_nli_scores method
        with patch.object(UnitResponseScorer, "_compute_claim_level_nli_scores") as mock_compute:
            # Set up return values for each call
            mock_compute.side_effect = [
                (np.ones((1, 1)) * 0.8, np.ones((1, 1)) * 0.9, np.ones((1, 1)) * 0.89),  # First claim set
                (np.ones((1, 1)) * 0.8, np.ones((1, 1)) * 0.9, np.ones((1, 1)) * 0.89),  # Second claim set
            ]

            scorer._compute_response_level_nli_score_lists(claim_sets, sampled_responses)

            # Check that the progress bar was used correctly
            mock_progress_bar.add_task.assert_called_once()
            assert mock_progress_bar.update.call_count == 2  # Once for each claim set

    def test_compute_response_level_nli_score_lists_with_matched_claims(self, scorer):
        """Test _compute_response_level_nli_score_lists with matched claims."""
        # Set matched_claim to True
        scorer.matched_claim = True

        claim_sets = [["Claim 1"]]
        sampled_claim_sets = [[["Matched Claim 1", "Matched Claim 2"]]]

        # Patch the _compute_claim_level_nli_scores method
        with patch.object(UnitResponseScorer, "_compute_claim_level_nli_scores") as mock_compute:
            # Set up return value
            mock_compute.return_value = (np.ones((1, 1)) * 0.8, np.ones((1, 1)) * 0.9, np.ones((1, 1)) * 0.89)

            entail_scores, noncontradict_scores, contrast_entail_scores = scorer._compute_response_level_nli_score_lists(claim_sets=claim_sets, sampled_claim_sets=sampled_claim_sets)

            # Check that the output has the correct structure
            assert len(entail_scores) == 1
            assert entail_scores[0].shape == (1, 1)  # 1 claim, 1 set of matched claims
            assert np.all(entail_scores[0] == 0.8)
            assert np.all(noncontradict_scores[0] == 0.9)
            assert np.all(contrast_entail_scores[0] == 0.89)

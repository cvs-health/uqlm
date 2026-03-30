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
from typing import List, Tuple, Optional, Union

from uqlm.longform.luq.baseclass.claims_scorer import ClaimScore, ClaimScores, ClaimScorer


class TestClaimScore:
    def test_initialization(self):
        """Test initialization of ClaimScore."""
        claim = "The Earth is round."
        original_response = True
        scores = {"entailment": 0.8, "contradiction": 0.1}
        scorer_type = "nli"

        score = ClaimScore(claim=claim, original_response=original_response, scores=scores, scorer_type=scorer_type)

        assert score.claim == claim
        assert score.original_response == original_response
        assert score.scores == scores
        assert score.scorer_type == scorer_type

    def test_pydantic_validation(self):
        """Test Pydantic validation for ClaimScore."""
        # Valid initialization
        score = ClaimScore(claim="Test claim", original_response=True, scores={"score1": 0.5}, scorer_type="test")
        assert score.claim == "Test claim"

        # Invalid initialization (missing required fields)
        with pytest.raises(Exception):
            ClaimScore(claim="Test claim")


class TestClaimScores:
    def test_initialization_empty(self):
        """Test initialization of ClaimScores with no arguments."""
        scores = ClaimScores()

        assert scores.entailment_score_lists is None
        assert scores.noncontradict_score_lists is None
        assert scores.contrasted_entailment_score_lists is None
        assert scores.cosine_similarity_lists is None
        assert scores.bert_score_lists is None

    def test_initialization_with_data(self):
        """Test initialization of ClaimScores with data."""
        entailment_scores = [np.array([[0.8, 0.7], [0.6, 0.5]])]
        noncontradict_scores = [np.array([[0.9, 0.8], [0.7, 0.6]])]

        scores = ClaimScores(entailment_score_lists=entailment_scores, noncontradict_score_lists=noncontradict_scores)

        assert scores.entailment_score_lists == entailment_scores
        assert scores.noncontradict_score_lists == noncontradict_scores
        assert scores.contrasted_entailment_score_lists is None
        assert scores.cosine_similarity_lists is None
        assert scores.bert_score_lists is None

    def test_to_dict_default(self):
        """Test to_dict method with default parameters."""
        entailment_scores = [np.array([[0.8, 0.7], [0.6, 0.5]])]
        noncontradict_scores = [np.array([[0.9, 0.8], [0.7, 0.6]])]

        scores = ClaimScores(entailment_score_lists=entailment_scores, noncontradict_score_lists=noncontradict_scores)

        result = scores.to_dict()

        # Check that only non-None scores are included
        assert "entailment" in result
        assert "noncontradiction" in result
        assert "contrasted_entailment" not in result
        assert "cosine_sim" not in result
        assert "bert_score" not in result

        # Check that means are calculated correctly (using np.isclose for floating point comparison)
        assert len(result["entailment"]) == 1
        assert len(result["entailment"][0]) == 2
        assert np.isclose(result["entailment"][0][0], 0.75)
        assert np.isclose(result["entailment"][0][1], 0.55)

        assert len(result["noncontradiction"]) == 1
        assert len(result["noncontradiction"][0]) == 2
        assert np.isclose(result["noncontradiction"][0][0], 0.85)
        assert np.isclose(result["noncontradiction"][0][1], 0.65)

    def test_to_dict_return_all(self):
        """Test to_dict method with return_all=True."""
        entailment_scores = [np.array([[0.8, 0.7], [0.6, 0.5]])]

        scores = ClaimScores(entailment_score_lists=entailment_scores)

        result = scores.to_dict(return_all=True)

        # Check that all values are returned, not just means
        assert len(result["entailment"]) == 1
        assert len(result["entailment"][0]) == 2
        assert len(result["entailment"][0][0]) == 2
        assert np.isclose(result["entailment"][0][0][0], 0.8)
        assert np.isclose(result["entailment"][0][0][1], 0.7)
        assert np.isclose(result["entailment"][0][1][0], 0.6)
        assert np.isclose(result["entailment"][0][1][1], 0.5)

    def test_format_result_empty(self):
        """Test _format_result with empty input."""
        result = ClaimScores._format_result([])
        assert result is None

    def test_format_result_default(self):
        """Test _format_result with default parameters."""
        score_arrays = [np.array([[0.8, 0.7], [0.6, 0.5]]), np.array([[0.9, 0.8]])]

        result = ClaimScores._format_result(score_arrays)

        # Check that means are calculated correctly (using np.isclose for floating point comparison)
        assert len(result) == 2
        assert len(result[0]) == 2
        assert np.isclose(result[0][0], 0.75)
        assert np.isclose(result[0][1], 0.55)

        assert len(result[1]) == 1
        assert np.isclose(result[1][0], 0.85)

    def test_format_result_return_all(self):
        """Test _format_result with return_all=True."""
        score_arrays = [np.array([[0.8, 0.7], [0.6, 0.5]]), np.array([[0.9, 0.8]])]

        result = ClaimScores._format_result(score_arrays, return_all=True)

        # Check that all values are returned, not just means
        assert len(result) == 2
        assert len(result[0]) == 2
        assert len(result[0][0]) == 2
        assert np.isclose(result[0][0][0], 0.8)
        assert np.isclose(result[0][0][1], 0.7)
        assert np.isclose(result[0][1][0], 0.6)
        assert np.isclose(result[0][1][1], 0.5)

        assert len(result[1]) == 1
        assert len(result[1][0]) == 2
        assert np.isclose(result[1][0][0], 0.9)
        assert np.isclose(result[1][0][1], 0.8)


# Create a concrete implementation of the abstract ClaimScorer for testing
class ConcreteClaimScorer(ClaimScorer):
    def __init__(self, matched_claim=False, progress_bar=None):
        self.nli = MagicMock()
        self.matched_claim = matched_claim
        self.progress_bar = progress_bar

        # Mock the NLI predict method to return a numpy array
        # Format: [contradiction, neutral, entailment]
        self.nli.predict.return_value = np.array([[0.1, 0.1, 0.8]])

    def evaluate(self, claim_sets: List[List[str]], sampled_responses: List[List[str]]) -> ClaimScores:
        """Implement the abstract method for testing."""
        entail_scores, noncontradict_scores, contrast_entail_scores = self._compute_response_level_nli_score_lists(claim_sets=claim_sets, sampled_responses=sampled_responses)
        return ClaimScores(entailment_score_lists=entail_scores, noncontradict_score_lists=noncontradict_scores, contrasted_entailment_score_lists=contrast_entail_scores)


class TestClaimScorer:
    @pytest.fixture
    def scorer(self):
        """Create a concrete ClaimScorer instance for testing."""
        return ConcreteClaimScorer()

    def test_get_nli_agreement_scores(self, scorer):
        """Test _get_nli_agreement_scores method."""
        claim = "The Earth is round."
        candidate = "The Earth is an oblate spheroid."

        # Patch the _get_nli_agreement_scores method to return fixed values for testing
        with patch.object(ConcreteClaimScorer, "_get_nli_agreement_scores", return_value=(0.8, 0.9, 0.89)):
            entail_prob, noncontradict_prob, contrast_entail_prob = scorer._get_nli_agreement_scores(claim, candidate)

            # Check that probabilities match our fixed values
            assert entail_prob == 0.8
            assert noncontradict_prob == 0.9
            assert contrast_entail_prob == 0.89

    def test_compute_claim_level_nli_scores(self, scorer):
        """Test _compute_claim_level_nli_scores method."""
        claims = ["Claim 1", "Claim 2"]
        candidates = ["Response 1", "Response 2"]

        # Patch the _get_nli_agreement_scores method to return fixed values
        with patch.object(ConcreteClaimScorer, "_get_nli_agreement_scores", return_value=(0.8, 0.9, 0.89)):
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
        with patch.object(ConcreteClaimScorer, "_get_nli_agreement_scores", return_value=(0.8, 0.9, 0.89)):
            entail_prob, noncontradict_prob, contrast_entail_prob = scorer._compute_matched_nli_scores(claim, candidate_claims)

            # Check that maximum probabilities are returned (which are our fixed values)
            assert entail_prob == 0.8
            assert noncontradict_prob == 0.9
            assert contrast_entail_prob == 0.89

    def test_compute_response_level_nli_score_lists(self, scorer):
        """Test _compute_response_level_nli_score_lists method."""
        claim_sets = [["Claim 1", "Claim 2"], ["Claim 3"]]
        sampled_responses = [["Response 1", "Response 2"], ["Response 3"]]

        # Patch the _compute_claim_level_nli_scores method to return fixed arrays
        with patch.object(ConcreteClaimScorer, "_compute_claim_level_nli_scores") as mock_compute:
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

    def test_compute_response_level_nli_score_lists_with_progress_bar(self):
        """Test _compute_response_level_nli_score_lists with progress bar."""
        # Create a mock progress bar
        mock_progress_bar = MagicMock()
        mock_progress_bar.add_task.return_value = "task_id"

        # Create a scorer with the mock progress bar
        scorer = ConcreteClaimScorer(progress_bar=mock_progress_bar)

        claim_sets = [["Claim 1"], ["Claim 2"]]
        sampled_responses = [["Response 1"], ["Response 2"]]

        # Patch the _compute_claim_level_nli_scores method
        with patch.object(ConcreteClaimScorer, "_compute_claim_level_nli_scores") as mock_compute:
            # Set up return values for each call
            mock_compute.side_effect = [
                (np.ones((1, 1)) * 0.8, np.ones((1, 1)) * 0.9, np.ones((1, 1)) * 0.89),  # First claim set
                (np.ones((1, 1)) * 0.8, np.ones((1, 1)) * 0.9, np.ones((1, 1)) * 0.89),  # Second claim set
            ]

            scorer._compute_response_level_nli_score_lists(claim_sets, sampled_responses)

            # Check that the progress bar was used correctly
            mock_progress_bar.add_task.assert_called_once()
            assert mock_progress_bar.update.call_count == 2  # Once for each claim set

    def test_compute_response_level_nli_score_lists_with_matched_claims(self):
        """Test _compute_response_level_nli_score_lists with matched claims."""
        # Create a scorer with matched_claim=True
        scorer = ConcreteClaimScorer(matched_claim=True)

        claim_sets = [["Claim 1"]]
        sampled_claim_sets = [[["Matched Claim 1", "Matched Claim 2"]]]

        # Patch the _compute_claim_level_nli_scores method
        with patch.object(ConcreteClaimScorer, "_compute_claim_level_nli_scores") as mock_compute:
            # Set up return value
            mock_compute.return_value = (np.ones((1, 1)) * 0.8, np.ones((1, 1)) * 0.9, np.ones((1, 1)) * 0.89)

            entail_scores, noncontradict_scores, contrast_entail_scores = scorer._compute_response_level_nli_score_lists(claim_sets=claim_sets, sampled_claim_sets=sampled_claim_sets)

            # Check that the output has the correct structure
            assert len(entail_scores) == 1
            assert entail_scores[0].shape == (1, 1)  # 1 claim, 1 set of matched claims
            assert np.all(entail_scores[0] == 0.8)
            assert np.all(noncontradict_scores[0] == 0.9)
            assert np.all(contrast_entail_scores[0] == 0.89)

    def test_evaluate(self, scorer):
        """Test the evaluate method."""
        claim_sets = [["Claim 1", "Claim 2"]]
        sampled_responses = [["Response 1", "Response 2"]]

        # Patch the _compute_response_level_nli_score_lists method
        with patch.object(ConcreteClaimScorer, "_compute_response_level_nli_score_lists") as mock_compute:
            # Set up return value
            mock_compute.return_value = (
                [np.ones((2, 2)) * 0.8],  # entail_scores
                [np.ones((2, 2)) * 0.9],  # noncontradict_scores
                [np.ones((2, 2)) * 0.89],  # contrast_entail_scores
            )

            result = scorer.evaluate(claim_sets, sampled_responses)

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

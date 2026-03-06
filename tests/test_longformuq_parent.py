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

from langchain_core.language_models.chat_models import BaseChatModel
from uqlm.scorers.baseclass.uncertainty import UncertaintyQuantifier
from uqlm.longform.decomposition import ResponseDecomposer
from uqlm.longform.uad import UncertaintyAwareDecoder
from uqlm.scorers.longform.baseclass.uncertainty import LongFormUQ


class TestLongFormUQ:
    @pytest.fixture
    def mock_llm(self):
        """Create a mock LLM."""
        mock = MagicMock(spec=BaseChatModel)
        mock.temperature = 0.7
        return mock

    @pytest.fixture
    def mock_decomposer(self):
        """Create a mock ResponseDecomposer."""
        mock = MagicMock(spec=ResponseDecomposer)
        mock.decompose_sentences.return_value = [["Sentence 1.1", "Sentence 1.2"], ["Sentence 2.1"]]
        mock.decompose_claims = AsyncMock(return_value=[["Claim 1.1", "Claim 1.2"], ["Claim 2.1"]])
        mock.decompose_candidate_sentences.return_value = [[["Sentence 1.1.1", "Sentence 1.1.2"], ["Sentence 1.2.1"]], [["Sentence 2.1.1"], ["Sentence 2.2.1"]]]
        mock.decompose_candidate_claims = AsyncMock(return_value=[[["Claim 1.1.1", "Claim 1.1.2"], ["Claim 1.2.1"]], [["Claim 2.1.1"], ["Claim 2.2.1"]]])
        return mock

    @pytest.fixture
    def mock_reconstructor(self):
        """Create a mock UncertaintyAwareDecoder."""
        mock = MagicMock(spec=UncertaintyAwareDecoder)
        mock.reconstruct_responses = AsyncMock(return_value={"refined_responses": ["Refined Response 1", "Refined Response 2"], "removed": [[False, True], [False]]})
        return mock

    @pytest.fixture
    def uq_default(self, mock_llm, mock_decomposer):
        """Create a LongFormUQ instance with default parameters."""
        with patch("uqlm.scorers.longform.baseclass.uncertainty.ResponseDecomposer", return_value=mock_decomposer):
            return LongFormUQ(llm=mock_llm, scorers=["entailment"])

    @pytest.fixture
    def uq_with_refinement(self, mock_llm, mock_decomposer, mock_reconstructor):
        """Create a LongFormUQ instance with response refinement enabled."""
        # Add claim_decomposition_llm attribute to the mock_decomposer
        mock_decomposer.claim_decomposition_llm = mock_llm

        with patch("uqlm.scorers.longform.baseclass.uncertainty.ResponseDecomposer", return_value=mock_decomposer), patch("uqlm.scorers.longform.baseclass.uncertainty.UncertaintyAwareDecoder", return_value=mock_reconstructor):
            return LongFormUQ(llm=mock_llm, scorers=["entailment", "noncontradiction"], response_refinement=True)

    def test_initialization_default(self, mock_llm):
        """Test initialization with default parameters."""
        with patch("uqlm.scorers.longform.baseclass.uncertainty.ResponseDecomposer") as mock_decomposer_class:
            uq = LongFormUQ(llm=mock_llm, scorers=["entailment"])

            # Check that ResponseDecomposer was initialized with the correct parameters
            mock_decomposer_class.assert_called_once_with(claim_decomposition_llm=mock_llm)

            # Check that attributes were set correctly
            assert uq.llm == mock_llm
            assert uq.claim_decomposition_llm is None
            assert uq.granularity == "claim"
            assert uq.scorers == ["entailment"]
            assert uq.aggregation == "mean"
            assert uq.response_refinement is False
            assert uq.claim_filtering_scorer is None
            assert uq.response_refinement_threshold is None

    def test_initialization_custom(self, mock_llm):
        """Test initialization with custom parameters."""
        custom_decomposition_llm = MagicMock(spec=BaseChatModel)

        with patch("uqlm.scorers.longform.baseclass.uncertainty.ResponseDecomposer") as mock_decomposer_class, patch("uqlm.scorers.longform.baseclass.uncertainty.UncertaintyAwareDecoder") as mock_reconstructor_class:
            uq = LongFormUQ(llm=mock_llm, scorers=["entailment", "noncontradiction"], granularity="claim", aggregation="min", claim_decomposition_llm=custom_decomposition_llm, response_refinement=True, claim_filtering_scorer="noncontradiction", device="cuda", system_prompt="Custom prompt", max_calls_per_min=10, use_n_param=True)

            # Check that ResponseDecomposer was initialized with the correct parameters
            mock_decomposer_class.assert_called_once_with(claim_decomposition_llm=custom_decomposition_llm)

            # Check that UncertaintyAwareDecoder was initialized with the correct parameters
            mock_reconstructor_class.assert_called_once()

            # Check that attributes were set correctly
            assert uq.llm == mock_llm
            assert uq.claim_decomposition_llm == custom_decomposition_llm
            assert uq.granularity == "claim"
            assert uq.scorers == ["entailment", "noncontradiction"]
            assert uq.aggregation == "min"
            assert uq.response_refinement is True
            assert uq.claim_filtering_scorer == "noncontradiction"
            assert uq.uad_scorer == "noncontradiction"
            assert uq.device == "cuda"
            assert uq.system_prompt == "Custom prompt"
            assert uq.max_calls_per_min == 10
            assert uq.use_n_param is True

    def test_initialization_invalid_granularity(self, mock_llm):
        """Test initialization with invalid granularity."""
        with pytest.raises(ValueError) as excinfo:
            LongFormUQ(llm=mock_llm, scorers=["entailment"], granularity="invalid")

        assert "Invalid granularity" in str(excinfo.value)

    def test_initialization_invalid_refinement_with_sentence(self, mock_llm):
        """Test initialization with response_refinement=True and granularity='sentence'."""
        with pytest.raises(ValueError) as excinfo:
            LongFormUQ(llm=mock_llm, scorers=["entailment"], granularity="sentence", response_refinement=True)

        assert "Uncertainty aware decoding is only possible with claim-level scoring" in str(excinfo.value)

    def test_initialization_default_claim_filtering_scorer(self, mock_llm, capfd):
        """Test initialization with response_refinement=True but no claim_filtering_scorer."""
        with patch("uqlm.scorers.longform.baseclass.uncertainty.ResponseDecomposer"), patch("uqlm.scorers.longform.baseclass.uncertainty.UncertaintyAwareDecoder"):
            uq = LongFormUQ(llm=mock_llm, scorers=["entailment", "noncontradiction"], response_refinement=True)

            # Check that a warning was printed
            out, _ = capfd.readouterr()
            assert "claim_filtering_scorer is not specified for response_refinement" in out

            # Check that uad_scorer was set to the first scorer
            assert uq.uad_scorer == "entailment"

    def test_initialization_invalid_claim_filtering_scorer(self, mock_llm, capfd):
        """Test initialization with response_refinement=True and invalid claim_filtering_scorer."""
        with patch("uqlm.scorers.longform.baseclass.uncertainty.ResponseDecomposer"), patch("uqlm.scorers.longform.baseclass.uncertainty.UncertaintyAwareDecoder"):
            uq = LongFormUQ(llm=mock_llm, scorers=["entailment", "noncontradiction"], response_refinement=True, claim_filtering_scorer="invalid_scorer")

            # Check that a warning was printed
            out, _ = capfd.readouterr()
            assert "claim_filtering_scorer is contained in list of scorers" in out

            # Check that uad_scorer was set to the first scorer
            assert uq.uad_scorer == "entailment"

    @pytest.mark.asyncio
    async def test_decompose_responses_claim(self, uq_default, mock_decomposer):
        """Test _decompose_responses method with claim granularity."""
        uq_default.responses = ["Response 1", "Response 2"]
        uq_default.progress_bar = MagicMock(spec=Progress)

        # Patch the _display_decomposition_header method
        with patch.object(LongFormUQ, "_display_decomposition_header") as mock_display_header:
            await uq_default._decompose_responses(show_progress_bars=True)

            # Check that _display_decomposition_header was called
            mock_display_header.assert_called_once_with(True)

            # Check that decompose_claims was called with the right arguments
            mock_decomposer.decompose_claims.assert_called_once_with(responses=["Response 1", "Response 2"], progress_bar=uq_default.progress_bar)

            # Check that claim_sets was set correctly
            assert uq_default.claim_sets == [["Claim 1.1", "Claim 1.2"], ["Claim 2.1"]]

    @pytest.mark.asyncio
    async def test_decompose_responses_sentence(self, uq_default, mock_decomposer):
        """Test _decompose_responses method with sentence granularity."""
        uq_default.responses = ["Response 1", "Response 2"]
        uq_default.progress_bar = MagicMock(spec=Progress)
        uq_default.granularity = "sentence"

        # Patch the _display_decomposition_header method
        with patch.object(LongFormUQ, "_display_decomposition_header") as mock_display_header:
            await uq_default._decompose_responses(show_progress_bars=True)

            # Check that _display_decomposition_header was called
            mock_display_header.assert_called_once_with(True)

            # Check that decompose_sentences was called with the right arguments
            mock_decomposer.decompose_sentences.assert_called_once_with(responses=["Response 1", "Response 2"], progress_bar=uq_default.progress_bar)

            # Check that claim_sets was set correctly
            assert uq_default.claim_sets == [["Sentence 1.1", "Sentence 1.2"], ["Sentence 2.1"]]

    @pytest.mark.asyncio
    async def test_decompose_candidate_responses_claim(self, uq_default, mock_decomposer):
        """Test _decompose_candidate_responses method with claim granularity."""
        uq_default.sampled_responses = [["Sample 1.1", "Sample 1.2"], ["Sample 2.1", "Sample 2.2"]]
        uq_default.progress_bar = MagicMock(spec=Progress)

        await uq_default._decompose_candidate_responses(show_progress_bars=True)

        # Check that decompose_candidate_claims was called with the right arguments
        mock_decomposer.decompose_candidate_claims.assert_called_once_with(sampled_responses=[["Sample 1.1", "Sample 1.2"], ["Sample 2.1", "Sample 2.2"]], progress_bar=uq_default.progress_bar)

        # Check that sampled_claim_sets was set correctly
        assert uq_default.sampled_claim_sets == [[["Claim 1.1.1", "Claim 1.1.2"], ["Claim 1.2.1"]], [["Claim 2.1.1"], ["Claim 2.2.1"]]]

    @pytest.mark.asyncio
    async def test_decompose_candidate_responses_sentence(self, uq_default, mock_decomposer):
        """Test _decompose_candidate_responses method with sentence granularity."""
        uq_default.sampled_responses = [["Sample 1.1", "Sample 1.2"], ["Sample 2.1", "Sample 2.2"]]
        uq_default.progress_bar = MagicMock(spec=Progress)
        uq_default.granularity = "sentence"

        await uq_default._decompose_candidate_responses(show_progress_bars=True)

        # Check that decompose_candidate_sentences was called with the right arguments
        mock_decomposer.decompose_candidate_sentences.assert_called_once_with(sampled_responses=[["Sample 1.1", "Sample 1.2"], ["Sample 2.1", "Sample 2.2"]], progress_bar=uq_default.progress_bar)

        # Check that sampled_claim_sets was set correctly
        assert uq_default.sampled_claim_sets == [[["Sentence 1.1.1", "Sentence 1.1.2"], ["Sentence 1.2.1"]], [["Sentence 2.1.1"], ["Sentence 2.2.1"]]]

    def test_aggregate_scores_mean(self, uq_default):
        """Test _aggregate_scores method with mean aggregation."""
        claim_scores = [[0.8, 0.6], [0.9]]

        result = uq_default._aggregate_scores(claim_scores)

        # Check the result
        assert result == [0.7, 0.9]

    def test_aggregate_scores_min(self, uq_default):
        """Test _aggregate_scores method with min aggregation."""
        uq_default.aggregation = "min"
        claim_scores = [[0.8, 0.6], [0.9]]

        result = uq_default._aggregate_scores(claim_scores)

        # Check the result
        assert result == [0.6, 0.9]

    def test_display_decomposition_header(self, uq_default):
        """Test _display_decomposition_header method."""
        uq_default.progress_bar = MagicMock(spec=Progress)

        uq_default._display_decomposition_header(show_progress_bars=True)

        # Check that progress_bar methods were called
        uq_default.progress_bar.start.assert_called_once()
        assert uq_default.progress_bar.add_task.call_count == 2
        uq_default.progress_bar.add_task.assert_any_call("")
        uq_default.progress_bar.add_task.assert_any_call("✂️ Decomposition")

    def test_display_reconstruction_header(self, uq_default):
        """Test _display_reconstruction_header method."""
        uq_default.progress_bar = MagicMock(spec=Progress)

        uq_default._display_reconstruction_header(show_progress_bars=True)

        # Check that progress_bar methods were called
        uq_default.progress_bar.start.assert_called_once()
        assert uq_default.progress_bar.add_task.call_count == 2
        uq_default.progress_bar.add_task.assert_any_call("")
        uq_default.progress_bar.add_task.assert_any_call("✅️ Refinement")

    @pytest.mark.asyncio
    async def test_uncertainty_aware_decode(self, uq_with_refinement, mock_reconstructor):
        """Test uncertainty_aware_decode method."""
        claim_sets = [["Claim 1.1", "Claim 1.2"], ["Claim 2.1"]]
        claim_scores = [[0.8, 0.3], [0.9]]
        uq_with_refinement.responses = ["Response 1", "Response 2"]
        uq_with_refinement.progress_bar = MagicMock(spec=Progress)

        # Set up claim_scores attribute
        uq_with_refinement.claim_scores = {"entailment": [[0.8, 0.3], [0.9]], "noncontradiction": [[0.9, 0.4], [0.95]]}

        # Set the reconstructor on the UQ object
        uq_with_refinement.reconstructor = mock_reconstructor

        # Patch the necessary methods
        with patch.object(LongFormUQ, "_construct_progress_bar") as mock_construct_progress, patch.object(LongFormUQ, "_display_reconstruction_header") as mock_display_header, patch.object(LongFormUQ, "_stop_progress_bar") as mock_stop_progress, patch.object(LongFormUQ, "_aggregate_scores") as mock_aggregate:
            # Set up return values for _aggregate_scores
            mock_aggregate.side_effect = [0.8, 0.9]  # For entailment and noncontradiction

            result = await uq_with_refinement.uncertainty_aware_decode(claim_sets=claim_sets, claim_scores=claim_scores, response_refinement_threshold=0.5, show_progress_bars=True)

            # Check that methods were called with the right arguments
            mock_construct_progress.assert_called_once_with(True)
            mock_display_header.assert_called_once_with(True)

            # Use assert_called_once() instead of assert_called_once_with() to avoid progress_bar mismatch
            mock_reconstructor.reconstruct_responses.assert_called_once()

            # Check the arguments separately
            args, kwargs = mock_reconstructor.reconstruct_responses.call_args
            assert kwargs["claim_sets"] == claim_sets
            assert kwargs["claim_scores"] == claim_scores
            assert kwargs["responses"] == ["Response 1", "Response 2"]
            assert kwargs["threshold"] == 0.5
            # Don't check progress_bar as it might be different

            mock_stop_progress.assert_called_once()

            # Check that response_refinement_threshold was set
            assert uq_with_refinement.response_refinement_threshold == 0.5

            # Check that _aggregate_scores was called for each scorer with filtered claims
            assert mock_aggregate.call_count == 2
            mock_aggregate.assert_any_call([[0.8], [0.9]])  # entailment (filtered)
            mock_aggregate.assert_any_call([[0.9], [0.95]])  # noncontradiction (filtered)

            # Check the result
            assert result == {"refined_responses": ["Refined Response 1", "Refined Response 2"], "removed": [[False, True], [False]], "refined_entailment": 0.8, "refined_noncontradiction": 0.9}

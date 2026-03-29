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
from langchain_core.messages import AIMessage

from uqlm.longform.uad import UncertaintyAwareDecoder


class TestUncertaintyAwareDecoder:
    @pytest.fixture
    def mock_llm(self):
        """Create a mock LLM for testing."""
        mock = AsyncMock()
        mock.ainvoke.return_value = AIMessage(content="Reconstructed response")
        return mock

    @pytest.fixture
    def decoder(self, mock_llm):
        """Create an UncertaintyAwareDecoder instance with a mock LLM."""
        return UncertaintyAwareDecoder(reconstructor_llm=mock_llm)

    @pytest.mark.asyncio
    async def test_reconstruct_single_response_with_claims(self, decoder):
        """Test reconstructing a single response with claims."""
        claim_set = ["Claim 1", "Claim 2"]
        result = await decoder._reconstruct_single_response(claim_set=claim_set)

        # Verify the LLM was called with the correct prompt
        decoder.reconstructor_llm.ainvoke.assert_called_once()

        # Verify the result is as expected
        assert result == "Reconstructed response"

    @pytest.mark.asyncio
    async def test_reconstruct_single_response_empty_claims(self, decoder):
        """Test reconstructing a single response with empty claims."""
        claim_set = []
        original_response = "Original response"
        result = await decoder._reconstruct_single_response(claim_set=claim_set, response=original_response)

        # Verify the LLM was not called
        decoder.reconstructor_llm.ainvoke.assert_not_called()

        # Verify the original response is returned
        assert result == "Original response"

    @pytest.mark.asyncio
    async def test_reconstruct_single_response_with_progress_bar(self, decoder):
        """Test reconstructing a single response with a progress bar."""
        claim_set = ["Claim 1"]
        progress_bar = MagicMock(spec=Progress)
        decoder.progress_task = "task_id"

        result = await decoder._reconstruct_single_response(claim_set=claim_set, progress_bar=progress_bar)

        # Verify progress bar was updated
        progress_bar.update.assert_called_once_with("task_id", advance=1)

        # Verify the result is as expected
        assert result == "Reconstructed response"

    @pytest.mark.asyncio
    async def test_reconstruct_responses_filtering(self, decoder):
        """Test filtering claims based on threshold."""
        claim_sets = [["Claim 1", "Claim 2", "Claim 3"], ["Claim 4", "Claim 5"]]
        claim_scores = [[0.2, 0.5, 0.1], [0.4, 0.2]]
        responses = ["Original 1", "Original 2"]
        threshold = 0.3

        # Mock the _reconstruct_single_response method
        with patch.object(decoder, "_reconstruct_single_response") as mock_reconstruct:
            mock_reconstruct.side_effect = ["Reconstructed 1", "Reconstructed 2"]

            result = await decoder.reconstruct_responses(claim_sets=claim_sets, claim_scores=claim_scores, responses=responses, threshold=threshold)

        # Verify filtering worked correctly
        assert mock_reconstruct.call_count == 2

        # First call should have only "Claim 2" (score 0.5 > threshold 0.3)
        assert mock_reconstruct.call_args_list[0][1]["claim_set"] == ["Claim 2"]

        # Second call should have only "Claim 4" (score 0.4 > threshold 0.3)
        assert mock_reconstruct.call_args_list[1][1]["claim_set"] == ["Claim 4"]

        # Check the result structure
        assert "refined_responses" in result
        assert "removed" in result
        assert result["refined_responses"] == ["Reconstructed 1", "Reconstructed 2"]
        assert result["removed"] == [[True, False, True], [False, True]]

    @pytest.mark.asyncio
    async def test_reconstruct_responses_empty_claims(self, decoder):
        """Test reconstructing responses with empty filtered claims."""
        claim_sets = [["Claim 1"], ["Claim 2"]]
        claim_scores = [[0.2], [0.2]]  # All below threshold
        responses = ["Original 1", "Original 2"]
        threshold = 0.3

        result = await decoder.reconstruct_responses(claim_sets=claim_sets, claim_scores=claim_scores, responses=responses, threshold=threshold)

        # All claims were filtered out, so original responses should be used
        assert result["refined_responses"] == ["Original 1", "Original 2"]
        assert result["removed"] == [[True], [True]]

    @pytest.mark.asyncio
    async def test_reconstruct_responses_with_progress_bar(self, decoder):
        """Test reconstructing responses with a progress bar."""
        claim_sets = [["Claim 1"], ["Claim 2"]]
        claim_scores = [[0.5], [0.5]]
        progress_bar = MagicMock(spec=Progress)

        with patch.object(decoder, "_reconstruct_single_response") as mock_reconstruct:
            mock_reconstruct.side_effect = ["Reconstructed 1", "Reconstructed 2"]

            await decoder.reconstruct_responses(claim_sets=claim_sets, claim_scores=claim_scores, progress_bar=progress_bar)

        # Verify progress bar was created
        progress_bar.add_task.assert_called_once()

    @pytest.mark.asyncio
    async def test_reconstruct_responses_default_responses(self, decoder):
        """Test reconstructing responses with default (None) responses."""
        claim_sets = [["Claim 1"], ["Claim 2"]]
        claim_scores = [[0.2], [0.2]]  # All below threshold
        threshold = 0.3

        # Mock to verify behavior with None responses
        with patch.object(decoder, "_reconstruct_single_response") as mock_reconstruct:
            mock_reconstruct.side_effect = [None, None]

            result = await decoder.reconstruct_responses(claim_sets=claim_sets, claim_scores=claim_scores, threshold=threshold)

        # Verify None was passed as response
        assert mock_reconstruct.call_args_list[0][1]["response"] is None
        assert mock_reconstruct.call_args_list[1][1]["response"] is None

    @pytest.mark.asyncio
    async def test_initialization(self, mock_llm):
        """Test proper initialization of the decoder."""
        decoder = UncertaintyAwareDecoder(reconstructor_llm=mock_llm)

        assert decoder.reconstructor_llm == mock_llm
        assert decoder.reconstruction_template is not None

    @pytest.mark.asyncio
    async def test_reconstruct_responses_with_different_thresholds(self, decoder):
        """Test reconstructing responses with different thresholds."""
        claim_sets = [["Claim 1", "Claim 2", "Claim 3"]]
        claim_scores = [[0.2, 0.4, 0.6]]

        # Test with threshold 0.3
        with patch.object(decoder, "_reconstruct_single_response") as mock_reconstruct:
            mock_reconstruct.return_value = "Reconstructed 0.3"
            result_0_3 = await decoder.reconstruct_responses(claim_sets=claim_sets, claim_scores=claim_scores, threshold=0.3)

            # Should include claims with scores 0.4 and 0.6
            assert mock_reconstruct.call_args[1]["claim_set"] == ["Claim 2", "Claim 3"]

        # Test with threshold 0.5
        with patch.object(decoder, "_reconstruct_single_response") as mock_reconstruct:
            mock_reconstruct.return_value = "Reconstructed 0.5"
            result_0_5 = await decoder.reconstruct_responses(claim_sets=claim_sets, claim_scores=claim_scores, threshold=0.5)

            # Should include only claim with score 0.6
            assert mock_reconstruct.call_args[1]["claim_set"] == ["Claim 3"]

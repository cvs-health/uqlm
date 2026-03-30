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
import re
from unittest.mock import AsyncMock, MagicMock, patch
from typing import List, Callable
from rich.progress import Progress
from langchain_core.messages import AIMessage

from uqlm.longform.decomposition import ResponseDecomposer


class TestResponseDecomposer:
    @pytest.fixture
    def mock_llm(self):
        """Create a mock LLM for testing."""
        mock = AsyncMock()
        mock.ainvoke.return_value = AIMessage(content="### Claim 1\n### Claim 2")
        return mock

    @pytest.fixture
    def mock_template(self):
        """Create a mock template function."""

        def template_func(response):
            return f"Decompose: {response}"

        return template_func

    @pytest.fixture
    def decomposer(self, mock_llm, mock_template):
        """Create a ResponseDecomposer instance with mock components."""
        return ResponseDecomposer(claim_decomposition_llm=mock_llm, response_template=mock_template)

    def test_initialization(self, mock_llm, mock_template):
        """Test proper initialization of the decomposer."""
        decomposer = ResponseDecomposer(claim_decomposition_llm=mock_llm, response_template=mock_template)

        assert decomposer.claim_decomposition_llm == mock_llm
        assert decomposer.response_template == mock_template

    def test_initialization_defaults(self):
        """Test initialization with default values."""
        decomposer = ResponseDecomposer()

        assert decomposer.claim_decomposition_llm is None
        assert callable(decomposer.response_template)

    def test_decompose_sentences(self):
        """Test decomposing responses into sentences."""
        decomposer = ResponseDecomposer()
        responses = ["This is sentence one. This is sentence two.", "Another response. With multiple sentences!"]

        result = decomposer.decompose_sentences(responses)

        assert len(result) == 2
        assert result[0] == ["This is sentence one.", "This is sentence two."]
        assert result[1] == ["Another response.", "With multiple sentences!"]

    def test_decompose_sentences_with_progress_bar(self):
        """Test decomposing sentences with a progress bar."""
        decomposer = ResponseDecomposer()
        responses = ["Sentence one. Sentence two."]
        progress_bar = MagicMock(spec=Progress)

        decomposer.decompose_sentences(responses, progress_bar=progress_bar)

        # Verify progress bar was created and updated
        progress_bar.add_task.assert_called_once()
        progress_bar.update.assert_called_once()

    def test_decompose_candidate_sentences(self):
        """Test decomposing candidate sentences."""
        decomposer = ResponseDecomposer()
        sampled_responses = [["Response 1.1. More text.", "Response 1.2."], ["Response 2.1.", "Response 2.2."]]

        with patch.object(decomposer, "decompose_sentences") as mock_decompose:
            mock_decompose.side_effect = [[["Response 1.1.", "More text."], ["Response 1.2."]], [["Response 2.1."], ["Response 2.2."]]]

            result = decomposer.decompose_candidate_sentences(sampled_responses)

            assert len(result) == 2
            assert mock_decompose.call_count == 2

    def test_decompose_candidate_sentences_with_progress_bar(self):
        """Test decomposing candidate sentences with a progress bar."""
        decomposer = ResponseDecomposer()
        sampled_responses = [["Response 1"], ["Response 2"]]
        progress_bar = MagicMock(spec=Progress)

        with patch.object(decomposer, "decompose_sentences") as mock_decompose:
            mock_decompose.return_value = [["Sentence"]]

            decomposer.decompose_candidate_sentences(sampled_responses, progress_bar=progress_bar)

            # Verify progress bar was created and updated
            progress_bar.add_task.assert_called_once()
            progress_bar.update.assert_called()

    @pytest.mark.asyncio
    async def test_decompose_claims(self, decomposer):
        """Test decomposing responses into claims."""
        responses = ["Response 1", "Response 2"]

        with patch.object(decomposer, "_decompose_claims") as mock_decompose:
            mock_decompose.return_value = [["Claim 1"], ["Claim 2"]]

            result = await decomposer.decompose_claims(responses)

            mock_decompose.assert_called_once()
            assert result == [["Claim 1"], ["Claim 2"]]

    @pytest.mark.asyncio
    async def test_decompose_claims_with_custom_template(self, decomposer):
        """Test decomposing claims with a custom template."""
        responses = ["Response"]
        custom_template = lambda x: f"Custom: {x}"

        with patch.object(decomposer, "_decompose_claims") as mock_decompose:
            mock_decompose.return_value = [["Claim"]]

            await decomposer.decompose_claims(responses, response_template=custom_template)

            # Verify template was updated
            assert decomposer.response_template == custom_template

    @pytest.mark.asyncio
    async def test_decompose_claims_no_llm(self):
        """Test decomposing claims without an LLM raises an error."""
        decomposer = ResponseDecomposer()  # No LLM provided
        responses = ["Response"]

        with pytest.raises(ValueError, match="llm must be provided"):
            await decomposer.decompose_claims(responses)

    @pytest.mark.asyncio
    async def test_decompose_claims_with_progress_bar(self, decomposer):
        """Test decomposing claims with a progress bar."""
        responses = ["Response"]
        progress_bar = MagicMock(spec=Progress)

        with patch.object(decomposer, "_decompose_claims") as mock_decompose:
            mock_decompose.return_value = [["Claim"]]

            await decomposer.decompose_claims(responses, progress_bar=progress_bar)

            # Verify progress bar was created
            progress_bar.add_task.assert_called_once()

    @pytest.mark.asyncio
    async def test_decompose_candidate_claims(self, decomposer):
        """Test decomposing candidate claims."""
        sampled_responses = [["Response 1.1", "Response 1.2"], ["Response 2.1", "Response 2.2"]]

        with patch.object(decomposer, "_decompose_claims") as mock_decompose:
            mock_decompose.side_effect = [[["Claim 1.1"], ["Claim 1.2"]], [["Claim 2.1"], ["Claim 2.2"]]]

            result = await decomposer.decompose_candidate_claims(sampled_responses)

            assert len(result) == 2
            assert mock_decompose.call_count == 2

    @pytest.mark.asyncio
    async def test_decompose_candidate_claims_no_llm(self):
        """Test decomposing candidate claims without an LLM raises an error."""
        decomposer = ResponseDecomposer()  # No LLM provided
        sampled_responses = [["Response"]]

        with pytest.raises(ValueError, match="llm must be provided"):
            await decomposer.decompose_candidate_claims(sampled_responses)

    @pytest.mark.asyncio
    async def test_decompose_candidate_claims_with_progress_bar(self, decomposer):
        """Test decomposing candidate claims with a progress bar."""
        sampled_responses = [["Response"]]
        progress_bar = MagicMock(spec=Progress)

        with patch.object(decomposer, "_decompose_claims") as mock_decompose:
            mock_decompose.return_value = [["Claim"]]

            await decomposer.decompose_candidate_claims(sampled_responses, progress_bar=progress_bar)

            # Verify progress bar was created
            progress_bar.add_task.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_claims_from_response(self, decomposer):
        """Test extracting claims from a response."""
        response = "Test response"
        decomposer.claim_decomposition_llm.ainvoke.return_value = AIMessage(content="### Claim 1\n### Claim 2\n### Claim 3")

        result = await decomposer._get_claims_from_response(response)

        # Verify LLM was called with correct template
        decomposer.claim_decomposition_llm.ainvoke.assert_called_once_with(decomposer.response_template(response))

        # Verify claims were extracted correctly
        assert result == ["Claim 1", "Claim 2", "Claim 3"]

    @pytest.mark.asyncio
    async def test_get_claims_from_response_with_progress_bar(self, decomposer):
        """Test extracting claims with a progress bar."""
        response = "Test response"
        progress_bar = MagicMock(spec=Progress)
        decomposer.progress_task = "task_id"

        await decomposer._get_claims_from_response(response, progress_bar=progress_bar)

        # Verify progress bar was updated
        progress_bar.update.assert_called_once_with("task_id", advance=1)

    @pytest.mark.asyncio
    async def test_get_claims_from_response_none_response(self, decomposer):
        """Test handling 'NONE' responses."""
        response = "Test response"
        decomposer.claim_decomposition_llm.ainvoke.return_value = AIMessage(content="### NONE")

        result = await decomposer._get_claims_from_response(response)

        # Verify empty list is returned for NONE response
        assert result == []

    def test_get_sentences_from_response(self):
        """Test sentence decomposition with various edge cases."""
        decomposer = ResponseDecomposer()

        # Test basic sentence splitting
        text = "This is sentence one. This is sentence two."
        result = decomposer._get_sentences_from_response(text)
        assert result == ["This is sentence one.", "This is sentence two."]

        # Test with abbreviations
        text = "Dr. Smith went to N.Y. City. He visited the U.S. Capitol."
        result = decomposer._get_sentences_from_response(text)
        assert result == ["Dr. Smith went to N.Y. City.", "He visited the U.S. Capitol."]

        # Test with decimal numbers
        text = "The price is 10.5 dollars. The weight is 3.75 kg."
        result = decomposer._get_sentences_from_response(text)
        assert result == ["The price is 10.5 dollars.", "The weight is 3.75 kg."]

        # Test with exclamation and question marks
        text = "Hello! How are you? I'm fine."
        result = decomposer._get_sentences_from_response(text)
        assert result == ["Hello!", "How are you?", "I'm fine."]

    def test_is_none_response(self):
        """Test detection of 'NONE' responses."""
        decomposer = ResponseDecomposer()

        # Test various forms of NONE responses
        assert decomposer._is_none_response("### NONE")
        assert decomposer._is_none_response("### None")
        assert decomposer._is_none_response("###NONE")
        assert decomposer._is_none_response("### none ###")

        # Test non-NONE responses
        assert not decomposer._is_none_response("### Claim 1")
        assert not decomposer._is_none_response("No claims found")
        assert not decomposer._is_none_response("The word none appears here")

    @pytest.mark.asyncio
    async def test_decompose_claims_helper(self, decomposer):
        """Test the _decompose_claims helper method."""
        responses = ["Response 1", "Response 2"]

        with patch.object(decomposer, "_get_claims_from_response") as mock_get_claims:
            mock_get_claims.side_effect = [["Claim 1"], ["Claim 2"]]

            result = await decomposer._decompose_claims(responses)

            assert mock_get_claims.call_count == 2
            assert result == [["Claim 1"], ["Claim 2"]]

    @pytest.mark.asyncio
    async def test_decompose_claims_helper_with_progress_bar(self, decomposer):
        """Test the _decompose_claims helper with progress bar and matched_claims=False."""
        responses = ["Response"]
        progress_bar = MagicMock(spec=Progress)
        decomposer.progress_task = "task_id"

        with patch.object(decomposer, "_get_claims_from_response") as mock_get_claims:
            mock_get_claims.return_value = ["Claim"]

            await decomposer._decompose_claims(responses, progress_bar=progress_bar, matched_claims=False)

            # Verify progress bar was updated for matched_claims=False
            progress_bar.update.assert_called_once_with("task_id", advance=1)

            # Verify _get_claims_from_response was called with None progress bar
            mock_get_claims.assert_called_once_with(response="Response", progress_bar=None)

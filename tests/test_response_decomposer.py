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
from unittest.mock import AsyncMock, Mock
from uqlm.longform.decomposition.response_decomposer import ResponseDecomposer


class TestResponseDecomposer:
    """Conservative test suite focusing on the three core claim decomposition scenarios."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.mock_llm = AsyncMock()
        self.decomposer = ResponseDecomposer(claim_decomposition_llm=self.mock_llm)

    @pytest.mark.asyncio
    async def test_decompose_claims_proper_format(self):
        """Test scenario 1: LLM responds with proper claim format."""
        mock_response = Mock()
        mock_response.content = "### The sky is blue.\n### Water is wet."
        self.mock_llm.ainvoke.return_value = mock_response

        responses = ["The sky is blue and water is wet."]
        result = await self.decomposer.decompose_claims(responses)

        assert len(result) == 1
        assert result[0] == ["The sky is blue.", "Water is wet."]

    @pytest.mark.asyncio
    async def test_decompose_claims_improper_format(self):
        """Test scenario 2: LLM does not respond in proper format."""
        mock_response = Mock()
        mock_response.content = "The passage contains facts but no ### markers."
        self.mock_llm.ainvoke.return_value = mock_response

        responses = ["Some response text."]
        result = await self.decomposer.decompose_claims(responses)

        assert len(result) == 1
        assert result[0] == []

    @pytest.mark.asyncio
    async def test_decompose_claims_no_claims_found(self):
        """Test scenario 3: LLM doesn't find any claims (responds with ### NONE)."""
        mock_response = Mock()
        mock_response.content = "### NONE"
        self.mock_llm.ainvoke.return_value = mock_response

        responses = ["Hi"]
        result = await self.decomposer.decompose_claims(responses)

        assert len(result) == 1
        assert result[0] == []
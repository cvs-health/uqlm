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
from unittest.mock import Mock
from uqlm.utils.nli import NLI, NLIResult
import numpy as np


class TestNLIInitialization:
    """Test NLI class initialization with different model types."""

    def test_default_initialization(self):
        """Test that NLI initializes with default HuggingFace model."""
        nli = NLI()
        assert nli.is_hf_model is True
        assert nli.max_length == 2000
        assert nli.label_mapping == ["contradiction", "neutral", "entailment"]
        assert nli.tokenizer is not None
        assert nli.model is not None
        assert nli._logprobs_warning_shown is False

    def test_custom_hf_model_initialization(self):
        """Test NLI initialization with custom HuggingFace model name."""
        nli = NLI(nli_model_name="microsoft/deberta-base-mnli")
        assert nli.is_hf_model is True
        assert nli.tokenizer is not None
        assert nli.model is not None

    def test_langchain_model_initialization(self):
        """Test NLI initialization with LangChain model."""
        mock_llm = Mock()
        nli = NLI(nli_llm=mock_llm)
        assert nli.is_hf_model is False
        assert nli.model == mock_llm
        assert nli.tokenizer is None
        assert nli.device is None

    def test_langchain_takes_precedence(self):
        """Test that nli_llm takes precedence over nli_model_name when both provided."""
        mock_llm = Mock()
        nli = NLI(nli_model_name="some-model", nli_llm=mock_llm)
        assert nli.is_hf_model is False
        assert nli.model == mock_llm

    def test_custom_max_length(self):
        """Test initialization with custom max_length."""
        nli = NLI(max_length=1000)
        assert nli.max_length == 1000

    def test_initialization_with_device(self):
        """Test initialization with device specification."""
        nli = NLI(device="cpu")
        assert nli.device == "cpu"


class TestNLIResultModel:
    """Test the NLIResult Pydantic model."""

    def test_binary_result_properties(self):
        """Test binary NLI result and its properties."""
        result = NLIResult(style="binary", binary_label=True, binary_probability=0.95)
        assert result.style == "binary"
        assert result.label is True
        assert result.entailment_probability == 0.95
        assert result.model_dump()["binary_label"] is True

    def test_ternary_result_properties(self):
        """Test ternary NLI result and its properties."""
        result = NLIResult(style="ternary", ternary_label="entailment", ternary_probabilities=(0.1, 0.2, 0.7))
        assert result.style == "ternary"
        assert result.label == "entailment"
        assert result.entailment_probability == 0.7
        assert result.contradiction_probability == 0.1
        assert result.neutral_probability == 0.2


class TestNLIPredictHuggingFaceTernary:
    """Test ternary NLI prediction with HuggingFace models."""

    def test_predict_returns_nli_result_with_probabilities(self):
        """Test that predict returns NLIResult with probabilities."""
        nli = NLI()
        result = nli.predict(hypothesis="The sky is blue.", premise="The sky is blue.", style="ternary", return_probabilities=True)

        assert isinstance(result, NLIResult)
        assert result.style == "ternary"
        assert result.ternary_label in ["contradiction", "neutral", "entailment"]
        assert result.ternary_probabilities is not None
        assert len(result.ternary_probabilities) == 3
        assert sum(result.ternary_probabilities) == pytest.approx(1.0)
        assert all(0 <= p <= 1 for p in result.ternary_probabilities)

    def test_predict_returns_nli_result_without_probabilities(self):
        """Test that predict returns NLIResult without probabilities."""
        nli = NLI()
        result = nli.predict(hypothesis="The sky is blue.", premise="The sky is blue.", style="ternary", return_probabilities=False)

        assert isinstance(result, NLIResult)
        assert result.style == "ternary"
        assert result.ternary_label in ["contradiction", "neutral", "entailment"]
        assert result.ternary_probabilities is None

    def test_predict_entailment_case(self):
        """Test prediction on an entailment case."""
        nli = NLI()
        result = nli.predict(hypothesis="The sky is blue.", premise="The sky is blue.", style="ternary")
        # Identical statements should be entailment
        assert result.label == "entailment"

    def test_predict_semantic_relationships(self):
        """Test prediction captures semantic relationships."""
        nli = NLI()
        # Similar sentences should favor entailment
        result = nli.predict(hypothesis="There is a full moon tonight.", premise="The moon is visible in the sky tonight.", style="ternary")
        assert np.argmax(result.ternary_probabilities) != 2  # Not entailment


class TestNLIPredictHuggingFaceBinary:
    """Test binary NLI prediction with HuggingFace models."""

    def test_predict_binary(self):
        """Test binary prediction with/without probabilities."""
        nli = NLI()

        # With probabilities
        result = nli.predict(hypothesis="The sky is blue.", premise="The sky is blue.", style="binary", return_probabilities=True)
        assert isinstance(result, NLIResult)
        assert result.style == "binary"
        assert result.label is True
        assert 0 <= result.binary_probability <= 1

        # Without probabilities
        result_no_prob = nli.predict(hypothesis="The sky is blue.", premise="The sky is blue.", style="binary", return_probabilities=False)
        assert result_no_prob.binary_probability is None


class TestNLIPredictLangChainTernary:
    """Test ternary NLI prediction with LangChain models."""

    def test_predict_probabilities_with_langchain(self):
        """Test that LangChain model returns NLIResult with probabilities."""
        mock_llm = Mock()

        # Mock responses for p_false, p_neutral, p_true queries
        mock_response_1 = Mock()
        mock_response_1.content = "No"
        mock_response_1.response_metadata = {}

        mock_response_2 = Mock()
        mock_response_2.content = "No"
        mock_response_2.response_metadata = {}

        mock_response_3 = Mock()
        mock_response_3.content = "Yes"
        mock_response_3.response_metadata = {}

        mock_llm.invoke.side_effect = [mock_response_1, mock_response_2, mock_response_3]

        nli = NLI(nli_llm=mock_llm)
        with pytest.warns(UserWarning, match="No logprobs found"):
            result = nli.predict(hypothesis="The sky is blue.", premise="The sky is blue.", style="ternary", return_probabilities=True)

        assert isinstance(result, NLIResult)
        assert result.style == "ternary"
        assert len(result.ternary_probabilities) == 3
        assert sum(result.ternary_probabilities) == pytest.approx(1.0)
        assert mock_llm.invoke.call_count == 3  # Called once for each probability

    def test_predict_probabilities_with_logprobs(self):
        """Test that LangChain model uses logprobs when available."""
        mock_llm = Mock()

        # Mock responses with OpenAI-style logprobs
        mock_responses = []
        for content, logprob in [("No", -0.1), ("No", -0.5), ("Yes", -0.1)]:
            mock_response = Mock()
            mock_response.content = content
            mock_response.response_metadata = {"logprobs": {"content": [{"token": content, "logprob": logprob}]}}
            mock_responses.append(mock_response)

        mock_llm.invoke.side_effect = mock_responses

        nli = NLI(nli_llm=mock_llm)
        result = nli.predict(hypothesis="The sky is blue.", premise="The sky is blue.", style="ternary", return_probabilities=True)

        assert isinstance(result, NLIResult)
        assert sum(result.ternary_probabilities) == pytest.approx(1.0)
        # Logprobs should give non-binary values
        assert not all(p in [0.0, 1.0] for p in result.ternary_probabilities)

    def test_predict_class_with_langchain(self):
        """Test that LangChain model returns NLIResult with class label."""
        mock_llm = Mock()
        mock_response = Mock()
        mock_response.content = "entailment"
        mock_llm.invoke.return_value = mock_response

        nli = NLI(nli_llm=mock_llm)
        result = nli.predict(hypothesis="The sky is blue.", premise="The sky is blue.", style="ternary", return_probabilities=False)

        assert isinstance(result, NLIResult)
        assert result.label == "entailment"
        assert result.ternary_probabilities is None
        assert mock_llm.invoke.call_count == 1

    def test_predict_semantic_relationships(self):
        """Test prediction captures semantic relationships."""
        nli = NLI()
        # Similar sentences should favor entailment
        result = nli.predict(hypothesis="There is a full moon tonight.", premise="The moon is visible in the sky tonight.", style="ternary")
        assert np.argmax(result.ternary_probabilities) != 2  # Not entailment

    def test_predict_error_handling(self):
        """Test that errors and unclear responses are handled gracefully."""
        mock_llm = Mock()

        # Test unclear response
        mock_response = Mock()
        mock_response.content = "I don't know"
        mock_llm.invoke.return_value = mock_response

        nli = NLI(nli_llm=mock_llm)
        with pytest.warns(UserWarning, match="Unclear NLI response"):
            result = nli.predict(hypothesis="Test", premise="Test", style="ternary", return_probabilities=False)
        assert result.label == "neutral"

        # Test exception handling
        mock_llm.invoke.side_effect = Exception("API Error")
        with pytest.warns(UserWarning):
            result = nli.predict(hypothesis="Test", premise="Test", style="ternary", return_probabilities=True)
        assert result.ternary_probabilities == pytest.approx((1 / 3, 1 / 3, 1 / 3))


class TestNLIPredictLangChainBinary:
    """Test binary NLI prediction with LangChain models."""

    def test_predict_binary(self):
        """Test binary prediction with LangChain."""
        mock_llm = Mock()

        # Test with probabilities and logprobs
        mock_response = Mock()
        mock_response.content = "Yes"
        mock_response.response_metadata = {"logprobs": {"content": [{"token": "Yes", "logprob": -0.1}]}}
        mock_llm.invoke.return_value = mock_response

        nli = NLI(nli_llm=mock_llm)
        result = nli.predict(hypothesis="Test", premise="Test", style="binary", return_probabilities=True)

        assert isinstance(result, NLIResult)
        assert result.binary_label is True
        assert result.binary_probability != 1.0  # Uses logprob, not binary

    def test_predict_binary_inverts_probability_for_no(self):
        """Test that No responses invert probability to represent entailment probability."""
        mock_llm = Mock()
        mock_response = Mock()
        mock_response.content = "No"
        mock_response.response_metadata = {"logprobs": {"content": [{"token": "No", "logprob": -0.1}]}}
        mock_llm.invoke.return_value = mock_response

        nli = NLI(nli_llm=mock_llm)
        result = nli.predict(hypothesis="Test", premise="Test", style="binary", return_probabilities=True)

        assert result.binary_label is False
        assert result.binary_probability < 0.2  # Inverted from high No probability


class TestNLIEdgeCases:
    """Test edge cases and error handling."""

    def test_long_text_truncation_warning(self):
        """Test that warning is issued for texts exceeding max_length."""
        nli = NLI(max_length=50)
        long_text = "This is a very long sentence " * 20

        with pytest.warns(UserWarning, match="Maximum response length exceeded"):
            nli.predict(hypothesis=long_text, premise="Short text")

    def test_logprobs_warning_shown_once(self):
        """Test that logprobs warning is only shown once."""
        import warnings as warnings_module

        mock_llm = Mock()
        mock_response = Mock()
        mock_response.content = "Yes"
        mock_response.response_metadata = {}  # No logprobs
        mock_llm.invoke.return_value = mock_response

        nli = NLI(nli_llm=mock_llm)

        # First call should show warning
        with pytest.warns(UserWarning, match="No logprobs found"):
            nli.predict("test", "test", style="binary", return_probabilities=True)

        # Second call should not show warning
        with warnings_module.catch_warnings(record=True) as warning_list:
            warnings_module.simplefilter("always")
            nli.predict("test", "test", style="binary", return_probabilities=True)
            assert not any("No logprobs found" in str(w.message) for w in warning_list)


@pytest.fixture
def mock_async_llm():
    """Fixture for a mock LangChain LLM with async support."""

    async def ainvoke_response(messages):
        mock_response = Mock()
        mock_response.content = "Yes"
        mock_response.response_metadata = {}
        return mock_response

    mock_llm = Mock()
    mock_llm.ainvoke = ainvoke_response
    return mock_llm


class TestNLIAsyncMethods:
    """Test async NLI prediction methods."""

    @pytest.mark.asyncio
    async def test_apredict_with_hf_model(self):
        """Test async prediction with HuggingFace models."""
        nli = NLI()
        result = await nli.apredict("The sky is blue.", "The sky is blue.")

        assert isinstance(result, NLIResult)
        assert sum(result.ternary_probabilities) == pytest.approx(1.0)

    @pytest.mark.asyncio
    async def test_apredict_with_langchain(self, mock_async_llm):
        """Test async prediction with LangChain model."""
        nli = NLI(nli_llm=mock_async_llm)
        with pytest.warns(UserWarning, match="No logprobs found"):
            result = await nli.apredict("Test", "Test", return_probabilities=True)

        assert isinstance(result, NLIResult)
        assert result.ternary_probabilities is not None

    @pytest.mark.asyncio
    async def test_apredict_error_handling(self):
        """Test async prediction handles errors gracefully."""
        mock_llm = Mock()

        async def ainvoke_error(messages):
            raise Exception("API Error")

        mock_llm.ainvoke = ainvoke_error
        nli = NLI(nli_llm=mock_llm)

        with pytest.warns(UserWarning):
            result = await nli.apredict("Test", "Test", return_probabilities=True)

        assert result.ternary_probabilities == pytest.approx((1 / 3, 1 / 3, 1 / 3))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

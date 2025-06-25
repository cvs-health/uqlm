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

import json
from unittest.mock import MagicMock, patch

import pytest
from langchain_openai import AzureChatOpenAI

from uqlm.utils.llm_config import save_llm_config, load_llm_config, _is_serializable


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
class TestLLMConfigUtils:
    def setup_method(self):
        """Set up test fixtures"""
        self.mock_llm = AzureChatOpenAI(deployment_name="test-deployment", temperature=0.7, max_tokens=1024, api_key="test-key", api_version="2024-05-01-preview", azure_endpoint="https://test.endpoint.com")

    @pytest.mark.filterwarnings("ignore::DeprecationWarning")
    def test_is_serializable(self):
        """Test _is_serializable helper function"""
        # Test serializable values
        assert _is_serializable("string") is True
        assert _is_serializable(123) is True
        assert _is_serializable(0.5) is True
        assert _is_serializable(True) is True
        assert _is_serializable(False) is True
        assert _is_serializable(None) is True
        assert _is_serializable([1, 2, 3]) is True
        assert _is_serializable({"key": "value"}) is True

        # Test non-serializable values
        assert _is_serializable(lambda x: x) is False
        assert _is_serializable(object()) is False
        assert _is_serializable(MagicMock()) is False

    @pytest.mark.filterwarnings("ignore::DeprecationWarning")
    def test_save_llm_config_basic(self):
        """Test save_llm_config saves basic configuration"""
        config = save_llm_config(self.mock_llm)

        # Check required fields
        assert config["class_name"] == "AzureChatOpenAI"
        assert config["module"] == "langchain_openai.chat_models.azure"

        # Check saved parameters
        assert config["temperature"] == 0.7
        assert config["max_tokens"] == 1024
        assert config["deployment_name"] == "test-deployment"
        # Note: api_key might not be saved as it's often a private attribute
        # Note: azure_endpoint should not be saved as it's excluded from configs

    @pytest.mark.filterwarnings("ignore::DeprecationWarning")
    def test_save_llm_config_filters_internal_attrs(self):
        """Test save_llm_config filters out internal LangChain attributes"""
        # Create a mock LLM with internal attributes
        mock_llm = MagicMock()
        mock_llm.__class__.__name__ = "TestLLM"
        mock_llm.__class__.__module__ = "test.module"
        mock_llm.temperature = 0.5
        mock_llm.config_specs = ["internal"]  # Should be filtered out
        mock_llm.lc_attributes = {"internal": "data"}  # Should be filtered out
        mock_llm.model_config = {"internal": "config"}  # Should be filtered out

        config = save_llm_config(mock_llm)

        # Internal attributes should be excluded
        assert "config_specs" not in config
        assert "lc_attributes" not in config
        assert "model_config" not in config

        # Valid attributes should be included
        assert config["temperature"] == 0.5
        assert config["class_name"] == "TestLLM"
        assert config["module"] == "test.module"

    @pytest.mark.filterwarnings("ignore::DeprecationWarning")
    def test_save_llm_config_filters_private_attrs(self):
        """Test save_llm_config filters out private attributes"""
        mock_llm = MagicMock()
        mock_llm.__class__.__name__ = "TestLLM"
        mock_llm.__class__.__module__ = "test.module"
        mock_llm.temperature = 0.5
        mock_llm._private_attr = "private"  # Should be filtered out
        mock_llm.__special_attr = "special"  # Should be filtered out

        config = save_llm_config(mock_llm)

        # Private attributes should be excluded
        assert "_private_attr" not in config
        assert "__special_attr" not in config

        # Valid attributes should be included
        assert config["temperature"] == 0.5

    @pytest.mark.filterwarnings("ignore::DeprecationWarning")
    def test_save_llm_config_filters_callable_attrs(self):
        """Test save_llm_config filters out callable attributes"""
        mock_llm = MagicMock()
        mock_llm.__class__.__name__ = "TestLLM"
        mock_llm.__class__.__module__ = "test.module"
        mock_llm.temperature = 0.5
        mock_llm.method = lambda x: x  # Should be filtered out
        mock_llm.function = MagicMock()  # Should be filtered out

        config = save_llm_config(mock_llm)

        # Callable attributes should be excluded
        assert "method" not in config
        assert "function" not in config

        # Valid attributes should be included
        assert config["temperature"] == 0.5

    @pytest.mark.filterwarnings("ignore::DeprecationWarning")
    def test_save_llm_config_handles_none_values(self):
        """Test save_llm_config excludes None values"""
        mock_llm = MagicMock()
        mock_llm.__class__.__name__ = "TestLLM"
        mock_llm.__class__.__module__ = "test.module"
        mock_llm.temperature = 0.5
        mock_llm.max_tokens = None  # Should be excluded
        mock_llm.api_key = None  # Should be excluded

        config = save_llm_config(mock_llm)

        # None values should be excluded
        assert "max_tokens" not in config
        assert "api_key" not in config

        # Valid values should be included
        assert config["temperature"] == 0.5

    @pytest.mark.filterwarnings("ignore::DeprecationWarning")
    def test_save_llm_config_handles_non_serializable_values(self):
        """Test save_llm_config excludes non-serializable values"""
        mock_llm = MagicMock()
        mock_llm.__class__.__name__ = "TestLLM"
        mock_llm.__class__.__module__ = "test.module"
        mock_llm.temperature = 0.5
        mock_llm.complex_obj = object()  # Non-serializable
        mock_llm.lambda_func = lambda x: x  # Non-serializable

        config = save_llm_config(mock_llm)

        # Non-serializable values should be excluded
        assert "complex_obj" not in config
        assert "lambda_func" not in config

        # Valid values should be included
        assert config["temperature"] == 0.5

    @pytest.mark.filterwarnings("ignore::DeprecationWarning")
    def test_save_llm_config_handles_attribute_errors(self):
        """Test save_llm_config handles AttributeError gracefully"""
        mock_llm = MagicMock()
        mock_llm.__class__.__name__ = "TestLLM"
        mock_llm.__class__.__module__ = "test.module"
        mock_llm.temperature = 0.5

        # Mock an attribute that raises AttributeError when accessed
        type(mock_llm).problematic_attr = MagicMock(side_effect=AttributeError("test"))

        config = save_llm_config(mock_llm)

        # Should not crash and should include valid attributes
        assert config["temperature"] == 0.5
        assert "problematic_attr" not in config

    @patch.dict("os.environ", {"AZURE_OPENAI_API_KEY": "test-key", "AZURE_OPENAI_ENDPOINT": "https://test.endpoint.com"})
    @pytest.mark.filterwarnings("ignore::DeprecationWarning")
    def test_load_llm_config_success(self):
        """Test load_llm_config successfully recreates LLM"""
        config = {"class_name": "AzureChatOpenAI", "module": "langchain_openai.chat_models.azure", "temperature": 0.5, "max_tokens": 512, "api_key": "test-key", "azure_endpoint": "https://test.endpoint.com", "api_version": "2024-05-01-preview"}

        recreated_llm = load_llm_config(config)

        assert isinstance(recreated_llm, AzureChatOpenAI)
        assert recreated_llm.temperature == 0.5
        assert recreated_llm.max_tokens == 512
        # Note: api_key might not be directly accessible as an attribute
        assert recreated_llm.azure_endpoint == "https://test.endpoint.com"

    @pytest.mark.filterwarnings("ignore::DeprecationWarning")
    def test_load_llm_config_import_error(self):
        """Test load_llm_config handles import errors"""
        config = {"class_name": "NonExistentLLM", "module": "non.existent.module", "temperature": 0.5}

        with pytest.raises(ValueError, match="Could not recreate LLM from config"):
            load_llm_config(config)

    @pytest.mark.filterwarnings("ignore::DeprecationWarning")
    def test_load_llm_config_class_not_found(self):
        """Test load_llm_config handles missing class"""
        config = {
            "class_name": "NonExistentClass",
            "module": "langchain_openai",  # Valid module
            "temperature": 0.5,
        }

        with pytest.raises(ValueError, match="Could not recreate LLM from config"):
            load_llm_config(config)

    @pytest.mark.filterwarnings("ignore::DeprecationWarning")
    def test_load_llm_config_missing_required_fields(self):
        """Test load_llm_config handles missing required fields"""
        # Missing class_name
        config = {"module": "langchain_openai.chat_models.azure", "temperature": 0.5}

        with pytest.raises(ValueError, match="Could not recreate LLM from config"):
            load_llm_config(config)

        # Missing module
        config = {"class_name": "AzureChatOpenAI", "temperature": 0.5}

        with pytest.raises(ValueError, match="Could not recreate LLM from config"):
            load_llm_config(config)

    @pytest.mark.filterwarnings("ignore::DeprecationWarning")
    @pytest.mark.filterwarnings("ignore::UserWarning")
    def test_load_llm_config_constructor_error(self):
        """Test load_llm_config handles constructor errors"""
        config = {
            "class_name": "AzureChatOpenAI",
            "module": "langchain_openai.chat_models.azure",
            "invalid_param": "invalid_value",  # Invalid parameter
        }

        with pytest.raises(ValueError, match="Could not recreate LLM from config"):
            load_llm_config(config)

    @patch.dict("os.environ", {"AZURE_OPENAI_API_KEY": "test-key", "AZURE_OPENAI_ENDPOINT": "https://test.endpoint.com"})
    @pytest.mark.filterwarnings("ignore::DeprecationWarning")
    def test_save_load_roundtrip(self):
        """Test complete save and load roundtrip"""
        # Create an AzureChatOpenAI instance
        azure_llm = AzureChatOpenAI(deployment_name="gpt-4o", openai_api_type="azure", openai_api_version="2024-02-15-preview", temperature=1, api_key="test-api-key", azure_endpoint="https://test.azure.endpoint.com")

        # Save the configuration
        config = save_llm_config(azure_llm)

        # Verify key parameters are saved
        assert config["class_name"] == "AzureChatOpenAI"
        assert config["module"] == "langchain_openai.chat_models.azure"
        assert config["deployment_name"] == "gpt-4o"
        assert config["openai_api_type"] == "azure"
        assert config["openai_api_version"] == "2024-02-15-preview"
        assert config["temperature"] == 1
        # Note: api_key might not be saved as it's often a private attribute
        # Note: azure_endpoint should not be saved as it's excluded from configs

        # Load and recreate the LLM
        recreated_llm = load_llm_config(config)

        # Verify the recreated LLM has the same parameters
        assert isinstance(recreated_llm, AzureChatOpenAI)
        assert recreated_llm.deployment_name == "gpt-4o"
        assert recreated_llm.openai_api_type == "azure"
        assert recreated_llm.openai_api_version == "2024-02-15-preview"
        assert recreated_llm.temperature == 1
        # Note: api_key might not be directly accessible as an attribute
        # Note: azure_endpoint should be loaded from environment variables

    @pytest.mark.filterwarnings("ignore::DeprecationWarning")
    def test_save_llm_config_with_complex_data_structures(self):
        """Test save_llm_config handles complex data structures"""
        mock_llm = MagicMock()
        mock_llm.__class__.__name__ = "TestLLM"
        mock_llm.__class__.__module__ = "test.module"
        mock_llm.temperature = 0.5
        mock_llm.list_param = [1, 2, 3]
        mock_llm.dict_param = {"key": "value", "nested": {"data": 123}}
        mock_llm.tuple_param = (1, 2, 3)

        config = save_llm_config(mock_llm)

        # Complex data structures should be saved
        assert config["list_param"] == [1, 2, 3]
        assert config["dict_param"] == {"key": "value", "nested": {"data": 123}}
        assert config["tuple_param"] == (1, 2, 3)
        assert config["temperature"] == 0.5

    @pytest.mark.filterwarnings("ignore::DeprecationWarning")
    def test_save_llm_config_json_serialization(self):
        """Test that saved config is JSON serializable"""
        config = save_llm_config(self.mock_llm)

        # Should be able to serialize to JSON
        json_str = json.dumps(config)
        assert isinstance(json_str, str)

        # Should be able to deserialize back
        deserialized = json.loads(json_str)
        assert deserialized["class_name"] == config["class_name"]
        assert deserialized["module"] == config["module"]
        assert deserialized["temperature"] == config["temperature"]

    @pytest.mark.filterwarnings("ignore::DeprecationWarning")
    def test_save_llm_config_excludes_endpoint_parameters(self):
        """Test save_llm_config excludes endpoint parameters"""
        mock_llm = MagicMock()
        mock_llm.__class__.__name__ = "TestLLM"
        mock_llm.__class__.__module__ = "test.module"
        mock_llm.temperature = 0.5

        # Add various endpoint-related attributes that should be excluded
        mock_llm.base_url = "https://api.example.com"
        mock_llm.endpoint = "https://endpoint.example.com"
        mock_llm.azure_endpoint = "https://azure.example.com"
        mock_llm.openai_api_base = "https://openai.example.com"
        mock_llm.api_base = "https://api-base.example.com"
        mock_llm.api_url = "https://api-url.example.com"
        mock_llm.url = "https://url.example.com"

        config = save_llm_config(mock_llm)

        # Endpoint parameters should be excluded
        assert "base_url" not in config
        assert "endpoint" not in config
        assert "azure_endpoint" not in config
        assert "openai_api_base" not in config
        assert "api_base" not in config
        assert "api_url" not in config
        assert "url" not in config

        # Valid parameters should be included
        assert config["temperature"] == 0.5
        assert config["class_name"] == "TestLLM"
        assert config["module"] == "test.module"

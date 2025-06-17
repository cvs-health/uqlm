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
import json
import tempfile
import os
from unittest.mock import patch, MagicMock
from uqlm.scorers import UQEnsemble
from uqlm.scorers.baseclass.uncertainty import UQResult
from langchain_openai import AzureChatOpenAI

datafile_path = "tests/data/scorers/ensemble_results_file.json"
with open(datafile_path, "r") as f:
    expected_result = json.load(f)

data = expected_result["data"]
metadata = expected_result["metadata"]
    
PROMPTS = data["prompts"]
MOCKED_RESPONSES = data["responses"]
MOCKED_SAMPLED_RESPONSES = data["sampled_responses"]
MOCKED_JUDGE_SCORES = data['judge_1']
MOCKED_LOGPROBS = metadata["logprobs"]
    
mock_object = AzureChatOpenAI(
    deployment_name="YOUR-DEPLOYMENT",
    temperature=1,
    api_key="SECRET_API_KEY",
    api_version="2024-05-01-preview",
    azure_endpoint="https://mocked.endpoint.com",
)

@pytest.mark.asyncio
async def test_ensemble(monkeypatch):
    uqe = UQEnsemble(
        llm=mock_object,
        scorers=[
            "exact_match",  
            "noncontradiction",  
            "min_probability",  
            mock_object,  
        ]
    )
    
    async def mock_generate_original_responses(*args, **kwargs):
        uqe.logprobs = MOCKED_LOGPROBS
        return MOCKED_RESPONSES
    
    async def mock_generate_candidate_responses(*args, **kwargs):
        uqe.multiple_logprobs = [MOCKED_LOGPROBS] * 5
        return MOCKED_SAMPLED_RESPONSES
    
    async def mock_judge_scores(*args, **kwargs):
        return UQResult({'data': {'judge_1': MOCKED_JUDGE_SCORES}})
    
    monkeypatch.setattr(uqe, "generate_original_responses", mock_generate_original_responses)
    monkeypatch.setattr(uqe, "generate_candidate_responses", mock_generate_candidate_responses)
    monkeypatch.setattr(uqe.judges_object, "score", mock_judge_scores)
    
    results = await uqe.generate_and_score(
        prompts=PROMPTS,
        num_responses=5,
    )
    
    assert all(
        [
            results.data["ensemble_scores"][i]
            == pytest.approx(data["ensemble_scores"][i])
            for i in range(len(PROMPTS))
        ]
    )

    assert all(
        [
            results.data["min_probability"][i]
            == pytest.approx(data["min_probability"][i])
            for i in range(len(PROMPTS))
        ]
    )

    assert all(
        [
            results.data["exact_match"][i] == pytest.approx(data["exact_match"][i])
            for i in range(len(PROMPTS))
        ]
    )

    assert all(
        [
            results.data["noncontradiction"][i]
            == pytest.approx(data["noncontradiction"][i])
            for i in range(len(PROMPTS))
        ]
    )

    assert all(
        [
            abs(results.data["judge_1"][i] - data["judge_1"][i]) < 1e-5
            for i in range(len(PROMPTS))
        ]
    )

    assert results.metadata == metadata


class TestUQEnsembleConfig:
    """Test suite for save/load configuration functionality"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.mock_llm = AzureChatOpenAI(
            deployment_name="test-deployment",
            temperature=0.7,
            max_tokens=1024,
            api_key="test-key",
            api_version="2024-05-01-preview",
            azure_endpoint="https://test.endpoint.com",
        )
        
    def test_save_llm_config(self):
        """Test _save_llm_config static method"""
        config = UQEnsemble._save_llm_config(self.mock_llm)
        
        assert config["class_name"] == "AzureChatOpenAI"
        assert config["module"] == "langchain_openai.chat_models.azure"
        assert config["temperature"] == 0.7
        assert config["max_tokens"] == 1024
        
    def test_save_llm_config_with_none_values(self):
        """Test _save_llm_config handles None values correctly"""
        mock_llm = MagicMock()
        mock_llm.__class__.__name__ = "TestLLM"
        mock_llm.__class__.__module__ = "test.module"
        mock_llm.temperature = 0.5
        mock_llm.max_tokens = None  # This should be excluded
        
        config = UQEnsemble._save_llm_config(mock_llm)
        
        assert "max_tokens" not in config  # None values should be excluded
        assert config["temperature"] == 0.5
        assert config["class_name"] == "TestLLM"
        assert config["module"] == "test.module"
        
    @patch.dict(os.environ, {"AZURE_OPENAI_API_KEY": "test-key", "AZURE_OPENAI_ENDPOINT": "https://test.endpoint.com"})
    def test_load_llm_config_success(self):
        """Test _load_llm_config successfully recreates LLM"""
        config = {
            "class_name": "AzureChatOpenAI",
            "module": "langchain_openai.chat_models.azure",
            "temperature": 0.5,
            "max_tokens": 512,
            "api_key": "test-key",
            "azure_endpoint": "https://test.endpoint.com",
            "api_version": "2024-05-01-preview"
        }
        
        recreated_llm = UQEnsemble._load_llm_config(config)
        
        assert isinstance(recreated_llm, AzureChatOpenAI)
        assert recreated_llm.temperature == 0.5
        assert recreated_llm.max_tokens == 512
        
    def test_load_llm_config_import_error(self):
        """Test _load_llm_config handles import errors"""
        config = {
            "class_name": "NonExistentLLM",
            "module": "non.existent.module",
            "temperature": 0.5
        }
        
        with pytest.raises(ValueError, match="Could not recreate LLM from config"):
            UQEnsemble._load_llm_config(config)
            
    def test_load_llm_config_class_not_found(self):
        """Test _load_llm_config handles missing class"""
        config = {
            "class_name": "NonExistentClass",
            "module": "langchain_openai",  # Valid module
            "temperature": 0.5
        }
        
        with pytest.raises(ValueError, match="Could not recreate LLM from config"):
            UQEnsemble._load_llm_config(config)
            
    def test_save_config_named_components_only(self):
        """Test save_config with only named string components"""
        ensemble = UQEnsemble(
            llm=self.mock_llm,
            scorers=["exact_match", "noncontradiction"],
            weights=[0.6, 0.4],
            thresh=0.75
        )
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            config_path = f.name
            
        try:
            ensemble.save_config(config_path)
            
            with open(config_path, 'r') as f:
                saved_config = json.load(f)
                
            assert saved_config["weights"] == [0.6, 0.4]
            assert saved_config["thresh"] == 0.75
            assert saved_config["components"] == ["exact_match", "noncontradiction"]
            assert saved_config["llm_config"]["class_name"] == "AzureChatOpenAI"
            assert saved_config["llm_config"]["module"] == "langchain_openai.chat_models.azure"
            assert saved_config["llm_scorers"] == {}
            
        finally:
            os.unlink(config_path)
            
    def test_save_config_with_llm_scorers(self):
        """Test save_config with LLM scorer components"""
        judge_llm = AzureChatOpenAI(
            deployment_name="judge-deployment",
            temperature=0.3,
            max_tokens=256,
            api_key="judge-key",
            api_version="2024-05-01-preview",
            azure_endpoint="https://judge.endpoint.com",
        )
        
        ensemble = UQEnsemble(
            llm=self.mock_llm,
            scorers=["exact_match", judge_llm],
            weights=[0.7, 0.3],
            thresh=0.6
        )
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            config_path = f.name
            
        try:
            ensemble.save_config(config_path)
            
            with open(config_path, 'r') as f:
                saved_config = json.load(f)
                
            assert saved_config["components"] == ["exact_match", "judge_1"]
            assert "judge_1" in saved_config["llm_scorers"]
            assert saved_config["llm_scorers"]["judge_1"]["temperature"] == 0.3
            assert saved_config["llm_scorers"]["judge_1"]["max_tokens"] == 256
            assert saved_config["llm_scorers"]["judge_1"]["module"] == "langchain_openai.chat_models.azure"
            
        finally:
            os.unlink(config_path)
            
    def test_save_config_no_main_llm(self):
        """Test save_config when no main LLM is provided"""
        ensemble = UQEnsemble(
            scorers=["exact_match", "noncontradiction"],
            weights=[0.5, 0.5],
            thresh=0.8
        )
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            config_path = f.name
            
        try:
            ensemble.save_config(config_path)
            
            with open(config_path, 'r') as f:
                saved_config = json.load(f)
                
            assert saved_config["llm_config"] is None
            
        finally:
            os.unlink(config_path)
            
    def test_save_config_invalid_component(self):
        """Test save_config with invalid component type"""
        invalid_component = {"invalid": "component"}
        
        ensemble = UQEnsemble(scorers=["exact_match"])
        ensemble.components = ["exact_match", invalid_component]  # Manually add invalid component
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            config_path = f.name
            
        try:
            with pytest.raises(ValueError, match="Cannot serialize component"):
                ensemble.save_config(config_path)
                
        finally:
            if os.path.exists(config_path):
                os.unlink(config_path)
                
    @patch.dict(os.environ, {"AZURE_OPENAI_API_KEY": "test-key", "AZURE_OPENAI_ENDPOINT": "https://test.endpoint.com"})
    def test_load_config_named_components_only(self):
        """Test load_config with only named components"""
        config = {
            "weights": [0.6, 0.4],
            "thresh": 0.75,
            "components": ["exact_match", "noncontradiction"],
            "llm_config": {
                "class_name": "AzureChatOpenAI",
                "module": "langchain_openai.chat_models.azure",
                "temperature": 0.5,
                "max_tokens": 512,
                "api_key": "test-key",
                "azure_endpoint": "https://test.endpoint.com",
                "api_version": "2024-05-01-preview"
            },
            "llm_scorers": {}
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            config_path = f.name
            json.dump(config, f)
            
        try:
            ensemble = UQEnsemble.load_config(config_path)
            
            assert ensemble.weights == [0.6, 0.4]
            assert ensemble.thresh == 0.75
            assert ensemble.components == ["exact_match", "noncontradiction"]
            assert ensemble.llm.temperature == 0.5
            assert ensemble.llm.max_tokens == 512
            
        finally:
            os.unlink(config_path)
            
    def test_load_config_with_provided_llm(self):
        """Test load_config with externally provided LLM"""
        config = {
            "weights": [0.6, 0.4],
            "thresh": 0.75,
            "components": ["exact_match", "noncontradiction"],
            "llm_config": {
                "class_name": "AzureChatOpenAI",
                "module": "langchain_openai.chat_models.azure",
                "temperature": 0.5
            },
            "llm_scorers": {}
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            config_path = f.name
            json.dump(config, f)
            
        try:
            external_llm = AzureChatOpenAI(
                deployment_name="external",
                temperature=0.9,
                api_key="external-key",
                api_version="2024-05-01-preview",
                azure_endpoint="https://external.endpoint.com",
            )
            
            ensemble = UQEnsemble.load_config(config_path, llm=external_llm)
            
            # Should use the provided LLM, not the one from config
            assert ensemble.llm.temperature == 0.9
            assert ensemble.llm.deployment_name == "external"
            
        finally:
            os.unlink(config_path)
            
    @patch.dict(os.environ, {"AZURE_OPENAI_API_KEY": "test-key", "AZURE_OPENAI_ENDPOINT": "https://test.endpoint.com"})
    def test_load_config_with_llm_scorers(self):
        """Test load_config with LLM scorer components"""
        config = {
            "weights": [0.7, 0.3],
            "thresh": 0.6,
            "components": ["exact_match", "judge_1"],
            "llm_config": {
                "class_name": "AzureChatOpenAI",
                "module": "langchain_openai.chat_models.azure",
                "temperature": 0.7,
                "api_key": "test-key",
                "azure_endpoint": "https://test.endpoint.com",
                "api_version": "2024-05-01-preview"
            },
            "llm_scorers": {
                "judge_1": {
                    "class_name": "AzureChatOpenAI",
                    "module": "langchain_openai.chat_models.azure",
                    "temperature": 0.3,
                    "max_tokens": 256,
                    "api_key": "test-key",
                    "azure_endpoint": "https://test.endpoint.com",
                    "api_version": "2024-05-01-preview"
                }
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            config_path = f.name
            json.dump(config, f)
            
        try:
            ensemble = UQEnsemble.load_config(config_path)
            
            assert len(ensemble.components) == 2
            assert ensemble.components[0] == "exact_match"
            assert isinstance(ensemble.components[1], AzureChatOpenAI)
            assert ensemble.components[1].temperature == 0.3
            assert ensemble.components[1].max_tokens == 256
            
        finally:
            os.unlink(config_path)
            
    def test_load_config_missing_llm_scorer(self):
        """Test load_config with missing LLM scorer configuration"""
        config = {
            "weights": [0.7, 0.3],
            "thresh": 0.6,
            "components": ["exact_match", "judge_1"],
            "llm_config": None,
            "llm_scorers": {}  # Missing judge_1 config
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            config_path = f.name
            json.dump(config, f)
            
        try:
            with pytest.raises(ValueError, match="Missing LLM config for judge_1"):
                UQEnsemble.load_config(config_path)
                
        finally:
            os.unlink(config_path)
            
    def test_load_config_no_llm_config(self):
        """Test load_config when no LLM config is provided and no external LLM"""
        config = {
            "weights": [1.0],
            "thresh": 0.5,
            "components": ["exact_match"],
            "llm_config": None,
            "llm_scorers": {}
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            config_path = f.name
            json.dump(config, f)
            
        try:
            ensemble = UQEnsemble.load_config(config_path)
            assert ensemble.llm is None
            
        finally:
            os.unlink(config_path)
            
    def test_load_config_file_not_found(self):
        """Test load_config with non-existent file"""
        with pytest.raises(FileNotFoundError):
            UQEnsemble.load_config("non_existent_config.json")
            
    def test_load_config_invalid_json(self):
        """Test load_config with invalid JSON file"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            config_path = f.name
            f.write("invalid json content")
            
        try:
            with pytest.raises(json.JSONDecodeError):
                UQEnsemble.load_config(config_path)
                
        finally:
            os.unlink(config_path)
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
from importlib import import_module
from typing import Any, Dict
from langchain_core.language_models.chat_models import BaseChatModel


def _is_serializable(value: Any) -> bool:
    """Check if a value is JSON serializable."""
    try:
        json.dumps(value)
        return True
    except (TypeError, ValueError):
        return False


def save_llm_config(llm: BaseChatModel) -> Dict[str, Any]:
    """
    Extract and save LLM configuration by capturing all available parameters.

    Parameters
    ----------
    llm : BaseChatModel
        The LLM instance to extract config from

    Returns
    -------
    dict
        Dictionary containing LLM configuration
    """
    config = {"class_name": llm.__class__.__name__, "module": llm.__class__.__module__}

    # Internal LangChain attributes that shouldn't be passed to constructors
    internal_attrs = {"config_specs", "lc_attributes", "lc_secrets", "model_computed_fields", "model_config", "model_kwargs", "disabled_params", "include_response_headers", "stream_usage", "validate_base_url", "disable_streaming"}

    # Endpoint and URL attributes that should not be saved (will be loaded from environment)
    endpoint_attrs = {"base_url", "endpoint", "azure_endpoint", "openai_api_base", "api_base", "api_url", "url"}

    # Save all attributes that are serializable and not None
    for attr_name in dir(llm):
        # Skip private attributes, methods, special attributes, internal LangChain attrs, and endpoint attrs
        if attr_name.startswith("_") or callable(getattr(llm, attr_name)) or attr_name in internal_attrs or attr_name in endpoint_attrs:
            continue

        try:
            value = getattr(llm, attr_name)
            if value is not None and _is_serializable(value):
                config[attr_name] = value
        except (AttributeError, TypeError):
            # Skip attributes that can't be accessed or would cause warnings
            continue

    return config


def load_llm_config(llm_config: Dict[str, Any]) -> BaseChatModel:
    """
    Recreate LLM instance from saved configuration.

    Parameters
    ----------
    llm_config : dict
        Dictionary containing LLM configuration

    Returns
    -------
    BaseChatModel
        Recreated LLM instance
    """
    try:
        # Import the LLM class
        module = import_module(llm_config["module"])
        llm_class = getattr(module, llm_config["class_name"])

        # Extract all parameters except class info
        llm_params = {k: v for k, v in llm_config.items() if k not in ["class_name", "module"]}

        # Create LLM instance
        return llm_class(**llm_params)
    except Exception as e:
        raise ValueError(f"Could not recreate LLM from config: {e}") from e

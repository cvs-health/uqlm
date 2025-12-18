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


from uqlm.utils.plots import plot_model_accuracies, plot_filtered_accuracy, plot_ranked_auc
from uqlm.utils.dataloader import load_dataset, load_example_dataset
from uqlm.utils.postprocessors import math_postprocessor, claims_dicts_to_lists
from uqlm.utils.response_generator import ResponseGenerator
from uqlm.utils.results import UQResult
from uqlm.utils.tuner import Tuner
from uqlm.utils.grader import LLMGrader
from uqlm.utils.llm_config import save_llm_config, load_llm_config
from uqlm.utils.display import ConditionalBarColumn, ConditionalTimeElapsedColumn, ConditionalTextColumn, ConditionalSpinnerColumn
from uqlm.utils.warn import beta_warning, deprecation_warning
from uqlm.utils.device import get_best_device
from uqlm.utils.prompts import TEMPLATE_TO_INSTRUCTION, TEMPLATE_TO_INSTRUCTION_WITH_EXPLANATIONS, SCORING_CONFIG, COMMON_INSTRUCTIONS, PROMPT_TEMPLATES, create_instruction, get_claim_breakdown_prompt, get_entailment_prompt

__all__ = [
    "plot_model_accuracies",
    "plot_filtered_accuracy",
    "plot_ranked_auc",
    "load_example_dataset",
    "load_dataset",
    "load_example_dataset",
    "math_postprocessor",
    "ResponseGenerator",
    "UQResult",
    "Tuner",
    "LLMGrader",
    "save_llm_config",
    "load_llm_config",
    "ConditionalBarColumn",
    "ConditionalTimeElapsedColumn",
    "ConditionalTextColumn",
    "ConditionalSpinnerColumn",
    "beta_warning",
    "deprecation_warning",
    "get_best_device",
    "TEMPLATE_TO_INSTRUCTION",
    "TEMPLATE_TO_INSTRUCTION_WITH_EXPLANATIONS",
    "SCORING_CONFIG",
    "COMMON_INSTRUCTIONS",
    "PROMPT_TEMPLATES",
    "create_instruction",
    "get_claim_breakdown_prompt",
    "get_entailment_prompt",
    "claims_dicts_to_lists",
]

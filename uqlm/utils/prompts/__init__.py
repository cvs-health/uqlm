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

"""
Prompt templates and utilities for the UQLM library.
"""

from uqlm.utils.prompts.judge_prompts import TEMPLATE_TO_INSTRUCTION, TEMPLATE_TO_INSTRUCTION_WITH_EXPLANATIONS, SCORING_CONFIG, COMMON_INSTRUCTIONS, PROMPT_TEMPLATES, create_instruction
from uqlm.utils.prompts.decomposition import get_claim_breakdown_prompt, get_factoid_breakdown_template
from uqlm.utils.prompts.reconstruction import get_response_reconstruction_prompt
from uqlm.utils.prompts.entailment_prompts import get_entailment_prompt
from uqlm.utils.prompts.factscore_prompts import FACTSCORE_SYSTEM_PROMPT, SUBJECTIVE_SYSTEM_PROMPT

__all__ = ["TEMPLATE_TO_INSTRUCTION", "TEMPLATE_TO_INSTRUCTION_WITH_EXPLANATIONS", "SCORING_CONFIG", "COMMON_INSTRUCTIONS", "PROMPT_TEMPLATES", "create_instruction", "get_claim_breakdown_prompt", "get_response_reconstruction_prompt", "get_entailment_prompt", "FACTSCORE_SYSTEM_PROMPT", "SUBJECTIVE_SYSTEM_PROMPT"]

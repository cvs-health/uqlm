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


import time
from typing import Any, Dict, List, Optional
import numpy as np
from rich.progress import Progress
from langchain_core.language_models.chat_models import BaseChatModel
from uqlm.utils.response_generator import ResponseGenerator

PTRUE_SYSTEM_PROMPT = """
Your task is to determine whether a given answer to a question is correct.

Guidelines for your evaluation:
- Do NOT penalize phrasing differences
- Respond with EXACTLY one word: "True" or "False"
- Answer "True" if the response is correct
- Answer "False" if the response is incorrect
- Do not explain your reasoning or provide any additional commentary
"""


class PTrueScorer:
    def __init__(self, llm: BaseChatModel, max_calls_per_min: Optional[int] = None) -> None:
        llm.logprobs = True
        self.response_generator = ResponseGenerator(llm, max_calls_per_min=max_calls_per_min)
        self.response_generator.response_generator_type = "p_true"

    async def evaluate(self, prompts: List[str], responses: List[str], sampled_responses: Optional[List[List[str]]] = None, progress_bar: Optional[Progress] = None) -> Dict[str, float]:
        if not sampled_responses:
            sampled_responses = [None] * len(responses)

        ptrue_prompts = [self._construct_ptrue_prompt(original_prompt=original_prompt_i, original_response=original_response_i, sampled_responses=sampled_responses_i) for original_prompt_i, original_response_i, sampled_responses_i in zip(prompts, responses, sampled_responses)]
        ptrue_responses = await self.response_generator.generate_responses(prompts=ptrue_prompts, system_prompt=PTRUE_SYSTEM_PROMPT, progress_bar=progress_bar)
        time.sleep(0.1)
        logprob_results = ptrue_responses["metadata"]["logprobs"]
        ptrue_scores = [self._extract_ptrue_from_logprobs_result(logprob_result) for logprob_result in logprob_results]
        return {"p_true": ptrue_scores}

    @staticmethod
    def _extract_ptrue_from_logprobs_result(logprobs_result: List[Dict[str, Any]]) -> float:
        first_token_data = logprobs_result[0]
        token = first_token_data.get("token", "").strip().lower()
        logprob = first_token_data.get("logprob", None)

        if logprob is not None:
            prob = np.exp(logprob)
            if token.startswith("true"):
                return prob  # High prob means high P_true
            elif token.startswith("false"):
                return 1.0 - prob  # High prob of False means low P_true
            else:
                return np.nan

    @staticmethod
    def _construct_ptrue_prompt(original_prompt: str, original_response: str, sampled_responses: Optional[List[str]] = None) -> str:
        proposed_answers_text = ""
        if sampled_responses:
            unique_responses = list(set(sampled_responses + [original_response]))

            if len(unique_responses) > 1:
                proposed_answers_text = "\n\nHere are some possible answers:\n"
                for possible_answer in unique_responses:
                    proposed_answers_text += possible_answer + "\n"

        ptrue_prompt = f"""
    Question: {original_prompt}
    {proposed_answers_text}
    Proposed Answer: {original_response}

    Is the proposed answer to the question true or false? Answer with only one word true/false.

    True or False:
        """
        return ptrue_prompt

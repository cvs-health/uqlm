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
from typing import List, Optional
from rich.progress import Progress
from langchain_core.language_models.chat_models import BaseChatModel
from uqlm.utils.response_generator import ResponseGenerator

GRADER_SYSTEM_PROMPT = """
You are an expert grading assistant designed to evaluate answers against a provided answer key. Your task is to determine whether a proposed answer is correct by comparing it to the ground truth answer(s).

## Your Responsibilities:

1. **Accept the ground truth as absolute**: The provided answer key contains the gold standard answer(s) and must be treated as correct, regardless of your own knowledge or beliefs.

2. **Evaluate the proposed answer**: Determine if the proposed answer aligns with any of the ground truth answers in terms of factual content, not just wording.

3. **Focus on semantic equivalence**: Look for meaning rather than exact wording. Two answers can be expressed differently but still be semantically equivalent.

4. **Provide ONLY a binary judgment**: Your entire response must be either the single word "yes" or "no" based solely on the answer's alignment with any of the ground truth answers. Answer "yes" if correct, "no" if incorrect.

5. **Avoid any explanation or reasoning**: Do not provide any justification, commentary, or additional text beyond the single word judgment.

6. **Be charitable but accurate**: Give credit when the proposed answer captures the essential elements of any of the ground truth answers, but don't overlook substantive differences.

Remember: You must return ONLY the word "yes" or "no" with no additional text. The ground truth answer(s) must be treated as correct even if you believe otherwise.
"""


class LLMGrader:
    def __init__(self, llm: BaseChatModel, max_calls_per_min: Optional[int] = None) -> None:
        llm.logprobs = True
        self.response_generator = ResponseGenerator(llm, max_calls_per_min=max_calls_per_min)
        self.response_generator.response_generator_type = "grader"

    async def grade_responses(self, prompts: List[str], responses: List[str], answers: List[str], progress_bar: Optional[Progress] = None) -> List[bool]:
        grader_prompts = [self._construct_grader_prompt(prompt, response, answer) for prompt, response, answer in zip(prompts, responses, answers)]
        grader_responses = await self.response_generator.generate_responses(prompts=grader_prompts, system_prompt=GRADER_SYSTEM_PROMPT, progress_bar=progress_bar)
        time.sleep(0.1)
        bool_grades = [self._extract_grades(grader_response) for grader_response in grader_responses["data"]["response"]]
        return bool_grades

    @staticmethod
    def _extract_grades(grader_response: str) -> bool:
        grader_response_stripped = grader_response.strip().lower()
        if "yes" in grader_response_stripped:
            return True
        elif "no" in grader_response_stripped:
            return False
        else:
            return False

    @staticmethod
    def _construct_grader_prompt(prompt: str, response: str, acceptable_answers: List[str]) -> str:
        grader_prompt = f"""
        Your task is to grade the following proposed answer against the provided answer key. The ground truth is the gold standard regardless of any other information you may have. Return ONLY the word "yes" or "no", with no additional text, based on whether the proposed answer aligns with any of the ground truth answers. Answer "yes" if correct, "no" if incorrect.

        **Question:**
        {prompt}

        **Ground Truth Answers (Answer Key):**
        {acceptable_answers}

        **Proposed Answer to Grade:**
        {response}

        Now your answer is (yes or no):
        """
        return grader_prompt

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


from typing import List, Optional
from rich.progress import Progress
from uqlm.utils.response_generator import ResponseGenerator
from uqlm.utils.prompts.factscore_prompts import FACTSCORE_SYSTEM_PROMPT, SUBJECTIVE_SYSTEM_PROMPT


class FactScoreGrader:
    def __init__(self, llm, max_calls_per_min: int = None):
        self.rg = ResponseGenerator(llm, max_calls_per_min=max_calls_per_min)
        self.grader_system_prompt = FACTSCORE_SYSTEM_PROMPT
        self.subjective_system_prompt = SUBJECTIVE_SYSTEM_PROMPT

    def construct_entailment_prompt(self, claim: str, answer: str) -> str:
        return f"""
            Context: {answer}
            Claim: {claim}
            Is the claim supported by the context above?
            Answer only Yes or No:
            """
    
    def construct_subjective_prompt(self, claim: str) -> str:
        return f"""
            Input: {claim}
            Is the input subjective or objective?
            Answer only subjective or objective:
            """

    async def grade_claims(self, claim_sets: List[List[str]], answers: List[str], progress_bar: Optional[Progress] = None) -> List[List[bool]]:
        prompts = []
        indices = []
        for i, (claim_set, answer) in enumerate(zip(claim_sets, answers)):
            for j, claim in enumerate(claim_set):
                prompt = self.construct_entailment_prompt(claim=claim, answer=answer)
                prompts.append(prompt)
                indices.append((i, j))

        generations = await self.rg.generate_responses(prompts=prompts, system_prompt=self.grader_system_prompt, progress_bar=progress_bar)
        responses = generations["data"]["response"]
        formatted_grade_lists = self._format_outputs(flat_grades_list=responses, reference_structure=claim_sets)
        return formatted_grade_lists
    
    async def evaluate_claim_objectivity(self, claim_sets: List[List[str]], progress_bar: Optional[Progress] = None) -> List[List[bool]]:
        prompts = []
        indices = []
        for i, claim_set in enumerate(claim_sets):
            for j, claim in enumerate(claim_set):
                prompt = self.construct_subjective_prompt(claim=claim)
                prompts.append(prompt)
                indices.append((i, j))

        self.generations = await self.rg.generate_responses(prompts=prompts, system_prompt=self.subjective_system_prompt, progress_bar=progress_bar)
        self.responses = self.generations["data"]["response"]
        formatted_grade_lists = self._format_outputs(flat_grades_list=self.responses, reference_structure=claim_sets, strings_to_check=["objective", "subjective"])
        return formatted_grade_lists    
        
    def _str_to_bool(self, response: str, strings_to_check: List[str] = ["yes", "no"]) -> bool:
        """Parse LLM response to extract Yes/No answer and convert to boolean"""
        response_text = response.strip().lower()
        if strings_to_check[0] in response_text: # either "yes" or "objective"
            return True
        elif strings_to_check[1] in response_text: # either "no" or "subjective"
            return False
        else:
            return False

    def _format_outputs(self, flat_grades_list: List[str], reference_structure: List[List[str]], strings_to_check: List[str] = ["yes", "no"]) -> List[bool]:
        """
        Reshape a flat list into a nested list structure that matches the reference structure.

        Args:
            flat_list: A flat list of elements with length equal to the sum of all inner list lengths in reference_structure
            reference_structure: A list of lists with varying inner list lengths

        Returns:
            A nested list with the same structure as reference_structure, containing elements from flat_list
        """
        formatted_grades = []
        flat_index = 0
        for inner_list in reference_structure:
            inner_length = len(inner_list)
            new_inner_list = flat_grades_list[flat_index : flat_index + inner_length]
            new_inner_list_bool = [self._str_to_bool(r, strings_to_check=strings_to_check) for r in new_inner_list]
            formatted_grades.append(new_inner_list_bool)
            flat_index += inner_length
        return formatted_grades
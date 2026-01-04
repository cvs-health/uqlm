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
from uqlm.utils.prompts.claim_qa import get_question_template, get_multiple_question_template
from langchain_core.language_models.chat_models import BaseChatModel
from uqlm.utils.response_generator import ResponseGenerator


class QuestionGenerator:
    def __init__(self, question_generator_llm: BaseChatModel, max_calls_per_min: Optional[int] = None) -> None:
        """
        Class for decomposing responses into individual claims or sentences. This class is used as an intermediate
        step for longform UQ methods.

        Parameters
        ----------
        question_generator_llm : langchain `BaseChatModel`, default=None
            A langchain llm `BaseChatModel`. User is responsible for specifying temperature and other
            relevant parameters to the constructor of their `llm` object.

        max_calls_per_min : int, default=None
            Specifies how many api calls to make per minute to avoid a rate limit error. By default, no
            limit is specified.
        """
        self.rg = ResponseGenerator(llm=question_generator_llm, max_calls_per_min=max_calls_per_min)
        self.rg.response_generator_type = "question_generator"

    async def generate_questions(self, claim_sets: List[List[str]], responses: Optional[List[str]] = None, num_questions: int = 1, progress_bar: Optional[Progress] = None) -> List[str]:
        """
        Parameters
        ----------
        claim_sets : List[List[str]]
            List of original responses decomposed into lists of claims

        responses : Optional[List[str]], default=None
            List of original responses to which the claim_sets belong

        num_questions : int, default=1
            The number of questions to generate for each claim/sentence.

        progress_bar : rich.progress.Progress, default=None
            If provided, displays a progress bar while scoring responses
        """
        self.num_questions = num_questions

        question_generation_prompts = self._construct_question_generation_prompts(claim_sets=claim_sets, responses=responses, num_questions=self.num_questions)

        question_generations = await self.rg.generate_responses(prompts=question_generation_prompts, progress_bar=progress_bar)

        claim_questions = self._extract_questions_from_generations(question_generations, num_questions=self.num_questions)

        return claim_questions

    @staticmethod
    def _construct_question_generation_prompts(claim_sets: List[List[str]], num_questions: int, responses: Optional[List[str]] = None) -> List[str]:
        """Construct prompts to generate questions for each claim/sentence"""
        question_generation_prompts = []
        for i, factoid_set in enumerate(claim_sets):
            response = None if not responses else responses[i]
            for factoid_i in factoid_set:
                if num_questions == 1:
                    question_generation_prompts.append(get_question_template(factoid_i))
                else:
                    question_generation_prompts.append(get_multiple_question_template(factoid_i, num_questions, response=response))
        return question_generation_prompts

    @staticmethod
    def _extract_questions_from_generations(question_generations: List[str], num_questions: int) -> List[str]:
        """Extract question texts from generations"""
        generated_texts = question_generations["data"]["response"]
        if num_questions > 1:
            generated_questions_list = []
            for i in range(len(generated_texts)):
                generated_questions_list += [tmp_x for tmp_x in generated_texts[i].split("###") if len(tmp_x) > 0]
            return generated_questions_list
        return generated_texts

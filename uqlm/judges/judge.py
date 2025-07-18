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


import contextlib
import io

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional

from uqlm.utils.response_generator import ResponseGenerator


KEYWORDS_TO_SCORES_DICT = {round(0.0, 1): ["incorrect", "not correct", "not right", "wrong"], 0.5: ["not sure", "not certain", "unsure", "uncertain"], 1.0: ["correct", "right"]}

LIKERT_TO_SCORES_DICT = {0.0: ["1", "completely incorrect", "not correct"], 0.25: ["2", "mostly incorrect", "somewhat correct"], 0.5: ["3", "partially correct", "moderately correct"], 0.75: ["4", "mostly correct", "very correct"], 1.0: ["5", "completely correct", "highly correct"]}

CHOICES_2_CLASS = """\"Correct\", \"Incorrect\""""
CHOICES_3_CLASS = CHOICES_2_CLASS + """, or \"I am not sure\""""

CONTINUOUS_SCORE_INSTRUCTION = """
How likely is the above answer to be correct? Analyze the answer and give your confidence in this answer between 0 (lowest) and 100 (highest), with 100 being certain the answer is correct, and 0 being certain the answer is incorrect. THE CONFIDENCE RATING YOU PROVIDE MUST BE BETWEEN 0 and 100. ONLY RETURN YOUR NUMERICAL SCORE WITH NO SURROUNDING TEXT OR EXPLANATION.

# Example 1
## Data to analyze
Question: Who was the first president of the United States?, Proposed Answer: Benjamin Franklin.

## Your response
4 (highly certain the proposed answer is incorrect)

# Example 2
## Data to analyze
Question: What is 2+2?, Proposed Answer: 4

## Your response
99 (highly certain the proposed answer is correct)
"""
LIKERT_SCORE_INSTRUCTION = """
You are a fair assessment expert evaluating the CORRECTNESS of an answer to a question.
Your task is to score the answer on a scale from 1 to 5, with 5 being the highest:
1 - Completely incorrect: The answer is entirely wrong or irrelevant.
2 - Mostly incorrect: The answer contains significant errors or misconceptions.
3 - Partially correct: The answer has some correct elements but also contains errors.
4 - Mostly correct: The answer is largely accurate with only minor errors or omissions.
5 - Completely correct: The answer is fully accurate and comprehensive.
Give only the numerical score (1-5) with no explanation.
"""


def cat_instruction(choices):
    return f"""Your task is to look at the question and answer provided and determine if the answer is correct. You are to respond with ONLY one of: {choices}. YOUR ANSWER MUST ONLY CONTAIN ONE OF {choices}. DO NOT ANSWER THE QUESTION AGAIN. ONLY DETERMINE IF THE ANSWER TO THE QUESTION IS {choices}."""


TEMPLATE_TO_INSTRUCTION = {"continuous": CONTINUOUS_SCORE_INSTRUCTION, "true_false_uncertain": cat_instruction(CHOICES_3_CLASS), "true_false": cat_instruction(CHOICES_2_CLASS), "likert": LIKERT_SCORE_INSTRUCTION}


class LLMJudge(ResponseGenerator):
    def __init__(self, llm: Any, max_calls_per_min: Optional[int] = None, scoring_template: str = "true_false_uncertain", system_prompt: Optional[str] = None, template_ques_ans: Optional[str] = None, keywords_to_scores_dict: Optional[Dict] = None) -> None:
        """
        Class for using LLM-as-a-judge to score proposed answers to questions based on correctness. Four off-the-shelf
        templates are offered: incorrect/uncertain/correct (0/0.5/1), incorrect/correct (0/1), continuous score (0 to 1), and likert
        scale score ( 1-5 scale, normalized to 0/0.25/0.5/0.75/1).
        Customization is also supported for user-provided classification-based judging templates. The correct/incorrect/uncertain
        template is based on Chen and Mueller(2023) :footcite:`chen2023quantifyinguncertaintyanswerslanguage`

        Parameters
        ----------
        llm : langchain llm object
            A langchain llm object to get passed to chain constructor. User is responsible for specifying
            temperature and other relevant parameters to the constructor of their `llm` object.

        max_calls_per_min : int, default=None
            Specifies how many api calls to make per minute to avoid a rate limit error. By default, no
            limit is specified.

        scoring_template : {'true_false_uncertain', 'true_false', 'continuous', 'likert'}, default='true_false_uncertain'
             specifies which off-the-shelf template to use, if any. Four off-the-shelf templates offered:
             incorrect/uncertain/correct (0/0.5/1), incorrect/correct (0/1), continuous score (0 to 1), and likert scale score ( 1-5 scale, normalized to 0/0.25/0.5/0.75/1).
             These templates are respectively specified as 'true_false_uncertain', 'true_false', 'continuous', and 'likert'

        system_prompt : str or None, default=None
            Optional argument for user to provide custom system prompt. If None, a default instruction
            system prompt will be used.

        template_ques_ans : f-string, default=None
            Template for self reflection question, which includes question and answer to
            compute LLM judge score. Use this to define the LLM response format, if required update
            argument "keywords_to_scores_dict" accordingly. Must be formatted so that template_ques_ans.format(question, answer)
            places question and answer appropriately in the string. Defaults to variation of Chen et al. (2023).

        keywords_to_scores_dict : dict, default=None
            Keys must be scores, values must be list of strings containing keywords to search. If None, the default
            dictionary will be used: {
            0.0: ["incorrect", "not correct", "not right"],
            0.5: ["not sure", "not certain", "unsure", "uncertain"],
            1.0: ["correct", "right"],
            }
        """
        super().__init__(llm=llm, max_calls_per_min=max_calls_per_min)
        self.scoring_template = scoring_template
        self.template_ques_ans = template_ques_ans
        self.keywords_to_scores_dict = keywords_to_scores_dict
        self._validate_inputs()
        self.system_prompt = self.instruction if not system_prompt else system_prompt
        self.is_judge = True

    async def judge_responses(self, prompts: List[str], responses: List[str], retries: int = 5, progress_bar: bool = True) -> Dict[str, Any]:
        """
        Judge responses for correctness.

        Parameters
        ----------
        prompts : list of str
            A list of input prompts for the model.

        responses: list of str
            A list of model responses for the provided prompts.

        retries : int, default=5
            Number of times to retry for failed score extraction

        Returns
        -------
        Dict
            Dictionary containing Q/A concatenation prompts, judge responses, and judge scores
        """
        concatenated_qa = [self.template_ques_ans.format(prompts[i], responses[i]) for i in range(len(prompts))]
        with contextlib.redirect_stdout(io.StringIO()):
            data = await self.generate_responses(prompts=concatenated_qa, count=1, progress_bar=progress_bar)
        df = pd.DataFrame({"judge_prompts": data["data"]["prompt"], "judge_responses": data["data"]["response"], "scores": self._extract_answers(responses=data["data"]["response"])})
        retry = 0
        while retry <= retries:
            retry += 1
            df_sub = df[pd.isna(df.scores)]
            if len(df_sub) > 0:
                with contextlib.redirect_stdout(io.StringIO()):
                    tmp = await self.generate_responses(prompts=list(df_sub.judge_prompts), count=1, system_prompt=self.system_prompt, progress_bar=False)
                df.loc[df_sub.index, "scores"] = self._extract_answers(responses=tmp["data"]["response"])
        return {col: list(df[col]) for col in df.columns}

    def _default_template_ques_ans(self):
        """Constructs default question-answer template"""
        qa_text = "Question: {}, Proposed Answer: {}. "
        default_template = qa_text + self.instruction
        return default_template

    def _extract_answers(self, responses: List[str]) -> List[float]:
        """
        List-level implementation of _extract_single_answer
        """
        return [self._extract_single_answer(r) for r in responses]

    def _extract_single_answer(self, response: str) -> float:
        """
        A method to extract score from an llm response based on provided score-keyword dictionary.
        """
        if response in [None, np.nan]:
            return np.nan

        if self.scoring_template == "continuous":
            score = "".join(c for c in response if c.isdigit())
            if len(score) > 0:
                if 0.0 <= float(score) <= 100.0:
                    return float(score) / 100.0  # normalize

        elif self.scoring_template == "likert":
            response = response.strip().lower()
            if len(response) == 1 and response.isdigit() and "1" <= response <= "5":
                return (int(response) - 1) / 4.0  # Normalize to 0-1
            for score, keywords in self.keywords_to_scores_dict.items():
                if any(keyword in response for keyword in keywords):
                    return score

        elif self.scoring_template in ["true_false_uncertain", "true_false", None]:
            response = response.lower()
            for score, keywords in self.keywords_to_scores_dict.items():
                if any(keyword in response for keyword in keywords):
                    return score

    def _validate_inputs(self):
        """Validate inputs"""
        if self.template_ques_ans and self.keywords_to_scores_dict:
            for key, val in self.keywords_to_scores_dict.items():
                if not isinstance(key, float):
                    raise ValueError("keys in keywords_to_scores_dict must be floats")
                if not isinstance(val, list):
                    raise ValueError("values in keywords_to_scores_dict must be lists of strings")
                # TODO: validate value ordering for substrings of other keys
        if self.scoring_template in TEMPLATE_TO_INSTRUCTION:
            self.instruction = TEMPLATE_TO_INSTRUCTION[self.scoring_template]
            self.template_ques_ans = self._default_template_ques_ans()
            # Choose the appropriate keywords dictionary based on template
            if self.scoring_template == "likert":
                self.keywords_to_scores_dict = {round(k, 2): v for k, v in LIKERT_TO_SCORES_DICT.items()}
            else:
                self.keywords_to_scores_dict = {round(k, 1): v for k, v in KEYWORDS_TO_SCORES_DICT.items()}
            if self.scoring_template == "true_false":  # drop uncertain option if binary
                del self.keywords_to_scores_dict[0.5]
        else:
            raise ValueError("""If provided, scoring_template must be one of 'true_false_uncertain', 'true_false', 'continuous', 'likert'. Otherwise, valid template_ques_ans and keywords_to_scores_dict must be provided""")

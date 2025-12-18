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
import rich
from typing import Any, Dict, List, Optional, Union, Tuple

from uqlm.utils.response_generator import ResponseGenerator
from uqlm.utils.prompts import TEMPLATE_TO_INSTRUCTION, TEMPLATE_TO_INSTRUCTION_WITH_EXPLANATIONS


KEYWORDS_TO_SCORES_DICT = {round(0.0, 1): ["incorrect", "not correct", "not right", "wrong"], 0.5: ["not sure", "not certain", "unsure", "uncertain"], 1.0: ["correct", "right"]}

LIKERT_TO_SCORES_DICT = {0.0: ["1", "completely incorrect", "not correct"], 0.25: ["2", "mostly incorrect", "somewhat correct"], 0.5: ["3", "partially correct", "moderately correct"], 0.75: ["4", "mostly correct", "very correct"], 1.0: ["5", "completely correct", "highly correct"]}


class LLMJudge(ResponseGenerator):
    def __init__(self, llm: Any, max_calls_per_min: Optional[int] = None, scoring_template: str = "true_false_uncertain", additional_context: Optional[str] = None, keywords_to_scores_dict: Optional[Dict] = None) -> None:
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

        additional_context : str or None, default=None
            Optional argument to provide additional context to inform LLM-as-a-Judge evaluations.

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
        self.keywords_to_scores_dict = keywords_to_scores_dict
        self._validate_inputs()
        self.response_generator_type = "judge"
        self.system_prompt = additional_context
        self.is_judge = True

    async def judge_responses(self, prompts: List[str], responses: List[str], retries: int = 5, progress_bar: Optional[rich.progress.Progress] = None, explanations: bool = False) -> Dict[str, Any]:
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

        progress_bar : rich.progress.Progress, default=None
            If provided, displays a progress bar while scoring responses

        explanations : bool, default=False
            If True, the judge will be instructed to provide explanations along with scores.

        Returns
        -------
        Dict
            Dictionary containing Q/A concatenation prompts, judge responses, judge scores, and optionally explanations
        """
        instruction = self.explanation_instruction if explanations else self.standard_instruction
        concatenated_qa = [self._default_template_ques_ans(instruction).format(prompts[i], responses[i]) for i in range(len(prompts))]
        with contextlib.redirect_stdout(io.StringIO()):
            data = await self.generate_responses(prompts=concatenated_qa, count=1, system_prompt=self.system_prompt, progress_bar=progress_bar)

        # Extract scores and explanations
        extracted_data = self._extract_answers(responses=data["data"]["response"], explanations=explanations)

        if explanations:
            scores, explanations_data = extracted_data
            df = pd.DataFrame({"judge_prompts": data["data"]["prompt"], "judge_responses": data["data"]["response"], "scores": scores, "explanations": explanations_data})
        else:
            scores = extracted_data
            df = pd.DataFrame({"judge_prompts": data["data"]["prompt"], "judge_responses": data["data"]["response"], "scores": scores})

        # Retry logic for failed extractions
        retry = 0
        while retry <= retries:
            retry += 1

            # Find any failures
            score_failures = df[pd.isna(df.scores)]
            explanation_failures = df[pd.isna(df.explanations)] if explanations else pd.DataFrame()

            # If ANY failures exist, retry BOTH score and explanation
            if len(score_failures) > 0 or len(explanation_failures) > 0:
                # Get all failure indices
                failure_indices = set(score_failures.index) | set(explanation_failures.index)

                with contextlib.redirect_stdout(io.StringIO()):
                    tmp = await self.generate_responses(prompts=list(df.loc[list(failure_indices), "judge_prompts"]), count=1, system_prompt=self.system_prompt, progress_bar=False)

                retry_data = self._extract_answers(responses=tmp["data"]["response"], explanations=explanations)

                if explanations:
                    retry_scores, retry_explanations = retry_data
                    df.loc[list(failure_indices), "scores"] = retry_scores
                    df.loc[list(failure_indices), "explanations"] = retry_explanations
                else:
                    df.loc[list(failure_indices), "scores"] = retry_data

            # Exit if no more failures
            if len(score_failures) == 0 and (not explanations or len(explanation_failures) == 0):
                break
        return {col: list(df[col]) for col in df.columns}

    def _default_template_ques_ans(self, instruction: str):
        """Constructs default question-answer template with provided instruction"""
        qa_text = "Question: {}, Proposed Answer: {}. "
        return qa_text + instruction

    def _extract_answers(self, responses: List[str], explanations: bool = False) -> Union[List[float], Tuple[List[float], List[str]]]:
        """
        List-level implementation of _extract_single_answer
        """
        if explanations:
            scores, explanations_data = zip(*[self._extract_single_answer(r, explanations=explanations) for r in responses])
            return list(scores), list(explanations_data)
        else:
            return [self._extract_single_answer(r, explanations=explanations) for r in responses]

    def _extract_single_answer(self, response: str, explanations: bool = False) -> Union[float, Tuple[float, str]]:
        """
        A method to extract score and optionally explanation from an llm response.
        Returns (score, explanation) if explanations=True, otherwise returns score only.
        """
        if response in [None, np.nan]:
            return (np.nan, np.nan) if explanations else np.nan

        if explanations:
            return self._parse_structured_response(response)
        else:
            return self._extract_score_from_text(response)

    def _parse_structured_response(self, response: str) -> Tuple[float, str]:
        """
        Parse structured response format: "Score: X\nExplanation: Y"
        """
        try:
            if "Score:" in response:
                # Extract score part
                if "Explanation:" in response:
                    score_part = response.split("Score:")[1].split("Explanation:")[0].strip()
                    explanation_part = response.split("Explanation:")[1].strip()
                else:
                    # Only score, no explanation
                    score_part = response.split("Score:")[1].strip()
                    explanation_part = "No explanation provided"

                # Extract score
                score = self._extract_score_from_text(score_part)

                return score, explanation_part
            else:
                # Fallback: try to extract score from entire response
                score = self._extract_score_from_text(response)
                return score, "No explanation provided"

        except Exception as e:
            import warnings

            warnings.warn(f"Failed to parse judge response: '{response[:50]}...'. Using NaN fallback. Error: {str(e)[:100]}")
            return np.nan, "Parsing failed - using NaN"

    def _extract_score_from_text(self, response: str) -> float:
        """
        Extract score from text using the standard extraction logic.
        Used for both structured responses and backward compatibility.
        """
        if self.scoring_template == "continuous":
            # Extract all digits and decimal points
            score = "".join(c for c in response if c.isdigit())
            if len(score) > 0:
                score_val = float(score)
                if 0.0 <= score_val <= 100.0:
                    return score_val / 100.0  # normalize
            return np.nan

        elif self.scoring_template == "likert":
            response = response.strip().lower()
            if len(response) == 1 and response.isdigit() and "1" <= response <= "5":
                return (int(response) - 1) / 4.0  # Normalize to 0-1
            for score, keywords in self.keywords_to_scores_dict.items():
                if any(keyword in response for keyword in keywords):
                    return score
            return np.nan

        elif self.scoring_template in ["true_false_uncertain", "true_false", None]:
            response = response.lower()
            for score, keywords in self.keywords_to_scores_dict.items():
                if any(keyword in response for keyword in keywords):
                    return score
            return np.nan

        return np.nan

    def _validate_inputs(self):
        """Validate inputs"""
        if self.scoring_template in TEMPLATE_TO_INSTRUCTION:
            # Store both standard and explanation templates for runtime selection
            self.standard_instruction = TEMPLATE_TO_INSTRUCTION[self.scoring_template]
            self.explanation_instruction = TEMPLATE_TO_INSTRUCTION_WITH_EXPLANATIONS[self.scoring_template]
            # Choose the appropriate keywords dictionary based on template
            if self.scoring_template == "likert":
                self.keywords_to_scores_dict = {round(k, 2): v for k, v in LIKERT_TO_SCORES_DICT.items()}
            else:
                self.keywords_to_scores_dict = {round(k, 1): v for k, v in KEYWORDS_TO_SCORES_DICT.items()}
            if self.scoring_template == "true_false":  # drop uncertain option if binary
                del self.keywords_to_scores_dict[0.5]
        else:
            raise ValueError("""If provided, scoring_template must be one of 'true_false_uncertain', 'true_false', 'continuous', 'likert'""")

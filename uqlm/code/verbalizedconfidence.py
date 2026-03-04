import contextlib
import io

import numpy as np
import pandas as pd
import rich
from typing import Any, Dict, List, Optional, Union, Tuple

from uqlm.utils.response_generator import ResponseGenerator


LIKERT_TO_SCORES_DICT = {
    0.0: "no chance",
    0.2: "little chance",
    0.4: "less than even",
    0.6: "fairly possible",
    0.8: "very good chance",
    1.0: "almost certain",
}


class VerbalizedConfidence(ResponseGenerator):
    def __init__(self, llm: Any, max_calls_per_min: Optional[int] = None) -> None:
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
        """
        super().__init__(llm=llm, max_calls_per_min=max_calls_per_min)
        self.response_generator_type = "judge"
        self.system_prompt = None
        self.is_judge = True

    async def judge_responses(
        self,
        prompts: List[str],
        responses: List[str],
        retries: int = 5,
        progress_bar: Optional[rich.progress.Progress] = None,
    ) -> Dict[str, Any]:
        """
        Judge responses for correctness.

        Parameters
        ----------
        prompts : list of str
            A list of input prompts for the model.

        claim_sets: list of str
            A list of model responses for the provided prompts.

        retries : int, default=5
            Number of times to retry for failed score extraction

        progress_bar : rich.progress.Progress, default=None
            If provided, displays a progress bar while scoring responses

        Returns
        -------
        Dict
            Dictionary containing Q/A concatenation prompts, judge responses, judge scores
        """
        judge_prompts = [
            self._construct_claim_prompt(
                original_question=prompts[i], claim=responses[i]
            )
            for i in range(len(prompts))
        ]
        with contextlib.redirect_stdout(io.StringIO()):
            data = await self.generate_responses(
                prompts=judge_prompts,
                count=1,
                system_prompt=self.system_prompt,
                progress_bar=progress_bar,
            )

        # Extract scores
        extracted_data = self._extract_answers(responses=data["data"]["response"])

        scores = extracted_data
        df = pd.DataFrame(
            {
                "judge_prompts": data["data"]["prompt"],
                "judge_responses": data["data"]["response"],
                "scores": scores,
            }
        )

        # Retry logic for failed extractions
        retry = 0
        while retry <= retries:
            retry += 1

            # Find any failures
            score_failures = df[pd.isna(df.scores)]

            # If ANY failures exist, retry
            if len(score_failures) > 0:
                # Get all failure indices
                failure_indices = set(score_failures.index)

                with contextlib.redirect_stdout(io.StringIO()):
                    tmp = await self.generate_responses(
                        prompts=list(df.loc[list(failure_indices), "judge_prompts"]),
                        count=1,
                        system_prompt=self.system_prompt,
                        progress_bar=False,
                    )

                retry_data = self._extract_answers(responses=tmp["data"]["response"])

                df.loc[list(failure_indices), "scores"] = retry_data

            # Exit if no more failures
            if len(score_failures) == 0:
                break

        return df.scores.tolist()

    def _construct_claim_prompt(self, original_question: str, claim: str) -> str:
        """Constructs default question-answer template with provided instruction"""
        claim_prompt = f"""
        We are writing a solution to this coding problem: {original_question}

        Describe how likely it is that the code below is correct as one of the following phrases: 

        No chance
        Little chance
        Less than even
        Fairly possible
        Very good chance
        Almost certain

        Give ONLY your confidence phrase, no other words or explanation. Your answer must contain ONLY one of the confidence phrases above.

        Here is the code: {claim}

        Now your answer is: 
        """
        return claim_prompt

    def _extract_answers(
        self, responses: List[str]
    ) -> Union[List[float], Tuple[List[float], List[str]]]:
        """
        List-level implementation of _extract_single_answer
        """
        return [self._extract_single_answer(r) for r in responses]

    def _extract_single_answer(self, response: str) -> Union[float, Tuple[float, str]]:
        """
        A method to extract score from an llm response.
        Returns score only.
        """
        if response in [None, np.nan]:
            return np.nan
        return self._extract_score_from_text(response)

    def _extract_score_from_text(self, response: str) -> float:
        """
        Extract score from text using the standard extraction logic.
        Used for both structured responses and backward compatibility.
        """
        for score, keywords in LIKERT_TO_SCORES_DICT.items():
            if keywords in response.strip().lower():
                return score
        return np.nan

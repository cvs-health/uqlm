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


import numpy as np
from typing import List, Optional, Union
from langchain_core.language_models.chat_models import BaseChatModel

from uqlm.judges.judge import LLMJudge
from uqlm.scorers.shortform.baseclass.uncertainty import ShortFormUQ
from uqlm.utils.results import UQResult


class LLMPanel(ShortFormUQ):
    def __init__(self, judges: List[Union[LLMJudge, BaseChatModel]], llm: Optional[BaseChatModel] = None, system_prompt: Optional[str] = None, max_calls_per_min: Optional[int] = None, scoring_templates: Optional[List[str]] = None, explanations: bool = False, additional_context: Optional[str] = None) -> None:
        """
        Class for aggregating multiple instances of LLMJudge using min, max, or majority voting

        Parameters
        ----------
        judges: list of LLMJudge or BaseChatModel
            Judges to use. If BaseChatModel, LLMJudge is instantiated using default parameters.

        llm : BaseChatModel
            A langchain llm `BaseChatModel`. User is responsible for specifying temperature and other
            relevant parameters to the constructor of their `llm` object.

        max_calls_per_min : int, default=None
            Used to control rate limiting. Will be used for original llm and any judges constructed
            from instances of BaseChatModel in judges

        system_prompt : str, default=None
            Optional argument for user to provide custom system prompt. If None, defaults to "You are a helpful assistant."

        scoring_templates : List[str], default=None
             Specifies which off-the-shelf template to use for each judge. Four off-the-shelf templates offered:
             incorrect/uncertain/correct (0/0.5/1), incorrect/correct (0/1), continuous score (0 to 1), and likert scale score ( 1-5 scale, normalized to 0/0.25/0.5/0.75/1).
             These templates are respectively specified as 'true_false_uncertain', 'true_false', 'continuous', and 'likert'
             If specified, must be of equal length to `judges` list. Defaults to 'true_false_uncertain' template
             used by Chen and Mueller (2023) :footcite:`chen2023quantifyinguncertaintyanswerslanguage` for each judge.

        explanations : bool, default=False
            If True, judges will be instructed to provide explanations along with scores.
            When enabled, explanation columns will be included in the output DataFrame.

        additional_context : str or None, default=None
            Optional argument to provide additional context to inform LLM-as-a-Judge evaluations.
        """
        super().__init__(llm=llm, max_calls_per_min=max_calls_per_min, system_prompt=system_prompt)
        self.explanations = explanations
        self.scoring_templates = scoring_templates
        if self.scoring_templates:
            if len(self.scoring_templates) != len(judges):
                raise ValueError("Length of scoring_templates list must be equal to length of judges list")
        else:
            self.scoring_templates = ["true_false_uncertain"] * len(judges)
        self.judges = []
        for judge, template in zip(judges, self.scoring_templates):
            if isinstance(judge, BaseChatModel):
                judge = LLMJudge(llm=judge, max_calls_per_min=max_calls_per_min, scoring_template=template, additional_context=additional_context)
            elif not isinstance(judge, LLMJudge):
                raise ValueError("judges must be a list containing instances of either LLMJudge or BaseChatModel")
            self.judges.append(judge)

    async def generate_and_score(self, prompts: List[str], show_progress_bars: Optional[bool] = True) -> UQResult:
        """
        Generate LLM responses to provided prompts and use panel of judges to score responses for correctness.

        Parameters
        ----------
        prompts : list of str
            A list of input prompts for the model.

        show_progress_bars : bool, default=True
            If True, displays a progress bar while generating and scoring responses

        Returns
        -------
        UQResult
            UQResult containing prompts, responses, Q/A concatenations, judge responses, and judge scores
        """
        if not all(isinstance(item, str) for item in prompts):
            raise ValueError("prompts must be list of strings when using LLMPanel")
        self._construct_progress_bar(show_progress_bars)
        self._display_generation_header(show_progress_bars)

        responses = await self.generate_original_responses(prompts, progress_bar=self.progress_bar)
        return await self.score(prompts=prompts, responses=responses, show_progress_bars=show_progress_bars)

    async def score(self, prompts: List[str], responses: Optional[List[str]] = None, show_progress_bars: bool = True, _display_header: bool = True) -> UQResult:
        """
        Use panel to of judges to score provided responses for correctness. Use if responses are already generated. Otherwise,
        use generate_and_score.

        Parameters
        ----------
        prompts : list of str
            A list of input prompts for the model.

        responses: list of str, default = None
            A list of LLM responses for the corresponding to the provided prompts.

        show_progress_bars : bool, default=True
            If True, displays a progress bar while scoring responses

        Returns
        -------
        UQResult
            UQResult containing prompts, responses, Q/A concatenations, judge responses, and judge scores
        """
        self.prompts = prompts
        self.responses = responses
        data = {"prompts": prompts, "responses": responses}

        self._construct_progress_bar(show_progress_bars)
        self._display_scoring_header(show_progress_bars and _display_header)

        judge_count = 1
        scores_lists = []
        for judge in self.judges:
            tmp = await judge.judge_responses(prompts=prompts, responses=responses, progress_bar=self.progress_bar, explanations=self.explanations)
            scores_lists.append(tmp["scores"])
            data[f"judge_{judge_count}"] = tmp["scores"]

            # Add explanation columns if explanations are enabled
            if self.explanations and "explanations" in tmp:
                data[f"judge_{judge_count}_explanation"] = tmp["explanations"]

            judge_count += 1

        scores_dict = {"avg": [np.mean(scores) for scores in zip(*scores_lists)], "max": [np.max(scores) for scores in zip(*scores_lists)], "min": [np.min(scores) for scores in zip(*scores_lists)], "median": [np.median(scores) for scores in zip(*scores_lists)]}
        data.update(scores_dict)
        result = {"data": data, "metadata": {"num_judges": len(self.judges), "temperature": None if not self.llm else self.llm.temperature}}

        self._stop_progress_bar()
        self.progress_bar = None  # if re-run ensure the same progress object is not used
        return UQResult(result)

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

from typing import Any, Dict, List, Optional
import math
import numpy as np
from langchain_core.language_models.chat_models import BaseChatModel

from uqlm.scorers.baseclass.uncertainty import UncertaintyQuantifier, UQResult


class WhiteBoxUQ(UncertaintyQuantifier):
    def __init__(self, llm: Optional[BaseChatModel] = None, system_prompt: str = "You are a helpful assistant.", max_calls_per_min: Optional[int] = None, scorers: Optional[List[str]] = None) -> None:
        """
        Class for computing white-box UQ confidence scores. This class offers two confidence scores, normalized
        probability :footcite:`malinin2021uncertaintyestimationautoregressivestructured` and minimum probability :footcite:`manakul2023selfcheckgptzeroresourceblackboxhallucination`.

        Parameters
        ----------
        llm : BaseChatModel
            A langchain llm object to get passed to chain constructor. User is responsible for specifying
            temperature and other relevant parameters to the constructor of their `llm` object.

        max_calls_per_min : int, default=None
            Used to control rate limiting.

        system_prompt : str or None, default="You are a helpful assistant."
            Optional argument for user to provide custom system prompt

        scorers : subset of {
            "imperplexity", "geometric_mean_probability", "min_probability", "max_probability",
        }, default=None
            Specifies which black box (consistency) scorers to include. If None, defaults to all.
        """
        super().__init__(llm=llm, max_calls_per_min=max_calls_per_min, system_prompt=system_prompt)
        self.scorers = scorers if scorers else self.white_box_names

    async def generate_and_score(self, prompts: List[str]) -> UQResult:
        """
        Generate responses and compute white-box confidence scores based on extracted token probabilities.

        Parameters
        ----------
        prompts : list of str
            A list of input prompts for the model.

        Returns
        -------
        UQResult
            UQResult containing prompts, responses, logprobs, and white-box UQ scores
        """
        assert hasattr(self.llm, "logprobs"), """
        BaseChatModel must have logprobs attribute and have logprobs=True
        """
        self.llm.logprobs = True
        responses = await self.generate_original_responses(prompts)
        return self.score(prompts=prompts, responses=responses, logprobs_results=self.logprobs)

    def score(self, logprobs_results: List[List[Dict[str, Any]]], prompts: Optional[List[str]] = None, responses: Optional[List[str]] = None) -> UQResult:
        """
        Compute white-box confidence scores from provided logprobs.

        Parameters
        ----------
        logprobs_results : list of logprobs_result
            List of dictionaries, each returned by BaseChatModel.agenerate

        prompts : list of str, default=None
            A list of input prompts for the model.

        responses : list of str, default=None
            A list of model responses for the prompts.

        Returns
        -------
        UQResult
            UQResult containing prompts, responses, logprobs, and white-box UQ scores
        """

        self.logprobs = logprobs_results
        self.prompts = prompts
        self.responses = responses

        data = {}
        if self.prompts:
            data["prompts"] = self.prompts
        if self.responses:
            data["responses"] = self.responses

        data["logprobs"] = self.logprobs
        scores = self._compute_scores(self.logprobs)
        for key in self.scorers:
            data[key] = scores[key]

        result = {"data": data, "metadata": {"temperature": None if not self.llm else self.llm.temperature}}
        return UQResult(result)

    def _compute_scores(self, logprobs_results: List[List[Dict[str, Any]]]) -> List[float]:
        """
        This method computes token-probability-based confidence scores.
        """
        return {"normalized_probability": [np.nan if not r else self._norm_prob(r) for r in logprobs_results], "min_probability": [np.nan if not r else self._min_prob(r) for r in logprobs_results]}

    def _norm_prob(self, logprobs: List[Dict[str, Any]]) -> float:
        """Compute normalized token probability"""
        return math.exp(self.avg_logprob(logprobs))

    def _min_prob(self, logprobs: List[Dict[str, Any]]) -> float:
        """Compute minimum token probability"""
        return min(self._get_probs(logprobs))

    def avg_logprob(self, logprobs: List[Dict[str, Any]]) -> float:
        "Compute average logprob"
        return np.mean(self.get_logprobs(logprobs))

    @staticmethod
    def _get_probs(logprobs):
        """Extract token probabilities"""
        return [math.exp(d["logprob"]) for d in logprobs]

    @staticmethod
    def get_logprobs(logprobs):
        """Extract log token probabilities"""
        return [d["logprob"] for d in logprobs]

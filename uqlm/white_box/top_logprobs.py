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
from typing import List, Dict, Any
from uqlm.white_box.baseclass.logprobs_scorer import LogprobsScorer


TOP_LOGPROBS_SCORER_NAMES = ["min_token_negentropy", "mean_token_negentropy", "probability_margin"]


class TopLogprobsScorer(LogprobsScorer):
    def __init__(self, scorers: List[str] = TOP_LOGPROBS_SCORER_NAMES):
        """Class for computing WhiteBox UQ scores with a single generation"""
        super().__init__()
        self.scorers = scorers

    def evaluate(self, logprobs_results: List[List[Dict[str, Any]]]) -> Dict[str, List[float]]:
        """Compute scores from top logprobs results"""
        scores_dict = {"mean_token_negentropy": self._compute_single_generation_scores(logprobs_results, self._mean_token_negentropy), "min_token_negentropy": self._compute_single_generation_scores(logprobs_results, self._min_token_negentropy), "probability_margin": self._compute_single_generation_scores(logprobs_results, self._probability_margin)}
        return {k: scores_dict[k] for k in self.scorers}

    def _compute_token_entropies(self, single_response_logprobs: List[Dict[str, Any]]) -> np.ndarray:
        """Compute entropy for each token in the sequence"""
        top_logprobs_list = self.extract_top_logprobs(single_response_logprobs)
        return np.array([self._entropy_from_logprobs(top_logprobs) for top_logprobs in top_logprobs_list])

    def _compute_token_negentropies(self, single_response_logprobs: List[Dict[str, Any]]) -> np.ndarray:
        """Compute negentropy for each token in the sequence"""
        entropies = self._compute_token_entropies(single_response_logprobs)
        top_logprobs_list = self.extract_top_logprobs(single_response_logprobs)
        k_values = np.array([len(top_logprobs) for top_logprobs in top_logprobs_list])
        max_entropies = np.log(k_values)
        negentropies = 1 - entropies / max_entropies
        return negentropies

    def _mean_token_negentropy(self, single_response_logprobs: List[Dict[str, Any]]) -> float:
        """Compute mean token negentropy across the sequence"""
        negentropies = self._compute_token_negentropies(single_response_logprobs)
        return np.mean(negentropies)

    def _min_token_negentropy(self, single_response_logprobs: List[Dict[str, Any]]) -> float:
        """Compute minimum token negentropy across the sequence"""
        negentropies = self._compute_token_negentropies(single_response_logprobs)
        return np.min(negentropies)

    def _probability_margin(self, single_response_logprobs: List[Dict[str, Any]]) -> float:
        """Compute mean probability margin (difference between top two probabilities)"""
        top_logprobs_list = self.extract_top_logprobs(single_response_logprobs)
        margins = []
        try:
            for top_logprobs in top_logprobs_list:
                probs = np.exp(top_logprobs)
                probs = np.sort(probs)[::-1]
                margin = probs[0] - probs[1]
                margins.append(margin)
            return np.mean(margins)
        except IndexError:
            print("top_logprobs were not available. Unable to compute associated scores.")
            return np.nan

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


SINGLE_LOGPROBS_SCORER_NAMES = ["normalized_probability", "min_probability", "sequence_probability"]


class SingleLogprobsScorer(LogprobsScorer):
    def __init__(self, scorers: List[str] = SINGLE_LOGPROBS_SCORER_NAMES):
        """Class for computing WhiteBox UQ scores with a single generation"""
        super().__init__()
        self.scorers = scorers

    def evaluate(self, logprobs_results: List[List[Dict[str, Any]]]) -> Dict[str, List[float]]:
        """Compute scores from logprobs results"""
        scores_dict = {"normalized_probability": self._compute_single_generation_scores(logprobs_results, self._norm_prob), "min_probability": self._compute_single_generation_scores(logprobs_results, self._min_prob), "sequence_probability": self._compute_single_generation_scores(logprobs_results, self._seq_prob)}
        return {k: scores_dict[k] for k in self.scorers}

    def _min_prob(self, single_response_logprobs: List[Dict[str, Any]]) -> float:
        """Compute minimum token probability"""
        probs = self.extract_probs(single_response_logprobs)
        return np.min(probs)

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


import bert_score
import numpy as np
from typing import List

from uqlm.black_box.baseclass.similarity_scorer import SimilarityScorer


class BertScorer(SimilarityScorer):
    def __init__(self) -> None:
        """
        Class for computing BERTScore values between original responses and candidates. For more on
        BERTScore, refer to Zhang et al.(2020) :footcite:`zhang2020bertscoreevaluatingtextgeneration`.
        """
        pass

    def evaluate(self, responses: List[str], sampled_responses: List[List[str]]) -> List[float]:
        """
        This method computes model-based text similarity metrics values for the provided pairs of texts.

        Parameters
        ----------
        responses : list of strings
            Original LLM response

        sampled_responses : list of list of strings
            Candidate responses to be compared to the original response

        Returns
        -------
        List of float
            Mean BertScore values
        """
        return [self._compute_score(response=responses[i], candidates=sampled_responses[i]) for i in range(len(responses))]

    @staticmethod
    def _compute_score(response: str, candidates: List[str]) -> float:
        """Compute mean BERTScore between a response and candidate responses"""
        duplicated_response = [response] * len(candidates)
        P, R, F1 = bert_score.score(list(duplicated_response), refs=list(candidates), lang="en")
        return np.mean([float(f) for f in F1])

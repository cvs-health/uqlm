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


from abc import ABC
import numpy as np
from typing import List, Dict, Any, Optional, Callable


class LogprobsScorer(ABC):
    def __init__(self):
        pass

    def _norm_prob(self, single_response_logprobs: List[Dict[str, Any]]) -> float:
        """Compute length-normalized sequence probability"""
        logprobs = self.extract_logprobs(single_response_logprobs)
        return np.exp(np.mean(logprobs))

    def _seq_prob(self, single_response_logprobs: List[Dict[str, Any]]) -> float:
        """Compute sequence probability"""
        probs = self.extract_probs(single_response_logprobs)
        return np.prod(probs)

    def _entropy_from_logprobs(self, logprobs_list: np.ndarray) -> float:
        """Compute entropy from list of logprobs"""
        probs_list = np.exp(logprobs_list)
        return self._entropy_from_probs(probs_list)

    def extract_probs(self, single_response_logprobs: List[Dict[str, Any]]) -> np.ndarray:
        """Extract probabilities from token data"""
        return np.exp(self.extract_logprobs(single_response_logprobs))

    @staticmethod
    def _compute_single_generation_scores(logprobs_results: List[List[Dict[str, Any]]], score_fn: Callable) -> List[float]:
        """Generic method to compute scores using the provided scoring function"""
        return [np.nan if not r else score_fn(r) for r in logprobs_results]

    @staticmethod
    def _entropy_from_probs(probs_list: np.ndarray, texts: Optional[List[str]] = None) -> float:
        """
        Compute entropy from a list of probabilities.
        """
        normalized_probs = probs_list / np.sum(probs_list)  # normalize probabilities to sum to 1

        if texts is None:
            # Case 1: If no responses are provided, treat all probabilities as distinct events
            logprobs = np.log(normalized_probs)
            return -np.sum(normalized_probs * logprobs)
        else:
            # Case 2: If responses, account for duplicates
            aggregated_probs = {}
            for text, prob in zip(texts, normalized_probs):
                if text in aggregated_probs:
                    aggregated_probs[text] += prob
                else:
                    aggregated_probs[text] = prob
            unique_probs = np.array(list(aggregated_probs.values()))
            logprobs = np.log(unique_probs)
            return -np.sum(unique_probs * logprobs)

    @staticmethod
    def extract_top_logprobs(single_response_logprobs: List[Dict[str, Any]]) -> List[np.ndarray]:
        """Extract top log probabilities for each token"""
        return [np.array([item["logprob"] for item in d["top_logprobs"]]) for d in single_response_logprobs]

    @staticmethod
    def extract_logprobs(single_response_logprobs: List[Dict[str, Any]]) -> np.ndarray:
        """Extract log probabilities from token data"""
        return np.array([d["logprob"] for d in single_response_logprobs])

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

from abc import ABC, abstractmethod
from typing import Any, List, Tuple, Optional, Union
from dataclasses import dataclass
import numpy as np
import time
from pydantic import BaseModel


class ClaimScore(BaseModel):
    claim: str
    original_response: bool
    scores: dict[str, float]
    scorer_type: str


@dataclass
class ClaimScores:
    """
    ClaimsScores is a dataclass that contains the aggregated score and the raw scores for each claim set.
    """

    def __init__(self, entailment_score_lists: List[np.ndarray] = None, noncontradict_score_lists: List[np.ndarray] = None, contrasted_entailment_score_lists: List[np.ndarray] = None, cosine_similarity_lists: List[np.ndarray] = None, bert_score_lists: List[np.ndarray] = None) -> None:
        self.entailment_score_lists = entailment_score_lists
        self.noncontradict_score_lists = noncontradict_score_lists
        self.contrasted_entailment_score_lists = contrasted_entailment_score_lists
        self.cosine_similarity_lists = cosine_similarity_lists
        self.bert_score_lists = bert_score_lists

    def to_dict(self, return_all: bool = False) -> dict:
        """Return results in dictionary form"""
        claim_scores_dict = {"entailment": self.entailment_score_lists, "noncontradiction": self.noncontradict_score_lists, "contrasted_entailment": self.contrasted_entailment_score_lists, "cosine_sim": self.cosine_similarity_lists, "bert_score": self.bert_score_lists}
        return {key: self._format_result(value, return_all) for key, value in claim_scores_dict.items() if value is not None}

    @staticmethod
    def _format_result(score_arrays: List[np.ndarray], return_all: bool = False) -> List[Any]:
        """Formats list of score arrays"""
        if not score_arrays:
            return None
        elif return_all:
            result = []
            for array in score_arrays:
                rows_as_lists = [row.tolist() for row in array]
                result.append(rows_as_lists)
            return result
        else:
            result = []
            for array in score_arrays:
                row_means = array.mean(axis=1).tolist()
                result.append(row_means)
        return result


class ClaimScorer(ABC):
    """Abstract class for text similarity scorers"""

    @abstractmethod
    def __init__(self):
        """Abstract constructor method"""

    @abstractmethod
    def evaluate(self, claim_sets: List[List[str]], sampled_responses: List[List[str]]) -> ClaimScores:
        """Abstract method for metric computation"""
        pass

    def _get_nli_agreement_scores(self, claim: str, candidate: str) -> float:
        """Compute probabilities from NLI model"""
        nli_result = self.nli.predict(hypothesis=claim, premise=candidate)
        # entail_prob = nli_result.entailment_probability
        # contradict_prob = nli_result.contradiction_probability
        entail_prob = nli_result[:, -1]
        contradict_prob = 1 - nli_result[:, 0]
        contrast_entail_prob = entail_prob / (entail_prob + contradict_prob)
        return entail_prob, (1 - contradict_prob), contrast_entail_prob

    def _compute_response_level_nli_score_lists(self, claim_sets: List[List[str]], sampled_responses: Optional[List[List[str]]] = None, sampled_claim_sets: Optional[List[List[List[str]]]] = None) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
        """Compute list of claim-level scores for each response"""
        if self.progress_bar:
            progress_task = self.progress_bar.add_task("  - Scoring claims/sentences with NLI...", total=len(claim_sets))
        n = len(claim_sets)
        claim_entail_score_lists, claim_noncontradict_score_lists, claim_constrast_entail_score_lists = [[]] * n, [[]] * n, [[]] * n
        for i, claim_set in enumerate(claim_sets):
            candidates = sampled_responses[i] if not self.matched_claim else sampled_claim_sets[i]
            claim_entail_score_lists[i], claim_noncontradict_score_lists[i], claim_constrast_entail_score_lists[i] = self._compute_claim_level_nli_scores(claims=claim_sets[i], candidates=candidates)
            if self.progress_bar:
                self.progress_bar.update(progress_task, advance=1)
        time.sleep(0.1)
        return claim_entail_score_lists, claim_noncontradict_score_lists, claim_constrast_entail_score_lists

    def _compute_claim_level_nli_scores(self, claims: List[str], candidates: Union[List[List[str]], List[str]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Evaluate the agreement scores for the provided claim with the provided set of candidate responses."""
        shape = (len(claims), len(candidates))
        entail_scores, noncontradict_scores, contrast_entail_scores = np.zeros(shape=shape), np.zeros(shape=shape), np.zeros(shape=shape)
        for i, claim in enumerate(claims):
            for j, candidate in enumerate(candidates):
                if self.matched_claim:
                    entail_scores[i, j], noncontradict_scores[i, j], contrast_entail_scores[i, j] = self._compute_matched_nli_scores(claim=claim, candidate_claims=candidate)
                else:
                    entail_scores[i, j], noncontradict_scores[i, j], contrast_entail_scores[i, j] = self._get_nli_agreement_scores(claim=claim, candidate=candidate)
        return entail_scores, noncontradict_scores, contrast_entail_scores

    def _compute_matched_nli_scores(self, claim: str, candidate_claims: List[str]) -> float:
        """Compute maximum matched-claim NLI score"""
        max_entailment_prob, max_noncontradict_prob, max_contrast_entail_prob = 0, 0, 0
        for candidate in candidate_claims:
            entail_prob, non_contradict_prob, contrast_entail_prob = self._get_nli_agreement_scores(claim=claim, candidate=candidate)
            max_entailment_prob = max(max_entailment_prob, float(entail_prob))
            max_noncontradict_prob = max(max_noncontradict_prob, float(non_contradict_prob))
            max_contrast_entail_prob = max(max_contrast_entail_prob, float(contrast_entail_prob))
        return max_entailment_prob, max_noncontradict_prob, max_contrast_entail_prob

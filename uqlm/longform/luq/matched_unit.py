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

from typing import List, Any, Optional, Tuple, Union

import numpy as np
import time
from rich.progress import Progress
from uqlm.black_box.cosine import CosineScorer
from uqlm.black_box.bert import BertScorer
from uqlm.longform.luq.baseclass.claims_scorer import ClaimScorer, ClaimScores
from uqlm.nli.nli import NLI


ALL_AGREEMENT_SCORER_NAMES = ["nli", "bert_score", "cosine_sim"]


class MatchedUnitScorer(ClaimScorer):
    def __init__(self, consistency_functions=["nli", "bert_score", "cosine_sim"], device: Any = None, transformer: str = "all-MiniLM-L6-v2", nli_model_name: str = "microsoft/deberta-large-mnli", max_length: int = 2000) -> None:
        """
        LUQScorer calculates variations of the LUQ, LUQ-Atomic, or LUQ-Pair scores.

        Parameters
        ----------
        consistency_functions: List[str], default=["nli", "bert_score", "cosine_sim"]
            Specifies which semantic consistency functions to use for scoring. Must be subset of ["nli", "bert_score", "cosine_sim"]

        device : torch.device input or torch.device object, default=None
            Specifies the device that classifiers use for prediction. Set to "cuda" for classifiers to be able to
            leverage the GPU.

        nli_model_name : str, default="microsoft/deberta-large-mnli"
            Specifies which NLI model to use. Must be acceptable input to AutoTokenizer.from_pretrained() and
            AutoModelForSequenceClassification.from_pretrained()

        max_length : int, default=2000
            Specifies the maximum allowed string length. Responses longer than this value will be truncated to
            avoid OutOfMemoryError
        """
        self.nli_model_name = nli_model_name
        self.consistency_functions = consistency_functions
        self.matched_claim = True
        if not set(consistency_functions).issubset(set(ALL_AGREEMENT_SCORER_NAMES)):
            raise ValueError("""consistency_functions must be subset of ["nli", "bertscore", "cosine_sim"]""")
        self.nli = NLI(device=device, nli_model_name=nli_model_name, max_length=max_length) if "nli" in consistency_functions else None
        self.cosine_scorer = CosineScorer(transformer=transformer) if "cosine_sim" in consistency_functions else None
        self.bert_scorer = BertScorer(device=device) if "bert_score" in consistency_functions else None
        self.progress_bar = None

    def evaluate(self, claim_sets: List[List[str]], sampled_claim_sets: List[List[List[str]]] = None, progress_bar: Optional[Progress] = None) -> ClaimScores:
        """
        Evaluate the LUQ score and claim scores for a list of claims from each original response and sampled responses.

        Parameters
        ----------
        claim_sets : list of list of strings
            List of original responses decomposed into lists of either claims or sentences

        sampled_claim_sets : list of list of list of strings
            Decomposed responses to be compared to the decomposed original responses

        progress_bar : rich.progress.Progress, default=None
            If provided, displays a progress bar while scoring responses

        Returns
        -------
        Instance of ClaimScores
            Contains claim-level entailment, non-contradiction, and contrasted entailment scores averaged across candidate responses.
        """
        entailment_score_lists, noncontradict_score_lists, contrasted_entailment_score_lists, cosine_similarity_lists, bert_score_lists = None, None, None, None, None
        self.progress_bar = progress_bar
        if len(claim_sets) != len(sampled_claim_sets):
            raise ValueError("claim_sets and sampled_claim_sets must be of equal length")
        if self.nli:
            entailment_score_lists, noncontradict_score_lists, contrasted_entailment_score_lists = self._compute_response_level_nli_score_lists(claim_sets=claim_sets, sampled_claim_sets=sampled_claim_sets)
        if self.cosine_scorer:
            cosine_similarity_lists = self._compute_response_level_cosine_score_lists(claim_sets=claim_sets, sampled_claim_sets=sampled_claim_sets)
        if self.bert_scorer:
            bert_score_lists = self._compute_response_level_bert_score_lists(claim_sets=claim_sets, sampled_claim_sets=sampled_claim_sets)
        return ClaimScores(entailment_score_lists=entailment_score_lists, noncontradict_score_lists=noncontradict_score_lists, contrasted_entailment_score_lists=contrasted_entailment_score_lists, cosine_similarity_lists=cosine_similarity_lists, bert_score_lists=bert_score_lists)

    def _compute_response_level_cosine_score_lists(self, claim_sets: List[List[str]], sampled_claim_sets: List[List[List[str]]]) -> List[List[float]]:
        """Compute list of claim-level scores for each response"""
        if self.progress_bar:
            progress_task = self.progress_bar.add_task("  - Scoring claims/sentences with cosine similarity...", total=len(claim_sets))
        n = len(claim_sets)
        cosine_similarity_lists = [[]] * n
        for i, claim_set in enumerate(claim_sets):
            cosine_similarity_lists[i] = self._compute_claim_level_cosine_scores(claims=claim_sets[i], candidates=sampled_claim_sets[i])
            if self.progress_bar:
                self.progress_bar.update(progress_task, advance=1)
        time.sleep(0.1)
        return cosine_similarity_lists

    def _compute_claim_level_cosine_scores(self, claims: List[str], candidates: Union[List[List[str]], List[str]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Evaluate the agreement scores for the provided claim with the provided set of candidate responses."""
        shape = (len(claims), len(candidates))
        cosine_scores = np.zeros(shape=shape)
        for i, claim in enumerate(claims):
            for j, candidate in enumerate(candidates):
                cosine_scores[i, j] = self._compute_matched_cosine_scores(claim, candidate)
        return cosine_scores

    def _compute_matched_cosine_scores(self, claim: str, candidate_claims: List[str]) -> float:
        """Compute maximum matched-unit cosine similarity score"""
        max_cosine_sim = 0
        for candidate in candidate_claims:
            cosine_sim = self.cosine_scorer._compute_score(claim, [candidate])
            max_cosine_sim = max(max_cosine_sim, float(cosine_sim))
        return max_cosine_sim

    def _compute_response_level_bert_score_lists(self, claim_sets: List[List[str]], sampled_claim_sets: List[List[List[str]]]) -> List[List[float]]:
        """Compute list of claim-level scores for each response"""
        if self.progress_bar:
            progress_task = self.progress_bar.add_task("  - Scoring claims/sentences with cosine similarity...", total=len(claim_sets))
        n = len(claim_sets)
        bert_score_lists = [[]] * n
        for i, claim_set in enumerate(claim_sets):
            bert_score_lists[i] = self._compute_claim_level_bert_scores(claims=claim_sets[i], candidates=sampled_claim_sets[i])
            if self.progress_bar:
                self.progress_bar.update(progress_task, advance=1)
        time.sleep(0.1)
        return bert_score_lists

    def _compute_claim_level_bert_scores(self, claims: List[str], candidates: Union[List[List[str]], List[str]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Evaluate the agreement scores for the provided claim with the provided set of candidate responses."""
        shape = (len(claims), len(candidates))
        bert_scores = np.zeros(shape=shape)
        for i, claim in enumerate(claims):
            for j, candidate in enumerate(candidates):
                bert_scores[i, j] = self._compute_matched_bert_scores(claim, candidate)
        return bert_scores

    def _compute_matched_bert_scores(self, claim: str, candidate_claims: List[str]) -> float:
        """Compute maximum matched-unit cosine similarity score"""
        max_bert_score = 0
        for candidate in candidate_claims:
            bert_score = self.bert_scorer._compute_score(claim, [candidate])
            max_bert_score = max(max_bert_score, float(bert_score))
        return max_bert_score

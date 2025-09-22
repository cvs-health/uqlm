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

from typing import List, Any, Tuple, Optional
import numpy as np
import time
from rich.progress import Progress
from uqlm.utils.nli import NLI
from uqlm.longform.black_box.baseclass.claims_scorer import ClaimScorer, ClaimScores


class LUQScorer(ClaimScorer):
    def __init__(self, nli_model_name: str = "microsoft/deberta-large-mnli", device: Any = None, max_length: int = 2000):
        """
        LUQScorer calculates variations of the LUQ, LUQ-Atomic, or LUQ-Pair scores.

        Parameters
        ----------
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
        self.device = device
        self.nli = NLI(device=device, nli_model_name=nli_model_name, max_length=max_length)

    def evaluate(self, claim_sets: List[List[str]], sampled_responses: Optional[List[List[str]]] = None, sampled_claims: Optional[List[List[List[str]]]] = None, progress_bar: Optional[Progress] = None) -> ClaimScores:
        """
        Evaluate the LUQ score and claim scores for a list of claims from each original response and sampled responses.

        Parameters
        ----------
        claim_sets : list of list of strings
            List of original responses decomposed into lists of either claims or sentences

        sampled_responses : list of list of strings
            Candidate responses to be compared to the decomposed original responses
            
        sampled_claims : list of list of list of strings
            Decomposed responses to be compared to the decomposed original responses

        progress_bar : rich.progress.Progress, default=None
            If provided, displays a progress bar while scoring responses

        Returns
        -------
        Instance of ClaimScores
            Contains claim-level entailment, non-contradiction, and contrasted entailment scores averaged across candidate responses.
        """
        if progress_bar:
            progress_task = progress_bar.add_task("  - Scoring claim/sentence sets with LUQ...", total=len(claim_sets))
        claim_entail_score_lists, claim_noncontradict_score_lists, claim_constrast_entail_score_lists = [], [], []
        if sampled_responses and sampled_claims:
            raise ValueError("Only one of sampled_responses and sampled_claims may be provided.")
        if sampled_responses: # For response-to-claim or response-to-sentence comparisons
            if sampled_claims:
                raise ValueError("Only one of sampled_responses and sampled_claims may be provided.")     
            samples_for_comparison = sampled_responses 
            matched_claims = False
        else:  # For matched claim-to-claim comparisons
            samples_for_comparison = sampled_claims   
            matched_claims = True
        for (claim_set, sample_for_comparison) in zip(claim_sets, samples_for_comparison):
            claim_entail_scores, claim_noncontradict_scores, claim_constrast_entail_scores = self._compute_claim_level_scores(
                claims=claim_set, candidates=sample_for_comparison, matched_claims=matched_claims
            )
            claim_entail_score_lists.append(claim_entail_scores)
            claim_noncontradict_score_lists.append(claim_noncontradict_scores)
            claim_constrast_entail_score_lists.append(claim_constrast_entail_scores)
            if progress_bar:
                progress_bar.update(progress_task, advance=1)
        time.sleep(0.1)
        return ClaimScores(claim_entail_scores=claim_entail_score_lists, claim_noncontradict_scores=claim_noncontradict_score_lists, claim_constrast_entail_scores=claim_constrast_entail_score_lists)

    def _compute_claim_level_scores(self, claims: List[str], candidates: List[str]| List[List[str]], matched_claims: bool) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Evaluate the LUQ score and claim scores for a list of claims and candidate responses."""
        shape=(len(claims), len(candidates))
        entail_scores = np.zeros(shape=shape)
        noncontradict_scores = np.zeros(shape=shape)
        contrast_entail_scores = np.zeros(shape=shape)
        for i, claim in enumerate(claims):
            for j, candidate in enumerate(candidates):
                if matched_claims:
                    entail_prob, noncontradict_prob, contrast_entail_prob = self._compute_matched_nli_scores(claim, candidate)
                else:
                    entail_prob, noncontradict_prob, contrast_entail_prob = self._compute_nli_scores(claim, candidate)
                entail_scores[i, j] = entail_prob
                noncontradict_scores[i, j] = noncontradict_prob
                contrast_entail_scores[i, j] = contrast_entail_prob
        claim_entail_scores = entail_scores.mean(axis=1)
        claim_noncontradict_scores = noncontradict_scores.mean(axis=1)
        claim_constrast_entail_scores = contrast_entail_scores.mean(axis=1)
        return claim_entail_scores, claim_noncontradict_scores, claim_constrast_entail_scores
    
    def _compute_matched_nli_scores(self, claim: str, candidate_claims: List[str]) -> float:
        """Compute maximum matched-claim NLI score"""
        max_entailment_prob, max_noncontradict_prob, max_contrast_entail_prob = 0, 0, 0
        for candidate in candidate_claims:
            entail_prob, non_contradict_prob, contrast_entail_prob = self._compute_nli_scores(
                claim=claim, candidate=candidate
            )
            max_entailment_prob = max(max_entailment_prob, float(entail_prob))
            max_noncontradict_prob = max(max_noncontradict_prob, float(non_contradict_prob))
            max_contrast_entail_prob = max(max_contrast_entail_prob, float(contrast_entail_prob))
        return max_entailment_prob, max_noncontradict_prob, max_contrast_entail_prob
        
        
    def _compute_nli_scores(self, claim: str, candidate: str) -> float:
        """Compute probabilities from NLI model"""
        nli_probabilities = self.nli.predict(hypothesis=candidate, premise=claim)
        entail_prob = nli_probabilities[:, 2]
        contradict_prob = nli_probabilities[:, 0]
        contrast_entail_prob = entail_prob / (entail_prob + contradict_prob)
        return entail_prob, (1 - contradict_prob), contrast_entail_prob

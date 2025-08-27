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
from uqlm.black_box.baseclass.claims_scorer import ClaimScorer, ClaimScores

class LUQScorer(ClaimScorer):
    """
    LUQScorer calculates the LUQ score .
    """
    def __init__(self, nli_model_name: str = "microsoft/deberta-large-mnli", device: Any = None, max_length: int = 2000):
        self.nli_model_name = nli_model_name
        self.device = device
        self.nli = NLI(device=device,
                       nli_model_name=nli_model_name,
                       max_length=max_length)
        
    def evaluate(self, claim_sets: List[List[str]], sampled_responses: List[List[str]], progress_bar: Optional[Progress] = None) -> ClaimScores:
        """
        Evaluate the LUQ score and claim scores for a list of claims from each original response and sampled responses.
        """
        if progress_bar:
            progress_task = progress_bar.add_task("  - [black]Scoring claim sets with LUQ...", total=len(claim_sets))
        luq_scores = np.zeros(len(claim_sets))
        entailment_scores = []
        claim_scores = []
        for i, (claim_set, candidates) in enumerate(zip(claim_sets, sampled_responses)):  
            luq_score, claim_scores_, entailment_scores_ = self._compute_luq_score(claim_set, candidates)
            luq_scores[i] = luq_score
            claim_scores.append(claim_scores_)
            entailment_scores.append(entailment_scores_)
            if progress_bar:
                progress_bar.update(progress_task, advance=1)
        time.sleep(0.1)
        return ClaimScores(response_scores=luq_scores, claim_scores=claim_scores, entailment_scores=entailment_scores)

    def _compute_luq_score(self, claims: List[str], candidate_responses: List[str]) -> Tuple[float, np.ndarray,np.ndarray]:
        """Evaluate the LUQ score and claim scores for a list of claims and candidate responses."""
        scores = np.zeros(shape=(len(claims), len(candidate_responses)))
        for i, claim in enumerate(claims):
            for j, candidate in enumerate(candidate_responses):
                scores[i, j] = self._compute_entailment_score(claim, candidate)
        claim_scores_ = scores.mean(axis=1)
        return claim_scores_.mean(), claim_scores_, scores
    
    def _compute_entailment_score(self, claim: str, sample: str) -> float:
        nli_proba = self.nli.predict(hypothesis=sample, premise=claim)
        nli_label = self.nli.label_mapping[nli_proba.argmax(axis=1)[0]]
        if nli_label == "entailment":
            return 1
        if nli_label == "neutral":
            return 0.5
        return 0
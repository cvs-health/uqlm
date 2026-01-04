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

from typing import List, Any, Optional
from rich.progress import Progress
from uqlm.nli.nli import NLI
from uqlm.longform.luq.baseclass.claims_scorer import ClaimScorer, ClaimScores


class UnitResponseScorer(ClaimScorer):
    def __init__(self, nli_model_name: str = "microsoft/deberta-large-mnli", device: Any = None, max_length: int = 2000) -> None:
        """
        LUQScorer calculates variations of the LUQ, LUQ-Atomic, or LUQ-Pair scores.

        Parameters
        ----------
        nli_model_name : str, default="microsoft/deberta-large-mnli"
            Specifies which NLI model to use. Must be acceptable input to AutoTokenizer.from_pretrained() and
            AutoModelForSequenceClassification.from_pretrained()

        device : torch.device input or torch.device object, default=None
            Specifies the device that classifiers use for prediction. Set to "cuda" for classifiers to be able to
            leverage the GPU.

        max_length : int, default=2000
            Specifies the maximum allowed string length. Responses longer than this value will be truncated to
            avoid OutOfMemoryError
        """
        self.nli_model_name = nli_model_name
        self.nli = NLI(device=device, nli_model_name=nli_model_name, max_length=max_length)
        self.progress_bar = None
        self.matched_claim = False

    def evaluate(self, claim_sets: List[List[str]], sampled_responses: List[List[str]], progress_bar: Optional[Progress] = None) -> ClaimScores:
        """
        Evaluate the LUQ score and claim scores for a list of claims from each original response and sampled responses.

        Parameters
        ----------
        claim_sets : list of list of strings
            List of original responses decomposed into lists of either claims or sentences

        sampled_responses : list of list of strings
            Candidate responses to be compared to the decomposed original responses

        progress_bar : rich.progress.Progress, default=None
            If provided, displays a progress bar while scoring responses

        Returns
        -------
        Instance of ClaimScores
            Contains claim-level entailment, non-contradiction, and contrasted entailment scores averaged across candidate responses.
        """
        self.progress_bar = progress_bar
        if len(claim_sets) != len(sampled_responses):
            raise ValueError("claim_sets and sampled_responses must be of equal length")
        entailment_score_lists, noncontradict_score_lists, contrasted_entailment_score_lists = self._compute_response_level_nli_score_lists(claim_sets=claim_sets, sampled_responses=sampled_responses)
        return ClaimScores(entailment_score_lists=entailment_score_lists, noncontradict_score_lists=noncontradict_score_lists, contrasted_entailment_score_lists=contrasted_entailment_score_lists)

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
from typing import Any, List, Optional
from bert_score import BERTScorer
import torch

from uqlm.black_box.baseclass.similarity_scorer import SimilarityScorer
from uqlm.utils.device import get_best_device

import time
from rich.progress import Progress


class BertScorer(SimilarityScorer):
    def __init__(self, device: Any = None) -> None:
        """
        Class for computing BERTScore values between original responses and candidates. For more on
        BERTScore, refer to Zhang et al.(2020) :footcite:`zhang2020bertscoreevaluatingtextgeneration`.

        Parameters
        ----------
        device : torch.device input or torch.device object, default=None
            Specifies the device that classifiers use for prediction. Set to "cuda" for classifiers to be able to
            leverage the GPU.
        """
        # Handle device detection
        if device is None:
            device = get_best_device()
        elif isinstance(device, str):
            device = torch.device(device)

        from transformers import logging

        logging.set_verbosity_error()
        self.bert_scorer = BERTScorer(device=device, lang="en")

    def evaluate(self, responses: List[str], sampled_responses: List[List[str]], progress_bar: Optional[Progress] = None) -> List[float]:
        """
        This method computes model-based text similarity metrics values for the provided pairs of texts.

        Parameters
        ----------
        responses : list of strings
            Original LLM response

        sampled_responses : list of list of strings
            Candidate responses to be compared to the original response

        progress_bar : rich.progress.Progress, default=None
            If provided, displays a progress bar while scoring responses

        Returns
        -------
        List of float
            Mean BertScore values
        """
        if progress_bar:
            progress_task = progress_bar.add_task("  - Scoring responses with BERTScore...", total=len(responses))
        results = []
        for i in range(len(responses)):
            score = self._compute_score(response=responses[i], candidates=sampled_responses[i])
            results.append(score)
            if progress_bar:
                progress_bar.update(progress_task, advance=1)
        time.sleep(0.1)
        return results

    def _compute_score(self, response: str, candidates: List[str]) -> float:
        """Compute mean BERTScore between a response and candidate responses"""
        num_responses = len(candidates)
        duplicated_response = [response] * num_responses
        P, R, F1 = self.bert_scorer.score(list(duplicated_response), refs=list(candidates))
        return np.mean([float(f) for f in F1])

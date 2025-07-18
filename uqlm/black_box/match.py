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
from typing import List, Optional

from uqlm.black_box.baseclass.similarity_scorer import SimilarityScorer

import time
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn


class MatchScorer(SimilarityScorer):
    def __init__(self) -> None:
        """
        Class for computing exact match rates between original responses and candidates. This
        method is based on Cole et al.(2023) :footcite:`cole2023selectivelyansweringambiguousquestions`.
        """
        pass

    def evaluate(self, responses: List[str], sampled_responses: List[List[str]], progress_bar: Optional[bool] = True) -> List[float]:
        """
        This method computes exact match rates for the provided pairs of texts.

        Parameters
        ----------
        responses : list of strings
            Original LLM response

        sampled_responses : list of list of strings
            Candidate responses to be compared to the original response

        progress_bar : bool, default=True
            If True, displays a progress bar while scoring responses

        Returns
        -------
        List of float
            Exact match rates
        """
        if progress_bar:
            with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), BarColumn(), TextColumn("[progress.percentage]{task.completed}/{task.total}"), TimeElapsedColumn()) as progress:
                task = progress.add_task("- [black]Scoring responses with Exact Match...", total=len(responses))
                results = []
                for i, (response, candidates) in enumerate(zip(responses, sampled_responses)):
                    score = self._compute_score(response=response, candidates=candidates)
                    results.append(score)
                    progress.update(task, advance=1)
                time.sleep(0.1)
                return results
        else:
            return [self._compute_score(response=o, candidates=mr) for o, mr in zip(responses, sampled_responses)]

    @staticmethod
    def _compute_score(response: str, candidates: List[str]) -> List[float]:
        """Get mean exact match rate between response and set of candidates"""
        return np.mean([1 if response == c else 0 for c in candidates])

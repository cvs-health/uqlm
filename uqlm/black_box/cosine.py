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


from typing import Any, List, Tuple

import numpy as np
from numpy.linalg import norm
from sentence_transformers import SentenceTransformer

from uqlm.black_box.baseclass.similarity_scorer import SimilarityScorer


class CosineScorer(SimilarityScorer):
    def __init__(self, transformer: str = "all-MiniLM-L6-v2") -> None:
        """Compute cosine similarity betwee original and candidate responses.

        Parameters
        ----------
        transformer : str (HuggingFace sentence transformer), default='all-MiniLM-L6-v2'
            Specifies which huggingface sentence transformer to use when computing cosine distance. See
            https://huggingface.co/sentence-transformers?sort_models=likes#models
            for more information. The recommended sentence transformer is 'all-MiniLM-L6-v2'.
        """
        self.transformer = transformer
        self.model = SentenceTransformer(f"sentence-transformers/{transformer}")

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
            Mean cosine similarity values
        """
        return [self._compute_score(response=responses[i], candidates=sampled_responses[i]) for i in range(len(responses))]

    def _get_embeddings(self, texts1: List[str], texts2: List[str]) -> Tuple[Any, Any]:
        """
        Helper function to get embeddings
        """
        embeddings1 = self.model.encode(texts1)
        embeddings2 = self.model.encode(texts2)
        return embeddings1, embeddings2

    def _compute_score(self, response: str, candidates: List[str]) -> float:
        """
        Helper function to get cosine dist
        """
        duplicate_responses = [response] * len(candidates)
        embeddings1, embeddings2 = self._get_embeddings(duplicate_responses, candidates)
        cosine_list = []
        for i in range(0, len(embeddings1)):
            cosine_i = np.dot(embeddings1[i], embeddings2[i]) / (norm(embeddings1[i]) * norm(embeddings2[i]))
            norm_cosine_i = 0.5 + cosine_i / 2
            cosine_list.append(norm_cosine_i)
        return np.mean(cosine_list)

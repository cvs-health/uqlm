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
from typing import Any, List, Optional, Dict
from uqlm.utils.results import UQResult
from uqlm.code.clusterer import CodeClusterer
from rich.progress import Progress
from uqlm.nli.entropy_utils import compute_response_probabilities, compute_semantic_entropy, normalize_cluster_probabilities, normalize_entropy


class FunctionalEntropy:
    def __init__(self, equivalence_llm: Any, system_prompt: Optional[str] = None, length_normalize: bool = False, language: str = "python", retries: int = 5) -> None:
        """
        Class for computing discrete and token-probability-based functional entropy and associated confidence scores. For more on functional entropy, refer to Farquhar et al.(2024) :footcite:`farquhar2024detectinghallucinations`.

        Parameters
        ----------
        equivalence_llm : Any
            A langchain llm `BaseChatModel`. User is responsible for specifying temperature and other
            relevant parameters to the constructor of their `equivalence_llm` object.

        system_prompt : str, default=None
            Optional argument for user to provide custom system prompt. If prompts are list of strings and system_prompt is None,
            defaults to "You are a helpful assistant."

        max_calls_per_min : int, default=None
            Specifies how many api calls to make per minute to avoid a rate limit error. By default, no
            limit is specified.

        length_normalize : bool, default=False
            Specifies whether to length normalize the logprobs.

        language : str, default="python"
            Specifies the language of the code. Must be one of python, java, sql.

        retries : int, default=5
            Specifies the number of retries to make if the equivalence score is not found.
        """
        self.logprobs = None
        self.multiple_logprobs = None
        self.length_normalize = length_normalize
        self.use_logprobs = False
        self.clusterer = CodeClusterer(llm=equivalence_llm, language=language, system_prompt=system_prompt, retries=retries)

    async def evaluate(self, responses: List[str] = None, sampled_responses: List[List[str]] = None, logprobs_results: Optional[List[List[Dict[str, Any]]]] = None, sampled_logprobs_results: Optional[List[List[List[Dict[str, Any]]]]] = None, progress_bar: Optional[Progress] = None) -> UQResult:
        """
        Evaluate functional entropy scores on the provided responses and sampled responses.

        Parameters
        ----------
        responses : list of str, default=None
            A list of model responses for the prompts.

        sampled_responses : list of list of str, default=None
            A list of lists of sampled model responses for each prompt. These will be used to compute consistency scores by comparing to
            the corresponding response from `responses`.

        logprobs_results : list of list of dict, default=None
            A list of lists of logprobs results for each prompt.

        sampled_logprobs_results : list of list of list of dict, default=None
            A list of lists of lists of logprobs results for each prompt.

        show_progress_bars : bool, default=True
            If True, displays a progress bar while evaluating the functional entropy scores.

        Returns
        -------
        UQResult
            UQResult, containing data (responses, sampled responses, and functional entropy scores) and metadata
        """
        self.responses = responses
        self.sampled_responses = sampled_responses
        self.num_responses = len(self.sampled_responses[0])
        self.logprobs = logprobs_results if logprobs_results else self.logprobs
        self.multiple_logprobs = sampled_logprobs_results if sampled_logprobs_results else self.multiple_logprobs

        n_prompts = len(self.responses)
        discrete_functional_entropy = [None] * n_prompts
        tokenprob_functional_entropy = [None] * n_prompts
        num_functional_sets = [None] * n_prompts

        cluster_result = await self.clusterer.evaluate(responses=responses, sampled_responses=sampled_responses, progress_bar=progress_bar)
        cluster_indices = cluster_result["cluster_indices"]
        original_equivalence_scores = cluster_result["original_equivalence_scores"]

        for i in range(n_prompts):
            candidate_logprobs = [list(self.logprobs[i])] + [list(ml) for ml in self.multiple_logprobs[i]] if (self.logprobs and self.multiple_logprobs) else None
            tmp = self._functional_entropy_process(single_prompt_cluster_indices=cluster_indices[i], logprobs_results=candidate_logprobs)
            discrete_functional_entropy[i], tokenprob_functional_entropy[i], num_functional_sets[i] = tmp

        data_to_return = dict()
        data_to_return["original_equivalence_scores"] = original_equivalence_scores
        data_to_return["functional_equivalence_rate"] = [np.mean(oes) for oes in original_equivalence_scores]
        data_to_return["discrete_entropy_values"] = discrete_functional_entropy
        data_to_return["functional_negentropy"] = [1 - ne for ne in normalize_entropy(discrete_functional_entropy, num_responses=self.num_responses)]
        data_to_return["num_functional_sets"] = num_functional_sets
        data_to_return["functional_sets_confidence"] = [(self.num_responses + 1 - num_functional_sets[i]) / (self.num_responses) for i in range(n_prompts)]
        data_to_return["cluster_indices"] = cluster_indices

        if tokenprob_functional_entropy[0] is not None:
            data_to_return["entropy_values_whitebox"] = tokenprob_functional_entropy
            data_to_return["functional_negentropy_whitebox"] = [1 - ne for ne in normalize_entropy(tokenprob_functional_entropy, num_responses=self.num_responses)]

        return data_to_return

    def _functional_entropy_process(self, single_prompt_cluster_indices: List[str], i: int = None, logprobs_results: List[List[Dict[str, Any]]] = None) -> Any:
        """
        Executes complete process for functional entropy and returns response, SE score, and dictionary
        of Equivalence scores for response pairs.
        """
        # Compute response probabilities
        tokenprob_response_probabilities, response_probabilities = compute_response_probabilities(logprobs_results=logprobs_results, num_responses=self.num_responses, length_normalize=self.length_normalize)

        # Compute Clusters and Equivalence scores
        cluster_probabilities = self._compute_cluster_probabilities(response_probabilities=response_probabilities, single_prompt_cluster_indices=single_prompt_cluster_indices)
        num_functional_sets = len(cluster_probabilities)

        # Compute discrete functional entropy
        discrete_functional_entropy = compute_semantic_entropy(cluster_probabilities=cluster_probabilities)

        # Compute token-level functional entropy
        tokenprob_functional_entropy = None
        if tokenprob_response_probabilities:
            tokenprob_cluster_probabilities = self._compute_cluster_probabilities(response_probabilities=tokenprob_response_probabilities, single_prompt_cluster_indices=single_prompt_cluster_indices)
            tokenprob_functional_entropy = compute_semantic_entropy(cluster_probabilities=tokenprob_cluster_probabilities)

        return (discrete_functional_entropy, tokenprob_functional_entropy, num_functional_sets)

    @staticmethod
    def _compute_cluster_probabilities(single_prompt_cluster_indices: List[List[int]], response_probabilities: List[float]) -> List[float]:
        """Compute cluster probabilities"""
        cluster_probabilities = [0] * len(single_prompt_cluster_indices)
        for i, cluster_index in enumerate(single_prompt_cluster_indices):
            cluster_probabilities[i] = sum([response_probabilities[j - 1] for j in cluster_index])
        return normalize_cluster_probabilities(cluster_probabilities=cluster_probabilities)

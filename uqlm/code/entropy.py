import math
import numpy as np
from typing import Any, List, Optional, Dict
from uqlm.utils.results import UQResult
from uqlm.code.clusterer import CodeClusterer


class FunctionalEntropy:
    def __init__(self, equivalence_llm: Any, system_prompt: Optional[str] = None, length_normalize: bool = False, language: str = "python", retries: int = 5) -> None:
        """
        Class for computing discrete and token-probability-based semantic entropy and associated confidence scores. For more on semantic entropy, refer to Farquhar et al.(2024) :footcite:`farquhar2024detectinghallucinations`.

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
        self.progress_bar = None

    async def evaluate(self, responses: List[str] = None, sampled_responses: List[List[str]] = None, logprobs_results: Optional[List[List[Dict[str, Any]]]] = None, sampled_logprobs_results: Optional[List[List[List[Dict[str, Any]]]]] = None, equivalence_indicators: List[Dict] = None, show_progress_bars: Optional[bool] = True, _display_header: bool = True) -> UQResult:
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

        if not equivalence_indicators:
            equivalence_indicators = [{} for i in responses]
        else:
            if len(equivalence_indicators) != len(responses):
                raise RuntimeError("UQLM: Length of equivalence_indicators and responses does not match.")
        self.equivalence_indicators = equivalence_indicators

        # self._construct_progress_bar(show_progress_bars)
        # self._display_scoring_header(show_progress_bars and _display_header)

        n_prompts = len(self.responses)
        discrete_semantic_entropy = [None] * n_prompts
        tokenprob_semantic_entropy = [None] * n_prompts
        num_semantic_sets = [None] * n_prompts

        cluster_result = await self.clusterer.evaluate(responses=responses, sampled_responses=sampled_responses, progress_bar=self.progress_bar)
        cluster_indices = cluster_result["cluster_indices"]
        original_equivalence_scores = cluster_result["original_equivalence_scores"]

        for i in range(n_prompts):
            candidate_logprobs = [list(self.logprobs[i])] + [list(ml) for ml in self.multiple_logprobs[i]] if (self.logprobs and self.multiple_logprobs) else None
            tmp = self._semantic_entropy_process(single_prompt_cluster_indices=cluster_indices[i], logprobs_results=candidate_logprobs)
            discrete_semantic_entropy[i], tokenprob_semantic_entropy[i], num_semantic_sets[i] = tmp

        data_to_return = dict()
        data_to_return["original_equivalence_scores"] = original_equivalence_scores
        data_to_return["equivalence_rate"] = [np.mean(oes) for oes in original_equivalence_scores]
        data_to_return["discrete_entropy_values"] = discrete_semantic_entropy
        data_to_return["discrete_confidence_scores"] = [1 - ne for ne in self._normalize_entropy(discrete_semantic_entropy)]
        data_to_return["num_semantic_sets"] = num_semantic_sets
        data_to_return["semantic_sets_confidence"] = [(self.num_responses + 1 - num_semantic_sets[i]) / (self.num_responses) for i in range(n_prompts)]
        data_to_return["cluster_indices"] = cluster_indices

        if tokenprob_semantic_entropy[0] is not None:
            data_to_return["tokenprob_entropy_values"] = tokenprob_semantic_entropy
            data_to_return["tokenprob_confidence_scores"] = [1 - ne for ne in self._normalize_entropy(tokenprob_semantic_entropy)]

        result = {"data": data_to_return, "metadata": {"parameters": {}}}

        # self._stop_progress_bar()
        self.progress_bar = None  # if re-run ensure the same progress object is not used
        return UQResult(result)

    def _semantic_entropy_process(self, single_prompt_cluster_indices: List[str], i: int = None, logprobs_results: List[List[Dict[str, Any]]] = None) -> Any:
        """
        Executes complete process for semantic entropy and returns response, SE score, and dictionary
        of Equivalence scores for response pairs.
        """
        # Compute response probabilities
        tokenprob_response_probabilities, response_probabilities = self._compute_response_probabilities(logprobs_results=logprobs_results, num_responses=self.num_responses)

        # Compute Clusters and Equivalence scores
        cluster_probabilities = self._compute_cluster_probabilities(response_probabilities=response_probabilities, single_prompt_cluster_indices=single_prompt_cluster_indices)
        num_semantic_sets = len(cluster_probabilities)

        # Compute discrete semantic entropy
        discrete_semantic_entropy = self._compute_semantic_entropy(cluster_probabilities=cluster_probabilities)

        # Compute token-level semantic entropy
        tokenprob_semantic_entropy = None
        if tokenprob_response_probabilities:
            tokenprob_cluster_probabilities = self._compute_cluster_probabilities(response_probabilities=tokenprob_response_probabilities, single_prompt_cluster_indices=single_prompt_cluster_indices)
            tokenprob_semantic_entropy = self._compute_semantic_entropy(cluster_probabilities=tokenprob_cluster_probabilities)

        return (discrete_semantic_entropy, tokenprob_semantic_entropy, num_semantic_sets)

    def _normalize_entropy(self, entropy_values):
        return [e / math.log(self.num_responses + 1) for e in entropy_values]

    def _compute_response_probabilities(self, logprobs_results: List[List[Dict[str, Any]]], num_responses: int = None) -> List[float]:
        """Compute response probabilities"""
        uniform_response_probabilities = [1 / num_responses] * num_responses
        tokenprob_response_probabilities = [self.length_norm_sequence_prob(logprobs_i, self.length_normalize) if logprobs_i else np.nan for logprobs_i in logprobs_results] if logprobs_results else None
        return tokenprob_response_probabilities, uniform_response_probabilities

    def _compute_cluster_probabilities(self, single_prompt_cluster_indices: List[List[int]], response_probabilities: List[float]) -> List[float]:
        """Compute cluster probabilities"""
        cluster_probabilities = [0] * len(single_prompt_cluster_indices)
        for i, cluster_index in enumerate(single_prompt_cluster_indices):
            cluster_probabilities[i] = sum([response_probabilities[j - 1] for j in cluster_index])
        return self._normalize_cluster_probabilities(cluster_probabilities=cluster_probabilities)

    @staticmethod
    def _compute_semantic_entropy(cluster_probabilities: List[float]) -> float:
        """
        Helper function to compute semantic entropy score from cluster probabilities
        """
        return abs(sum([p * math.log(p) if p > 0.0 else 0 for p in cluster_probabilities]))

    @staticmethod
    def length_norm_sequence_prob(logprobs: List[Dict[str, Any]], length_normalize: bool = True) -> float:
        "Compute length normalized sequence logprob"
        factor = 1 / len(logprobs) if length_normalize else 1
        return np.exp(np.sum([d["logprob"] for d in logprobs]) * factor)

    @staticmethod
    def _normalize_cluster_probabilities(cluster_probabilities: List[float]) -> float:
        """Normalize cluster probabilities"""
        total_probability = sum(cluster_probabilities)
        return [cp_i / total_probability for cp_i in cluster_probabilities]

from collections import deque, Counter
from typing import Any, Dict, List, Tuple
from uqlm.nli.nli import NLI
import numpy as np


class SemanticClusterer:
    def __init__(self, nli: NLI = None, length_normalize: bool = False):
        self.nli = nli
        self.length_normalize = length_normalize
        self.nli_scores = {"noncontradiction": dict(), "entailment": dict()}

    def evaluate(self, responses: List[str], prompt: str = None, response_probabilities: List[float] = None) -> Tuple[str, List[List[str]], List[float], Dict[Tuple[str, str], float]]:
        """
        Evaluate the cluster of responses.
        """
        clustered_responses, cluster_indices, noncontradiction_scores, entailment_scores = self.cluster_responses(responses=responses, prompt=prompt)
        self.nli_scores["noncontradiction"].update(noncontradiction_scores)
        self.nli_scores["entailment"].update(entailment_scores)
        cluster_probabilities = self.compute_cluster_probabilities(response_probabilities=response_probabilities, cluster_indices=cluster_indices)
        best_response = self.best_response_selection(clustered_responses=clustered_responses, cluster_probabilities=cluster_probabilities)
        return best_response, clustered_responses, cluster_probabilities, cluster_indices

    def cluster_responses(self, responses: List[str], prompt: str = None) -> Any:
        """
        This method create clusters from a list of responses based on the semantic meaning of each response.

        Parameters
        ----------
        responses : list of str, default=None
            A list of model responses

        prompt : str, default=None
            A prompt for the responses.

        Returns
        ----------
        A list of lists, where each list represents a cluster.
        """
        clusters, cluster_indices = [deque([responses[0]])], [deque([0])]
        noncontradiction_scores = {}
        entailments = {}
        entailment_scores = {}
        for i in range(1, len(responses)):
            new_cluster_indicator = True
            for j, cluster in enumerate(clusters):
                text1 = f"{prompt}\n{cluster[0]}" if prompt else cluster[0]
                text2 = f"{prompt}\n{responses[i]}" if prompt else responses[i]
                key, rev_key = (text1, text2), (text2, text1)
                if key in noncontradiction_scores:
                    # Do not recompute if pair already assessed
                    entailment = entailments[key]
                else:
                    # Compute nli score and entailment if pair not yet assessed
                    nli_result = self.nli.get_nli_results(response1=text1, response2=text2)
                    noncontradiction_score, entailment, entailment_score = nli_result["noncontradiction_score"], nli_result["entailment"], nli_result["entailment_score"]
                    noncontradiction_scores[key], noncontradiction_scores[rev_key] = noncontradiction_score, noncontradiction_score
                    entailments[key], entailments[rev_key] = entailment, entailment
                    entailment_scores[key], entailment_scores[rev_key] = entailment_score, entailment_score
                if entailment:
                    new_cluster_indicator = False
                    cluster.append(responses[i])
                    cluster_indices[j].append(i)

            if new_cluster_indicator:
                clusters.append(deque([responses[i]]))
                cluster_indices.append(deque([i]))

        # Arrange cluster so that first element is mode (if exists) else longest
        clusters = [self._sort_responses(list(cluster)) for cluster in clusters]

        return clusters, cluster_indices, noncontradiction_scores, entailment_scores

    def compute_response_probabilities(self, logprobs_results: List[List[Dict[str, Any]]], num_responses: int = None) -> List[float]:
        """Compute response probabilities"""
        uniform_response_probabilities = [1 / num_responses] * num_responses
        tokenprob_response_probabilities = [self.length_norm_sequence_prob(logprobs_i, self.length_normalize) if logprobs_i else np.nan for logprobs_i in logprobs_results] if logprobs_results else None
        return tokenprob_response_probabilities, uniform_response_probabilities

    def compute_cluster_probabilities(self, response_probabilities: List[float], cluster_indices: List[List[int]]) -> List[float]:
        """Compute cluster probabilities"""
        cluster_probabilities = [0] * len(cluster_indices)
        for i, cluster_index in enumerate(cluster_indices):
            cluster_probabilities[i] = sum([response_probabilities[j] for j in cluster_index])
        return self._normalize_cluster_probabilities(cluster_probabilities=cluster_probabilities)

    @staticmethod
    def length_norm_sequence_prob(logprobs: List[Dict[str, Any]], length_normalize: bool = True) -> float:
        "Compute length normalized sequence logprob"
        factor = 1 / len(logprobs) if length_normalize else 1
        return np.exp(np.sum([d["logprob"] for d in logprobs]) * factor)

    @staticmethod
    def best_response_selection(clustered_responses: List[List[str]], cluster_probabilities: List[float]) -> str:
        """Select the best response from the clustered responses based on the cluster probabilities"""
        return clustered_responses[cluster_probabilities.index(max(cluster_probabilities))][0]

    @staticmethod
    def _normalize_cluster_probabilities(cluster_probabilities: List[float]) -> float:
        """Normalize cluster probabilities"""
        total_probability = sum(cluster_probabilities)
        return [cp_i / total_probability for cp_i in cluster_probabilities]

    @staticmethod
    def _sort_responses(responses: List[str]) -> List[str]:
        """Sorts responses in a cluster"""
        counter = Counter(responses)
        mode_str, count = counter.most_common(1)[0]
        if count > 1:
            return sorted(responses, key=lambda x: (x != mode_str, x))
        else:
            return sorted(responses, key=len, reverse=True)

from collections import deque, Counter
from typing import Any, Dict, List, Tuple
from uqlm.nli.nli import NLIScorer
import numpy as np

class Cluster:
    def __init__(self, nli_scorer: NLIScorer=None):
        self.nli_scorer = nli_scorer

    def evaluate(self, responses: List[str], response_probabilities: List[float]) -> Tuple[str, List[List[str]], List[float], Dict[Tuple[str, str], float]]:
        """
        Evaluate the cluster of responses.
        """
        clustered_responses, cluster_indices, self.nli_scores = self.cluster_responses(responses=responses)
        cluster_probabilities = self.compute_cluster_probabilities(response_probabilities=response_probabilities, cluster_indices=cluster_indices)
        best_response = self.best_response_selection(clustered_responses=clustered_responses, cluster_probabilities=cluster_probabilities)
        return best_response, clustered_responses, cluster_probabilities, cluster_indices

    def cluster_responses(self, responses: List[str]) -> Any:
        """
        This method create clusters from a list of responses based on the semantic meaning of each response.

        Parameters
        ----------
        responses : list of str, default=None
            A list of model responses

        Returns
        ----------
        A list of lists, where each list represents a cluster.
        """
        clusters, cluster_indices = [deque([responses[0]])], [deque([0])]
        nli_scores = {}
        entailments = {}
        for i in range(1, len(responses)):
            new_cluster_indicator = True
            for j, cluster in enumerate(clusters):
                key, rev_key = (cluster[0], responses[i]), (responses[i], cluster[0])
                if key in nli_scores:
                    # Do not recompute if pair already assessed
                    entailment = entailments[key]
                else:
                    # Compute nli score and entailment if pair not yet assessed
                    nli_result = self.nli_scorer.get_nli_results(response1=cluster[0], response2=responses[i])
                    score, entailment = nli_result["score"], nli_result["entailment"]
                    nli_scores[key], nli_scores[rev_key] = score, score
                    entailments[key], entailments[rev_key] = entailment, entailment
                if entailment:
                    new_cluster_indicator = False
                    cluster.append(responses[i])
                    cluster_indices[j].append(i)

            if new_cluster_indicator:
                clusters.append(deque([responses[i]]))
                cluster_indices.append(deque([i]))

        # Arrange cluster so that first element is mode (if exists) else longest
        clusters = [self._sort_responses(list(cluster)) for cluster in clusters]

        return clusters, cluster_indices, nli_scores

    def compute_response_probabilities(self, logprobs_results: List[List[Dict[str, Any]]], num_responses: int = None) -> List[float]:
        """Compute response probabilities"""
        uniform_response_probabilities = [1 / num_responses] * num_responses
        tokenprob_response_probabilities = [self.avg_logprob(logprobs_i) if logprobs_i else np.nan for logprobs_i in logprobs_results] if logprobs_results else None
        return tokenprob_response_probabilities, uniform_response_probabilities

    def compute_cluster_probabilities(self, response_probabilities: List[float], cluster_indices: List[List[int]]) -> List[float]:
        """Compute cluster probabilities"""
        cluster_probabilities = [0] * len(cluster_indices)
        for i, cluster_index in enumerate(cluster_indices):
            cluster_probabilities[i] = sum([response_probabilities[j] for j in cluster_index])
        return self._normalize_cluster_probabilities(cluster_probabilities=cluster_probabilities)

    @staticmethod
    def avg_logprob(logprobs: List[Dict[str, Any]]) -> float:
        "Compute average logprob"
        return np.prod([np.exp(d["logprob"]) for d in logprobs])

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
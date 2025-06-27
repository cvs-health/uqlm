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


import math
import numpy as np
import warnings
from collections import deque, Counter
from typing import Any, Dict, List
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from uqlm.black_box.baseclass.similarity_scorer import SimilarityScorer


class NLIScorer(SimilarityScorer):
    def __init__(self, device: Any = None, verbose: bool = False, nli_model_name: str = "microsoft/deberta-large-mnli", max_length: int = 2000) -> None:
        """
        A class to computing NLI-based confidence scores. This class offers two types of confidence scores, namely
        noncontradiction probability :footcite:`chen2023quantifyinguncertaintyanswerslanguage` and semantic entropy
        :footcite:`farquhar2024detectinghallucinations`.

        Parameters
        ----------
        device : torch.device input or torch.device object, default=None
            Specifies the device that classifiers use for prediction. Set to "cuda" for classifiers to be able to
            leverage the GPU.

        verbose : bool, default=False
            Specifies whether to print verbose status updates of NLI scoring process

        nli_model_name : str, default="microsoft/deberta-large-mnli"
            Specifies which NLI model to use. Must be acceptable input to AutoTokenizer.from_pretrained() and
            AutoModelForSequenceClassification.from_pretrained()

        max_length : int, default=2000
            Specifies the maximum allowed string length. Responses longer than this value will be truncated to
            avoid OutOfMemoryError
        """
        self.device = device
        self.verbose = verbose
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(nli_model_name)
        model = AutoModelForSequenceClassification.from_pretrained(nli_model_name)
        self.model = model.to(self.device) if self.device else model
        self.label_mapping = ["contradiction", "neutral", "entailment"]

    def predict(self, response1: str, response2: str) -> Any:
        """
        This method compute probability of contradiction on the provide inputs.

        Parameters
        ----------
        response1 : str
            An input for the sequence classification DeBERTa model.

        response2 : str
            An input for the sequence classification DeBERTa model.

        Returns
        -------
        numpy.ndarray
            Probabilities computed by NLI model
        """
        if len(response1) > self.max_length or len(response2) > self.max_length:
            warnings.warn("Maximum response length exceeded for NLI comparison. Truncation will occur. To adjust, change the value of max_length")
        concat = response1[0 : self.max_length] + " [SEP] " + response2[0 : self.max_length]
        encoded_inputs = self.tokenizer(concat, padding=True, return_tensors="pt")
        if self.device:
            encoded_inputs = {name: tensor.to(self.device) for name, tensor in encoded_inputs.items()}
        logits = self.model(**encoded_inputs).logits
        np_logits = logits.detach().cpu().numpy() if self.device else logits.detach().numpy()
        probabilites = np.exp(np_logits) / np.exp(np_logits).sum(axis=-1, keepdims=True)
        return probabilites

    def evaluate(self, responses: List[str], sampled_responses: List[List[str]], use_best: bool, compute_entropy: bool = False) -> Dict[str, Any]:
        """
        Evaluate confidence scores on LLM responses.

        Parameters
        ----------
        responses : list of strings
            Original LLM response

        sampled_responses : list of list of strings
            Sampled candidate responses to be compared to the original response

        use_best : bool
            Specifies whether to swap the original response for the uncertainty-minimized response
            based on semantic entropy clusters.

        compute_entropy : bool, default=False
            Specifies whether to include semantic entropy in returned result

        Returns
        -------
        Dict
            Dictionary containing mean NLI and (optionally) semantic entropy scores.
            The dictionary will also contain original and multiple responses, updated if `use_best` is True
        """
        self.num_responses = len(sampled_responses[0])
        observed_consistency_data = {"noncontradiction": [], "semantic_negentropy": [], "responses": responses, "sampled_responses": sampled_responses}
        for i, response in enumerate(responses):
            oc_result_i = self._observed_consistency_i(original=response, candidates=sampled_responses[i], use_best=use_best, compute_entropy=compute_entropy)
            observed_consistency_data["noncontradiction"].append(oc_result_i["nli_score_i"])
            observed_consistency_data["semantic_negentropy"].append(oc_result_i["semantic_negentropy"])
            responses[i] = oc_result_i["response"]  # Replace with optimized response if use_best
            sampled_responses[i] = oc_result_i["candidates"]  # Replace with updated candidates if use_best

        if use_best:
            observed_consistency_data["responses"] = responses
            observed_consistency_data["sampled_responses"] = sampled_responses
        return observed_consistency_data

    def _observed_consistency_i(self, original: str, candidates: List[str], use_best: bool = False, compute_entropy: bool = False) -> Dict[str, Any]:
        """
        Compute observed consistency score on the provided original response and multiple candidates.
        """
        scores = {}
        nli_scores = []
        best_response = original
        semantic_negentropy = None
        if compute_entropy or use_best:
            all_responses = [original] + candidates
            tmp = self._semantic_entropy_process(candidates=all_responses)
            best_response, semantic_negentropy, scores = tmp
            if use_best:
                all_responses.remove(best_response)
                candidates = all_responses

        for candidate in candidates:
            if (candidate, best_response) in scores:
                nli_score = scores[(candidate, best_response)]
            else:
                nli_score = self._get_nli_results(response1=best_response, response2=candidate)["score"]
            nli_scores.append(nli_score)

        return {"nli_score_i": np.mean(nli_scores), "candidates": candidates, "response": best_response, "semantic_negentropy": semantic_negentropy}

    def _semantic_entropy_process(self, candidates: List[str], i: int = None, discrete=True) -> Any:
        """
        Executes complete process for semantic entropy and returns best response, SE score, and dictionary
        of NLI scores for response pairs
        """
        if self.verbose:
            print("Question No. - ", i + 1)
        clustered_responses, nli_scores = self._cluster_responses(responses=candidates)
        if discrete:
            response_probabilities = [[1] * len(cluster_i) for cluster_i in clustered_responses]
            cluster_probabilities = self._compute_cluster_probability(response_probabilities=response_probabilities)
            best_response = clustered_responses[cluster_probabilities.index(max(cluster_probabilities))][0]
            semantic_negentropy = self._compute_semantic_entropy(cluster_probabilities=cluster_probabilities)
        else:
            # TODO: enable continuous semantic entropy
            raise ValueError("SemanticEntropy currently only supports discrete evaluations")
        return (best_response, semantic_negentropy, nli_scores)

    def _get_nli_results(self, response1: str, response2: str) -> Dict[str, Any]:
        """This method computes mean NLI score and determines whether entailment exists."""
        if response1 == response2:
            avg_nli_score, entailment = 1, True
        else:
            left = self.predict(response1=response1, response2=response2)
            left_label = self.label_mapping[left.argmax(axis=1)[0]]

            right = self.predict(response1=response2, response2=response1)
            right_label = self.label_mapping[right.argmax(axis=1)[0]]
            s1, s2 = 1 - left[:, 0], 1 - right[:, 0]

            entailment = left_label == "entailment" or right_label == "entailment"
            avg_nli_score = ((s1 + s2) / 2)[0]
        return {"score": avg_nli_score, "entailment": entailment}

    def _cluster_responses(self, responses: List[str]) -> Any:
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
        clusters = [deque([responses[0]])]
        nli_scores = {}
        entailments = {}
        for i in range(1, len(responses)):
            new_cluster_indicator = True
            for cluster in clusters:
                key, rev_key = (cluster[0], responses[i]), (responses[i], cluster[0])
                if key in nli_scores:
                    # Do not recompute if pair already assessed
                    entailment = entailments[key]
                else:
                    # Compute nli score and entailment if pair not yet assessed
                    nli_result = self._get_nli_results(response1=cluster[0], response2=responses[i])
                    score, entailment = nli_result["score"], nli_result["entailment"]
                    nli_scores[key], nli_scores[rev_key] = score, score
                    entailments[key], entailments[rev_key] = entailment, entailment
                if entailment:
                    new_cluster_indicator = False
                    cluster.append(responses[i])

            if new_cluster_indicator:
                clusters.append(deque([responses[i]]))

        # Arrange cluster so that first element is mode (if exists) else longest
        clusters = [self._sort_responses(list(cluster)) for cluster in clusters]
        return clusters, nli_scores

    def _compute_semantic_entropy(self, cluster_probabilities: List[float]) -> float:
        """
        Helper function to compute semantic entropy score from cluster probabilities
        """
        return abs(sum([p * math.log(p) for p in cluster_probabilities]))

    def _normalize_entropy(self, entropy_values):
        return [e / math.log(self.num_responses + 1) for e in entropy_values]

    @staticmethod
    def _compute_cluster_probability(response_probabilities: List[float]) -> float:
        """Computes cluster probabilities from response probabilities"""
        total_probability = sum(map(sum, response_probabilities))
        cluster_probabilities = [sum(rp_i) / total_probability for rp_i in response_probabilities]
        return cluster_probabilities

    @staticmethod
    def _sort_responses(responses: List[str]) -> List[str]:
        """Sorts responses in a cluster"""
        counter = Counter(responses)
        mode_str, count = counter.most_common(1)[0]
        if count > 1:
            return sorted(responses, key=lambda x: (x != mode_str, x))
        else:
            return sorted(responses, key=len, reverse=True)

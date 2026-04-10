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
from typing import Any, Dict, List
import math
import numpy as np


def normalize_entropy(entropy_values, num_responses):
    """Helper function to normalize semantic entropy"""
    return [e / math.log(num_responses + 1) for e in entropy_values]


def compute_semantic_entropy(cluster_probabilities: List[float]) -> float:
    """
    Helper function to compute semantic entropy score from cluster probabilities
    """
    return abs(sum([p * math.log(p) if p > 0.0 else 0 for p in cluster_probabilities]))


def compute_response_probabilities(logprobs_results: List[List[Dict[str, Any]]], num_responses: int = None, length_normalize: bool = True) -> List[float]:
    """Compute response probabilities"""
    uniform_response_probabilities = [1 / num_responses] * num_responses
    tokenprob_response_probabilities = [length_norm_sequence_prob(logprobs_i, length_normalize) if logprobs_i else np.nan for logprobs_i in logprobs_results] if logprobs_results else None
    return tokenprob_response_probabilities, uniform_response_probabilities


def compute_cluster_probabilities(response_probabilities: List[float], cluster_indices: List[List[int]]) -> List[float]:
    """Compute cluster probabilities"""
    cluster_probabilities = [0] * len(cluster_indices)
    for i, cluster_index in enumerate(cluster_indices):
        cluster_probabilities[i] = sum([response_probabilities[j] for j in cluster_index])
    return normalize_cluster_probabilities(cluster_probabilities=cluster_probabilities)


def length_norm_sequence_prob(logprobs: List[Dict[str, Any]], length_normalize: bool = True) -> float:
    "Compute length normalized sequence logprob"
    factor = 1 / len(logprobs) if length_normalize else 1
    return np.exp(np.sum([d["logprob"] for d in logprobs]) * factor)


def best_response_selection(clustered_responses: List[List[str]], cluster_probabilities: List[float]) -> str:
    """Select the best response from the clustered responses based on the cluster probabilities"""
    return clustered_responses[cluster_probabilities.index(max(cluster_probabilities))][0]


def normalize_cluster_probabilities(cluster_probabilities: List[float]) -> float:
    """Normalize cluster probabilities"""
    total_probability = sum(cluster_probabilities)
    return [cp_i / total_probability for cp_i in cluster_probabilities]

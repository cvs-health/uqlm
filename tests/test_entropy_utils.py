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

import pytest

from uqlm.nli.entropy_utils import compute_semantic_entropy, normalize_cluster_probabilities


def test_normalize_cluster_probabilities_normal():
    result = normalize_cluster_probabilities([0.3, 0.7])
    assert isinstance(result, list)
    assert len(result) == 2
    assert abs(sum(result) - 1.0) < 1e-10


def test_normalize_cluster_probabilities_all_zeros():
    """All-zero input must not raise ZeroDivisionError (regression).

    When token logprobs are very negative, np.exp() underflows to 0.0 for every
    cluster. The function must return a valid probability distribution (uniform
    fallback) instead of crashing.
    """
    result = normalize_cluster_probabilities([0.0, 0.0, 0.0])
    assert isinstance(result, list)
    assert len(result) == 3
    assert abs(sum(result) - 1.0) < 1e-10


def test_normalize_cluster_probabilities_empty():
    result = normalize_cluster_probabilities([])
    assert result == []


@pytest.mark.parametrize("probabilities", [[float("nan"), 0.5], [float("nan"), float("nan")]])
def test_semantic_entropy_preserves_missing_probabilities(probabilities):
    assert math.isnan(compute_semantic_entropy(probabilities))


@pytest.mark.parametrize("probabilities, expected", [([0.0, 1.0], 0.0), ([0.5, 0.5], math.log(2))])
def test_semantic_entropy_valid_probabilities(probabilities, expected):
    assert compute_semantic_entropy(probabilities) == pytest.approx(expected)

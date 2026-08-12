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
import pytest
from unittest.mock import patch

from uqlm.black_box.consistency import ConsistencyScorer


class FakeNLI:
    """Deterministic NLI stand-in that mirrors the real read-through cache and records forward batches."""

    def __init__(self, scores=None):
        self.scores = scores or {}
        self.batch_calls = []
        self.probabilities = dict()
        self._result_cache = dict()

    def _compute(self, response1, response2):
        noncontradiction, entailment_score = self.scores.get((response1, response2), (0.5, 0.25))
        return {"noncontradiction_score": noncontradiction, "entailment": entailment_score > 0.5, "entailment_score": entailment_score}

    def get_nli_results_batch(self, response_pairs):
        pending = [pair for pair in response_pairs if pair[0] != pair[1] and pair not in self._result_cache]
        if pending:
            self.batch_calls.append(pending)
            for pair in pending:
                self._result_cache[pair] = self._compute(*pair)
        return [{"noncontradiction_score": 1, "entailment": True, "entailment_score": 1} if pair[0] == pair[1] else self._result_cache[pair] for pair in response_pairs]

    def get_nli_results(self, response1, response2):
        return self.get_nli_results_batch([(response1, response2)])[0]


@pytest.fixture
def fake_nli():
    return FakeNLI(scores={("A", "B"): (0.9, 0.8), ("A", "C"): (0.2, 0.1), ("X", "Y"): (0.4, 0.3)})


def test_injected_nli_is_used(fake_nli):
    """An injected NLI instance must be reused instead of loading a new model."""
    with patch("uqlm.black_box.consistency.NLI") as mock_nli_class:
        scorer = ConsistencyScorer(nli=fake_nli)
    mock_nli_class.assert_not_called()
    assert scorer.nli is fake_nli


def test_device_and_batch_size_forwarded():
    """ConsistencyScorer must honor device and nli_batch_size when constructing its own NLI."""
    with patch("uqlm.black_box.consistency.NLI") as mock_nli_class:
        ConsistencyScorer(device="cpu", nli_batch_size=16, max_length=1000)
    mock_nli_class.assert_called_once_with(nli_model_name="microsoft/deberta-large-mnli", max_length=1000, device="cpu", batch_size=16)


def test_evaluate_batches_all_candidates(fake_nli):
    """All unique candidate pairs must be evaluated in a single batched NLI call."""
    scorer = ConsistencyScorer(nli=fake_nli)
    result = scorer.evaluate(responses=["A"], sampled_responses=[["B", "C", "A"]])

    assert fake_nli.batch_calls == [[("A", "B"), ("A", "C")]]
    assert result["noncontradiction"][0] == pytest.approx(np.mean([0.9, 0.2, 1.0]))
    assert result["entailment"][0] == pytest.approx(np.mean([0.8, 0.1, 1.0]))


def test_evaluate_batches_across_prompts(fake_nli):
    """Pairs from different prompts must be evaluated together in one batched pass."""
    scorer = ConsistencyScorer(nli=fake_nli)
    result = scorer.evaluate(responses=["A", "X"], sampled_responses=[["B", "C"], ["Y"]])

    assert fake_nli.batch_calls == [[("A", "B"), ("A", "C"), ("X", "Y")]]
    assert result["noncontradiction"] == [pytest.approx(np.mean([0.9, 0.2])), pytest.approx(0.4)]
    assert result["entailment"] == [pytest.approx(np.mean([0.8, 0.1])), pytest.approx(0.3)]


def test_evaluate_uses_available_nli_scores(fake_nli):
    """Pairs already present in available_nli_scores must not be re-evaluated."""
    scorer = ConsistencyScorer(nli=fake_nli)
    available = {"noncontradiction": {("B", "A"): 0.7}, "entailment": {("B", "A"): 0.6}}
    result = scorer.evaluate(responses=["A"], sampled_responses=[["B", "C"]], available_nli_scores=available)

    assert fake_nli.batch_calls == [[("A", "C")]]
    assert result["noncontradiction"][0] == pytest.approx(np.mean([0.7, 0.2]))
    assert result["entailment"][0] == pytest.approx(np.mean([0.6, 0.1]))

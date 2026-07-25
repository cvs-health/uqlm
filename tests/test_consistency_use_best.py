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

from unittest.mock import MagicMock, patch


def test_observed_consistency_use_best_returns_list():
    """use_best=True must not crash — list.remove() returns None (regression).

    list.remove() is an in-place operation that returns None. Using its return
    value as the iteration target causes TypeError on the use_best=True path.
    After the fix, candidates must be a list (the remaining responses minus
    best_response).
    """
    from uqlm.black_box.consistency import ConsistencyScorer

    with patch("uqlm.black_box.consistency.NLI"):
        scorer = ConsistencyScorer(use_best=True)

    mock_clusterer = MagicMock()
    mock_clusterer.compute_response_probabilities.return_value = (
        None,
        [0.33, 0.33, 0.34],
    )
    mock_clusterer.evaluate.return_value = ("resp_B", None, None, None)
    mock_clusterer.nli_scores = {}

    with patch(
        "uqlm.black_box.consistency.SemanticClusterer",
        return_value=mock_clusterer,
    ):
        scorer.available_nli_scores = {}
        scorer.nli = MagicMock()
        scorer.nli.get_nli_results.return_value = {
            "noncontradiction_score": 0.9,
            "entailment_score": 0.8,
        }
        result = scorer._observed_consistency_i("resp_A", ["resp_B", "resp_C"])

    # Key assertions: candidates is a list (not None) with best_response removed
    assert isinstance(result["candidates"], list)
    assert "resp_B" not in result["candidates"]
    assert result["response"] == "resp_B"

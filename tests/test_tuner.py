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

from math import isclose
from uqlm.utils.tuner import Tuner


class TestTuner:
    def setup_method(self):
        # Setup common test data
        self.y_scores = [0.1, 0.4, 0.35, 0.8]
        self.correct_indicators = [0, 1, 0, 1]
        self.score_lists = [[0.1, 0.4, 0.35, 0.8], [0.2, 0.5, 0.3, 0.7], [0.15, 0.45, 0.25, 0.75]]

    def test_initialization(self):
        # Test default initialization
        tuner = Tuner()
        assert list(tuner.objective_to_func.keys()) == ["fbeta_score", "accuracy_score", "balanced_accuracy_score", "log_loss", "roc_auc"]

    def test_tune_threshold(self):
        tuner = Tuner()
        threshold = tuner.tune_threshold(y_scores=self.y_scores, correct_indicators=self.correct_indicators)
        assert isinstance(threshold, float)
        assert 0 <= threshold <= 1
        assert isclose(threshold, 0.35, abs_tol=10**-4)

    def test_tune_params(self):
        tuner = Tuner()
        result = tuner.tune_params(score_lists=self.score_lists, correct_indicators=self.correct_indicators)
        assert "weights" in result
        assert "thresh" in result
        assert isinstance(result["weights"], tuple)
        assert isinstance(result["thresh"], float)
        # can't check exact values because of random nature of optimization
        # print(f"result: {result}")

    def test_normalize_weights(self):
        weights = [0.2, 0.3, 0.5]
        normalized_weights = Tuner._normalize_weights(weights)
        assert abs(sum(normalized_weights) - 1.0) < 1e-9
        assert len(normalized_weights) == len(weights)

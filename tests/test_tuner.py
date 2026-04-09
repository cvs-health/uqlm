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
import pytest
import numpy as np
from unittest.mock import MagicMock
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
        assert list(tuner.objective_to_func.keys()) == ["fbeta_score", "accuracy_score", "balanced_accuracy_score", "log_loss", "roc_auc", "average_precision", "brier_score"]

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

    def test_validation_errors_and_optimization_paths(self):
        # test input validation (k=1)
        with pytest.raises(ValueError) as e:
            Tuner().tune_params(score_lists=[self.score_lists[0]], correct_indicators=self.correct_indicators)
        assert "Tuning only applies if more than scorer component is present." in str(e.value)

        # test unsupported weights_objective
        with pytest.raises(ValueError) as e:
            Tuner().tune_params(score_lists=self.score_lists, correct_indicators=self.correct_indicators, weights_objective="invalid")
        assert "Only 'fbeta_score', 'accuracy_score', 'balanced_accuracy_score', 'roc_auc_score', 'log_loss', 'average_precision', and 'brier_score' are supported for tuning objectives." in str(e.value)

        # test unsupported thresh_objective
        with pytest.raises(ValueError) as e:
            Tuner().tune_params(score_lists=self.score_lists, correct_indicators=self.correct_indicators, thresh_objective="invalid")
        assert "Only 'fbeta_score', 'accuracy_score', 'balanced_accuracy_score' are supported for tuning objectives." in str(e.value)

        # test thresh_objective must match weights_objective for any threshold-dependent weights_objective
        with pytest.raises(ValueError) as e:
            Tuner().tune_params(self.score_lists, self.correct_indicators, weights_objective="fbeta_score", thresh_objective="accuracy_score")
        assert "thresh_objective must match weights_objective for any threshold-dependent weights_objective." in str(e.value)

        # test threshold optimization with different paths
        # cover  tune_threshold() method  and different objective function evaluations
        for obj in ["accuracy_score", "balanced_accuracy_score", "roc_auc"]:
            Tuner().tune_threshold(self.y_scores, self.correct_indicators, thresh_objective=obj)

        # k=2: different objectives (optimize_jointly=False path)
        Tuner().tune_params(self.score_lists[:2], self.correct_indicators, weights_objective="roc_auc", thresh_objective="fbeta_score")
        # k=2: same objectives (optimize_jointly=True, grid search)
        Tuner().tune_params(self.score_lists[:2], self.correct_indicators, weights_objective="fbeta_score", thresh_objective="fbeta_score")
        # k=3: same objectives (optimize_jointly=True, grid search)
        Tuner().tune_params(self.score_lists, self.correct_indicators, weights_objective="fbeta_score", thresh_objective="fbeta_score")
        # k=3: different objectives (optimize_jointly=False, separate optimization)
        Tuner().tune_params(self.score_lists, self.correct_indicators, weights_objective="accuracy_score", thresh_objective="accuracy_score")
        # k>3: Optuna path
        extended_lists = self.score_lists + [[0.25, 0.55, 0.35, 0.65]]
        Tuner().tune_params(extended_lists, self.correct_indicators)
        # log_loss objective (obj_multiplier = -1 path)
        Tuner().tune_params(self.score_lists, self.correct_indicators, weights_objective="log_loss", thresh_objective="fbeta_score")


@pytest.fixture
def mock_progress():
    progress = MagicMock()
    progress.add_task.return_value = "task_id"
    return progress


def test_tune_threshold_with_progress(mock_progress):
    tuner = Tuner()
    y_scores = [0.1, 0.5, 0.9]
    correct_indicators = [False, True, True]

    result = tuner.tune_threshold(y_scores=y_scores, correct_indicators=correct_indicators, progress_bar=mock_progress)

    assert 0 <= result <= 1
    mock_progress.add_task.assert_called_once()
    # Should update for each threshold value
    assert mock_progress.update.call_count > 0


def test_optimize_objective_with_progress_joint(mock_progress):
    tuner = Tuner()
    score_lists = [[0.2, 0.8], [0.5, 0.6]]
    correct_indicators = [True, False]
    # k=2 triggers grid search with progress bar
    tuner.tune_params(score_lists=score_lists, correct_indicators=correct_indicators, weights_objective="fbeta_score", thresh_objective="fbeta_score", progress_bar=mock_progress, n_trials=5)

    mock_progress.add_task.assert_called()
    assert mock_progress.update.call_count > 0


def test_grid_search_weights_thresh_progress(mock_progress):
    tuner = Tuner()
    tuner.k = 2
    tuner.step_size = 0.5
    tuner.thresh_bounds = (0, 1)
    tuner.correct_indicators = np.array([True, False])
    tuner.score_lists = np.array([[0.2, 0.8], [0.5, 0.6]])
    tuner.weights_tuning_objective = tuner._f_score
    tuner.obj_multiplier = -1
    tuner.progress_bar = mock_progress
    tuner.fscore_beta = 1
    result = tuner._grid_search_weights_thresh()
    assert len(result) == 3  # two weights + threshold
    mock_progress.add_task.assert_called_once()
    assert mock_progress.update.call_count > 0


def test_grid_search_weights_progress_k2(mock_progress):
    tuner = Tuner()
    tuner.k = 2
    tuner.step_size = 0.5
    tuner.correct_indicators = np.array([True, False])
    tuner.score_lists = np.array([[0.2, 0.8], [0.5, 0.6]])
    tuner.weights_tuning_objective = tuner._f_score
    tuner.obj_multiplier = -1
    tuner.progress_bar = mock_progress
    tuner._evaluate_objective = MagicMock(return_value=0.0)
    result = tuner._grid_search_weights()
    assert len(result) == 2
    mock_progress.add_task.assert_called_once()
    assert mock_progress.update.call_count > 0


def test_grid_search_weights_progress_k3(mock_progress):
    tuner = Tuner()
    tuner.k = 3
    tuner.step_size = 0.5
    tuner.correct_indicators = np.array([True, False])
    tuner.score_lists = np.array([[0.2, 0.8], [0.5, 0.6], [0.3, 0.7]])
    tuner.weights_tuning_objective = tuner._f_score
    tuner.obj_multiplier = -1
    tuner.progress_bar = mock_progress
    tuner._evaluate_objective = MagicMock(return_value=0.0)
    result = tuner._grid_search_weights()
    assert len(result) == 3
    mock_progress.add_task.assert_called_once()
    assert mock_progress.update.call_count > 0


def test_optimize_objective_progress_joint_k3(mock_progress):
    """k > 2, optimize_jointly=True, with progress_bar covers Optuna path (lines 155, 159)."""
    tuner = Tuner()
    score_lists = [[0.2, 0.8], [0.5, 0.6], [0.3, 0.9]]  # k=3
    correct_indicators = [True, False]
    tuner.tune_params(
        score_lists=score_lists,
        correct_indicators=correct_indicators,
        weights_objective="fbeta_score",
        thresh_objective="fbeta_score",  # same → optimize_jointly=True, k>2 → Optuna
        progress_bar=mock_progress,
        n_trials=2,
    )
    mock_progress.add_task.assert_called()


def test_optimize_objective_progress_separate_k4(mock_progress):
    """k > 3, optimize_jointly=False, with progress_bar covers Optuna path (lines 174, 179)."""
    tuner = Tuner()
    score_lists = [[0.2, 0.8], [0.5, 0.6], [0.3, 0.9], [0.1, 0.7]]  # k=4
    correct_indicators = [True, False]
    tuner.tune_params(
        score_lists=score_lists,
        correct_indicators=correct_indicators,
        weights_objective="roc_auc",  # different → optimize_jointly=False, k>3 → Optuna
        thresh_objective="fbeta_score",
        progress_bar=mock_progress,
        n_trials=2,
    )
    mock_progress.add_task.assert_called()

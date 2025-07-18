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
from numpy.typing import ArrayLike
import optuna
from typing import Any, Dict, List, Tuple
import time
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
from sklearn.metrics import fbeta_score, balanced_accuracy_score, accuracy_score, roc_auc_score, log_loss, average_precision_score, brier_score_loss

optuna.logging.set_verbosity(optuna.logging.WARNING)


class Tuner:
    def __init__(self) -> None:
        """
        Class for tuning weights and threshold for UQEnsemble
        """
        self.objective_to_func = {"fbeta_score": self._f_score, "accuracy_score": accuracy_score, "balanced_accuracy_score": balanced_accuracy_score, "log_loss": log_loss, "roc_auc": roc_auc_score, "average_precision": average_precision_score, "brier_score": brier_score_loss}

    def tune_threshold(self, y_scores: List[float], correct_indicators: List[bool], thresh_objective: str = "fbeta_score", fscore_beta: float = 1, bounds: Tuple[float, float] = (0, 1), step_size: int = 0.01, progress_bar: bool = False) -> float:
        """
        Conducts 1-dimensional grid search for threshold.

        Parameters
        ----------
        y_scores : list of floats
            List of confidence scores.

        correct_indicators : list of bool
            A list of boolean indicators of whether self.original_responses are correct.

        thresh_objective: {'fbeta_score', 'accuracy_score', 'balanced_accuracy_score'}, default='fbeta_score'
            Objective function for threshold optimization via grid search.

        fscore_beta : float, default=1
            Value of beta in fbeta_score.

        bounds : tuple of floats, default=(0,1)
            Bounds to search for threshold.

        step_size : float, default=0.01
            Indicates step size in grid search, if used.

        progress_bar : bool, default=False
            If True, displays a progress bar during optimization routine

        Returns
        -------
        float
            Optimized threshold.
        """

        self.fscore_beta = fscore_beta
        self.progress_bar = progress_bar
        threshold_tuning_objective = self.objective_to_func[thresh_objective]
        threshold_values = np.arange(bounds[0], bounds[1], step=step_size)

        y_scores_array = np.array(y_scores)
        y_pred_matrix = (y_scores_array[:, np.newaxis] > threshold_values).astype(int)
        values = np.zeros(len(threshold_values))
        with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), BarColumn(), TextColumn("[progress.percentage]{task.percentage:>3.0f}%"), TimeElapsedColumn()) as progress:
            if self.progress_bar:
                progress_task = progress.add_task("- [black]Optimizing threshold with grid search...", total=len(threshold_values))
            for i, y_pred in enumerate(y_pred_matrix.T):
                values[i] = -threshold_tuning_objective(np.array(correct_indicators), y_pred)
                if self.progress_bar:
                    progress.update(progress_task, advance=1)
            time.sleep(0.1)
        values = -np.array([threshold_tuning_objective(np.array(correct_indicators), y_pred) for y_pred in y_pred_matrix.T])

        best_index = np.argmin(values)
        best_threshold = threshold_values[best_index]
        return best_threshold

    def tune_params(self, score_lists: List[List[float]], correct_indicators: List[bool], weights_objective: str = "roc_auc", thresh_objective: str = "fbeta_score", thresh_bounds: Tuple[float, float] = (0, 1), n_trials: int = 100, step_size: float = 0.01, fscore_beta: float = 1, progress_bar: bool = False) -> Dict[str, Any]:
        """
        Tunes weights and threshold parameters on a set of user-provided graded responses.

        Parameters
        ----------
        score_lists : list of lists of floats
            A list of lists of floats. Each interior list is a list of component-specific scores.

        correct_indicators : list of bool
            A list of boolean indicators of whether self.original_responses are correct.

        weights_objective : {'fbeta_score', 'accuracy_score', 'balanced_accuracy_score', 'roc_auc', 'log_loss', 'average_precision', 'brier_score'}, default='roc_auc'
            Objective function for optimization of weights. Must match thresh_objective if one of 'fbeta_score',
            'accuracy_score', 'balanced_accuracy_score'. If same as thresh_objective, joint optimization will be done.

        thresh_objective : {'fbeta_score', 'accuracy_score', 'balanced_accuracy_score'}, default='fbeta_score'
            Objective function for threshold optimization via grid search.

        thresh_bounds : tuple of floats, default=(0,1)
            Bounds to search for threshold.

        n_trials : int, default=100
            Indicates how many candidates to search over with optuna optimizer.

        step_size : float, default=0.01
            Indicates step size in grid search, if used.

        fscore_beta : float, default=1
            Value of beta in fbeta_score.

        progress_bar : bool, default=False
            If True, displays a progress bar during optimization routine

        Returns
        -------
        Dict
            Dictionary containing optimized weights and threshold.
        """
        self.score_lists = np.stack([np.array(sl) for sl in score_lists])
        self.k = len(score_lists)
        self.correct_indicators = np.array(correct_indicators)
        self.weights_objective = weights_objective
        self.thresh_bounds = thresh_bounds
        self.thresh_objective = thresh_objective
        self.n_trials = n_trials
        self.step_size = step_size
        self.fscore_beta = fscore_beta
        self.optimize_jointly = weights_objective == thresh_objective
        self.obj_multiplier = 1 if weights_objective in ["logloss", "brier_score"] else -1
        self.progress_bar = progress_bar

        self._validate_tuning_inputs()
        self.weights_tuning_objective = self.objective_to_func[self.weights_objective]
        self.threshold_tuning_objective = self.objective_to_func[self.thresh_objective]

        params = self._optimize_objective()
        return {"weights": params[:-1], "thresh": params[-1]}

    def _optimize_objective(self):
        """Runs optimization routine as specified by user"""
        if self.optimize_jointly:
            if self.k > 2:
                study = optuna.create_study()
                with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), BarColumn(), TextColumn("[progress.percentage]{task.percentage:>3.0f}%"), TimeElapsedColumn()) as progress:
                    if self.progress_bar:
                        progress_task = progress.add_task("- [black]Optimizing weights...", total=self.n_trials)

                    def callback(study, trial):
                        if self.progress_bar:
                            progress.update(progress_task, advance=1)

                    study.optimize(self._optuna_objective, n_trials=self.n_trials, callbacks=[callback])
                    params = tuple(study.best_params.values())
                    best_weights = self._normalize_weights(params[:-1])
                time.sleep(0.1)
                return tuple(best_weights) + (params[-1],)
            else:
                params = self._grid_search_weights_thresh()
                best_weights = self._normalize_weights(params[:-1])
                return tuple(best_weights) + (params[-1],)
        else:
            if self.k > 3:
                study = optuna.create_study()
                with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), BarColumn(), TextColumn("[progress.percentage]{task.percentage:>3.0f}%"), TimeElapsedColumn()) as progress:
                    if self.progress_bar:
                        progress_task = progress.add_task("- [black]Optimizing weights...", total=self.n_trials)

                    def callback(study, trial):
                        if self.progress_bar:
                            progress.update(progress_task, advance=1)

                    study.optimize(self._optuna_objective, n_trials=self.n_trials, callbacks=[callback])
                    best_weights_raw = tuple(study.best_params.values())
                    best_weights = self._normalize_weights(best_weights_raw)
                    time.sleep(0.1)
            else:
                best_weights = self._grid_search_weights()

            # rprint("[blue]Optimizing threshold with grid search...")
            new_scores = self._update_scores(np.array(best_weights))
            best_threshold = self.tune_threshold(y_scores=new_scores, correct_indicators=self.correct_indicators, thresh_objective=self.thresh_objective, fscore_beta=self.fscore_beta, progress_bar=self.progress_bar)
            return tuple(best_weights) + (best_threshold,)

    def _f_score(self, y_true, y_pred):
        """Helper function to compute f-beta score."""
        return fbeta_score(y_true, y_pred, beta=self.fscore_beta)

    def _validate_tuning_inputs(self):
        """Helper function to validate tuning inputs."""
        if self.k == 1:
            raise ValueError("Tuning only applies if more than scorer component is present.")

        if self.weights_objective not in self.objective_to_func:
            raise ValueError(
                """
                Only 'fbeta_score', 'accuracy_score', 'balanced_accuracy_score', 'roc_auc_score', 'log_loss', 'average_precision', and 'brier_score' are supported for tuning objectives.
                """
            )
        if self.thresh_objective not in ["fbeta_score", "accuracy_score", "balanced_accuracy_score"]:
            raise ValueError(
                """
                Only 'fbeta_score', 'accuracy_score', 'balanced_accuracy_score' are supported for tuning objectives.
                """
            )
        if self.weights_objective in ["fbeta_score", "accuracy_score", "balanced_accuracy_score"]:
            if not self.optimize_jointly:
                raise ValueError(
                    """
                    thresh_objective must match weights_objective for any threshold-dependent weights_objective.
                    """
                )

    def _optuna_objective(self, trial) -> float:
        """Helper function to define optuna objective."""
        thresh = None
        raw_weights = [trial.suggest_float(f"weight_{i}", 0, 1) for i in range(self.k)]
        weights = self._normalize_weights(raw_weights)
        if self.optimize_jointly:
            thresh = trial.suggest_float("thresh", self.thresh_bounds[0], self.thresh_bounds[1])
        ensemble_scores = self._compute_ensemble_scores(weights=np.array(weights), score_lists=self.score_lists)
        return self._evaluate_objective(y_true=self.correct_indicators, y_pred=ensemble_scores, thresh=thresh)

    def _evaluate_objective(self, y_true, y_pred, thresh=None):
        """Helper function to define evaluate objective function for weights."""
        if thresh is not None:
            y_pred = y_pred > thresh
        return self.obj_multiplier * self.weights_tuning_objective(y_true, y_pred)

    def _update_scores(self, weights: List[float]) -> List[float]:
        """Update confidence scores"""
        return self._compute_ensemble_scores(weights, score_lists=self.score_lists)

    def _compute_ensemble_scores(self, weights: List[float], score_lists: List[List[float]]) -> List[float]:
        """Helper function to compute dot product for getting ensemble scores."""
        valid_mask = ~np.isnan(score_lists)
        adjusted_weights = weights[:, None] * valid_mask
        normalized_weights = adjusted_weights / np.sum(adjusted_weights, axis=0, keepdims=True)
        stacked_nonan = np.nan_to_num(score_lists, nan=0.0)
        ensemble_scores = np.sum(stacked_nonan * normalized_weights, axis=0)
        return np.clip(ensemble_scores, 0, 1)

    def _grid_search_weights_thresh(self):
        """
        Joint grid search optimization for k weights and threshold.
        Use only if k==2, only one weight is free (the second is 1 - w1).
        """
        weight_grid = np.linspace(0, 1, int(1 / self.step_size))
        threshold_grid = np.linspace(self.thresh_bounds[0] + self.step_size, self.thresh_bounds[1] - self.step_size, int((self.thresh_bounds[1] - self.thresh_bounds[0]) / (self.step_size) - 1))
        best_cost = np.inf
        with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), BarColumn(), TextColumn("[progress.percentage]{task.percentage:>3.0f}%"), TimeElapsedColumn()) as progress:
            if self.progress_bar:
                progress_task = progress.add_task("- [black]Jointly optimizing weights and threshold using grid search...", total=len(weight_grid) * len(threshold_grid))
            for w in weight_grid:
                weights = np.array([w, 1 - w])
                for thresh in threshold_grid:
                    cost = self._evaluate_objective(y_true=self.correct_indicators, y_pred=self._update_scores(weights), thresh=thresh)
                    if self.progress_bar:
                        progress.update(progress_task, advance=1)
                    if cost < best_cost:
                        best_cost = cost
                        best_weights = weights
                        best_thresh = thresh
            time.sleep(0.1)
        return tuple(best_weights) + (best_thresh,)

    def _grid_search_weights(self):
        """
        Grid search for weights only.
        For k==2: one free weight.
        For k==3: two free weights where the third is 1 - (w1+w2).
        Only consider feasible regions (weights are non-negative and sum to 1).
        """
        best_cost = np.inf
        if self.k == 2:
            w_grid = np.linspace(0, 1, int(1 / self.step_size))
            with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), BarColumn(), TextColumn("[progress.percentage]{task.percentage:>3.0f}%"), TimeElapsedColumn()) as progress:
                if self.progress_bar:
                    progress_task = progress.add_task("- [black]Optimizing weights using grid search...", total=len(w_grid))
                for w in w_grid:
                    weights = np.array([w, 1 - w])
                    cost = self._evaluate_objective(y_true=self.correct_indicators, y_pred=self._update_scores(weights))
                    if self.progress_bar:
                        progress.update(progress_task, advance=1)
                    if cost < best_cost:
                        best_cost = cost
                        best_weights = weights
                time.sleep(0.1)
        elif self.k == 3:
            w1_grid = np.linspace(0, 1, int(1 / self.step_size))
            w2_grid = np.linspace(0, 1, int(1 / self.step_size))
            with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), BarColumn(), TextColumn("[progress.percentage]{task.percentage:>3.0f}%"), TimeElapsedColumn()) as progress:
                if self.progress_bar:
                    progress_task = progress.add_task("- [black]Optimizing weights using grid search...", total=len(w1_grid) * len(w2_grid))
                for w1 in w1_grid:
                    for w2 in w2_grid:
                        if self.progress_bar:
                            progress.update(progress_task, advance=1)
                        w3 = 1 - w1 - w2
                        if w3 < 0:  # infeasible, skip
                            continue
                        weights = np.array([w1, w2, w3])
                        cost = self._evaluate_objective(y_true=self.correct_indicators, y_pred=self._update_scores(weights))
                        if cost < best_cost:
                            best_cost = cost
                            best_weights = weights
                time.sleep(0.1)
        return tuple(best_weights)

    @staticmethod
    def _normalize_weights(weights: ArrayLike) -> ArrayLike:
        """Helper function to ensure weights sum to 1."""
        weights_array = np.asarray(weights)
        return weights_array / np.sum(weights_array)

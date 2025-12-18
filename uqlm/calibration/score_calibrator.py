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

"""
Score calibration module for uncertainty quantification confidence scores.

This module provides calibration methods to transform raw confidence scores
into better-calibrated probabilities using Platt Scaling and Isotonic Regression.
"""

import numpy as np
from typing import Literal, List, Union
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from uqlm.utils.results import UQResult

Ignore_Columns = ["prompts", "responses", "sampled_responses", "raw_sampled_responses", "raw_responses", "logprob"]


class ScoreCalibrator:
    """
    A class for calibrating confidence scores using Platt Scaling or Isotonic Regression.

    Confidence scores from uncertainty quantification methods may not be well-calibrated
    probabilities. This class provides methods to transform raw scores into calibrated
    probabilities that better reflect the true likelihood of correctness.

    Parameters
    ----------
    method : {'platt', 'isotonic'}, default='platt'
        The calibration method to use:
        - 'platt': Platt scaling using logistic regression
        - 'isotonic': Isotonic regression (non-parametric, monotonic)

    Attributes
    ----------
    method : str
        The calibration method used.
    calibrator_ : sklearn estimator
        The fitted calibration model.
    is_fitted_ : bool
        Whether the calibrator has been fitted.
    """

    def __init__(self, method: Literal["platt", "isotonic"] = "platt"):
        self.method = method
        self.calibrators = {}
        self.is_fitted_ = False

    def fit(self, uq_result: UQResult, correct_indicators: Union[List[bool], List[int], np.ndarray]) -> None:
        """
        Fit the calibration model using scores and binary correctness labels.

        Parameters
        ----------
        uq_result : UQResult
            The UQResult object to fit the calibrator on.
        correct_indicators : array-like of shape (n_samples,)
            Binary labels indicating correctness (True/False or 1/0).

        Returns
        -------
        self : ScoreCalibrator
            The fitted calibrator instance.
        """
        correct_indicators = np.array(correct_indicators, dtype=int)
        if not np.all(np.isin(correct_indicators, [0, 1])):
            raise ValueError("correct_indicators must be binary (True/False or 1/0)")

        for scorer in uq_result.data:
            if scorer not in Ignore_Columns:
                scores = np.array(uq_result.data[scorer])
                if len(scores) != len(correct_indicators):
                    raise ValueError("scores and correct_indicators must have the same length")

                if not np.all((scores >= 0) & (scores <= 1)):
                    raise ValueError("scores must be between 0 and 1 inclusive")

                if self.method == "platt":
                    # Reshape scores to 2D array for LogisticRegression
                    scores_2d = scores.reshape(-1, 1)
                    self.calibrators[scorer] = LogisticRegression()
                    self.calibrators[scorer].fit(scores_2d, correct_indicators)
                elif self.method == "isotonic":
                    self.calibrators[scorer] = IsotonicRegression(out_of_bounds="clip")
                    self.calibrators[scorer].fit(scores, correct_indicators)
                else:
                    raise ValueError(f"Unknown method: {self.method}")

        self.is_fitted_ = True

    def transform(self, uq_result: UQResult) -> None:
        """
        Transform raw scores into calibrated probabilities.

        Parameters
        ----------
        uq_result : UQResult
            The UQResult object to transform.

        Returns
        -------
        None
        """
        if not self.is_fitted_:
            raise ValueError("Calibrator must be fitted before transform")

        tmp = {}
        for scorer in uq_result.data:
            if scorer not in Ignore_Columns:
                if scorer not in self.calibrators.keys():
                    raise ValueError("Scorer outputs contained in the provided UQResult do not match the scorers used for calibration")
                scores = np.array(uq_result.data[scorer])
                if self.method == "platt":
                    # LogisticRegression needs 2D input and returns probabilities for class 1
                    scores_2d = scores.reshape(-1, 1)
                    tmp["calibrated_" + scorer] = self.calibrators[scorer].predict_proba(scores_2d)[:, 1]
                elif self.method == "isotonic":
                    # IsotonicRegression can handle 1D arrays
                    tmp["calibrated_" + scorer] = self.calibrators[scorer].predict(scores)
                else:
                    raise ValueError(f"Unknown method: {self.method}")
        uq_result.data.update(tmp)

    def fit_transform(self, uq_result: UQResult, correct_indicators: Union[List[bool], List[int], np.ndarray]) -> None:
        """
        Fit the calibrator and transform the scores in one step.

        Parameters
        ----------
        uq_result : UQResult
            The UQResult object to fit and transform.
        correct_indicators : array-like of shape (n_samples,)
            Binary labels indicating correctness (True/False or 1/0).

        Returns
        -------
        None
        """
        self.fit(uq_result, correct_indicators)
        self.transform(uq_result)

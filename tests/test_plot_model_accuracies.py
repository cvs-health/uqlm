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

import os
import pytest
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from uqlm.utils.plots import plot_model_accuracies


def test_plot_model_accuracies_basic():
    """Test that the function runs successfully with valid inputs"""
    scores = np.array([0.1, 0.4, 0.35, 0.8])
    correct_indicators = np.array([True, False, True, True])
    thresholds = np.linspace(0, 0.9, num=10)

    try:
        plot_model_accuracies(scores, correct_indicators, thresholds)
    except Exception as e:
        pytest.fail(f"plot_model_accuracies raised an exception {e}")
    plt.close("all")


def test_plot_model_accuracies_value_error():
    """Test that the function raises ValueError when inputs have different lengths"""
    scores = np.array([0.1, 0.4, 0.35])
    correct_indicators = np.array([True, False, True, True])
    thresholds = np.linspace(0, 0.9, num=10)

    with pytest.raises(ValueError):
        plot_model_accuracies(scores, correct_indicators, thresholds)


def test_plot_model_accuracies_with_write_path():
    """Test that the function works when saving the plot to a file"""
    scores = np.array([0.1, 0.4, 0.35, 0.8])
    correct_indicators = np.array([True, False, True, True])
    thresholds = np.linspace(0, 0.9, num=10)
    write_path = "test_plot.png"

    try:
        plot_model_accuracies(scores, correct_indicators, thresholds, write_path=write_path)
    except Exception as e:
        pytest.fail(f"plot_model_accuracies raised an exception {e}")
    plt.close("all")
    assert os.path.exists(write_path)
    os.remove(write_path)

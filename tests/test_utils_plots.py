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
from unittest.mock import patch
from uqlm.utils.results import UQResult
from uqlm.utils.plots import plot_filtered_accuracy, plot_model_accuracies, plot_ranked_auc

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# Dummy Method_Names and Ignore_Columns for testing
Method_Names = {"semantic_negentropy": "Semantic Negentropy", "normalized_probability": "Normalized Probability", "ensemble_scores": "Ensemble Scores"}
Ignore_Columns = ["responses"]

# Tests for plot_model_accuracies()


def test_plot_model_accuracies_basic():
    """Test that the function runs successfully with valid inputs"""
    scores = np.array([0.1, 0.4, 0.35, 0.8])
    correct_indicators = np.array([True, False, True, True])
    thresholds = np.linspace(0, 0.9, num=10)

    plot_model_accuracies(scores, correct_indicators, thresholds)
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

    plot_model_accuracies(scores, correct_indicators, thresholds, write_path=write_path)
    plt.close("all")
    assert os.path.exists(write_path)
    os.remove(write_path)


# Tests for plot_ranked_auc()
@pytest.fixture
def string_response_uq_result():
    return UQResult(result={"data": {"responses": ["The Eiffel Tower is in Berlin.", "Water boils at 100°C.", "The moon is made of cheese.", "Paris is the capital of France."], "semantic_negentropy": [0.9, 0.1, 0.8, 0.2], "normalized_probability": [0.85, 0.15, 0.75, 0.25], "ensemble_scores": [0.88, 0.12, 0.78, 0.22]}, "metadata": {}})


# Input validation tests
def test_missing_correct_indicators(string_response_uq_result):
    with pytest.raises(ValueError, match="correct_indicators must be provided"):
        plot_ranked_auc(string_response_uq_result, None)


def test_length_mismatch(string_response_uq_result):
    with pytest.raises(ValueError, match="correct_responses must be the same length"):
        plot_ranked_auc(string_response_uq_result, [True, False])


def test_invalid_metric_type(string_response_uq_result):
    with pytest.raises(ValueError, match="metric_type must be one of"):
        plot_ranked_auc(string_response_uq_result, [False, True, False, True], metric_type="invalid")


# Plot rendering tests
@patch("matplotlib.pyplot.show")
def test_plot_auroc_only(mock_show, string_response_uq_result):
    plot_ranked_auc(string_response_uq_result, [False, True, False, True], metric_type="auroc")
    mock_show.assert_called_once()


@patch("matplotlib.pyplot.show")
def test_plot_auprc_only(mock_show, string_response_uq_result):
    plot_ranked_auc(string_response_uq_result, [False, True, False, True], metric_type="auprc")
    mock_show.assert_called_once()


@patch("matplotlib.pyplot.show")
def test_plot_both_metrics(mock_show, string_response_uq_result):
    plot_ranked_auc(string_response_uq_result, [False, True, False, True], metric_type="both")
    mock_show.assert_called_once()


# Scorer categorization and naming
@patch("matplotlib.pyplot.show")
def test_scorer_categorization_and_naming(mock_show):
    uq_result = UQResult(result={"data": {"responses": ["A", "B", "C", "D"], "semantic_negentropy": [0.9, 0.1, 0.8, 0.2], "normalized_probability": [0.85, 0.15, 0.75, 0.25], "ensemble_scores": [0.88, 0.12, 0.78, 0.22]}, "metadata": {}})
    correct = [False, True, False, True]

    plot_ranked_auc(uq_result, correct, scorers_names=["semantic_negentropy", "normalized_probability", "ensemble_scores"], metric_type="auroc")
    mock_show.assert_called_once()


# Default scorer selection (scorers_names=None)
@patch("matplotlib.pyplot.show")
def test_default_scorer_selection(mock_show):
    uq_result = UQResult(result={"data": {"responses": ["A", "B", "C", "D"], "semantic_negentropy": [0.9, 0.1, 0.8, 0.2], "normalized_probability": [0.85, 0.15, 0.75, 0.25], "ensemble_scores": [0.88, 0.12, 0.78, 0.22], "prompts": ["Q1", "Q2", "Q3", "Q4"]}, "metadata": {}})
    correct = [False, True, False, True]

    plot_ranked_auc(uq_result, correct, scorers_names=None, metric_type="auroc")
    mock_show.assert_called_once()


# Tests for plot_filtered_accuracy()
@pytest.fixture
def sample_uq_result():
    return UQResult(result={"data": {"responses": ["The Eiffel Tower is in Berlin.", "Water boils at 100°C.", "The moon is made of cheese.", "Paris is the capital of France."], "semantic_negentropy": [0.9, 0.1, 0.8, 0.2], "normalized_probability": [0.85, 0.15, 0.75, 0.25], "ensemble_scores": [0.88, 0.12, 0.78, 0.22]}, "metadata": {}})


def test_missing_correct_indicators_raises(sample_uq_result):
    with pytest.raises(ValueError, match="correct_indicators must be provided"):
        plot_filtered_accuracy(sample_uq_result, None)


def test_mismatched_length_raises(sample_uq_result):
    wrong_length = [1, 0]
    with pytest.raises(ValueError, match="correct_responses must be the same length"):
        plot_filtered_accuracy(sample_uq_result, wrong_length)


@patch("matplotlib.pyplot.show")
def test_plot_runs_successfully(mock_show, sample_uq_result):
    correct = [0, 1, 0, 1]
    plot_filtered_accuracy(sample_uq_result, correct)
    assert mock_show.called


@patch("matplotlib.pyplot.savefig")
@patch("matplotlib.pyplot.show")
def test_plot_saves_image(mock_show, mock_savefig, sample_uq_result, tmp_path):
    correct = [0, 1, 0, 1]
    out_path = tmp_path / "accuracy_plot.png"
    plot_filtered_accuracy(sample_uq_result, correct, write_path=str(out_path))
    mock_savefig.assert_called_once_with(str(out_path), dpi=300)


@patch("matplotlib.pyplot.show")
def test_plot_with_specific_scorers(mock_show, sample_uq_result):
    correct = [0, 1, 0, 1]
    plot_filtered_accuracy(sample_uq_result, correct, scorers_names=["semantic_negentropy", "ensemble_scores"])
    assert mock_show.called


@patch("matplotlib.pyplot.show")
def test_plot_with_custom_title_and_font(mock_show, sample_uq_result):
    correct = [0, 1, 0, 1]
    plot_filtered_accuracy(sample_uq_result, correct, title="Custom Accuracy Plot", fontsize=12, fontname="Arial")
    assert mock_show.called


@patch("matplotlib.pyplot.show")
def test_plot_excludes_ignore_columns(mock_show):
    uq_result = UQResult(
        result={
            "data": {
                "responses": ["A", "B", "C", "D"],
                "semantic_negentropy": [0.9, 0.1, 0.8, 0.2],
                "normalized_probability": [0.85, 0.15, 0.75, 0.25],
                "ensemble_scores": [0.88, 0.12, 0.78, 0.22],
                "metadata": [0, 0, 0, 0],  # Should be ignored
            },
            "metadata": {},
        }
    )
    Ignore_Columns.append("metadata")
    correct = [0, 1, 0, 1]
    plot_filtered_accuracy(uq_result, correct, scorers_names=["semantic_negentropy", "normalized_probability", "ensemble_scores"])
    Ignore_Columns.remove("metadata")  # Clean up
    assert mock_show.called

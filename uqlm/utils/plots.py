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


import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import ArrayLike
from typing import Optional


def scale(values, upper, lower):
    """Helper function to scale valuees in plot"""
    max_v, min_v = max(values), min(values)
    return [lower + (val - min_v) * (upper - lower) / (max_v - min_v) for val in values]


def plot_model_accuracies(scores: ArrayLike, correct_indicators: ArrayLike, thresholds: ArrayLike = np.linspace(0, 0.9, num=10), axis_buffer: float = 0.1, title: str = "LLM Accuracy by Confidence Score Threshold", write_path: Optional[str] = None, bar_width=0.05, display_percentage: bool = False):
    """
    Parameters
    ----------
    scores : list of float
        A list of confidence scores from an uncertainty quantifier

    correct_indicators : list of bool
        A list of boolean indicators of whether self.original_responses are correct.

    thresholds : ArrayLike, default=np.linspace(0, 1, num=10)
        A correspoding list of threshold values for accuracy computation

    axis_buffer : float, default=0.1
        Specifies how much of a buffer to use for vertical axis

    title : str, default="LLM Accuracy by Confidence Score Threshold"
        Chart title

    write_path : Optional[str], default=None
        Destination path for image file.

    bar_width : float, default=0.05
        The width of the bars in the plot

    display_percentage : bool, default=False
        Whether to display the sample size as a percentage

    Returns
    -------
    None
    """
    n_samples = len(scores)
    if n_samples != len(correct_indicators):
        raise ValueError("scores and correct_indicators must be the same length")

    accuracies, sample_sizes = [], []
    denominator = n_samples / 100 if display_percentage else 1
    for t in thresholds:
        grades_t = [correct_indicators[i] for i in range(0, len(scores)) if scores[i] >= t]
        accuracies.append(np.mean(grades_t))
        sample_sizes.append(len(grades_t) / denominator)

    min_acc = min(accuracies)
    max_acc = max(accuracies)

    # Create a single figure and axis
    _, ax = plt.subplots()

    # Plot the first dataset (original)
    ax.scatter(thresholds, accuracies, s=15, marker="s", label="Accuracy", color="blue")
    ax.plot(thresholds, accuracies, color="blue")

    # Calculate sample proportion for the first dataset
    normalized_sample_1 = scale(sample_sizes, upper=max_acc, lower=min_acc)

    # Adjust x positions for the first dataset
    bar_positions = np.array(thresholds)
    label = "Sample Size" if not display_percentage else "Sample Size (%)"
    pps1 = ax.bar(bar_positions, normalized_sample_1, label=label, alpha=0.2, width=bar_width)

    # Annotate the bars for the first dataset
    count = 0
    for p in pps1:
        height = p.get_height()
        s_ = "{:.0f} %".format(sample_sizes[count]) if display_percentage else "{:.0f}".format(sample_sizes[count])
        ax.text(x=p.get_x() + p.get_width() / 2, y=height - (height - min_acc * (1 - axis_buffer)) / 50, s=s_, ha="center", fontsize=8, rotation=90, va="top")
        count += 1

    # Set x and y ticks, limits, labels, and title
    ax.set_xticks(np.arange(0, 1, 0.1))
    ax.set_xlim([-0.04, 0.95])
    ax.set_ylim([min_acc * (1 - axis_buffer), max_acc * (1 + axis_buffer)])
    ax.legend()
    ax.set_xlabel("Thresholds")
    ax.set_ylabel("LLM Accuracy (Filtered)")
    ax.set_title(f"{title}", fontsize=10)
    if write_path:
        plt.savefig(f"{write_path}", dpi=300)
    plt.show()


def ranked_bar_plot(scores: dict, weights: ArrayLike = None, title: str = None, write_path: Optional[str] = None, bar_colors: list = ["C0", "C2", "C3", "C4"], fs: int = 10, fn: str = "Arial") -> plt.Axes:
    """
    Parameters
    ----------
    scores : dict of dict
        A dictionary where each key is a method name and each value is a dictionary
        containing information about different scorers.
        Example:
            {
                "White-box": {"scorer1": 0.85, "scorer2": 0.72, ...},
                "Black-box": {"scorer1": 0.85, "scorer2": 0.72, ...},
                "Judges": {"scorer1": 0.85, "scorer2": 0.72, ...},
                "Ensemble": {"scorer": 0.85},
            }

    write_path : Optional[str], default=None
        The path to save the plot

    weights : ArrayLike, default=None
        The weights of the scorers

    fs : int, default=10
        The font size of the plot

    fn : str, default="Arial"
        The font name of the plot

    title : str, default=None
        The title of the plot

    Returns
    -------
    ax : matplotlib.axes.Axes
        The axes object of the plot
    """
    _, ax = plt.subplots()
    cols, values = [], []
    for key in scores:
        for scorer in scores[key]:
            cols.append(scorer)
            values.append(scores[key][scorer])

    if weights is not None:
        sorted_tuples = sorted(zip(values, cols, weights))
        sorted_values, sorted_cols, sorted_weights = zip(*sorted_tuples)
    else:
        sorted_values, sorted_cols = zip(*sorted(zip(values, cols)))

    for i in range(len(sorted_values)):
        if sorted_cols[i] in scores.get("Black-box", {}):
            c = bar_colors[0]
        elif sorted_cols[i] in scores.get("White-box", {}):
            c = bar_colors[1]
        elif sorted_cols[i] in scores.get("Judges", {}):
            c = bar_colors[2]
        else:
            c = bar_colors[3]
        ax.barh(sorted_cols[i], sorted_values[i], color=c)

    ax.set_xlim(sorted_values[0] - 0.2, sorted_values[-1] + 0.04)
    ax.tick_params(axis="x", labelsize=fs - 3)
    ax.tick_params(axis="y", labelsize=fs - 3)
    ax.grid()
    ax.set_title(title, fontsize=fs, y=-0.22, fontname=fn)

    # Add labels to the right of each bar
    if weights is not None:
        for i in range(len(sorted_values)):
            bar_value = sorted_values[i]
            bar_label = f"{sorted_cols[i]} ({sorted_weights[i]:.2f})"
            ax.text(bar_value + 0.01, sorted_cols[i], bar_label, va="center", ha="left", fontsize=fs - 3, fontname=fn)

    if write_path:
        plt.savefig(f"{write_path}", dpi=300)
    plt.show()
    return ax


def plot_filtered_accuracy(top_scores: dict, response_correct: list, fs=12) -> plt.Axes:
    """
    Plot the filtered accuracy for the given top_scores.
    top_scores: dict
        Dictionary containing the top scores for each technique.
        Example:
        {
            'White-Box UQ': [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0],
            'Black-Box UQ': [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0],
            'Judge': [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]
        }
    response_correct: list
        List of response correctness.
    fs: int
        Font size for the plot.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The axes object of the plot
    """
    _, ax = plt.subplots()
    thresholds = np.arange(0, 1, 0.1)

    accuracy = {}
    for key in top_scores:
        y_true = response_correct
        y_score = top_scores[key]
        accuracy[key] = list()
        for thresh in thresholds:
            accuracy[key].append(np.mean([y_true[i] for i in range(0, len(y_true)) if y_score[i] >= thresh]))

    marker = {"White-Box UQ": "o", "Black-Box UQ": "s", "Judge": "^"}
    color = {"White-Box UQ": "C2", "Black-Box UQ": "C0", "Judge": "C3"}
    ax.hlines(accuracy[key][0], 0, 0.9, color="k", linestyles="dashed", label="Baseline LLM Accuracy")
    for key in accuracy:
        ax.plot(thresholds, accuracy[key], marker=marker[key], label=key, color=color[key])
    ax.hlines(accuracy[key][0], 0, 0.9, color="k", linestyles="dashed")

    ax.set_xlim(-0.05, 0.95)
    ax.tick_params(axis="both", labelsize=fs - 3)  # Increase tick label font size
    ax.set_xlabel("Confidence Score Threshold", fontsize=fs)
    ax.set_ylabel("LLM Filtered Accuracy", fontsize=fs)
    ax.legend(fontsize=fs - 3, bbox_to_anchor=(1.05, 1.1), ncol=4)
    ax.grid()
    plt.show()
    return ax

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

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
from matplotlib.patches import Rectangle
import numpy as np
from numpy.typing import ArrayLike
from typing import List, Optional
from sklearn.metrics import roc_auc_score, average_precision_score

from uqlm.utils.results import UQResult

Black_Box_Scorers = ["semantic_negentropy", "noncontradiction", "exact_match", "cosine_sim"]
White_Box_Scorers = ["normalized_probability", "min_probability"]
Ensemble = ["ensemble_scores"]
Ignore_Columns = ["prompts", "responses", "sampled_responses", "raw_sampled_responses", "raw_responses", "logprobs"]
Method_Names = {"semantic_negentropy": "Semantic Negentropy", "noncontradiction": "Non-Contradiction", "exact_match": "Exact Match", "cosine_sim": "Cosine Similarity", "normalized_probability": "Normalized Probability", "min_probability": "Min Probability", "ensemble_scores": "Ensemble"}


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


def plot_ranked_auc(uq_result: UQResult, correct_indicators: ArrayLike, scorers_names: List[str] = None, write_path: Optional[str] = None, title: str = "Hallucination Detection: Scorer-specific AUROC", fontsize: int = 10, fontname: str = None, metric_type="auroc"):
    """
    Plot the ranked bar plot for hallucination detection AUROC/AUPRC of the given scorers.

    Parameters
    ----------
    uq_result : UQResult
        The UQResult object to plot

    correct_indicators : ArrayLike
        The correct indicators of the responses

    scorers_names : List[str], default=None
        The names of the scorers to plot

    title : str, default="Hallucination Detection: Scorer-specific AUROC"
        The title of the plot. Adjusted based on the metric type

    write_path : Optional[str], default=None
        The path to save the plot

    fontsize : int, default=10
        The font size of the plot

    fontname : str, default=None
        The font name of the plot

    metric_type: str, default="auroc"
        Type of metric(s) to compute and plot:
       - "auroc": Plot only AUROC scores (Area Under ROC Curve)
       - "auprc": Plot only AUPRC scores (Area Under Precision-Recall Curve)
       - "both": Plot both AUROC and AUPRC side by side in subplots

    Returns
    -------
    None
    """
    bar_colors: list = ["C0", "C2", "C3", "C4"]

    if correct_indicators is None:
        raise ValueError("correct_indicators must be provided")
    if len(correct_indicators) != len(uq_result.data["responses"]):
        raise ValueError("correct_responses must be the same length as the number of responses")

    if scorers_names is None:
        scorers_names = [col for col in uq_result.data.keys() if col not in Ignore_Columns]

    if metric_type not in ["auroc", "auprc", "both"]:
        raise ValueError("metric_type must be one of 'both', 'auroc', 'auprc'")

    # Determine which metrics to compute
    metrics = ["auroc", "auprc"] if metric_type == "both" else [metric_type]

    # Initialize score dictionaries for each metric
    scores = {}
    for metric in metrics:
        scores[metric] = {"Black-box": {}, "White-box": {}, "Judges": {}, "Ensemble": {}}

    if metric_type in ["both", "auprc"]:
        # For AUPRC, we need flipped labels
        incorrect_indicators = [not ci for ci in correct_indicators]

    for col in scorers_names:
        if col in uq_result.data.keys():
            # Get uncertainty scores
            uncertainty_scores = [1 - cs for cs in uq_result.data[col]]

            for metric in metrics:
                # Calculate metric score
                if metric == "auprc":
                    score_value = average_precision_score(incorrect_indicators, uncertainty_scores)
                else:  # auroc
                    score_value = roc_auc_score(correct_indicators, uq_result.data[col])
                # Determine category and store score
                if col in Black_Box_Scorers:
                    category = "Black-box"
                elif col in White_Box_Scorers:
                    category = "White-box"
                elif col[:6] == "judge_":
                    category = "Judges"
                elif col in Ensemble:
                    category = "Ensemble"
                method_name = Method_Names.get(col, col.replace("_", " ").title())
                scores[metric][category][method_name] = score_value
    # Remove empty categories
    for metric in metrics:
        empty_keys = [k for k, v in scores[metric].items() if not v]
        for k in empty_keys:
            del scores[metric][k]

    # Create plots
    if len(metrics) == 2:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        _plot_single_metric(ax1, scores["auroc"], bar_colors, "AUROC", fontsize, fontname)
        _plot_single_metric(ax2, scores["auprc"], bar_colors, "AUPRC", fontsize, fontname)
        fig.suptitle(title.replace("AUROC", "AUROC & AUPRC"), fontsize=fontsize + 2, fontname=fontname)
        plt.tight_layout()
    else:
        _, ax = plt.subplots(figsize=(10, 6))
        metric = metrics[0]
        metric_name = "AUPRC" if metric == "auprc" else "AUROC"
        _plot_single_metric(ax, scores[metric], bar_colors, metric_name, fontsize, fontname)
        ax.set_title(title.replace("AUROC", metric_name), fontsize=fontsize, fontname=fontname)
    if write_path:
        plt.savefig(f"{write_path}", dpi=300)
    plt.show()


def plot_filtered_accuracy(uq_result: UQResult, correct_indicators: ArrayLike, scorers_names: List[str] = None, write_path: Optional[str] = None, title: str = "LLM Accuracy by Confidence Score Threshold", fontsize: int = 10, fontname: str = None):
    """
    Plot the filtered accuracy for the given scorers.

    uq_result : UQResult
        The UQResult object to plot

    correct_indicators : ArrayLike
        The correct indicators of the responses

    scorers_names : List[str], default=None
        The names of the scorers to plot

    write_path : Optional[str], default=None
        The path to save the plot

    title : str, default="LLM Accuracy by Confidence Score Threshold"
        The title of the plot

    fontsize : int, default=10
        The font size of the plot

    fontname : str, default=None
        The font name of the plot

    Returns
    -------
    None
    """
    if correct_indicators is None:
        raise ValueError("correct_indicators must be provided")
    if len(correct_indicators) != len(uq_result.data["responses"]):
        raise ValueError("correct_responses must be the same length as the number of responses")

    if scorers_names is None:
        scorers_names = [col for col in uq_result.data.keys() if col not in Ignore_Columns]

    _, ax = plt.subplots()
    thresholds = np.arange(0, 1, 0.1)

    accuracy = {}
    for key in scorers_names:
        if key in uq_result.data.keys():
            y_true = correct_indicators
            y_score = uq_result.data[key]
            accuracy[key] = list()
            for thresh in thresholds:
                filtered = [y_true[i] for i in range(len(y_true)) if y_score[i] >= thresh]
                accuracy[key].append(np.mean(filtered) if filtered else np.nan)

    for key in accuracy:
        label_ = f"Judge {key[6:]}" if key[:6] == "judge_" else Method_Names[key]
        ax.plot(thresholds, accuracy[key], label=label_)
    ax.hlines(accuracy[key][0], 0, 0.9, color="k", linestyles="dashed", label="Baseline LLM Accuracy")

    ax.set_xlim(-0.05, 0.95)
    ax.tick_params(axis="both", labelsize=fontsize - 3)  # Increase tick label font size
    ax.set_xlabel("Thresholds", fontsize=fontsize - 2, fontname=fontname)
    ax.set_ylabel("LLM Accuracy (Filtered)", fontsize=fontsize - 2, fontname=fontname)
    ax.legend(fontsize=fontsize - 2)
    ax.grid()
    ax.set_title(f"{title}", fontsize=fontsize, fontname=fontname)
    if write_path:
        plt.savefig(f"{write_path}", dpi=300)
    plt.show()


def _plot_single_metric(ax, scores, bar_colors, metric_name, fontsize, fontname):
    """Helper function to plot a single metric"""

    cols, values = [], []
    for key in scores:
        for scorer in scores[key]:
            cols.append(scorer)
            values.append(scores[key][scorer])
    # Sort values
    sorted_values, sorted_cols = zip(*sorted(zip(values, cols)))
    # Assign colors based on category
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
    legend_elements = []
    legend_labels = []
    # Check which categories are present and add to legend
    if any(col in scores.get("Black-box", {}) for col in sorted_cols):
        legend_elements.append(Rectangle((0, 0), 1, 1, facecolor=bar_colors[0]))
        legend_labels.append("Black-Box UQ")
    if any(col in scores.get("White-box", {}) for col in sorted_cols):
        legend_elements.append(Rectangle((0, 0), 1, 1, facecolor=bar_colors[1]))
        legend_labels.append("White-Box UQ")
    if any(col in scores.get("Judges", {}) for col in sorted_cols):
        legend_elements.append(Rectangle((0, 0), 1, 1, facecolor=bar_colors[2]))
        legend_labels.append("LLM Judge")
    if any(col in scores.get("Ensemble", {}) for col in sorted_cols):
        legend_elements.append(Rectangle((0, 0), 1, 1, facecolor=bar_colors[3]))
        legend_labels.append("Ensemble")
    # Add legend if there are multiple categories
    if len(legend_elements) > 1:
        ax.legend(legend_elements, legend_labels, loc="lower right", fontsize=fontsize - 2)
    ax.set_xlim(sorted_values[0] - 0.2, sorted_values[-1] + 0.04)
    ax.tick_params(axis="x", labelsize=fontsize - 3)
    ax.tick_params(axis="y", labelsize=fontsize - 3)
    ax.grid()
    ax.set_xlabel(f"{metric_name} Score", fontsize=fontsize - 2, fontname=fontname)

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


def accuracy_score(y_true: list, y_pred: list) -> float:
    """
    Calculates the accuracy score.

    Parameters
    ----------
    y_true : list
        A list of True labels.
    y_pred : list
        A list of Predicted labels.

    Returns
    -------
    float
        Accuracy score or the number of correct predictions.
    """
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length.")

    return sum(true == pred for true, pred in zip(y_true, y_pred))/len(y_true)


def balanced_accuracy_score(y_true: list, y_pred: list) -> float:
    """
    Computes the balanced accuracy score.

    Parameters
    ----------
    y_true : list
        A list of True labels.
    y_pred : list
        A list of Predicted labels.

    Returns
    -------
    float
        Balanced accuracy score.
    """
    classes = np.unique(y_true)
    recall_scores = []
    for cls in classes:
        y_true_cls = (y_true == cls)
        y_pred_cls = (y_pred == cls)
        true_positives = np.sum(y_true_cls * y_pred_cls)
        actual_positives = np.sum(y_true_cls)
        
        if actual_positives > 0:
             recall = true_positives / actual_positives
        else:
            recall = 0.0
        recall_scores.append(recall)
    return np.mean(recall_scores)


def fbeta_score(y_true: list, y_pred: list, beta: float = 0.5) -> float:
    """
    Computes the F-beta score.

    Parameters
    ----------
    y_true : list
        A list of True labels.
    y_pred : list
        A list of Predicted labels.
    beta : float, default = 0.5
        Beta value for weighting precision and recall.

    Returns
    -------
    float
        F-beta score.
    """
    
    # Ensure inputs are numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Calculate confusion matrix elements
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    # Calculate precision and recall
    if (tp + fp) == 0:
        precision = 0
    else:
        precision = tp / (tp + fp)

    if (tp + fn) == 0:
        recall = 0
    else:
        recall = tp / (tp + fn)
    
    # Calculate F-beta score
    if (precision + recall) == 0:
        fbeta = 0
    else:
        fbeta = (1 + beta**2) * (precision * recall) / ((beta**2 * precision) + recall)

    return fbeta


def log_loss(y_true, y_pred, eps=1e-15):
    """
    Compute log loss (cross-entropy loss) between true labels and predicted probabilities.

    Parameters
    ----------
    y_true : list
        A list of True labels.
    y_pred : list
        A list of Predicted labels.
    eps : float, default = 1e-15
        A small value to avoid log(0) errors.
 
    Returns
    -------
    float
        Log loss value.
    """
    # Clip probabilities to avoid log(0) errors
    y_true = np.array(y_true)
    y_pred = np.clip(y_pred, eps, 1 - eps)

    # Calculate log loss for each sample
    log_loss_values = - (y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    # Return the average log loss
    return np.mean(log_loss_values)


def roc_auc_score(y_true, y_score):
    """
    Compute Area Under the Receiver Operating Characteristic Curve (ROC AUC).

    Parameters
    ----------
    y_true : list
        A list of True labels.
    y_score : list
        Predicted probabilities or scores.

    Returns
    -------
    float
        ROC AUC score.
    """
    # Combine true labels and scores, then sort by scores in descending order
    data = sorted(zip(y_score, y_true), reverse=True)
    
    # Initialize variables
    tp, fp = 0, 0
    tpr_list, fpr_list = [0], [0]
    
    num_tp, num_fp = sum(y_true), len(y_true) - sum(y_true)
    # Iterate through sorted data to calculate TPR and FPR
    for score, label in data:
        if label == 1:
            tp += 1
        else:
            fp += 1
        tpr_list.append(tp / num_tp if num_tp > 0 else 0)
        fpr_list.append(fp / num_fp if num_fp > 0 else 0)
    
    # Calculate AUC using trapezoidal rule
    auc = 0
    for i in range(1, len(fpr_list)):
        auc += (fpr_list[i] - fpr_list[i-1]) * (tpr_list[i] + tpr_list[i-1]) / 2
    
    return auc
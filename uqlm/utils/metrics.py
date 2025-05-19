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
import warnings


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
    _check_lists(list1=y_true, list2=y_pred)

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
    _check_lists(list1=y_true, list2=y_pred)
    
    classes = np.unique(y_true)
    recall_scores = []
    for cls in classes:
        y_true_cls = (y_true == cls)
        y_pred_cls = (y_pred == cls)
        true_positives = np.sum(y_true_cls * y_pred_cls)
        actual_positives = np.sum(y_true_cls)
        
        if actual_positives == 0:
            warnings.warn(f"No actual positives in the class {cls}; recall cannot be computed. Instead using np.nan.")
            recall = np.nan
        else:
            recall = true_positives / actual_positives
        recall_scores.append(recall)
    return np.mean(recall_scores)


def f1_score(y_true: list, y_pred: list):
    """
    Computes the F1-score.

    Parameters
    ----------
    y_true : list
        A list of True labels.
    y_pred : list
        A list of Predicted labels.

    Returns
    -------
    float
        F1-score.
    """
    return fbeta_score(y_true=y_true, y_pred=y_pred, beta=1.0)


def fbeta_score(y_true: list, y_pred: list, beta: float = 1.0) -> float:
    """
    Computes the F-beta score.

    Parameters
    ----------
    y_true : list
        A list of True labels.
    y_pred : list
        A list of Predicted labels.
    beta : float, default = 1
        Beta value for weighting precision and recall.

    Returns
    -------
    float
        F-beta score.
    """
    precision = precision_score(y_true=y_true, y_pred=y_pred)
    recall = recall_score(y_true=y_true, y_pred=y_pred)
    
    # Calculate F-beta score
    if (precision + recall) == 0:
        return 0.0
    return (1 + beta**2) * (precision * recall) / ((beta**2 * precision) + recall)


def log_loss(y_true: list, y_pred: list, eps: float=1e-15):
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
    _check_lists(list1=y_true, list2=y_pred)
    
    # Clip probabilities to avoid log(0) errors
    y_true = np.array(y_true)
    y_pred = np.clip(y_pred, eps, 1 - eps)

    # Calculate log loss for each sample
    log_loss_values = - (y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    # Return the average log loss
    return np.mean(log_loss_values)


def precision_score(y_true: list, y_pred: list):
    """
    Calculate precision score.

    Parameters
    ----------
    y_true : list
        A list of True labels.
    y_pred : list
        A list of Predicted labels.

    Returns
    -------
    float
        Precision score.
    """
    _check_lists(list1=y_true, list2=y_pred)
    
    true_positives = 0
    num_positive_preds = 0

    for true, pred in zip(y_true, y_pred):
        if pred == 1:
            num_positive_preds += 1
            if true == pred:
                true_positives += 1
    
    if num_positive_preds == 0:
        warnings.warn("No predicted positives in the sample; precision cannot be computed. Returning np.nan.")
        return np.nan
    
    return true_positives / num_positive_preds


def recall_score(y_true: list, y_pred: list):
    """
    Calculates the recall score.

    Parameters
    ----------
    y_true : list
        A list of True labels.
    y_pred : list
        A list of Predicted labels.

    Returns
    -------
    float 
        Recall score.
    """
    _check_lists(list1=y_true, list2=y_pred)
    
    true_positives = 0
    num_actual_positives = 0

    for true, pred in zip(y_true, y_pred):
        if true == 1:
            num_actual_positives += 1
            if true == pred:
                true_positives += 1

    if num_actual_positives == 0:
        warnings.warn("No actual positives in the sample; recall cannot be computed. Returning np.nan.")
        return np.nan
    
    return true_positives / num_actual_positives


def roc_auc_score(y_true: list, y_score: list):
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
    _check_lists(list1=y_true, list2=y_score, list2_name="y_score")
    
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


def _check_lists(list1: list, list2: list, list2_name:str = "y_pred"):
    if len(list1) != len(list2):
        raise ValueError(f"y_true and {list2_name} must have the same length.")
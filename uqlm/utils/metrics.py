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
import warnings


def accuracy_score(y_true: ArrayLike, y_pred: ArrayLike, check_inputs: bool=True) -> float:
    """
    Calculates the accuracy score.

    Parameters
    ----------
    y_true : ArrayLike
        A list or numpy.ndarray of True labels.
    y_pred : ArrayLike
        A list of Predicted labels.
    check_inputs : bool
        Whether to validate the input arrays.

    Returns
    -------
    float
        Accuracy score or the number of correct predictions.
    """
    if check_inputs:
        y_true, y_pred = _check_arrays(array1=y_true, array2=y_pred)

    return np.sum(y_true == y_pred)/len(y_true)


def balanced_accuracy_score(y_true: ArrayLike, y_pred: ArrayLike, check_inputs: bool=True, value_if_undefined: float=0.0) -> float:
    """
    Computes the balanced accuracy score.

    Parameters
    ----------
    y_true : ArrayLike
        A list or numpy.ndarray of True labels.
    y_pred : ArrayLike
        A list of Predicted labels.
    check_inputs : bool
        Whether to validate the input arrays.
    value_if_undefined : float
        Value to return if no actual positives are present in the sample.

    Returns
    -------
    float
        Balanced accuracy score.
    """
    if check_inputs:
        y_true, y_pred = _check_arrays(array1=y_true, array2=y_pred)

    classes = np.array([0, 1])
    
    # Create binary masks for each class
    y_true_masks = y_true == classes[:, np.newaxis]  # Shape: (n_classes, n_samples)
    y_pred_masks = y_pred == classes[:, np.newaxis]  # Shape: (n_classes, n_samples)

    # Calculate true positives and actual positives for all classes at once
    true_positives = np.sum(y_true_masks * y_pred_masks, axis=1)
    actual_positives = np.sum(y_true_masks, axis=1)
    
    # # Handle cases where there are no actual positives
    recall_scores = np.divide(true_positives, actual_positives, out=np.array([value_if_undefined]*len(classes)), where=actual_positives!=0)
    
    undefined_ind = actual_positives == 0
    if np.count_nonzero(undefined_ind) > 0:
        warnings.warn(f"No actual positives in the class {classes[undefined_ind]}; recall cannot be computed. Instead using {value_if_undefined}. To modify this behavior, use `value_if_undefined` parameter.")

    return np.mean(recall_scores)


def f1_score(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """
    Computes the F1-score.

    Parameters
    ----------
    y_true : ArrayLike
        A list or numpy.ndarray of True labels.
    y_pred : ArrayLike
        A list of Predicted labels.

    Returns
    -------
    float
        F1-score.
    """
    return fbeta_score(y_true=y_true, y_pred=y_pred, beta=1.0)


def fbeta_score(y_true: ArrayLike, y_pred: ArrayLike, check_inputs: bool=True, beta: float=1.0) -> float:
    """
    Computes the F-beta score.

    Parameters
    ----------
    y_true : ArrayLike
        A list or numpy.ndarray of True labels.
    y_pred : ArrayLike
        A list of Predicted labels.
    check_inputs : bool
        Whether to validate the input arrays.
    beta : float, default = 1
        Beta value for weighting precision and recall.

    Returns
    -------
    float
        F-beta score.
    """
    if check_inputs:
        y_true, y_pred = _check_arrays(array1=y_true, array2=y_pred)

    precision = precision_score(y_true=y_true, y_pred=y_pred, check_inputs=False)
    recall = recall_score(y_true=y_true, y_pred=y_pred, check_inputs=False)
    
    # Calculate F-beta score
    if (precision + recall) == 0:
        return 0.0
    return (1 + beta**2) * (precision * recall) / ((beta**2 * precision) + recall)


def log_loss(y_true: ArrayLike, y_pred: ArrayLike, check_inputs: bool=True, eps: float=1e-15) -> float:
    """
    Compute log loss (cross-entropy loss) between true labels and predicted probabilities.

    Parameters
    ----------
    y_true : ArrayLike
        A list or numpy.ndarray of True labels.
    y_pred : ArrayLike
        A list of Predicted probabilities.
    check_inputs : bool
        Whether to validate the input arrays.
    eps : float, default = 1e-15
        A small value to avoid log(0) errors.
 
    Returns
    -------
    float
        Log loss value.
    """
    if check_inputs:
        y_true, y_pred = _check_arrays(array1=y_true, array2=y_pred, array2_name="y_score")
    
    # Clip probabilities to avoid log(0) errors
    y_pred = np.clip(y_pred, eps, 1 - eps)

    # Calculate log loss for each sample
    log_loss_values = - (y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    # Return the average log loss
    return np.mean(log_loss_values)


def precision_score(y_true: ArrayLike, y_pred: ArrayLike, check_inputs: bool=True, value_if_undefined: float=0.0) -> float:
    """
    Calculate precision score.

    Parameters
    ----------
    y_true : ArrayLike
        A list or numpy.ndarray of True labels.
    y_pred : ArrayLike
        A list of Predicted labels.
    check_inputs : bool
        Whether to validate the input arrays.
    value_if_undefined : float
        Value to return if no predicted positives are present in the sample.

    Returns
    -------
    float
        Precision score.
    """
    if check_inputs:
        y_true, y_pred = _check_arrays(array1=y_true, array2=y_pred)
        
    true_positives = np.count_nonzero((y_true == 1) & (y_pred == 1))
    predicted_positives = np.count_nonzero(y_pred == 1)
    
    if predicted_positives == 0:
        warnings.warn(f"No predicted positives in the sample; precision cannot be computed. A value of {value_if_undefined} will be returned. To modify this behavior, use `value_if_undefined` parameter.")
        return value_if_undefined
    
    return true_positives / predicted_positives


def recall_score(y_true: ArrayLike, y_pred: ArrayLike, check_inputs: bool=True, value_if_undefined: float=0.0) -> float:
    """
    Calculates the recall score.

    Parameters
    ----------
    y_true : ArrayLike
        A list or numpy.ndarray of True labels.
    y_pred : ArrayLike
        A list of Predicted labels.
    check_inputs : bool
        Whether to validate the input arrays.
    value_if_undefined : float
        Value to return if no actual positives are present in the sample.

    Returns
    -------
    float 
        Recall score.
    """
    if check_inputs:
        y_true, y_pred = _check_arrays(array1=y_true, array2=y_pred)
    
    true_positives = np.count_nonzero((y_true == 1) & (y_pred == 1))
    actual_positives = np.count_nonzero(y_true == 1)

    if actual_positives == 0:
        warnings.warn(f"No actual positives in the sample; recall cannot be computed. A value of {value_if_undefined} will be returned. To modify this behavior, use `value_if_undefined` parameter.")
        return value_if_undefined
    
    return true_positives / actual_positives


def roc_auc_score(y_true: ArrayLike, y_score: ArrayLike, check_inputs: bool=True) -> float:
    """
    Compute Area Under the Receiver Operating Characteristic Curve (ROC AUC).

    Parameters
    ----------
    y_true : ArrayLike
        A list or numpy.ndarray of True labels.
    y_score : ArrayLike
        Predicted probabilities or scores.
    check_inputs : bool
        Whether to validate the input arrays.

    Returns
    -------
    float
        ROC AUC score.
    """
    if check_inputs:
        y_true, y_score = _check_arrays(array1=y_true, array2=y_score, array2_name="y_score")
        
    # Sort indices by predicted scores in descending order
    desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
    y_true = y_true[desc_score_indices]

    # Calculate cumulative true positive and false positive counts
    cumulative_tp = np.cumsum(y_true)
    cumulative_fp = np.cumsum(1 - y_true)

    num_tp = np.count_nonzero(y_true) 
    num_fp = len(y_true) - num_tp
    # Calculate true positive rates and false positive rates
    tpr = np.concatenate([[0], cumulative_tp / num_tp]) if num_tp > 0 else np.array([0, 0])
    fpr = np.concatenate([[0], cumulative_fp / num_fp]) if num_fp > 0 else np.array([0, 0])

    # Calculate AUC using trapezoidal rule
    return np.trapz(tpr, fpr)


def _check_arrays(array1: ArrayLike, array2: ArrayLike, array2_name:str = "y_pred") -> None:
    if len(array1) != len(array2):
        raise ValueError(f"y_true and {array2_name} must have the same length.")
    
    array1 = np.asarray(array1)
    array2 = np.asarray(array2)
        
    arrays_to_check_elements = [array1, array2] if array2_name=="y_pred" else [array1]
    for array_ in arrays_to_check_elements:
        unique_values = np.unique(array_)
        if not np.all(np.isin(unique_values, [0, 1])) or not np.all(np.isin(unique_values, [True, False])):
            raise AssertionError("Array contains values other than [0, 1] or [True, False].")
            
    return array1, array2
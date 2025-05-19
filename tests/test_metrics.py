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


import pytest
import random
from uqlm.utils.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    fbeta_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)


@pytest.fixture
def y_true():  
    random.seed(1)
    return [random.choice([True, False]) for _ in range(10)]


@pytest.fixture
def y_pred():  
    random.seed(2)
    return [random.choice([True, False]) for _ in range(10)]


@pytest.fixture
def y_score():  
    random.seed(3)
    return [random.random() for _ in range(10)]


def test_accuracy_score(y_true, y_pred):
    assert accuracy_score(y_true, y_pred) == 0.6
    
    
def test_balanced_accuracy_score(y_true, y_pred):
    assert balanced_accuracy_score(y_true, y_pred) == 0.6000000000000001
    

def test_f1_score(y_true, y_pred):
    assert f1_score(y_true, y_pred) == 0.6666666666666666

    
def test_fbeta_score(y_true, y_pred):
    assert fbeta_score(y_true, y_pred, beta=0.5) == 0.6060606060606061

    
def test_log_loss(y_true, y_score):
    assert log_loss(y_true, y_score) == 0.8691546801559589


def test_precision_score(y_true, y_pred):
    assert precision_score(y_true, y_pred) == 0.5714285714285714


def test_recall_score(y_true, y_pred):
    assert recall_score(y_true, y_pred) == 0.8


def test_roc_auc_score(y_true, y_score):
    assert roc_auc_score(y_true, y_score) == 0.48
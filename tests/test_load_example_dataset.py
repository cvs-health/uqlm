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
import pandas as pd
from uqlm.utils.dataloader import load_example_dataset, list_dataset_names


def test_list_dataset_names():
    datasets = list_dataset_names()
    assert isinstance(datasets, list)
    assert "gsm8k" in datasets


def test_load_full_dataset():
    df = load_example_dataset("gsm8k")
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0


def test_load_limited_rows():
    df = load_example_dataset("gsm8k", n=100)
    assert df.shape[0] == 100


def test_load_specific_columns():
    df = load_example_dataset("gsm8k", cols=["question", "answer"])
    assert list(df.columns) == ["question", "answer"]


def test_load_nonexistent_dataset():
    with pytest.raises(FileNotFoundError):
        load_example_dataset("nonexistent_dataset")


def test_load_dataset_with_processing():
    df = load_example_dataset("gsm8k", n=100, cols=["question", "answer"])
    assert df.shape[0] == 100
    assert list(df.columns) == ["question", "answer"]

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
from uqlm.utils.dataloader import load_example_dataset, list_dataset_names, _combine_question_and_choices
from uqlm.utils.dataloader import _dataset_processing


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


def test_dataset_processing_type_error():
    # tests that _dataset_processing raises a TypeError when passed something other than a pandas DataFrame
    with pytest.raises(TypeError):
        _dataset_processing("not_a_dataframe")


def test_combine_question_and_choices_type_error():
    # tests that _combine_question_and_choices raises a TypeError when choice_col parameter is not a string or a list
    df = pd.DataFrame({"question": ["Q1"]})
    with pytest.raises(TypeError):
        _combine_question_and_choices(df, "question", 123)


def test_combine_question_and_choices_list_case():
    # test the elif isinstance(choice_col, list) branch
    df = pd.DataFrame({"question": ["Q1"], "choiceA": ["A"]})
    result = _combine_question_and_choices(df, "question", ["choiceA"])
    assert len(result) == 1


def test_load_example_dataset_with_concat_all():
    # test the if split == “all”: branch and concatenate_datasets call
    df = load_example_dataset("svamp")  # svamp has concat=“all” in config
    assert len(df) > 0


def test_dataset_processing_rename_columns():
    # test the column renaming functionality
    df = pd.DataFrame({"old_name": ["A", "B"]})
    result = _dataset_processing(df, rename_columns={"old_name": "new_name"})
    assert "new_name" in result.columns


def test_dataset_processing_strip_non_numeric():
    # test removing non-numeric characters from specified columns
    df = pd.DataFrame({"answer": ["A123B", "C456D"]})
    result = _dataset_processing(df, strip_non_numeric=["answer"])
    assert "123" in result["answer"].values


def test_dataset_processing_strip_whitespace():
    # test removing leading/trailing whitespaces from specified columns
    df = pd.DataFrame({"answer": [" hello ", " world "]})
    result = _dataset_processing(df, strip_whitespace=["answer"])
    assert "hello" in result["answer"].values


def test_dataset_processing_to_upper():
    # test converting text to uppercase for both string columns and list columns
    # test string case
    df = pd.DataFrame({"answer": ["hello"]})
    result = _dataset_processing(df, to_upper=["answer"])
    assert result["answer"].iloc[0] == "HELLO"
    # test list case
    df = pd.DataFrame({"answer": [["hello", "world"]]})
    result = _dataset_processing(df, to_upper=["answer"])
    assert result["answer"].iloc[0] == ["HELLO", "WORLD"]


def test_dataset_processing_to_lower():
    # test converting text to uppercase for both string columns and list columns
    # test string case
    df = pd.DataFrame({"answer": ["HELLO"]})
    result = _dataset_processing(df, to_lower=["answer"])
    assert result["answer"].iloc[0] == "hello"
    # test list case
    df = pd.DataFrame({"answer": [["HELLO", "WORLD"]]})
    result = _dataset_processing(df, to_lower=["answer"])
    assert result["answer"].iloc[0] == ["hello", "world"]


def test_dataset_processing_combine_question_and_choices():
    # testing _combine_question_and_choices()
    df = pd.DataFrame({"question": ["Q1"], "choices": ["A"]})
    combine_params = {"question_col": "question", "choice_col": "choices"}
    result = _dataset_processing(df, combine_question_and_choices=combine_params)
    assert len(result) == 1


def test_dataset_processing_regex_filters_no_group():
    # test regex matching when no group is specified
    df = pd.DataFrame({"answer": ["A123", "B456"]})
    regex_filters = [{"pattern": r"(\d+)", "col": "answer", "operation": "search"}]
    result = _dataset_processing(df, regex_filters=regex_filters)
    assert len(result) == 2


def test_dataset_processing_subset_columns_string():
    # test column subsetting when single column name is provided as a string
    df = pd.DataFrame({"question": ["Q1"], "answer": ["A1"], "other": ["O1"]})
    result = _dataset_processing(df, subset_columns="question")
    assert list(result.columns) == ["question"]


def test_dataset_processing_subset_columns_warning():
    # test warnings when trying to subset columns that do not exist in dataset
    df = pd.DataFrame({"question": ["Q1"], "answer": ["A1"]})
    result = _dataset_processing(df, subset_columns=["question", "missing_col"])
    assert len(result.columns) >= 1


def test_combine_question_and_choices_save_original():
    # test saving original question when combining  questions and choices
    df = pd.DataFrame({"question": ["What is 2+2?"], "choices": ["4"]})
    result = _combine_question_and_choices(df, "question", "choices", save_original_question=True)
    assert "original_question" in result.columns


def test_combine_question_and_choices_dict_format():
    # test handling complex choice formats with text and label columns
    df = pd.DataFrame({"question": ["What is 2+2?"], "choices": [{"text": "Four", "label": "D"}], "text": ["Four"], "label": ["D"]})
    result = _combine_question_and_choices(df, "question", "choices", choice_text_col="text", choice_label_col="label")
    assert len(result) == 1

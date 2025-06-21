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

import pandas as pd
from typing import Optional, Union
from datasets import load_dataset, concatenate_datasets
from datasets import disable_progress_bars
import re
import numpy as np
from copy import deepcopy

"""This module uses the _dataset_default_params dict to control what datasets load_example_dataset can load and how they are loaded.

You can add new datasets to the loader by adding a new entry in the _dataset_default_params dict like:

'my_dataset_name': {

    'load_params': {'path': 'hf_hub_org_name/org_dataset_name', # HF dataset ref
                    'name': 'if_hf_dataset_has_multi_files', # needed for specific datasets
                    'split': 'train'}, # optional
    'extra_processing': {} # optional, see examples in _dataset_default_params dict

}
"""
_dataset_default_params = {
    "ai2_arc": {
        "load_params": {
            "path": "allenai/ai2_arc",  # HF Hub dataset name
            "name": "ARC-Easy",  # HF Hub filename
            "split": "test",
        },  # HF Hub dataset split
        "extra_processing": {
            "rename_columns": {"answerKey": "answer"},  # renaming is always the first operation
            "strip_whitespace": ["answer"],  # notice we're referencing the renamed col name
            "to_upper": ["answer"],
            "combine_question_and_choices": {"question_col": "question", "choice_col": "choices", "choice_text_col": "text", "choice_label_col": "label"},
            "subset_columns": ["question", "answer"],
        },
    },
    "csqa": {"load_params": {"path": "skrishna/CSQA_preprocessed", "split": "train"}, "extra_processing": {"rename_columns": {"question": "q_only", "answerKey": "answer", "inputs": "question"}, "to_upper": ["answer"], "subset_columns": ["question", "answer"]}},
    "gsm8k": {
        "load_params": {"path": "openai/gsm8k", "name": "main", "split": "train"},
        "extra_processing": {
            "regex_filters": [
                {
                    "pattern": r"#### ([-+]?\d*\.\d+|[-+]?\d+)",  # regex pattern
                    "col": "answer",  # dataset col to apply pattern to
                    "operation": "search",  # type of regex operation
                    "group": 1,
                }
            ],  # capture group desired
            "subset_columns": ["question", "answer"],
        },
    },
    "nq_open": {"load_params": {"path": "google-research-datasets/nq_open", "split": "validation"}, "extra_processing": {"to_lower": ["answer"], "subset_columns": ["question", "answer"]}},
    "popqa": {"load_params": {"path": "akariasai/PopQA", "split": "test"}, "extra_processing": {"rename_columns": {"possible_answers": "answer"}, "to_lower": ["answer"], "subset_columns": ["question", "answer"]}},
    "svamp": {
        "load_params": {"path": "Chilled/SVAMP"},
        "extra_processing": {
            "concat": "all",
            "rename_columns": {"question_concat": "question", "Answer": "answer"},
            "regex_filters": [
                {
                    "pattern": r"#### ([-+]?\d*\.\d+|[-+]?\d+)",  # get numbers only (like gsm8k)
                    "col": "answer",
                    "operation": "search",
                    "group": 1,
                }
            ],
            "subset_columns": ["question", "answer"],
        },
    },
}


def list_dataset_names() -> list:
    """
    List all available example dataset names in uqlm.

    Returns
    -------
    list
        A list of available datasets.

    Example
    -------
    >>> from uqlm.utils.dataloader import list_dataset_names
    >>> list_dataset_names()
    ['ai2_arc', 'csqa', 'dialogue_sum', 'gsm8k', 'nq_open', 'popqa', 'svamp', 'triviaqa']
    """
    return list(_dataset_default_params.keys())


def load_example_dataset(name: str, n: int = None, cols: Optional[Union[list, str]] = None) -> pd.DataFrame:
    """
    Load a dataset for testing purposes.

    Parameters
    ----------
    name : str
        The name of the dataset to load. Must be one of "svamp", "gsm8k", "ai2_arc",
        "csqa", "nq_open", "popqa"

    n : int, optional
        Number of rows to load from the dataset.

    Returns
    -------
    pd.DataFrame
        The loaded dataset.

    Example
    -------
    >>> from uqlm.utils.dataloader import load_example_dataset
    >>> df = load_example_dataset("gsm8k", n=1000)
    >>> df.shape
    (1000, 2)
    """
    dataset_dict = deepcopy(_dataset_default_params)
    if name in dataset_dict.keys():  # loads from huggingface hub
        disable_progress_bars()  # disable hf tqdm bars b/c it's a little ugly
        print(f"Loading dataset - {name}...")
        ds = load_dataset(**dataset_dict[name]["load_params"])
        print("Processing dataset...")
        extras = dataset_dict[name].get("extra_processing", dict())
        if extras:
            if extras.get("concat"):  # combine different splits into one
                split = extras.get("concat")
                if split == "all":  # combine splits into one dataset
                    ds = concatenate_datasets([ds[s] for s in ds])
                extras.pop("concat")
        df = ds.to_pandas()
        if cols:
            extras["subset_columns"] = cols
        if extras:
            df = _dataset_processing(df=df, **extras)  # data wrangling on single df
        if isinstance(n, int):
            df = df.iloc[:n]
        print("Dataset ready!")
        return df
    else:
        raise FileNotFoundError(f"uqlm could not find the dataset '{name}'.\nPlease use `uqlm.utils.dataloader.list_dataset_names()` for available sample datasets.")


def _dataset_processing(df: pd.DataFrame, rename_columns: dict = None, subset_columns: list = None, to_upper: list = None, to_lower: list = None, combine_question_and_choices: dict = None, strip_non_numeric: list = None, strip_whitespace: list = None, regex_filters: list[dict] = None) -> pd.DataFrame:
    """
    Process a dataset with various operations.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to process.
    rename_columns : dict, optional
        A dictionary mapping old column names to new column names.
    subset_columns : list, optional
        A list of columns to keep in the dataframe.
    to_upper : list, optional
        A list of columns whose string values should be converted to uppercase.
    to_lower: list, optional
        A list of columns whose string values should be converted to lowercase.
    combine_question_and_choices : dict, optional
        A dictionary with parameters to combine question and question choice columns.
    strip_non_numeric : list, optional
        A list of columns from which to strip non-numeric characters.
    strip_whitespace : list, optional
        A list of columns from which to strip whitespace characters.
    regex_filters: list[dict]
        A list of dictionaries like `{'pattern':r'', 'col':''}` to describe regex transformations to apply on the dataset.
    Returns
    -------
    pd.DataFrame
        The processed dataframe.

    Raises
    ------
    TypeError
        If the input `df` is not a pandas DataFrame.

    Example
    -------
    >>> df = pd.DataFrame({"A": ["a", "b", "c"], "B": ["1", "2", "3"]})
    >>> _dataset_processing(df, rename_columns={"A": "a"}, to_upper=["a"])
       a  B
    0  A  1
    1  B  2
    2  C  3
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Dataset processing requires 'pd.DataFrame' but received '{type(df)}'")

    if rename_columns:
        df = df.rename(columns=rename_columns)
    if strip_non_numeric:
        for col in strip_non_numeric:
            df[col] = df[col].apply(lambda x: "".join(c for c in x if c.isdigit()))
    if strip_whitespace:
        for col in strip_whitespace:
            df[col] = df[col].str.replace(" ", "")
    if to_upper:
        for col in to_upper:
            if isinstance(df[col][0], (list, np.ndarray)):
                df[col] = df[col].apply(lambda x: [s.upper() for s in x])
            else:
                df[col] = df[col].str.upper()
    if to_lower:
        for col in to_lower:
            if isinstance(df[col][0], (list, np.ndarray)):
                df[col] = df[col].apply(lambda x: [s.lower() for s in x])
            else:
                df[col] = df[col].str.lower()
    if combine_question_and_choices:
        df = _combine_question_and_choices(df, **combine_question_and_choices)
    if regex_filters:
        for rfilter in regex_filters:
            if rfilter["operation"] == "search":
                if not rfilter.get("group", None):
                    rfilter["group"] = 0

                df[rfilter["col"]] = df[rfilter["col"]].apply(lambda x: re.search(rfilter["pattern"], x).group(rfilter["group"]) if re.search(rfilter["pattern"], x) else x)
    if subset_columns:
        cols = subset_columns
        if isinstance(cols, (list, str)):
            if isinstance(cols, str):
                cols = [cols]
            cols_in_df = [x for x in cols if x in df.columns]
            df = df[cols_in_df]
            if df.shape[1] != len(cols):
                print("WARNING: some specified columns not found in dataset...")

    return df


def _combine_question_and_choices(df: pd.DataFrame, question_col: str, choice_col: Union[str, list] = None, choice_text_col: str = None, choice_label_col: str = None, save_original_question: bool = False) -> pd.DataFrame:
    """
    Combine question and choices columns in a dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to process.
    question_col : str
        The name of the question column.
    choice_col : Union[str, list], optional
        The name(s) of the choice column(s). If a string, it should be the name of a column containing dictionaries.
        If a list, it should be the names of columns containing choices.
    choice_text_col : str, optional
        The name of the choice text column, used when `choice_col` is a string.
    choice_label_col : str, optional
        The name of the choice label column, used when `choice_col` is a string.
    save_original_question : bool, optional
        Whether to save the original question column as 'original_question'.

    Returns
    -------
    pd.DataFrame
        The processed dataframe with combined question and choices.

    Raises
    ------
    TypeError
        If `choice_col` is not a string or a list.

    Example
    -------
    >>> df = pd.DataFrame({"question": ["What is the capital of France?", "What is 2+2?"], "choices": [{"text": ["Paris", "London"], "label": ["A", "B"]}, {"text": ["3", "4"], "label": ["A", "B"]}]})
    >>> _combine_question_and_choices(df, question_col="question", choice_col="choices", choice_text_col="text", choice_label_col="label")
    >>> df
                               question
    0  What is the capital of France? A) Paris B) London
    1                          What is 2+2? A) 3 B) 4
    """
    if save_original_question and question_col == "question":
        df["original_question"] = df["question"].copy()
    if isinstance(choice_col, str):
        if isinstance(df[choice_col][0], dict):  # example of this format is allenai/ai2_arc
            df["question"] = df[question_col] + " " + df[choice_col].apply(lambda x: " ".join([f"{label}) {text}" for text, label in zip(x[choice_text_col], x[choice_label_col])]))
    elif isinstance(choice_col, list):
        # TODO when a dataset needs this... ex would be where cols are like answerA, answerB
        pass
    else:
        raise TypeError(f"'choice_col' must be str or list, but received '{type(choice_col)}'")
    return df

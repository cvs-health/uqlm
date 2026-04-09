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

import json
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from uqlm.utils.code_evaluation import ensure_list_of_dicts, evaluate_row_unified, sanitize_llm_output


# ---------------------------------------------------------------------------
# sanitize_llm_output
# ---------------------------------------------------------------------------


def test_sanitize_none_returns_empty_string():
    assert sanitize_llm_output(None) == ""


def test_sanitize_plain_code_no_fences():
    code = "def foo():\n    return 1"
    assert sanitize_llm_output(code) == code


def test_sanitize_strips_surrounding_backticks():
    assert sanitize_llm_output("`hello`") == "hello"


def test_sanitize_extracts_python_fenced_block():
    raw = "Here is code:\n```python\ndef foo():\n    return 1\n```"
    result = sanitize_llm_output(raw)
    assert "def foo():" in result
    assert "```" not in result


def test_sanitize_extracts_generic_fenced_block():
    raw = "```\ndef bar():\n    pass\n```"
    result = sanitize_llm_output(raw)
    assert "def bar():" in result
    assert "```" not in result


def test_sanitize_picks_longest_fenced_block():
    raw = "```python\nx = 1\n```\n```python\ndef long_function():\n    x = 1\n    y = 2\n    return x + y\n```"
    result = sanitize_llm_output(raw)
    assert "long_function" in result


def test_sanitize_normalizes_windows_newlines():
    raw = "def foo():\r\n    return 1"
    result = sanitize_llm_output(raw)
    assert "\r" not in result


def test_sanitize_unescapes_html_entities():
    raw = "x &gt; 0"
    result = sanitize_llm_output(raw)
    assert ">" in result
    assert "&gt;" not in result


def test_sanitize_malformed_fence_stripped():
    raw = "```def foo():\n    pass"
    result = sanitize_llm_output(raw)
    assert "```" not in result


# ---------------------------------------------------------------------------
# ensure_list_of_dicts
# ---------------------------------------------------------------------------


def test_ensure_list_of_dicts_with_list():
    data = [{"input": "a", "output": "b"}]
    assert ensure_list_of_dicts(data) == data


def test_ensure_list_of_dicts_with_json_string():
    data = [{"input": "a", "output": "b"}]
    assert ensure_list_of_dicts(json.dumps(data)) == data


def test_ensure_list_of_dicts_with_invalid_string():
    assert ensure_list_of_dicts("not json") == []


def test_ensure_list_of_dicts_with_non_list_non_string():
    assert ensure_list_of_dicts(42) == []


def test_ensure_list_of_dicts_with_empty_list():
    assert ensure_list_of_dicts([]) == []


# ---------------------------------------------------------------------------
# evaluate_row_unified
# ---------------------------------------------------------------------------


def _make_row(response="def foo(): return 1", func_name="foo", test_cases=None):
    if test_cases is None:
        test_cases = [{"input": "", "output": "1"}]
    return {
        "response": response,
        "metadata": {"func_name": func_name},
        "public_test_cases": test_cases,
    }


@patch("uqlm.utils.code_evaluation.subprocess.run")
def test_evaluate_row_unified_parses_valid_json(mock_run):
    payload = {"unit_test_passed": 1, "results": [], "meta": {}}
    mock_run.return_value = MagicMock(stdout=json.dumps(payload), stderr="")
    row = _make_row()
    result = evaluate_row_unified(row)
    assert result["unit_test_passed"] == 1


@patch("uqlm.utils.code_evaluation.subprocess.run")
def test_evaluate_row_unified_handles_invalid_json(mock_run):
    mock_run.return_value = MagicMock(stdout="not-json", stderr="error detail")
    row = _make_row()
    result = evaluate_row_unified(row)
    assert result["unit_test_passed"] == 0
    assert "Non-JSON stdout" in result["meta"]["error_message"]


@patch("uqlm.utils.code_evaluation.subprocess.run")
def test_evaluate_row_unified_no_func_name(mock_run):
    """Rows without func_name should not include fn_name in the subprocess payload."""
    payload = {"unit_test_passed": 0, "results": []}
    mock_run.return_value = MagicMock(stdout=json.dumps(payload), stderr="")
    row = {"response": "print(1)", "metadata": {}, "public_test_cases": []}
    evaluate_row_unified(row)
    call_input = json.loads(mock_run.call_args.kwargs["input"])
    assert "fn_name" not in call_input


@patch("uqlm.utils.code_evaluation.subprocess.run")
def test_evaluate_row_unified_includes_fn_name_when_present(mock_run):
    payload = {"unit_test_passed": 1, "results": []}
    mock_run.return_value = MagicMock(stdout=json.dumps(payload), stderr="")
    row = _make_row(func_name="foo")
    evaluate_row_unified(row)
    call_input = json.loads(mock_run.call_args.kwargs["input"])
    assert call_input["fn_name"] == "foo"


@patch("uqlm.utils.code_evaluation.subprocess.run")
def test_evaluate_row_unified_respects_timeout(mock_run):
    payload = {"unit_test_passed": 1}
    mock_run.return_value = MagicMock(stdout=json.dumps(payload), stderr="")
    row = _make_row()
    evaluate_row_unified(row, timeout=10)
    call_input = json.loads(mock_run.call_args.kwargs["input"])
    assert call_input["timeout"] == 10


@patch("uqlm.utils.code_evaluation.subprocess.run")
def test_evaluate_python_code_adds_columns(mock_run):
    """evaluate_python_code should add unit_test_passed, stderr, stdout columns."""
    from uqlm.utils.code_evaluation import evaluate_python_code

    payload = {"unit_test_passed": 1, "results": [], "stderr": "", "stdout": ""}
    mock_run.return_value = MagicMock(stdout=json.dumps(payload), stderr="")

    df = pd.DataFrame(
        {
            "response": ["def foo(): return 1"],
            "public_test_cases": [json.dumps([{"input": "", "output": "1"}])],
            "metadata": [json.dumps({"func_name": "foo"})],
        }
    )

    result_df = evaluate_python_code(df)
    assert "unit_test_passed" in result_df.columns
    assert "stderr" in result_df.columns
    assert "stdout" in result_df.columns

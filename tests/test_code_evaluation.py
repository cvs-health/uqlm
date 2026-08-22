import pytest

from uqlm.code.code_evaluation import sanitize_llm_output, ensure_list_of_dicts


# sanitize_llm_output


def test_sanitize_extracts_python_fence():
    raw = "Sure!\n```python\ndef f(): return 1\n```\nDone."
    assert sanitize_llm_output(raw) == "def f(): return 1"


def test_sanitize_picks_longest_block():
    raw = "```python\nx = 1\n```\nand\n```python\ndef f():\n    return 1\n```"
    assert "def f()" in sanitize_llm_output(raw)


def test_sanitize_handles_no_fence():
    assert sanitize_llm_output("def f(): return 1") == "def f(): return 1"


def test_sanitize_handles_none():
    assert sanitize_llm_output(None) == ""


# ensure_list_of_dicts


def test_ensure_list_passes_through_list():
    assert ensure_list_of_dicts([{"a": 1}]) == [{"a": 1}]


def test_ensure_list_parses_json_string():
    assert ensure_list_of_dicts('[{"a": 1}]') == [{"a": 1}]


# Regression for bug #11: silent [] return → loud ValueError


def test_ensure_list_raises_on_bad_json():
    with pytest.raises(ValueError):
        ensure_list_of_dicts("not json")


def test_ensure_list_raises_on_wrong_type():
    with pytest.raises(ValueError):
        ensure_list_of_dicts(42)

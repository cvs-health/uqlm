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

"""Tests for miscellaneous prompt templates not covered elsewhere."""

import pytest

from uqlm.utils.prompts.claims_prompts import get_claim_breakdown_prompt as claims_get_claim_breakdown_prompt
from uqlm.utils.prompts.codegen import python_prompt_template, python_prompt_template_stdio

RESPONSE = "Einstein developed the theory of general relativity in 1915."


# ---------------------------------------------------------------------------
# claims_prompts.get_claim_breakdown_prompt
# ---------------------------------------------------------------------------


def test_claims_get_claim_breakdown_prompt_returns_string():
    result = claims_get_claim_breakdown_prompt(RESPONSE)
    assert isinstance(result, str)
    assert len(result) > 0


def test_claims_get_claim_breakdown_prompt_includes_response():
    result = claims_get_claim_breakdown_prompt(RESPONSE)
    assert RESPONSE in result


# ---------------------------------------------------------------------------
# codegen prompt templates
# ---------------------------------------------------------------------------


def test_python_prompt_template_returns_string():
    result = python_prompt_template("Sort a list.", "def sort_list(lst):")
    assert isinstance(result, str)
    assert "Sort a list." in result
    assert "def sort_list(lst):" in result


def test_python_prompt_template_stdio_returns_string():
    result = python_prompt_template_stdio("Read two numbers and print their sum.")
    assert isinstance(result, str)
    assert "Read two numbers" in result
    assert "stdin" in result.lower() or "input" in result.lower()


def test_python_prompt_template_stdio_different_problems():
    r1 = python_prompt_template_stdio("Problem A")
    r2 = python_prompt_template_stdio("Problem B")
    assert r1 != r2

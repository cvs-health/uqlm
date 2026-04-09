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

from uqlm.utils.prompts.claim_qa import get_answer_template, get_multiple_question_template, get_question_template

CLAIM = "Marie Curie won the Nobel Prize in Chemistry in 1911."


# ---------------------------------------------------------------------------
# get_question_template
# ---------------------------------------------------------------------------


def test_get_question_template_returns_string():
    result = get_question_template(CLAIM)
    assert isinstance(result, str)
    assert len(result) > 0


def test_get_question_template_includes_claim():
    result = get_question_template(CLAIM)
    assert CLAIM in result


def test_get_question_template_different_claims_differ():
    r1 = get_question_template("Einstein developed relativity.")
    r2 = get_question_template("Newton discovered gravity.")
    assert r1 != r2


# ---------------------------------------------------------------------------
# get_multiple_question_template
# ---------------------------------------------------------------------------


def test_get_multiple_question_template_without_response():
    result = get_multiple_question_template(CLAIM, num_questions=3)
    assert isinstance(result, str)
    assert CLAIM in result
    assert "3" in result


def test_get_multiple_question_template_with_response():
    response = "Marie Curie was a pioneering scientist."
    result = get_multiple_question_template(CLAIM, num_questions=2, response=response)
    assert isinstance(result, str)
    assert CLAIM in result
    assert response in result


def test_get_multiple_question_template_default_num_questions():
    result = get_multiple_question_template(CLAIM)
    assert isinstance(result, str)
    assert "2" in result


def test_get_multiple_question_template_with_response_uses_different_template():
    without = get_multiple_question_template(CLAIM, num_questions=2)
    with_resp = get_multiple_question_template(CLAIM, num_questions=2, response="Some context.")
    assert without != with_resp


# ---------------------------------------------------------------------------
# get_answer_template
# ---------------------------------------------------------------------------


def test_get_answer_template_basic():
    result = get_answer_template("Who won the Nobel Prize?")
    assert isinstance(result, str)
    assert len(result) > 0


def test_get_answer_template_with_original_question():
    result = get_answer_template("What year?", original_question="Tell me about Marie Curie.")
    assert "Tell me about Marie Curie." in result


def test_get_answer_template_with_original_response():
    result = get_answer_template("What year?", original_response="Marie Curie was a scientist.")
    assert "Marie Curie was a scientist." in result
    assert "What year?" in result


def test_get_answer_template_with_both():
    result = get_answer_template("What year?", original_question="About Curie?", original_response="She was a scientist.")
    assert "About Curie?" in result
    assert "She was a scientist." in result


def test_get_answer_template_no_original_response_uses_short_format():
    result = get_answer_template("What year?")
    assert "few words" in result.lower() or "answer" in result.lower()

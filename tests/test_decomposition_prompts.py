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

from uqlm.utils.prompts.decomposition import get_claim_breakdown_prompt, get_factoid_breakdown_template

SAMPLE_RESPONSE = "Marie Curie was a Polish physicist who won two Nobel Prizes."


def test_get_claim_breakdown_prompt_returns_string():
    result = get_claim_breakdown_prompt(SAMPLE_RESPONSE)
    assert isinstance(result, str)
    assert len(result) > 0


def test_get_claim_breakdown_prompt_includes_response():
    result = get_claim_breakdown_prompt(SAMPLE_RESPONSE)
    assert SAMPLE_RESPONSE in result


def test_get_claim_breakdown_prompt_includes_hash_format_instruction():
    result = get_claim_breakdown_prompt(SAMPLE_RESPONSE)
    assert "###" in result


def test_get_claim_breakdown_prompt_instructs_atomic_facts():
    result = get_claim_breakdown_prompt(SAMPLE_RESPONSE)
    assert "fact" in result.lower()


def test_get_claim_breakdown_prompt_different_inputs_produce_different_prompts():
    r1 = get_claim_breakdown_prompt("Einstein developed the theory of relativity.")
    r2 = get_claim_breakdown_prompt("Newton discovered gravity.")
    assert r1 != r2


def test_get_factoid_breakdown_template_returns_string():
    result = get_factoid_breakdown_template(SAMPLE_RESPONSE)
    assert isinstance(result, str)
    assert len(result) > 0


def test_get_factoid_breakdown_template_includes_response():
    result = get_factoid_breakdown_template(SAMPLE_RESPONSE)
    assert SAMPLE_RESPONSE in result


def test_get_factoid_breakdown_template_includes_hash_format_instruction():
    result = get_factoid_breakdown_template(SAMPLE_RESPONSE)
    assert "###" in result


def test_get_factoid_breakdown_template_different_inputs_produce_different_prompts():
    r1 = get_factoid_breakdown_template("Einstein developed the theory of relativity.")
    r2 = get_factoid_breakdown_template("Newton discovered gravity.")
    assert r1 != r2

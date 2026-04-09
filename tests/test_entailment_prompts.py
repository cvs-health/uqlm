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

from uqlm.utils.prompts.entailment_prompts import get_entailment_prompt

CLAIM = "The sky is blue."
SOURCE = "The sky appears blue due to Rayleigh scattering."


@pytest.mark.parametrize(
    "style,expected_fragment",
    [
        ("binary", "Is the claim entailed by the context above?"),
        ("p_true", "Is the claim supported by the source text?"),
        ("p_false", "Is the claim contradicted by the source text?"),
        ("p_neutral", "Is the claim neutral to the source text?"),
        ("nli_classification", "entailment, contradiction, or neutral"),
        ("default_fallback", "true - if the claim is entailed by the source text"),
    ],
)
def test_get_entailment_prompt_returns_string(style, expected_fragment):
    result = get_entailment_prompt(CLAIM, SOURCE, style)
    assert isinstance(result, str)
    assert len(result) > 0


@pytest.mark.parametrize(
    "style,expected_fragment",
    [
        ("binary", "Is the claim entailed by the context above?"),
        ("p_true", "Is the claim supported by the source text?"),
        ("p_false", "Is the claim contradicted by the source text?"),
        ("p_neutral", "Is the claim neutral to the source text?"),
        ("nli_classification", "entailment, contradiction, or neutral"),
        ("default_fallback", "true - if the claim is entailed by the source text"),
    ],
)
def test_get_entailment_prompt_contains_expected_fragment(style, expected_fragment):
    result = get_entailment_prompt(CLAIM, SOURCE, style)
    assert expected_fragment in result


@pytest.mark.parametrize("style", ["binary", "p_true", "p_false", "p_neutral", "nli_classification", "other"])
def test_get_entailment_prompt_includes_claim_and_source(style):
    result = get_entailment_prompt(CLAIM, SOURCE, style)
    assert CLAIM in result
    assert SOURCE in result


def test_get_entailment_prompt_binary_asks_yes_or_no():
    result = get_entailment_prompt(CLAIM, SOURCE, "binary")
    assert "Yes or No" in result


def test_get_entailment_prompt_nli_classification_lists_all_labels():
    result = get_entailment_prompt(CLAIM, SOURCE, "nli_classification")
    assert "entailment" in result
    assert "contradiction" in result
    assert "neutral" in result


def test_get_entailment_prompt_unknown_style_returns_default():
    result = get_entailment_prompt(CLAIM, SOURCE, "unknown_style_xyz")
    assert result is not None
    assert CLAIM in result

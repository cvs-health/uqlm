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

"""
Lightweight tests for CodeGenUQ initialization and scorer validation.
Kept separate from test_codegen.py to avoid the heavy-transformer skip in conftest.py.
"""

from unittest.mock import MagicMock, patch

import pytest

from uqlm.scorers.shortform.codegen import CodeGenUQ


@pytest.fixture
def mock_llm():
    m = MagicMock()
    m.logprobs = False
    return m


@pytest.fixture(autouse=True)
def patch_heavy_deps():
    """Patch transformer-heavy constructors in codegen module to avoid device/model loading."""
    with (
        patch("uqlm.scorers.shortform.codegen.CosineScorer", MagicMock()),
        patch("uqlm.scorers.shortform.codegen.CodeBLEU", MagicMock()),
        patch("uqlm.scorers.shortform.codegen.VerbalizedConfidence", MagicMock()),
        patch("uqlm.scorers.shortform.codegen.FunctionalEntropy", MagicMock()),
        patch("uqlm.scorers.shortform.codegen.WhiteBoxUQ", MagicMock()),
    ):
        yield


def test_validate_scorers_defaults_when_none(mock_llm):
    cg = CodeGenUQ(llm=mock_llm, scorers=None)
    assert cg.scorers == ["functional_equivalence_rate", "cosine_sim"]


def test_validate_scorers_raises_on_invalid(mock_llm):
    with pytest.raises(ValueError, match="Invalid scorers"):
        CodeGenUQ(llm=mock_llm, scorers=["nonexistent_scorer"])


def test_validate_scorers_cosine_sim_only(mock_llm):
    cg = CodeGenUQ(llm=mock_llm, scorers=["cosine_sim"])
    assert "cosine_sim" in cg.scorers
    assert hasattr(cg, "cos")


def test_validate_scorers_no_wbuq_scorers_when_none_selected(mock_llm):
    cg = CodeGenUQ(llm=mock_llm, scorers=["cosine_sim"])
    assert cg.wbuq_scorers == []


def test_codegen_default_attributes(mock_llm):
    cg = CodeGenUQ(llm=mock_llm)
    assert cg.sampling_temperature == 1.0
    assert cg.top_k_logprobs == 15
    assert cg.length_normalize is True
    assert cg.language == "python"
    assert cg.retries == 5


def test_validate_scorers_equivalence_llm_defaults_to_llm(mock_llm):
    cg = CodeGenUQ(llm=mock_llm, scorers=["cosine_sim"])
    assert cg.equivalence_llm is mock_llm


def test_validate_scorers_consistency_and_confidence_narrows_scorers(mock_llm):
    """consistency_and_confidence branch narrows scorers list (line 165)."""
    cg = CodeGenUQ(llm=mock_llm, scorers=["consistency_and_confidence", "cosine_sim", "sequence_probability"])
    # After line 165, only cosine_sim and sequence_probability should remain
    assert "consistency_and_confidence" not in cg.scorers



def test_validate_scorers_verbalized_confidence_creates_vc(mock_llm):
    """verbalized_confidence scorer constructs VerbalizedConfidence (line 169)."""
    cg = CodeGenUQ(llm=mock_llm, scorers=["verbalized_confidence"])
    assert hasattr(cg, "vc")


def test_validate_scorers_wbuq_scorers_creates_wbuq(mock_llm):
    """White-box scorers in list trigger WhiteBoxUQ construction (line 175)."""
    cg = CodeGenUQ(llm=mock_llm, scorers=["sequence_probability"])
    assert len(cg.wbuq_scorers) > 0
    assert hasattr(cg, "wbuq")

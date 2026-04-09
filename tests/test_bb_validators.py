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

"""Tests for BlackBoxUQ._validate_scorers uncovered branches.

File is named test_bb_validators.py (not test_blackboxuq) to avoid the
conftest HEAVY_TESTS skip list.
"""

import sys
from unittest.mock import MagicMock, patch

import pytest

# Pre-mock heavy deps so black_box.py can be imported in isolation
_mock_bert_score = MagicMock()
_mock_bert_score.BERTScorer = MagicMock(return_value=MagicMock())
sys.modules.setdefault("bert_score", _mock_bert_score)

_mock_st = MagicMock()
_mock_st.SentenceTransformer = MagicMock(return_value=MagicMock())
sys.modules.setdefault("sentence_transformers", _mock_st)

from uqlm.scorers.shortform.black_box import BlackBoxUQ  # noqa: E402


@pytest.fixture
def mock_llm():
    return MagicMock()


def _make_bbq(mock_llm, scorers, **kwargs):
    """Construct BlackBoxUQ with heavy model constructors mocked out."""
    with (
        patch("uqlm.scorers.shortform.black_box.BertScorer") as MockBert,
        patch("uqlm.scorers.shortform.black_box.CosineScorer") as MockCosine,
        patch("uqlm.scorers.shortform.black_box.SemanticEntropy") as MockSE,
    ):
        MockBert.return_value = MagicMock()
        MockCosine.return_value = MagicMock()
        MockSE.return_value = MagicMock()
        bbq = BlackBoxUQ(llm=mock_llm, scorers=scorers, **kwargs)
        return bbq


# ---------------------------------------------------------------------------
# _validate_scorers — line 220: scorers=None uses default
# ---------------------------------------------------------------------------


def test_validate_scorers_none_uses_default(mock_llm):
    """When scorers=None, falls back to default_black_box_names (line 220)."""
    bbq = _make_bbq(mock_llm, scorers=None)
    assert bbq.scorers is not None
    assert len(bbq.scorers) > 0


# ---------------------------------------------------------------------------
# _validate_scorers — lines 227-228: bert_score branch
# ---------------------------------------------------------------------------


def test_validate_scorers_bert_score(mock_llm):
    """bert_score scorer constructs BertScorer (lines 227-228)."""
    with (
        patch("uqlm.scorers.shortform.black_box.BertScorer") as MockBert,
    ):
        MockBert.return_value = MagicMock()
        bbq = BlackBoxUQ(llm=mock_llm, scorers=["bert_score"])
    assert "bert_score" in bbq.scorer_objects
    MockBert.assert_called_once()


# ---------------------------------------------------------------------------
# _validate_scorers — lines 230-231: cosine_sim branch
# ---------------------------------------------------------------------------


def test_validate_scorers_cosine_sim(mock_llm):
    """cosine_sim scorer constructs CosineScorer (lines 230-231)."""
    with (
        patch("uqlm.scorers.shortform.black_box.CosineScorer") as MockCosine,
    ):
        MockCosine.return_value = MagicMock()
        bbq = BlackBoxUQ(llm=mock_llm, scorers=["cosine_sim"])
    assert "cosine_sim" in bbq.scorer_objects
    MockCosine.assert_called_once()


# ---------------------------------------------------------------------------
# _validate_scorers — lines 234-235: entropy scorer branch
# ---------------------------------------------------------------------------


def test_validate_scorers_semantic_negentropy(mock_llm):
    """semantic_negentropy scorer adds to entropy_scorer_names (lines 234-235, 247)."""
    with (
        patch("uqlm.scorers.shortform.black_box.SemanticEntropy") as MockSE,
    ):
        MockSE.return_value = MagicMock()
        bbq = BlackBoxUQ(llm=mock_llm, scorers=["semantic_negentropy"])
    assert "semantic_negentropy" in bbq.entropy_scorer_names
    assert "semantic_negentropy" in bbq.scorer_objects
    MockSE.assert_called_once()


def test_validate_scorers_semantic_sets_confidence(mock_llm):
    """semantic_sets_confidence also appends to entropy_scorer_names (line 234-235)."""
    with (
        patch("uqlm.scorers.shortform.black_box.SemanticEntropy") as MockSE,
    ):
        MockSE.return_value = MagicMock()
        bbq = BlackBoxUQ(llm=mock_llm, scorers=["semantic_sets_confidence"])
    assert "semantic_sets_confidence" in bbq.entropy_scorer_names


# ---------------------------------------------------------------------------
# _validate_scorers — lines 237-239: bleurt deprecated + ValueError
# ---------------------------------------------------------------------------


def test_validate_scorers_raises_on_invalid(mock_llm):
    """Invalid scorer name raises ValueError (line 239); bleurt prints deprecation (237-238)."""
    with pytest.raises(ValueError):
        _make_bbq(mock_llm, scorers=["unknown_scorer_xyz"])


def test_validate_scorers_bleurt_prints_and_raises(mock_llm, capsys):
    """bleurt scorer triggers deprecation message then raises (lines 237-239)."""
    with pytest.raises(ValueError):
        _make_bbq(mock_llm, scorers=["bleurt"])
    captured = capsys.readouterr()
    assert "deprecated" in captured.out.lower()

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

"""Tests for uqlm/black_box/bert.py and uqlm/black_box/cosine.py"""

import sys
import time
from types import ModuleType
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

# ---------------------------------------------------------------------------
# Pre-mock heavy deps so bert.py / cosine.py can import without loading models
# ---------------------------------------------------------------------------

# bert_score mock
_mock_bert_score = MagicMock()
_mock_bert_score_instance = MagicMock()
_mock_bert_score.BERTScorer = MagicMock(return_value=_mock_bert_score_instance)
sys.modules.setdefault("bert_score", _mock_bert_score)

# sentence_transformers mock
_mock_st = MagicMock()
_mock_st_model = MagicMock()
_mock_st.SentenceTransformer = MagicMock(return_value=_mock_st_model)
sys.modules.setdefault("sentence_transformers", _mock_st)

# Now it's safe to import the modules under test
from uqlm.black_box.bert import BertScorer  # noqa: E402
from uqlm.black_box.cosine import CosineScorer  # noqa: E402


# ===========================================================================
# BertScorer
# ===========================================================================


@pytest.fixture
def bert_scorer():
    """BertScorer with a fresh mock BERTScorer instance."""
    with (
        patch("uqlm.black_box.bert.BERTScorer") as MockBERTScorer,
        patch("uqlm.black_box.bert.get_best_device", return_value=MagicMock()),
    ):
        mock_instance = MagicMock()
        MockBERTScorer.return_value = mock_instance
        scorer = BertScorer(device=None)
        scorer.bert_scorer = mock_instance
        yield scorer, mock_instance


# ---------------------------------------------------------------------------
# Constructor (lines 41-49)
# ---------------------------------------------------------------------------


def test_bert_scorer_init_device_none():
    """device=None → get_best_device() is called."""
    with (
        patch("uqlm.black_box.bert.BERTScorer") as MockBERTScorer,
        patch("uqlm.black_box.bert.get_best_device") as mock_device,
    ):
        mock_device.return_value = MagicMock()
        MockBERTScorer.return_value = MagicMock()
        scorer = BertScorer(device=None)
        mock_device.assert_called_once()
        assert scorer.bert_scorer is MockBERTScorer.return_value


def test_bert_scorer_init_string_device():
    """device='cpu' → torch.device('cpu') is called."""
    with (
        patch("uqlm.black_box.bert.BERTScorer") as MockBERTScorer,
        patch("uqlm.black_box.bert.torch") as mock_torch,
    ):
        mock_torch.device.return_value = MagicMock()
        MockBERTScorer.return_value = MagicMock()
        scorer = BertScorer(device="cpu")
        mock_torch.device.assert_called_once_with("cpu")


# ---------------------------------------------------------------------------
# evaluate (lines 71-80)
# ---------------------------------------------------------------------------


def test_bert_scorer_evaluate_returns_floats(bert_scorer):
    scorer, mock_instance = bert_scorer
    mock_instance.score.return_value = (
        torch.tensor([0.9]),
        torch.tensor([0.85]),
        torch.tensor([0.87]),
    )
    result = scorer.evaluate(responses=["The sky is blue."], sampled_responses=[["Blue sky.", "Sky is blue."]])
    assert len(result) == 1
    assert isinstance(result[0], float)


def test_bert_scorer_evaluate_with_progress_bar(bert_scorer):
    scorer, mock_instance = bert_scorer
    mock_instance.score.return_value = (
        torch.tensor([0.9, 0.8]),
        torch.tensor([0.85, 0.75]),
        torch.tensor([0.87, 0.77]),
    )
    progress_bar = MagicMock()
    progress_bar.add_task.return_value = MagicMock()

    result = scorer.evaluate(responses=["A", "B"], sampled_responses=[["A1", "A2"], ["B1"]], progress_bar=progress_bar)
    assert len(result) == 2
    progress_bar.add_task.assert_called_once()
    assert progress_bar.update.call_count == 2


# ---------------------------------------------------------------------------
# _compute_score (lines 84-87)
# ---------------------------------------------------------------------------


def test_bert_scorer_compute_score(bert_scorer):
    scorer, mock_instance = bert_scorer
    mock_instance.score.return_value = (
        torch.tensor([0.9, 0.9]),
        torch.tensor([0.8, 0.8]),
        torch.tensor([0.85, 0.85]),
    )
    score = scorer._compute_score("hello", ["hello there", "hi there"])
    assert isinstance(score, float)
    assert abs(score - 0.85) < 0.01


# ===========================================================================
# CosineScorer
# ===========================================================================


@pytest.fixture
def cosine_scorer():
    """CosineScorer with SentenceTransformer mocked."""
    with patch("uqlm.black_box.cosine.CosineScorer.__init__", lambda self, *a, **kw: None):
        scorer = CosineScorer.__new__(CosineScorer)
    mock_model = MagicMock()
    scorer.model = mock_model
    scorer.max_length = 2000
    scorer.transformer = "mock-transformer"
    scorer.pair_scores = []
    return scorer, mock_model


# ---------------------------------------------------------------------------
# Constructor (lines 41-45)
# ---------------------------------------------------------------------------


def test_cosine_scorer_init():
    """Constructor initializes transformer, model, and max_length."""
    mock_st_module = MagicMock()
    mock_model = MagicMock()
    mock_st_module.SentenceTransformer.return_value = mock_model

    with patch.dict(sys.modules, {"sentence_transformers": mock_st_module}):
        # Force re-import by removing cached module first
        uqlm_cosine = sys.modules.pop("uqlm.black_box.cosine", None)
        try:
            from uqlm.black_box.cosine import CosineScorer as _CS

            scorer = _CS(transformer="test-model", max_length=500)
            assert scorer.max_length == 500
            assert scorer.transformer == "test-model"
            mock_st_module.SentenceTransformer.assert_called_once_with("test-model", trust_remote_code=True)
        finally:
            # Restore the cached module so other tests use the already-imported version
            if uqlm_cosine is not None:
                sys.modules["uqlm.black_box.cosine"] = uqlm_cosine


# ---------------------------------------------------------------------------
# evaluate (lines 67-76)
# ---------------------------------------------------------------------------


def test_cosine_scorer_evaluate_returns_floats(cosine_scorer):
    scorer, mock_model = cosine_scorer
    mock_model.encode.side_effect = lambda texts: np.ones((len(texts), 4))
    result = scorer.evaluate(responses=["resp A"], sampled_responses=[["cand 1", "cand 2"]])
    assert len(result) == 1
    assert 0.0 <= result[0] <= 1.0


def test_cosine_scorer_evaluate_with_progress_bar(cosine_scorer):
    scorer, mock_model = cosine_scorer
    mock_model.encode.side_effect = lambda texts: np.ones((len(texts), 4))
    progress_bar = MagicMock()
    progress_bar.add_task.return_value = MagicMock()

    result = scorer.evaluate(responses=["A", "B"], sampled_responses=[["c1"], ["c2"]], progress_bar=progress_bar)
    assert len(result) == 2
    progress_bar.add_task.assert_called_once()
    assert progress_bar.update.call_count == 2


# ---------------------------------------------------------------------------
# _get_embeddings (lines 82-84)
# ---------------------------------------------------------------------------


def test_cosine_scorer_get_embeddings(cosine_scorer):
    scorer, mock_model = cosine_scorer
    mock_model.encode.side_effect = lambda texts: np.ones((len(texts), 8))
    emb1, emb2 = scorer._get_embeddings(["t1", "t2"], ["t3", "t4"])
    assert emb1.shape == (2, 8)
    assert emb2.shape == (2, 8)


# ---------------------------------------------------------------------------
# _compute_score (lines 90-99)
# ---------------------------------------------------------------------------


def test_cosine_scorer_compute_score(cosine_scorer):
    scorer, mock_model = cosine_scorer
    vec = np.array([1.0, 0.0, 0.0, 0.0])
    mock_model.encode.side_effect = lambda texts: np.array([vec for _ in texts])
    scores = scorer._compute_score("response", ["cand1", "cand2"])
    assert len(scores) == 2
    # identical unit vectors → cosine=1 → normalized = 1.0
    assert abs(scores[0] - 1.0) < 1e-5


def test_cosine_scorer_compute_score_truncates(cosine_scorer):
    scorer, mock_model = cosine_scorer
    scorer.max_length = 5
    mock_model.encode.side_effect = lambda texts: np.ones((len(texts), 4))
    scores = scorer._compute_score("hello world", ["hi there world"])
    assert len(scores) == 1

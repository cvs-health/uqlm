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
import numpy as np
from uqlm.white_box.baseclass.logprobs_scorer import LogprobsScorer
from uqlm.white_box.single_logprobs import SingleLogprobsScorer, SINGLE_LOGPROBS_SCORER_NAMES


@pytest.fixture
def mock_single_response_logprobs():
    """Fixture to provide mock single response logprobs."""
    return [{"logprob": -0.1, "top_logprobs": [{"logprob": -0.1}, {"logprob": -1.0}, {"logprob": -2.0}]}, {"logprob": -0.2, "top_logprobs": [{"logprob": -0.2}, {"logprob": -0.5}, {"logprob": -1.5}]}]


@pytest.fixture
def mock_logprobs_results(mock_single_response_logprobs):
    """Fixture to provide mock logprobs results."""
    return [mock_single_response_logprobs, mock_single_response_logprobs]


@pytest.fixture
def scorer():
    """Fixture to create a LogprobsScorer instance."""
    return LogprobsScorer()


def test_norm_prob(mock_single_response_logprobs, scorer):
    """Test the _norm_prob method."""
    result = scorer._norm_prob(mock_single_response_logprobs)
    assert isinstance(result, float)
    assert result > 0.0 and result <= 1.0


def test_seq_prob(mock_single_response_logprobs, scorer):
    """Test the _seq_prob method."""
    result = scorer._seq_prob(mock_single_response_logprobs)
    assert isinstance(result, float)
    assert result > 0.0 and result <= 1.0


def test_entropy_from_logprobs(scorer):
    """Test the _entropy_from_logprobs method."""
    logprobs_list = np.array([-0.1, -0.2, -0.3])
    result = scorer._entropy_from_logprobs(logprobs_list)
    assert isinstance(result, float)
    assert result >= 0.0


def test_entropy_from_probs(scorer):
    """Test the _entropy_from_probs method."""
    probs_list = np.array([0.5, 0.3, 0.2])
    result = scorer._entropy_from_probs(probs_list)
    assert isinstance(result, float)
    assert result >= 0.0


def test_entropy_from_probs_with_texts(scorer):
    """Test the _entropy_from_probs method with texts."""
    probs_list = np.array([0.5, 0.3, 0.2])
    texts = ["a", "b", "a"]
    result = scorer._entropy_from_probs(probs_list, texts)
    assert isinstance(result, float)
    assert result >= 0.0


def test_extract_probs(mock_single_response_logprobs, scorer):
    """Test the extract_probs method."""
    result = scorer.extract_probs(mock_single_response_logprobs)
    assert isinstance(result, np.ndarray)
    assert result.shape == (len(mock_single_response_logprobs),)
    assert np.all(result > 0.0) and np.all(result <= 1.0)


def test_extract_logprobs(mock_single_response_logprobs, scorer):
    """Test the extract_logprobs method."""
    result = scorer.extract_logprobs(mock_single_response_logprobs)
    assert isinstance(result, np.ndarray)
    assert result.shape == (len(mock_single_response_logprobs),)
    assert np.all(result < 0.0)  # Logprobs should be negative


def test_extract_top_logprobs(mock_single_response_logprobs, scorer):
    """Test the extract_top_logprobs method."""
    result = scorer.extract_top_logprobs(mock_single_response_logprobs)
    assert isinstance(result, list)
    assert len(result) == len(mock_single_response_logprobs)
    for top_logprobs in result:
        assert isinstance(top_logprobs, np.ndarray)
        assert top_logprobs.shape[0] > 0


def test_single_logprobs_scorer_default_no_key_error(mock_logprobs_results):
    """SingleLogprobsScorer default evaluate must not raise KeyError (normalized_probability removed)."""
    assert "normalized_probability" not in SINGLE_LOGPROBS_SCORER_NAMES
    scorer = SingleLogprobsScorer()
    result = scorer.evaluate(mock_logprobs_results)
    assert "min_probability" in result
    assert "sequence_probability" in result
    assert "normalized_probability" not in result


def test_compute_single_generation_scores(mock_logprobs_results, scorer):
    """Test the _compute_single_generation_scores method."""

    def mock_score_fn(single_response_logprobs):
        return 0.9

    result = scorer._compute_single_generation_scores(mock_logprobs_results, mock_score_fn)
    assert isinstance(result, list)
    assert len(result) == len(mock_logprobs_results)
    assert all(score == 0.9 for score in result)

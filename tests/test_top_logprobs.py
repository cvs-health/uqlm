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
from uqlm.white_box.top_logprobs import TopLogprobsScorer, TOP_LOGPROBS_SCORER_NAMES


@pytest.fixture
def mock_logprobs_results():
    """Fixture to provide mock logprobs results."""
    return [[{"token": "a", "logprobs": [-0.1, -1.0, -2.0]}, {"token": "b", "logprobs": [-0.2, -0.5, -1.5]}], [{"token": "c", "logprobs": [-0.3, -0.7, -1.2]}, {"token": "d", "logprobs": [-0.4, -0.8, -1.0]}]]


@pytest.fixture
def scorer():
    """Fixture to create a TopLogprobsScorer instance."""
    return TopLogprobsScorer(top_k_logprobs=3)


def test_evaluate(mock_logprobs_results, scorer, monkeypatch):
    """Test the evaluate method of TopLogprobsScorer."""
    # Mock the extract_top_logprobs method to return only the logprobs
    monkeypatch.setattr(scorer, "extract_top_logprobs", lambda logprobs: [logprob["logprobs"] for logprob in logprobs])

    # Mock the _entropy_from_logprobs method to return a fixed entropy value
    monkeypatch.setattr(scorer, "_entropy_from_logprobs", lambda logprobs: 0.5)

    result = scorer.evaluate(mock_logprobs_results)

    # Verify the result contains all scorer names
    assert set(result.keys()) == set(TOP_LOGPROBS_SCORER_NAMES)

    # Verify the length of the results matches the number of sequences
    for key in result:
        assert len(result[key]) == len(mock_logprobs_results)


def test_mean_token_negentropy(mock_logprobs_results, scorer, monkeypatch):
    """Test the _mean_token_negentropy method."""
    # Mock the extract_top_logprobs method
    monkeypatch.setattr(scorer, "extract_top_logprobs", lambda logprobs: [logprob["logprobs"] for logprob in logprobs])

    # Mock the _entropy_from_logprobs method
    monkeypatch.setattr(scorer, "_entropy_from_logprobs", lambda logprobs: 0.5)
    # scorer.top_k_logprobs = 3
    result = scorer._mean_token_negentropy(mock_logprobs_results[0])
    assert isinstance(result, float)
    assert result >= 0.0 and result <= 1.0


def test_min_token_negentropy(mock_logprobs_results, scorer, monkeypatch):
    """Test the _min_token_negentropy method."""
    # Mock the extract_top_logprobs method
    monkeypatch.setattr(scorer, "extract_top_logprobs", lambda logprobs: [logprob["logprobs"] for logprob in logprobs])

    # Mock the _entropy_from_logprobs method
    monkeypatch.setattr(scorer, "_entropy_from_logprobs", lambda logprobs: 0.5)
    # scorer.top_k_logprobs = 3
    result = scorer._min_token_negentropy(mock_logprobs_results[0])
    assert isinstance(result, float)
    assert result >= 0.0 and result <= 1.0


def test_probability_margin(mock_logprobs_results, scorer, monkeypatch):
    """Test the _probability_margin method."""
    # Mock the extract_top_logprobs method
    monkeypatch.setattr(scorer, "extract_top_logprobs", lambda logprobs: [logprob["logprobs"] for logprob in logprobs])

    result = scorer._probability_margin(mock_logprobs_results[0])
    assert isinstance(result, float)
    assert result >= 0.0 and result <= 1.0


def test_probability_margin_with_empty_logprobs(scorer):
    """Test the _probability_margin method with empty logprobs."""
    result = scorer._probability_margin([])
    assert np.isnan(result)

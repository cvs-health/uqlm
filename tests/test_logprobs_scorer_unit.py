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

"""Lightweight unit tests for SampledLogprobsScorer (no model loading).

test_sampled_logprobs.py is in the HEAVY_TESTS skip list, so it is never run.
This file covers the method bodies that are unreachable from the skipped file,
using full mocking so no transformer models are loaded.
"""

from unittest.mock import MagicMock, patch
import pytest
from uqlm.white_box.sampled_logprobs import SampledLogprobsScorer


# ---------------------------------------------------------------------------
# Shared test data
# ---------------------------------------------------------------------------

RESPONSES = ["response1", "response2"]
SAMPLED = [["s1", "s2"], ["s3", "s4"]]
LOGPROBS = [[{"token": "a", "logprob": -0.1}], [{"token": "b", "logprob": -0.2}]]
SAMPLED_LOGPROBS = [[[{"token": "c", "logprob": -0.15}]], [[{"token": "d", "logprob": -0.25}]]]
PROMPTS = ["prompt1", "prompt2"]


def make_scorer(scorers=None):
    mock_llm = MagicMock()
    if scorers:
        return SampledLogprobsScorer(llm=mock_llm, scorers=scorers)
    return SampledLogprobsScorer(llm=mock_llm)


# ---------------------------------------------------------------------------
# evaluate() branches — lines 81, 83, 85
# ---------------------------------------------------------------------------


def test_evaluate_consistency_and_confidence_branch():
    """evaluate() with 'consistency_and_confidence' scorer covers line 81."""
    scorer = make_scorer(scorers=["consistency_and_confidence"])
    with patch.object(scorer, "compute_consistency_confidence", return_value=[0.5, 0.6]) as mock_cc:
        result = scorer.evaluate(
            responses=RESPONSES,
            sampled_responses=SAMPLED,
            logprobs_results=LOGPROBS,
            sampled_logprobs_results=SAMPLED_LOGPROBS,
        )
    mock_cc.assert_called_once()
    assert result == {"consistency_and_confidence": [0.5, 0.6]}


def test_evaluate_semantic_negentropy_branch():
    """evaluate() with 'semantic_negentropy' scorer covers line 83."""
    scorer = make_scorer(scorers=["semantic_negentropy"])
    with patch.object(scorer, "compute_semantic_negentropy", return_value=[0.7, 0.8]) as mock_sn:
        result = scorer.evaluate(
            responses=RESPONSES,
            sampled_responses=SAMPLED,
            logprobs_results=LOGPROBS,
            sampled_logprobs_results=SAMPLED_LOGPROBS,
            prompts=PROMPTS,
        )
    mock_sn.assert_called_once()
    assert result == {"semantic_negentropy": [0.7, 0.8]}


def test_evaluate_semantic_density_branch():
    """evaluate() with 'semantic_density' scorer covers line 85."""
    scorer = make_scorer(scorers=["semantic_density"])
    with patch.object(scorer, "compute_semantic_density", return_value=[0.9, 1.0]) as mock_sd:
        result = scorer.evaluate(
            responses=RESPONSES,
            sampled_responses=SAMPLED,
            logprobs_results=LOGPROBS,
            sampled_logprobs_results=SAMPLED_LOGPROBS,
        )
    mock_sd.assert_called_once()
    assert result == {"semantic_density": [0.9, 1.0]}


# ---------------------------------------------------------------------------
# compute_consistency_confidence — lines 89-93
# ---------------------------------------------------------------------------


def test_compute_consistency_confidence():
    """compute_consistency_confidence mocks CosineScorer and _compute_single_generation_scores (lines 89-93)."""
    scorer = make_scorer()

    mock_cosine = MagicMock()
    mock_cosine.evaluate.return_value = [0.8, 0.9]

    with patch("uqlm.white_box.sampled_logprobs.CosineScorer", return_value=mock_cosine), \
         patch.object(scorer, "_compute_single_generation_scores", return_value=[0.5, 0.4]):
        result = scorer.compute_consistency_confidence(
            responses=RESPONSES,
            sampled_responses=SAMPLED,
            logprobs_results=LOGPROBS,
        )

    assert len(result) == 2
    # 0.8 * 0.5 = 0.4, 0.9 * 0.4 = 0.36
    assert abs(result[0] - 0.4) < 1e-9
    assert abs(result[1] - 0.36) < 1e-9


# ---------------------------------------------------------------------------
# compute_semantic_negentropy — lines 106-110
# ---------------------------------------------------------------------------


def test_compute_semantic_negentropy():
    """compute_semantic_negentropy patches SemanticEntropy to avoid loading NLI (lines 106-110)."""
    scorer = make_scorer()

    mock_se = MagicMock()
    mock_se_result = MagicMock()
    mock_se_result.to_dict.return_value = {"data": {"tokenprob_confidence_scores": [0.75, 0.85]}}
    mock_se.score.return_value = mock_se_result

    with patch("uqlm.white_box.sampled_logprobs.SemanticEntropy", return_value=mock_se):
        result = scorer.compute_semantic_negentropy(
            responses=RESPONSES,
            prompts=PROMPTS,
            sampled_responses=SAMPLED,
            logprobs_results=LOGPROBS,
            sampled_logprobs_results=SAMPLED_LOGPROBS,
        )

    assert result == [0.75, 0.85]
    mock_se.score.assert_called_once()


# ---------------------------------------------------------------------------
# compute_semantic_density — lines 113-122 (two branches)
# ---------------------------------------------------------------------------


def test_compute_semantic_density_without_negentropy_scorer():
    """Without prior semantic_negentropy_scorer, covers the else-branch (lines 117-120)."""
    scorer = make_scorer()
    assert scorer.semantic_negentropy_scorer is None

    mock_sd = MagicMock()
    mock_sd.nli = MagicMock()
    mock_sd.nli.probabilities = {}
    mock_sd_result = MagicMock()
    mock_sd_result.to_dict.return_value = {"data": {"semantic_density_values": [0.6, 0.7]}}
    mock_sd.score.return_value = mock_sd_result

    with patch("uqlm.white_box.sampled_logprobs.SemanticDensity", return_value=mock_sd):
        result = scorer.compute_semantic_density(
            responses=RESPONSES,
            sampled_responses=SAMPLED,
            logprobs_results=LOGPROBS,
            sampled_logprobs_results=SAMPLED_LOGPROBS,
            prompts=PROMPTS,
        )

    assert result == [0.6, 0.7]
    # Else-branch sets progress_bar and show_progress_bars=True
    assert mock_sd.progress_bar is None  # progress_bar arg was None (default)
    mock_sd.score.assert_called_once()


def test_compute_semantic_density_with_existing_negentropy_scorer():
    """With prior semantic_negentropy_scorer, covers the if-branch (lines 114-116)."""
    scorer = make_scorer()

    # Simulate a previously run semantic_negentropy_scorer
    mock_clusterer = MagicMock()
    mock_clusterer.nli.probabilities = {"key1": [0.1, 0.2]}
    mock_negentropy = MagicMock()
    mock_negentropy.clusterer = mock_clusterer
    scorer.semantic_negentropy_scorer = mock_negentropy

    mock_sd = MagicMock()
    mock_sd.nli = MagicMock()
    mock_sd_result = MagicMock()
    mock_sd_result.to_dict.return_value = {"data": {"semantic_density_values": [0.55, 0.65]}}
    mock_sd.score.return_value = mock_sd_result

    with patch("uqlm.white_box.sampled_logprobs.SemanticDensity", return_value=mock_sd):
        result = scorer.compute_semantic_density(
            responses=RESPONSES,
            sampled_responses=SAMPLED,
            logprobs_results=LOGPROBS,
            sampled_logprobs_results=SAMPLED_LOGPROBS,
        )

    assert result == [0.55, 0.65]
    # If-branch assigns probabilities from negentropy_scorer's NLI cache
    assert mock_sd.nli.probabilities == {"key1": [0.1, 0.2]}

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

"""Targeted tests for uncovered branches in UncertaintyQuantifier base class and TopLogprobsScorer."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from uqlm.scorers.baseclass.uncertainty import UncertaintyQuantifier
from uqlm.white_box.top_logprobs import TopLogprobsScorer


# ===========================================================================
# UncertaintyQuantifier._display_generation_header — "claim_qa" branch
# (lines 184-185)
# ===========================================================================


class _ConcreteUQ(UncertaintyQuantifier):
    """Minimal concrete implementation to test the base class methods."""

    async def generate_and_score(self, *args, **kwargs):
        pass  # pragma: no cover

    async def score(self, *args, **kwargs):
        pass  # pragma: no cover


def test_display_generation_header_claim_qa():
    """generation_type='claim_qa' hits lines 184-185."""
    uq = _ConcreteUQ(llm=MagicMock())
    mock_pb = MagicMock()
    uq.progress_bar = mock_pb

    uq._display_generation_header(show_progress_bars=True, generation_type="claim_qa")

    mock_pb.add_task.assert_called_once()
    call_args = mock_pb.add_task.call_args[0][0]
    assert "Claim-QA" in call_args or "claim" in call_args.lower()


def test_display_generation_header_no_progress_bar():
    """show_progress_bars=False → no add_task call."""
    uq = _ConcreteUQ(llm=MagicMock())
    uq.progress_bar = None
    # Should not raise
    uq._display_generation_header(show_progress_bars=False, generation_type="claim_qa")


# ===========================================================================
# TopLogprobsScorer._probability_margin — IndexError path
# (lines 70-72)
# ===========================================================================


@pytest.fixture
def top_scorer():
    return TopLogprobsScorer()


def test_probability_margin_index_error(top_scorer):
    """Single-element probs list triggers IndexError → returns np.nan (lines 70-72)."""
    # extract_top_logprobs returns a list with one entry that has only 1 element
    # so probs[1] raises IndexError
    top_scorer.extract_top_logprobs = lambda logprobs: [[np.log(0.9)]]  # only 1 prob
    result = top_scorer._probability_margin([{"token": "hi", "logprob": -0.1, "top_logprobs": [{"token": "hi", "logprob": -0.1}]}])
    assert np.isnan(result)


def test_probability_margin_index_error_empty_inner(top_scorer, capsys):
    """Empty inner probs list also triggers IndexError → prints message and returns nan."""
    top_scorer.extract_top_logprobs = lambda logprobs: [[]]  # empty inner → probs[0] raises
    result = top_scorer._probability_margin([{"anything": True}])
    assert np.isnan(result)
    captured = capsys.readouterr()
    assert "top_logprobs were not available" in captured.out

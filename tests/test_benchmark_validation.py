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

"""Tests for benchmark validation."""

import pytest
from typing import List, Optional

from uqlm.benchmarks.validation import validate_benchmark_scorers, get_valid_scorers_for_category, BenchmarkValidationError, LONGFORM_SCORERS, BLACK_BOX_SCORERS, WHITE_BOX_SCORERS
from uqlm.benchmarks.datasets import BaseBenchmark


class MockValidBenchmark(BaseBenchmark):
    """A valid mock benchmark for testing."""

    def __init__(self):
        super().__init__()

    def get_prompts(self) -> List[str]:
        return ["test prompt 1", "test prompt 2"]

    @property
    def category(self) -> str:
        return "longform"

    @property
    def dataset_name(self) -> str:
        return "mock/dataset"

    @property
    def version(self) -> Optional[str]:
        return "1.0.0"


def test_validate_benchmark_scorers_valid_longform():
    """Test validation passes for valid longform scorers."""
    benchmark = MockValidBenchmark()
    scorers = ["response_claim_entail", "response_sent_entail"]

    # Should not raise
    validate_benchmark_scorers(benchmark, scorers, "longform")


def test_validate_benchmark_scorers_invalid_longform():
    """Test validation fails when using black-box scorers with longform benchmark."""
    benchmark = MockValidBenchmark()
    scorers = ["semantic_negentropy", "exact_match"]  # Black-box scorers

    with pytest.raises(BenchmarkValidationError) as exc_info:
        validate_benchmark_scorers(benchmark, scorers, "longform")

    error_msg = str(exc_info.value)
    assert "INCOMPATIBLE SCORERS" in error_msg
    assert "semantic_negentropy" in error_msg
    assert "exact_match" in error_msg
    assert "longform" in error_msg


def test_validate_benchmark_scorers_mixed_invalid():
    """Test validation fails when mixing valid and invalid scorers."""
    benchmark = MockValidBenchmark()
    scorers = [
        "response_claim_entail",  # Valid for longform
        "semantic_negentropy",  # Invalid for longform
    ]

    with pytest.raises(BenchmarkValidationError) as exc_info:
        validate_benchmark_scorers(benchmark, scorers, "longform")

    error_msg = str(exc_info.value)
    assert "semantic_negentropy" in error_msg
    # Should only list the invalid scorer, not the valid one
    assert "response_claim_entail" not in error_msg or "❌" not in error_msg


def test_validate_benchmark_scorers_empty_list():
    """Test validation fails for empty scorer list."""
    benchmark = MockValidBenchmark()

    with pytest.raises(BenchmarkValidationError) as exc_info:
        validate_benchmark_scorers(benchmark, [], "longform")

    error_msg = str(exc_info.value)
    assert "No scorers provided" in error_msg


def test_validate_benchmark_scorers_unknown_category():
    """Test validation fails for unknown category."""
    benchmark = MockValidBenchmark()
    scorers = ["response_claim_entail"]

    with pytest.raises(BenchmarkValidationError) as exc_info:
        validate_benchmark_scorers(benchmark, scorers, "unknown_category")

    error_msg = str(exc_info.value)
    assert "Unknown benchmark category" in error_msg
    assert "unknown_category" in error_msg


def test_get_valid_scorers_for_category_longform():
    """Test getting valid scorers for longform category."""
    scorers = get_valid_scorers_for_category("longform")

    assert isinstance(scorers, set)
    assert len(scorers) == len(LONGFORM_SCORERS)
    assert "response_claim_entail" in scorers
    assert "response_sent_entail" in scorers
    assert "matched_claim_entail" in scorers


def test_get_valid_scorers_for_category_short_form():
    """Test getting valid scorers for short_form category."""
    scorers = get_valid_scorers_for_category("short_form")

    assert isinstance(scorers, set)
    # Should include both black-box and white-box scorers
    assert "semantic_negentropy" in scorers
    assert "exact_match" in scorers
    assert "normalized_probability" in scorers


def test_get_valid_scorers_for_category_unknown():
    """Test getting valid scorers for unknown category raises error."""
    with pytest.raises(ValueError) as exc_info:
        get_valid_scorers_for_category("unknown")

    error_msg = str(exc_info.value)
    assert "Unknown benchmark category" in error_msg


def test_scorer_constants_no_overlap():
    """Test that longform, black-box, and white-box scorers don't overlap."""
    # Longform should be distinct from black-box and white-box
    assert LONGFORM_SCORERS.isdisjoint(BLACK_BOX_SCORERS)
    assert LONGFORM_SCORERS.isdisjoint(WHITE_BOX_SCORERS)
    # Black-box and white-box should also be distinct
    assert BLACK_BOX_SCORERS.isdisjoint(WHITE_BOX_SCORERS)


def test_all_longform_scorers_defined():
    """Test that all expected longform scorers are defined."""
    expected_longform = {"response_sent_entail", "response_sent_noncontradict", "response_sent_contrast_entail", "response_claim_entail", "response_claim_noncontradict", "response_claim_contrast_entail", "matched_claim_entail", "matched_claim_noncontradict", "matched_claim_contrast_entail"}

    assert LONGFORM_SCORERS == expected_longform

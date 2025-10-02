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

"""Validation utilities for benchmark configurations."""

from typing import List, Set, Dict

from uqlm.benchmarks.datasets.base import BaseBenchmark


# Define available scorers by category
LONGFORM_SCORERS = {
    # Sentence-level
    "response_sent_entail",
    "response_sent_noncontradict",
    "response_sent_contrast_entail",
    # Claim-level
    "response_claim_entail",
    "response_claim_noncontradict",
    "response_claim_contrast_entail",
    # Matched-claim
    "matched_claim_entail",
    "matched_claim_noncontradict",
    "matched_claim_contrast_entail",
}

BLACK_BOX_SCORERS = {"semantic_negentropy", "noncontradiction", "exact_match", "cosine_sim", "bert_score"}

WHITE_BOX_SCORERS = {"normalized_probability", "min_probability"}

# Map categories to their valid scorers
CATEGORY_SCORERS: Dict[str, Set[str]] = {"longform": LONGFORM_SCORERS, "short_form": BLACK_BOX_SCORERS | WHITE_BOX_SCORERS}

# Human-readable descriptions for error messages
SCORER_DESCRIPTIONS = {
    "longform": ("LongForm scorers include:\n  - Sentence-level: response_sent_entail, response_sent_noncontradict, response_sent_contrast_entail\n  - Claim-level: response_claim_entail, response_claim_noncontradict, response_claim_contrast_entail\n  - Matched-claim: matched_claim_entail, matched_claim_noncontradict, matched_claim_contrast_entail"),
    "short_form": ("Short-form scorers include:\n  - Black-box: semantic_negentropy, noncontradiction, exact_match, cosine_sim, bert_score\n  - White-box: normalized_probability, min_probability"),
}


class BenchmarkValidationError(Exception):
    """Raised when benchmark configuration is invalid."""

    pass


def validate_benchmark_scorers(benchmark: BaseBenchmark, scorers: List[str], benchmark_category: str) -> None:
    """
    Validate that the requested scorers are compatible with the benchmark category.

    Parameters:
    -----------
    benchmark : BaseBenchmark
        The benchmark instance being run
    scorers : List[str]
        List of scorer names requested by the user
    benchmark_category : str
        The benchmark category (from benchmark.category)

    Raises:
    -------
    BenchmarkValidationError
        If scorers are incompatible with the benchmark category
    """
    if not scorers:
        raise BenchmarkValidationError(f"No scorers provided. Please specify at least one scorer for the '{benchmark_category}' benchmark category.")

    # Get valid scorers for this category
    valid_scorers = CATEGORY_SCORERS.get(benchmark_category)
    if valid_scorers is None:
        raise BenchmarkValidationError(f"Unknown benchmark category: '{benchmark_category}'. Supported categories: {list(CATEGORY_SCORERS.keys())}")

    # Check each scorer
    invalid_scorers = []
    for scorer in scorers:
        if scorer not in valid_scorers:
            invalid_scorers.append(scorer)

    if invalid_scorers:
        benchmark_name = benchmark.name
        scorer_desc = SCORER_DESCRIPTIONS.get(benchmark_category, "")

        raise BenchmarkValidationError(f"\n{'=' * 70}\nINCOMPATIBLE SCORERS for {benchmark_name}\n{'=' * 70}\n\nThe benchmark '{benchmark_name}' supports category '{benchmark_category}',\nbut the following scorers are not compatible:\n\n  ❌ {', '.join(invalid_scorers)}\n\n{scorer_desc}\n\nPlease select scorers that match the benchmark category.\n{'=' * 70}")


def get_valid_scorers_for_category(category: str) -> Set[str]:
    """
    Get the set of valid scorers for a given benchmark category.

    Parameters:
    -----------
    category : str
        Benchmark category (e.g., "longform", "short_form")

    Returns:
    --------
    Set[str]
        Set of valid scorer names for this category

    Raises:
    -------
    ValueError
        If category is not recognized
    """
    if category not in CATEGORY_SCORERS:
        raise ValueError(f"Unknown benchmark category: '{category}'. Supported categories: {list(CATEGORY_SCORERS.keys())}")
    return CATEGORY_SCORERS[category].copy()

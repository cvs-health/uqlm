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

"""Pydantic models for benchmark data structures."""

from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime
import hashlib
import json


class BenchmarkConfig(BaseModel):
    """Configuration for a benchmark run."""

    benchmark_name: str
    benchmark_category: str  # e.g., "longform", "short_form"
    llm_names: List[str]  # Extracted from LLM objects for storage
    scorer_names: List[str]
    dataset_name: str
    dataset_version: Optional[str] = None
    sampling_temperature: float = 0.7
    num_responses: int = 5
    additional_params: Dict[str, Any] = Field(default_factory=dict)

    def compute_hash(self) -> str:
        """
        Compute deterministic hash for caching.

        Returns a hash based on all configuration parameters to identify
        duplicate runs.
        """
        # Create a sorted, deterministic representation
        hash_dict = {"benchmark_name": self.benchmark_name, "benchmark_category": self.benchmark_category, "llm_names": sorted(self.llm_names), "scorer_names": sorted(self.scorer_names), "dataset_name": self.dataset_name, "dataset_version": self.dataset_version, "sampling_temperature": self.sampling_temperature, "num_responses": self.num_responses, "additional_params": self.additional_params}

        # Create deterministic JSON string
        hash_string = json.dumps(hash_dict, sort_keys=True)

        # Return SHA256 hash
        return hashlib.sha256(hash_string.encode()).hexdigest()


class RunMetadata(BaseModel):
    """Metadata about a benchmark run."""

    run_id: str
    config_hash: str
    created_at: datetime
    completed_at: Optional[datetime] = None
    status: str = "pending"  # pending, running, completed, failed
    config: BenchmarkConfig
    error_message: Optional[str] = None


class PromptResult(BaseModel):
    """Results for a single prompt."""

    prompt_id: int
    prompt: str
    llm_name: str
    original_response: str
    sampled_responses: List[str]
    scores: Dict[str, float] = Field(default_factory=dict)  # scorer_name -> score
    metadata: Dict[str, Any] = Field(default_factory=dict)


class BenchmarkRun(BaseModel):
    """Complete benchmark run with all results."""

    metadata: RunMetadata
    results: List[PromptResult] = Field(default_factory=list)

    def to_dict(self) -> dict:
        """Serialize for storage."""
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: dict) -> "BenchmarkRun":
        """Deserialize from storage."""
        return cls.model_validate(data)

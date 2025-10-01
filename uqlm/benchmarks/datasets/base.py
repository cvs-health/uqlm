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

"""Base class for benchmark dataset implementations."""

from abc import ABC, abstractmethod
from typing import List, Optional


class BaseBenchmark(ABC):
    """
    Abstract base class for benchmark dataset implementations.

    Subclasses should implement:
    1. get_prompts() - Return the list of prompts for evaluation
    2. get_supported_category() - Declare which benchmark category this supports

    The BenchmarkRunner handles UQ instantiation and scoring based
    on the benchmark category. The benchmark implementation only needs
    to provide the prompts and declare its compatible category.

    Example:
        class MyBenchmark(BaseBenchmark):
            def __init__(self, judge_llm, max_samples=None):
                self.judge_llm = judge_llm
                self.max_samples = max_samples

            def get_prompts(self) -> List[str]:
                # Load and return prompts from dataset
                ds = load_dataset("my_dataset", split="test")
                if self.max_samples:
                    ds = ds.select(range(self.max_samples))
                return ds["prompt"]

            @classmethod
            def get_supported_category(cls) -> str:
                # Declare this benchmark works with longform scorers
                return "longform"
    """

    @abstractmethod
    def get_prompts(self) -> List[str]:
        """
        Get the list of prompts for this benchmark.

        Returns:
        --------
        List[str]
            List of prompts to evaluate
        """
        pass

    @classmethod
    @abstractmethod
    def get_supported_category(cls) -> str:
        """
        Get the benchmark category this dataset supports.

        This declares which type of UQ scorer is compatible with this benchmark.
        For example, "longform" benchmarks work with LongFormUQ scorers,
        while "short_form" benchmarks would work with BlackBoxUQ scorers.

        Returns:
        --------
        str
            The supported benchmark category (e.g., "longform", "short_form")
        """
        pass

    def get_dataset_name(self) -> str:
        """
        Get the HuggingFace dataset name for this benchmark.

        Returns:
        --------
        str
            The dataset name/path (e.g., "dskar/FActScore")
        """
        raise NotImplementedError(f"{self.__class__.__name__} must implement get_dataset_name()")

    def get_dataset_version(self) -> Optional[str]:
        """
        Get the dataset version if available.

        For HuggingFace datasets, this is typically extracted from dataset info
        after loading. Override this method to provide version information.

        Returns:
        --------
        Optional[str]
            Dataset version string, or None if not available
        """
        return None

    def get_name(self) -> str:
        """
        Get the name of this benchmark.

        Returns:
        --------
        str
            Benchmark name
        """
        return self.__class__.__name__

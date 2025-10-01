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

from datasets import load_dataset_builder


class BaseBenchmark(ABC):
    """
    Abstract base class for benchmark dataset implementations.

    Subclasses must implement the following abstract properties:
    1. dataset_name - The HuggingFace dataset identifier
    2. category - The benchmark category (e.g., "longform", "short_form")
    
    And the abstract method:
    3. get_prompts() - Return the list of prompts for evaluation

    Additional properties available:
    - name: Benchmark name (defaults to class name, can override)
    - version: Dataset version (automatically extracted from HF metadata)

    All properties are available immediately upon instantiation (before dataset loading).
    The version property uses HuggingFace's load_dataset_builder() to fetch metadata
    without downloading the full dataset.

    Dataset Loading Convention:
    ---------------------------
    When loading HuggingFace datasets, subclasses should follow this pattern:
    - Store the raw HF dataset object in self._dataset_raw (for metadata access)
    - Store the processed data (e.g., pandas DataFrame) in self._dataset
    
    Example:
        class MyBenchmark(BaseBenchmark):
            def __init__(self, judge_llm, max_samples=None):
                super().__init__()
                self.judge_llm = judge_llm
                self.max_samples = max_samples

            @property
            def dataset_name(self) -> str:
                return "my_dataset"
            
            @property
            def category(self) -> str:
                return "longform"

            def _load_dataset(self):
                if self._dataset is None:
                    ds = load_dataset(self.dataset_name, split="test")
                    self._dataset_raw = ds
                    df = ds.to_pandas()
                    if self.max_samples:
                        df = df.head(self.max_samples)
                    self._dataset = df

            def get_prompts(self) -> List[str]:
                if self._dataset is None:
                    self._load_dataset()
                return self._dataset["prompt"].tolist()
    """

    def __init__(self):
        """
        Initialize the benchmark with standard dataset attributes.
        
        Subclasses should call super().__init__() in their constructors.
        """
        self._dataset_raw = None  # Raw HuggingFace dataset object
        self._dataset = None      # Processed dataset (e.g., pandas DataFrame)

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

    @property
    @abstractmethod
    def dataset_name(self) -> str:
        """
        The HuggingFace dataset name for this benchmark.

        This property should be available immediately upon instantiation.

        Returns:
        --------
        str
            The dataset name/path (e.g., "dskar/FActScore")
        """
        pass

    @property
    @abstractmethod
    def category(self) -> str:
        """
        The benchmark category this dataset supports.

        This declares which type of UQ scorer is compatible with this benchmark.
        For example, "longform" benchmarks work with LongFormUQ scorers,
        while "short_form" benchmarks would work with BlackBoxUQ scorers.

        This property should be available immediately upon instantiation.

        Returns:
        --------
        str
            The supported benchmark category (e.g., "longform", "short_form")
        """
        pass

    @property
    def version(self) -> Optional[str]:
        """
        The dataset version if available.

        This property attempts to get version information in the following order:
        1. From self._dataset_raw if already loaded (the raw HF dataset object)
        2. By loading the dataset builder (metadata only, no data download)
        3. Returns None if version is unavailable
        
        This property is available immediately upon instantiation (uses lazy metadata loading).
        Override this property if you need custom version logic.

        Returns:
        --------
        Optional[str]
            Dataset version string, or None if not available
        """
        # First check if dataset is already loaded
        if self._dataset_raw is not None:
            try:
                if hasattr(self._dataset_raw, "info") and hasattr(self._dataset_raw.info, "version"):
                    return str(self._dataset_raw.info.version)
            except Exception:
                pass
        
        # Try loading just the metadata (no data download)
        try:
            builder = load_dataset_builder(self.dataset_name)
            
            if hasattr(builder, "info") and hasattr(builder.info, "version"):
                return str(builder.info.version)
        except Exception:
            # If we can't get metadata, that's okay
            pass
        
        return None

    @property
    def name(self) -> str:
        """
        The name of this benchmark.

        Defaults to the class name. Override this property if you want a custom name.

        Returns:
        --------
        str
            Benchmark name
        """
        return self.__class__.__name__.lower()


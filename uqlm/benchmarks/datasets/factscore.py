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

"""FactScore benchmark dataset for longform generation evaluation."""

from typing import List, Optional
from langchain_core.language_models.chat_models import BaseChatModel
from datasets import load_dataset
import logging

from uqlm.benchmarks.datasets.base import BaseBenchmark

logger = logging.getLogger(__name__)


class FactScoreBenchmark(BaseBenchmark):
    """
    FactScore benchmark for evaluating longform generation quality.

    Uses the FActScore dataset to evaluate factual accuracy in biographical
    text generation. This benchmark is designed for longform UQ scorers that
    work at the claim/sentence level.

    Supported category: "longform"
    Compatible scorers: response_claim_entail, response_sent_entail,
                       matched_claim_entail, etc.
    """

    def __init__(self, judge_llm: BaseChatModel, dataset_split: str = "test", max_samples: Optional[int] = None):
        """
        Initialize FactScore benchmark.

        Parameters:
        -----------
        judge_llm : BaseChatModel
            LLM to use for judging/entailment (used by some scoring methods)
        dataset_split : str
            Dataset split to use (default: "test")
        max_samples : int, optional
            Maximum number of samples to use (for testing)
        """
        super().__init__()
        self.judge_llm = judge_llm
        self.dataset_split = dataset_split
        self.max_samples = max_samples
        self._prompts = None

    @property
    def category(self) -> str:
        """
        FactScore supports longform UQ evaluation.

        Returns:
        --------
        str
            "longform" - This benchmark works with LongFormUQ scorers
        """
        return "longform"

    @property
    def dataset_name(self) -> str:
        """
        The HuggingFace dataset name.

        Returns:
        --------
        str
            "dskar/FActScore"
        """
        return "dskar/FActScore"

    def _load_dataset(self):
        """Load the FActScore dataset."""
        if self._dataset is None:
            logger.info(f"Loading FActScore dataset (split={self.dataset_split})")
            ds = load_dataset("dskar/FActScore", split=self.dataset_split)

            # Store the original dataset object to preserve metadata
            self._dataset_raw = ds
            df = ds.to_pandas()

            if self.max_samples is not None:
                df = df.head(self.max_samples)

            self._dataset = df
            self._prompts = df["factscore_prompt"].tolist()

            logger.info(f"Loaded {len(self._prompts)} prompts from FActScore")

    def get_prompts(self) -> List[str]:
        """
        Get the list of prompts for this benchmark.

        Returns:
        --------
        List[str]
            List of prompts from the FActScore dataset
        """
        if self._prompts is None:
            self._load_dataset()

        return self._prompts

    def get_entities(self) -> List[str]:
        """
        Get the list of entities (biographical subjects).

        Returns:
        --------
        List[str]
            List of entity names
        """
        if self._dataset is None:
            self._load_dataset()

        return self._dataset["entity"].tolist()

    def get_source_texts(self) -> List[str]:
        """
        Get the source Wikipedia texts for each entity.

        Returns:
        --------
        List[str]
            List of source texts
        """
        if self._dataset is None:
            self._load_dataset()

        return self._dataset["wikipedia_text"].tolist()

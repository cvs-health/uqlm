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

"""Base class for benchmark implementations."""

from abc import ABC, abstractmethod
from typing import List, Any, Dict
from uqlm.benchmarks.models import BenchmarkConfig, PromptResult


class BaseBenchmark(ABC):
    """
    Abstract base class for benchmark implementations.
    
    Subclasses should implement the evaluate method to define
    how the benchmark is executed.
    
    Example:
        class MyBenchmark(BaseBenchmark):
            def __init__(self, judge_llm):
                self.judge_llm = judge_llm
            
            async def evaluate(self, config: BenchmarkConfig) -> List[PromptResult]:
                # Implementation here
                pass
    """
    
    @abstractmethod
    async def evaluate(
        self,
        config: BenchmarkConfig,
        progress_callback: Any = None
    ) -> List[PromptResult]:
        """
        Execute the benchmark evaluation.
        
        Parameters:
        -----------
        config : BenchmarkConfig
            Configuration for this benchmark run
        progress_callback : Any, optional
            Optional callback for progress updates
        
        Returns:
        --------
        List[PromptResult]
            List of results for each prompt/LLM combination
        """
        pass
    
    def get_name(self) -> str:
        """
        Get the name of this benchmark.
        
        Returns:
        --------
        str
            Benchmark name
        """
        return self.__class__.__name__


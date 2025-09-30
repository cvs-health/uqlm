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

"""
UQLM Benchmarking Framework

Main classes:
- BenchmarkRunner: Run benchmarks with caching and incremental saving
- BenchmarkAnalyzer: Analyze and visualize benchmark results

Benchmark implementations:
- FactScoreBenchmark: Longform generation benchmark using FActScore dataset
- BaseBenchmark: Base class for creating custom benchmarks
"""

from uqlm.benchmarks.runner import BenchmarkRunner
from uqlm.benchmarks.analyzer import BenchmarkAnalyzer
from uqlm.benchmarks.implementations import BaseBenchmark, FactScoreBenchmark

__all__ = [
    "BenchmarkRunner",
    "BenchmarkAnalyzer",
    "BaseBenchmark",
    "FactScoreBenchmark",
]


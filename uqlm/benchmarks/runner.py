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

"""Benchmark execution and orchestration."""

import logging
import uuid
from pathlib import Path
from typing import Any, List, Optional, Dict
from datetime import datetime

from uqlm.benchmarks.models import BenchmarkConfig, BenchmarkRun, RunMetadata
from uqlm.benchmarks.storage import BenchmarkResultsDB

logger = logging.getLogger(__name__)


class BenchmarkRunner:
    """
    Run benchmarks with automatic caching and result persistence.
    
    Handles benchmark execution, caching, and incremental saving
    to prevent data loss during long-running evaluations.
    
    Example:
        runner = BenchmarkRunner(storage_path="~/.uqlm/benchmark_results")
        
        results = await runner.run_benchmark(
            benchmark_name="factscore",
            benchmark_implementation=fs_benchmark,
            llm_names=["gemini-2.5-flash"],
            scorer_names=["LUQ"],
            dataset_name="dskar/FActScore",
            sampling_temperature=0.4,
            num_responses=5,
            use_cache=True,
            save_results=True
        )
    """
    
    def __init__(self, storage_path: str = "~/.uqlm/benchmark_results"):
        """
        Initialize benchmark runner.
        
        Parameters:
        -----------
        storage_path : str
            Directory where benchmark database is stored
        """
        self.storage_path = Path(storage_path).expanduser()
        self.db = BenchmarkResultsDB(self.storage_path)
    
    async def run_benchmark(
        self,
        benchmark_name: str,
        benchmark_implementation: Any,
        llm_names: List[str],
        scorer_names: List[str],
        dataset_name: str,
        sampling_temperature: float = 0.7,
        num_responses: int = 5,
        dataset_version: Optional[str] = None,
        use_cache: bool = True,
        save_results: bool = True,
        save_interval: int = 10,
        **additional_params
    ) -> BenchmarkRun:
        """
        Run a benchmark evaluation.
        
        Parameters:
        -----------
        benchmark_name : str
            Name of the benchmark (e.g., "factscore")
        benchmark_implementation : Any
            Instance of benchmark class (e.g., FactScoreBenchmark)
        llm_names : List[str]
            List of LLM names to evaluate
        scorer_names : List[str]
            List of scorer names to use
        dataset_name : str
            Name/path of dataset
        sampling_temperature : float, default=0.7
            Temperature for sampling responses
        num_responses : int, default=5
            Number of sampled responses per prompt
        dataset_version : Optional[str]
            Specific dataset version
        use_cache : bool, default=True
            If True, return cached results if available
        save_results : bool, default=True
            If True, save results to database after completion
        save_interval : int, default=10
            Save intermediate results every N prompts (only if save_results=True)
        **additional_params
            Any additional benchmark-specific parameters
        
        Returns:
        --------
        BenchmarkRun
            Complete run object with metadata and results
        """
        # Build config
        config = BenchmarkConfig(
            benchmark_name=benchmark_name,
            llm_names=llm_names,
            scorer_names=scorer_names,
            dataset_name=dataset_name,
            dataset_version=dataset_version,
            sampling_temperature=sampling_temperature,
            num_responses=num_responses,
            additional_params=additional_params
        )
        
        # Check cache
        if use_cache:
            cached_run = self.db.find_matching_run(config)
            if cached_run:
                logger.info(f"Using cached run: {cached_run.metadata.run_id}")
                return cached_run
        
        # Create new run
        run_id = str(uuid.uuid4())
        logger.info(f"Starting new benchmark run: {run_id}")
        
        # Create run metadata
        metadata = RunMetadata(
            run_id=run_id,
            config_hash=config.compute_hash(),
            created_at=datetime.now(),
            status="running",
            config=config
        )
        
        run = BenchmarkRun(metadata=metadata, results=[])
        
        # Save initial run state
        if save_results:
            try:
                self.db.save_run(run)
            except Exception as e:
                logger.warning(f"Failed to save initial run state: {e}")
        
        # Execute benchmark
        try:
            # Delegate to the benchmark implementation
            # The implementation should return a BenchmarkRun or dict of results
            results = await self._execute_benchmark(
                benchmark_implementation=benchmark_implementation,
                config=config,
                run_id=run_id,
                save_results=save_results,
                save_interval=save_interval
            )
            
            # Update run with results
            run.results = results
            run.metadata.status = "completed"
            run.metadata.completed_at = datetime.now()
            
            # Save final results
            if save_results:
                self.db.save_run(run, update_if_exists=True)
                logger.info(f"Completed and saved run: {run_id}")
            
            return run
            
        except Exception as e:
            logger.error(f"Benchmark run {run_id} failed: {e}")
            run.metadata.status = "failed"
            run.metadata.error_message = str(e)
            
            if save_results:
                try:
                    self.db.update_run_status(
                        run_id=run_id,
                        status="failed",
                        error_message=str(e)
                    )
                except Exception as save_error:
                    logger.error(f"Failed to save error status: {save_error}")
            
            raise
    
    async def _execute_benchmark(
        self,
        benchmark_implementation: Any,
        config: BenchmarkConfig,
        run_id: str,
        save_results: bool,
        save_interval: int
    ) -> List:
        """
        Execute the benchmark implementation.
        
        This method will need to be adapted based on the specific
        interface of benchmark implementations. For now, it provides
        a template.
        
        Parameters:
        -----------
        benchmark_implementation : Any
            The benchmark instance to execute
        config : BenchmarkConfig
            Configuration for the run
        run_id : str
            Unique run identifier
        save_results : bool
            Whether to save incremental results
        save_interval : int
            How often to save (every N prompts)
        
        Returns:
        --------
        List
            List of PromptResult objects
        """
        # This is a placeholder - actual implementation will depend on
        # the interface provided by benchmark implementations
        # For now, we'll raise NotImplementedError to be filled in later
        
        logger.info(f"Executing benchmark: {config.benchmark_name}")
        logger.info(f"Config: {config.model_dump()}")
        
        # TODO: Implement actual benchmark execution logic
        # This will involve:
        # 1. Loading the dataset
        # 2. Iterating through prompts
        # 3. Getting responses from LLMs
        # 4. Computing scores
        # 5. Saving incrementally if requested
        
        raise NotImplementedError(
            "Benchmark execution logic needs to be implemented. "
            "This will be connected to actual benchmark implementations."
        )
    
    def get_run(self, run_id: str) -> BenchmarkRun:
        """
        Get complete details for a specific run.
        
        Parameters:
        -----------
        run_id : str
            Run ID to retrieve
        
        Returns:
        --------
        BenchmarkRun
            Complete run object
        """
        return self.db.load_run(run_id)
    
    def delete_run(self, run_id: str) -> None:
        """
        Delete a run from the database.
        
        Parameters:
        -----------
        run_id : str
            Run ID to delete
        """
        self.db.delete_run(run_id)
        logger.info(f"Deleted run: {run_id}")


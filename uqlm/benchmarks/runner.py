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
from langchain_core.language_models.chat_models import BaseChatModel

from uqlm.benchmarks.models import BenchmarkConfig, BenchmarkRun, RunMetadata, PromptResult
from uqlm.benchmarks.storage import BenchmarkResultsDB
from uqlm.benchmarks.validation import validate_benchmark_implementation, validate_benchmark_scorers, BenchmarkValidationError
from uqlm.benchmarks.datasets import BaseBenchmark
from uqlm.scorers.longform import LongFormUQ
from uqlm.utils.llm_config import get_llm_name

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

    async def run_benchmark(self, benchmark: BaseBenchmark, llms: List[BaseChatModel], scorers: List[str], sampling_temperature: float = 0.7, num_responses: int = 5, use_cache: bool = True, save_results: bool = True, save_interval: int = 10, resume_run_id: Optional[str] = None, **additional_params) -> BenchmarkRun:
        """
        Run a benchmark evaluation.

        Parameters:
        -----------
        benchmark : BaseBenchmark
            Instance of benchmark class (e.g., FactScoreBenchmark).
            The benchmark provides its name, category, dataset name, and version.
        llms : List[BaseChatModel]
            List of instantiated LLM objects to evaluate
        scorers : List[str]
            List of scorer names to use (e.g., ["response_claim_entail"])
        sampling_temperature : float, default=0.7
            Temperature for sampling responses
        num_responses : int, default=5
            Number of sampled responses per prompt
        use_cache : bool, default=True
            If True, return cached results if available. When True, also attempts
            to resume incomplete runs with matching configuration.
        save_results : bool, default=True
            If True, save results to database after completion
        save_interval : int, default=10
            Save intermediate results every N prompts (only if save_results=True)
        resume_run_id : Optional[str], default=None
            Explicitly resume a specific run by ID. If provided, the run must exist
            and have status 'pending' or 'failed'. Completed runs cannot be resumed.
            If None, automatic resumption based on config matching (via use_cache)
            may still occur.
        **additional_params
            Any additional benchmark-specific parameters

        Returns:
        --------
        BenchmarkRun
            Complete run object with metadata and results

        Raises:
        -------
        ValueError
            If resume_run_id is provided but the run doesn't exist, is completed,
            or has mismatched configuration.
        BenchmarkValidationError
            If benchmark or scorers are invalid for the category.
        """
        # Validate benchmark implementation
        logger.info("Validating benchmark configuration...")
        try:
            validate_benchmark_implementation(benchmark)
        except BenchmarkValidationError as e:
            logger.error(f"Benchmark validation failed: {e}")
            raise

        # Get all benchmark metadata from the benchmark itself
        benchmark_name = benchmark.get_name()
        benchmark_category = benchmark.get_supported_category()
        dataset_name = benchmark.get_dataset_name()
        dataset_version = benchmark.get_dataset_version()

        # Validate scorers are compatible with benchmark category
        try:
            validate_benchmark_scorers(benchmark=benchmark, scorers=scorers, benchmark_category=benchmark_category)
        except BenchmarkValidationError as e:
            logger.error(f"Scorer validation failed: {e}")
            raise

        logger.info(f"✓ Validation passed: {benchmark_name} ({benchmark_category}) with scorers {scorers}")

        # Extract LLM names for config
        llm_names = [get_llm_name(llm) for llm in llms]

        # Build config
        config = BenchmarkConfig(benchmark_name=benchmark_name, benchmark_category=benchmark_category, llm_names=llm_names, scorer_names=scorers, dataset_name=dataset_name, dataset_version=dataset_version, sampling_temperature=sampling_temperature, num_responses=num_responses, additional_params=additional_params)

        # Handle explicit resume request
        existing_run = None
        if resume_run_id:
            logger.info(f"Attempting to resume run: {resume_run_id}")
            try:
                existing_run = self.db.load_run(resume_run_id)

                # Validate run can be resumed
                if existing_run.metadata.status == "completed":
                    raise ValueError(f"Cannot resume run {resume_run_id}: already completed. Use use_cache=True to reuse completed results.")

                # Verify config matches
                if existing_run.metadata.config_hash != config.compute_hash():
                    raise ValueError(f"Cannot resume run {resume_run_id}: configuration mismatch. Existing config hash: {existing_run.metadata.config_hash}, Current config hash: {config.compute_hash()}")

                logger.info(f"Resuming run {resume_run_id} (status: {existing_run.metadata.status}, {len(existing_run.results)} existing results)")

            except ValueError:
                raise  # Re-raise ValueError for resume errors
            except Exception as e:
                raise ValueError(f"Failed to load run {resume_run_id}: {e}")

        # Check cache for matching runs (if not explicitly resuming)
        elif use_cache:
            cached_run = self.db.find_matching_run(config, only_completed=False)
            if cached_run:
                if cached_run.metadata.status == "completed":
                    logger.info(f"Using cached completed run: {cached_run.metadata.run_id}")
                    return cached_run
                else:
                    # Found incomplete run - resume it
                    existing_run = cached_run
                    logger.info(f"Found incomplete run with matching config: {existing_run.metadata.run_id} (status: {existing_run.metadata.status}, {len(existing_run.results)} existing results). Resuming...")

        # Determine run_id and setup
        if existing_run:
            # Resuming existing run
            run_id = existing_run.metadata.run_id
            run = existing_run
            # Update status to running
            run.metadata.status = "running"
            run.metadata.error_message = None  # Clear previous error
            if save_results:
                self.db.update_run_status(run_id, "running")
        else:
            # Create new run
            run_id = str(uuid.uuid4())
            logger.info(f"Starting new benchmark run: {run_id}")

            # Create run metadata
            metadata = RunMetadata(run_id=run_id, config_hash=config.compute_hash(), created_at=datetime.now(), status="running", config=config)
            run = BenchmarkRun(metadata=metadata, results=[])

            # Save initial run state
            if save_results:
                try:
                    self.db.save_run(run)
                except Exception as e:
                    logger.warning(f"Failed to save initial run state: {e}")

        # Get completed prompts if resuming
        completed_prompts = {}
        if existing_run:
            completed_prompts = self.db.get_completed_prompt_ids(run_id)
            logger.info(f"Skipping {sum(len(v) for v in completed_prompts.values())} already completed prompt/LLM combinations")

        # Execute benchmark
        try:
            # Delegate to the execution logic
            new_results = await self._execute_benchmark(benchmark_category=benchmark_category, benchmark=benchmark, llms=llms, scorers=scorers, config=config, run_id=run_id, save_results=save_results, save_interval=save_interval, completed_prompts=completed_prompts)

            # Merge results (existing + new)
            if existing_run:
                # Combine existing results with new ones
                all_results = list(existing_run.results) + new_results
                run.results = all_results
            else:
                run.results = new_results

            run.metadata.status = "completed"
            run.metadata.completed_at = datetime.now()

            # Save final results
            if save_results:
                if existing_run:
                    # Use append_results for incremental update
                    self.db.append_results(run_id, new_results)
                    self.db.update_run_status(run_id, "completed", completed_at=datetime.now())
                else:
                    # Save complete run
                    self.db.save_run(run, update_if_exists=True)
                logger.info(f"Completed and saved run: {run_id}")

            logger.info(f"Benchmark run {run_id} completed: {len(new_results)} new results, {len(run.results)} total")
            return run

        except Exception as e:
            logger.error(f"Benchmark run {run_id} failed: {e}")
            run.metadata.status = "failed"
            run.metadata.error_message = str(e)

            if save_results:
                try:
                    self.db.update_run_status(run_id=run_id, status="failed", error_message=str(e))
                except Exception as save_error:
                    logger.error(f"Failed to save error status: {save_error}")

            raise

    async def _execute_benchmark(self, benchmark_category: str, benchmark: Any, llms: List[BaseChatModel], scorers: List[str], config: BenchmarkConfig, run_id: str, save_results: bool, save_interval: int, completed_prompts: Dict[str, set] = None) -> List[PromptResult]:
        """
        Execute the benchmark based on category.

        Parameters:
        -----------
        benchmark_category : str
            Category determines which UQ class to use
        benchmark : Any
            The benchmark instance to execute
        llms : List[BaseChatModel]
            List of LLM objects
        scorers : List[str]
            List of scorer names
        config : BenchmarkConfig
            Configuration for the run
        run_id : str
            Unique run identifier
        save_results : bool
            Whether to save incremental results
        save_interval : int
            How often to save (every N prompts)
        completed_prompts : Dict[str, set], optional
            Mapping of llm_name -> set of already completed prompt_ids
            Used when resuming incomplete runs

        Returns:
        --------
        List[PromptResult]
            List of PromptResult objects
        """
        logger.info(f"Executing {benchmark_category} benchmark: {config.benchmark_name}")

        if completed_prompts is None:
            completed_prompts = {}

        if benchmark_category == "longform":
            return await self._execute_longform_benchmark(benchmark=benchmark, llms=llms, scorers=scorers, config=config, run_id=run_id, save_results=save_results, save_interval=save_interval, completed_prompts=completed_prompts)
        else:
            raise NotImplementedError(f"Benchmark category '{benchmark_category}' is not yet implemented. Supported categories: 'longform'")

    async def _execute_longform_benchmark(self, benchmark: Any, llms: List[BaseChatModel], scorers: List[str], config: BenchmarkConfig, run_id: str, save_results: bool, save_interval: int, completed_prompts: Dict[str, set] = None) -> List[PromptResult]:
        """
        Execute a longform benchmark using LongFormUQ.

        This method:
        1. Creates LongFormUQ instances for each LLM
        2. Loads the dataset from the benchmark
        3. Runs generate_and_score for each prompt (skipping completed ones if resuming)
        4. Collects results and creates PromptResult objects

        Parameters:
        -----------
        completed_prompts : Dict[str, set], optional
            Mapping of llm_name -> set of already completed prompt_ids
            Used when resuming incomplete runs to skip already processed prompts
        """
        if completed_prompts is None:
            completed_prompts = {}

        logger.info("Initializing LongFormUQ scorers")

        # Create UQ instances for each LLM
        uq_instances = {}
        for llm in llms:
            llm_name = get_llm_name(llm)
            uq = LongFormUQ(llm=llm, scorers=scorers, sampling_temperature=config.sampling_temperature)
            uq_instances[llm_name] = uq

        # Get dataset from benchmark
        logger.info(f"Loading dataset: {config.dataset_name}")

        if hasattr(benchmark, "get_prompts"):
            prompts = benchmark.get_prompts()
        else:
            raise NotImplementedError(f"Benchmark {benchmark.__class__.__name__} must implement get_prompts() method")

        # Execute benchmark for each LLM
        all_results = []

        for llm_name, uq in uq_instances.items():
            # Determine which prompts need to be processed for this LLM
            completed_for_llm = completed_prompts.get(llm_name, set())

            # Create list of (original_index, prompt) for prompts that need processing
            prompts_to_process = [(i, prompt) for i, prompt in enumerate(prompts) if i not in completed_for_llm]

            if not prompts_to_process:
                logger.info(f"All prompts already completed for LLM: {llm_name}, skipping")
                continue

            logger.info(f"Running benchmark for LLM: {llm_name} ({len(prompts_to_process)}/{len(prompts)} prompts remaining)")

            # Extract just the prompts (without indices) for UQ scoring
            prompts_list = [p for _, p in prompts_to_process]

            # Run UQ scoring on remaining prompts only
            uq_result = await uq.generate_and_score(prompts=prompts_list, num_responses=config.num_responses, show_progress_bars=True)

            # Convert UQ results to PromptResult objects, using original indices
            for result_idx, (original_idx, prompt) in enumerate(prompts_to_process):
                prompt_result = PromptResult(
                    prompt_id=original_idx,  # Use original index from dataset
                    prompt=prompt,
                    llm_name=llm_name,
                    original_response=uq_result.data["response"][result_idx],
                    sampled_responses=uq_result.data["sampled_responses"][result_idx],
                    scores={scorer: uq_result.data[scorer][result_idx] for scorer in scorers if scorer in uq_result.data},
                    metadata={"temperature": uq_result.metadata.get("temperature"), "sampling_temperature": uq_result.metadata.get("sampling_temperature")},
                )
                all_results.append(prompt_result)

            # Save incrementally if requested
            if save_results and len(all_results) % save_interval == 0:
                logger.info(f"Saving incremental results ({len(all_results)} completed)")
                self.db.append_results(run_id, all_results[-save_interval:])

        logger.info(f"Benchmark execution completed. Total new results: {len(all_results)}")
        return all_results

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

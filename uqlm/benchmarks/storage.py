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

"""SQLite storage backend for benchmark results."""

import sqlite3
import json
import logging
from pathlib import Path
from typing import Optional, List, Tuple, Dict
from datetime import datetime

from uqlm.benchmarks.models import BenchmarkRun, BenchmarkConfig, RunMetadata, PromptResult

logger = logging.getLogger(__name__)


class BenchmarkResultsDB:
    """
    SQLite-backed storage for benchmark results.

    Manages persistence of benchmark runs with support for querying,
    caching, and incremental updates.
    """

    def __init__(self, storage_path: Path):
        """
        Initialize database connection.

        Parameters:
        -----------
        storage_path : Path
            Directory containing the SQLite database
        """
        self.storage_path = Path(storage_path).expanduser()
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.db_path = self.storage_path / "benchmarks.db"
        self._init_db()

    def _init_db(self) -> None:
        """Create database tables if they don't exist."""
        conn = self._get_connection()
        cursor = conn.cursor()

        # Runs metadata table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS runs (
                run_id TEXT PRIMARY KEY,
                config_hash TEXT NOT NULL,
                benchmark_name TEXT NOT NULL,
                benchmark_category TEXT NOT NULL,
                created_at TEXT NOT NULL,
                completed_at TEXT,
                status TEXT NOT NULL,
                config_json TEXT NOT NULL,
                error_message TEXT,
                UNIQUE(config_hash)
            )
        """)

        # Results table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS results (
                result_id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                prompt_id INTEGER NOT NULL,
                llm_name TEXT NOT NULL,
                prompt TEXT NOT NULL,
                original_response TEXT NOT NULL,
                sampled_responses_json TEXT NOT NULL,
                scores_json TEXT NOT NULL,
                metadata_json TEXT NOT NULL,
                FOREIGN KEY (run_id) REFERENCES runs (run_id) ON DELETE CASCADE
            )
        """)

        # Create indices for common queries
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_runs_config_hash ON runs(config_hash)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_runs_benchmark ON runs(benchmark_name)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_runs_category ON runs(benchmark_category)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_runs_created ON runs(created_at)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_results_run_id ON results(run_id)")

        conn.commit()
        conn.close()

    def _get_connection(self) -> sqlite3.Connection:
        """Get a database connection."""
        return sqlite3.connect(str(self.db_path))

    def save_run(self, run: BenchmarkRun, update_if_exists: bool = False) -> None:
        """
        Save a complete benchmark run.

        Parameters:
        -----------
        run : BenchmarkRun
            Complete run object to save
        update_if_exists : bool
            If True, update existing run; otherwise skip if exists
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        try:
            # Save run metadata
            cursor.execute(
                """
                INSERT OR REPLACE INTO runs 
                (run_id, config_hash, benchmark_name, benchmark_category, created_at, completed_at, 
                 status, config_json, error_message)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (run.metadata.run_id, run.metadata.config_hash, run.metadata.config.benchmark_name, run.metadata.config.benchmark_category, run.metadata.created_at.isoformat(), run.metadata.completed_at.isoformat() if run.metadata.completed_at else None, run.metadata.status, json.dumps(run.metadata.config.model_dump()), run.metadata.error_message),
            )

            # Delete existing results for this run (if updating)
            if update_if_exists:
                cursor.execute("DELETE FROM results WHERE run_id = ?", (run.metadata.run_id,))

            # Save results
            for result in run.results:
                cursor.execute(
                    """
                    INSERT INTO results 
                    (run_id, prompt_id, llm_name, prompt, original_response, 
                     sampled_responses_json, scores_json, metadata_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (run.metadata.run_id, result.prompt_id, result.llm_name, result.prompt, result.original_response, json.dumps(result.sampled_responses), json.dumps(result.scores), json.dumps(result.metadata)),
                )

            conn.commit()
            logger.info(f"Saved run {run.metadata.run_id} with {len(run.results)} results")

        except sqlite3.IntegrityError as e:
            logger.warning(f"Run {run.metadata.run_id} already exists: {e}")
            raise
        finally:
            conn.close()

    def load_run(self, run_id: str) -> BenchmarkRun:
        """
        Load a specific run by ID.

        Parameters:
        -----------
        run_id : str
            Run ID to load

        Returns:
        --------
        BenchmarkRun
            Complete run object with all results
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        try:
            # Load metadata
            cursor.execute(
                """
                SELECT run_id, config_hash, created_at, completed_at, 
                       status, config_json, error_message
                FROM runs WHERE run_id = ?
            """,
                (run_id,),
            )

            row = cursor.fetchone()
            if not row:
                raise ValueError(f"Run {run_id} not found")

            metadata = RunMetadata(run_id=row[0], config_hash=row[1], created_at=datetime.fromisoformat(row[2]), completed_at=datetime.fromisoformat(row[3]) if row[3] else None, status=row[4], config=BenchmarkConfig.model_validate(json.loads(row[5])), error_message=row[6])

            # Load results
            cursor.execute(
                """
                SELECT prompt_id, llm_name, prompt, original_response,
                       sampled_responses_json, scores_json, metadata_json
                FROM results WHERE run_id = ?
                ORDER BY prompt_id, llm_name
            """,
                (run_id,),
            )

            results = []
            for row in cursor.fetchall():
                result = PromptResult(prompt_id=row[0], llm_name=row[1], prompt=row[2], original_response=row[3], sampled_responses=json.loads(row[4]), scores=json.loads(row[5]), metadata=json.loads(row[6]))
                results.append(result)

            return BenchmarkRun(metadata=metadata, results=results)

        finally:
            conn.close()

    def find_matching_run(self, config: BenchmarkConfig, only_completed: bool = True) -> Optional[BenchmarkRun]:
        """
        Find an existing run matching this config.

        Parameters:
        -----------
        config : BenchmarkConfig
            Configuration to match
        only_completed : bool, default=True
            If True, only return completed runs. If False, also return
            pending/failed runs that can be resumed.

        Returns:
        --------
        Optional[BenchmarkRun]
            Matching run if found, None otherwise
        """
        config_hash = config.compute_hash()
        conn = self._get_connection()
        cursor = conn.cursor()

        try:
            if only_completed:
                cursor.execute(
                    """
                    SELECT run_id FROM runs 
                    WHERE config_hash = ? AND status = 'completed'
                    ORDER BY created_at DESC
                    LIMIT 1
                """,
                    (config_hash,),
                )
            else:
                # Include pending/failed runs that can be resumed
                cursor.execute(
                    """
                    SELECT run_id FROM runs 
                    WHERE config_hash = ?
                    ORDER BY 
                        CASE status 
                            WHEN 'completed' THEN 1
                            WHEN 'pending' THEN 2
                            WHEN 'failed' THEN 3
                            ELSE 4
                        END,
                        created_at DESC
                    LIMIT 1
                """,
                    (config_hash,),
                )

            row = cursor.fetchone()
            if row:
                run = self.load_run(row[0])
                logger.info(f"Found {'cached' if run.metadata.status == 'completed' else 'resumable'} run with hash {config_hash}: {row[0]} (status: {run.metadata.status})")
                return run

            return None

        finally:
            conn.close()

    def get_completed_prompt_ids(self, run_id: str) -> Dict[str, set]:
        """
        Get completed prompt IDs for each LLM in a run.

        Parameters:
        -----------
        run_id : str
            Run ID to check

        Returns:
        --------
        Dict[str, set]
            Mapping of llm_name -> set of completed prompt_ids
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        try:
            cursor.execute(
                """
                SELECT llm_name, prompt_id
                FROM results
                WHERE run_id = ?
                ORDER BY llm_name, prompt_id
                """,
                (run_id,),
            )

            completed_prompts = {}
            for row in cursor.fetchall():
                llm_name, prompt_id = row
                if llm_name not in completed_prompts:
                    completed_prompts[llm_name] = set()
                completed_prompts[llm_name].add(prompt_id)

            return completed_prompts

        finally:
            conn.close()

    def update_run_status(self, run_id: str, status: str, error_message: Optional[str] = None, completed_at: Optional[datetime] = None) -> None:
        """
        Update the status of a run.

        Parameters:
        -----------
        run_id : str
            Run ID to update
        status : str
            New status (pending, running, completed, failed)
        error_message : Optional[str]
            Error message if status is failed
        completed_at : Optional[datetime]
            Completion timestamp (for completed/failed status)
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        try:
            cursor.execute(
                """
                UPDATE runs
                SET status = ?, error_message = ?, completed_at = ?
                WHERE run_id = ?
                """,
                (status, error_message, completed_at.isoformat() if completed_at else None, run_id),
            )
            conn.commit()
            logger.info(f"Updated run {run_id} status to {status}")

        finally:
            conn.close()

    def append_results(self, run_id: str, results: List[PromptResult]) -> None:
        """
        Append new results to an existing run.

        This is used for incremental saving during benchmark execution
        or when resuming incomplete runs.

        Parameters:
        -----------
        run_id : str
            Run ID to append to
        results : List[PromptResult]
            New results to add
        """
        if not results:
            return

        conn = self._get_connection()
        cursor = conn.cursor()

        try:
            for result in results:
                # Check if this result already exists (prompt_id + llm_name combo)
                cursor.execute(
                    """
                    SELECT 1 FROM results
                    WHERE run_id = ? AND prompt_id = ? AND llm_name = ?
                    """,
                    (run_id, result.prompt_id, result.llm_name),
                )

                if cursor.fetchone():
                    # Update existing result
                    cursor.execute(
                        """
                        UPDATE results
                        SET prompt = ?, original_response = ?,
                            sampled_responses_json = ?, scores_json = ?,
                            metadata_json = ?
                        WHERE run_id = ? AND prompt_id = ? AND llm_name = ?
                        """,
                        (result.prompt, result.original_response, json.dumps(result.sampled_responses), json.dumps(result.scores), json.dumps(result.metadata), run_id, result.prompt_id, result.llm_name),
                    )
                else:
                    # Insert new result
                    cursor.execute(
                        """
                        INSERT INTO results 
                        (run_id, prompt_id, llm_name, prompt, original_response, 
                         sampled_responses_json, scores_json, metadata_json)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (run_id, result.prompt_id, result.llm_name, result.prompt, result.original_response, json.dumps(result.sampled_responses), json.dumps(result.scores), json.dumps(result.metadata)),
                    )

            conn.commit()
            logger.info(f"Appended {len(results)} results to run {run_id}")

        finally:
            conn.close()

    def query_runs(self, benchmark_name: Optional[str] = None, benchmark_category: Optional[str] = None, llm_names: Optional[List[str]] = None, scorer_names: Optional[List[str]] = None, date_range: Optional[Tuple[datetime, datetime]] = None, status: Optional[str] = None, limit: int = 100) -> List[RunMetadata]:
        """
        Query runs with filters.

        Parameters:
        -----------
        benchmark_name : Optional[str]
            Filter by benchmark name
        benchmark_category : Optional[str]
            Filter by benchmark category (e.g., "longform")
        llm_names : Optional[List[str]]
            Filter by LLM names (returns runs containing any of these)
        scorer_names : Optional[List[str]]
            Filter by scorer names (returns runs containing any of these)
        date_range : Optional[Tuple[datetime, datetime]]
            Filter by date range (start, end)
        status : Optional[str]
            Filter by status
        limit : int
            Maximum number of results to return

        Returns:
        --------
        List[RunMetadata]
            List of matching run metadata
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        query = "SELECT run_id, config_hash, created_at, completed_at, status, config_json, error_message FROM runs WHERE 1=1"
        params = []

        if benchmark_name:
            query += " AND benchmark_name = ?"
            params.append(benchmark_name)

        if benchmark_category:
            query += " AND benchmark_category = ?"
            params.append(benchmark_category)

        if status:
            query += " AND status = ?"
            params.append(status)

        if date_range:
            query += " AND created_at >= ? AND created_at <= ?"
            params.extend([date_range[0].isoformat(), date_range[1].isoformat()])

        # For LLM and scorer filtering, we need to parse the config JSON
        # This is less efficient but SQLite doesn't have good JSON query support
        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)

        try:
            cursor.execute(query, params)
            rows = cursor.fetchall()

            metadata_list = []
            for row in rows:
                config = BenchmarkConfig.model_validate(json.loads(row[5]))

                # Apply LLM filter
                if llm_names and not any(llm in config.llm_names for llm in llm_names):
                    continue

                # Apply scorer filter
                if scorer_names and not any(scorer in config.scorer_names for scorer in scorer_names):
                    continue

                metadata = RunMetadata(run_id=row[0], config_hash=row[1], created_at=datetime.fromisoformat(row[2]), completed_at=datetime.fromisoformat(row[3]) if row[3] else None, status=row[4], config=config, error_message=row[6])
                metadata_list.append(metadata)

            return metadata_list

        finally:
            conn.close()

    def delete_run(self, run_id: str) -> None:
        """
        Delete a run from the database.

        Parameters:
        -----------
        run_id : str
            Run ID to delete
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        try:
            cursor.execute("DELETE FROM runs WHERE run_id = ?", (run_id,))
            conn.commit()
            logger.info(f"Deleted run {run_id}")
        finally:
            conn.close()

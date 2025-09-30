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

"""Benchmark result analysis and visualization."""

import logging
import pandas as pd
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime
from collections import defaultdict

from uqlm.benchmarks.models import BenchmarkRun
from uqlm.benchmarks.storage import BenchmarkResultsDB

logger = logging.getLogger(__name__)


class BenchmarkAnalyzer:
    """
    Analyze and visualize benchmark results.
    
    Provides methods for comparing LLMs, scorers, computing aggregates,
    and generating visualizations from stored benchmark results.
    
    Example:
        analyzer = BenchmarkAnalyzer(storage_path="~/.uqlm/benchmark_results")
        
        # Compare LLMs
        comparison = analyzer.compare_llms(
            benchmark_name="factscore",
            scorer_names=["LUQ"]
        )
        
        # Visualize
        analyzer.plot_results(comparison, plot_type="bar")
    """
    
    def __init__(self, storage_path: str = "~/.uqlm/benchmark_results"):
        """
        Initialize benchmark analyzer.
        
        Parameters:
        -----------
        storage_path : str
            Directory where benchmark database is stored
        """
        self.storage_path = Path(storage_path).expanduser()
        self.db = BenchmarkResultsDB(self.storage_path)
    
    def compare_scorers(
        self,
        run_ids: Optional[List[str]] = None,
        benchmark_name: Optional[str] = None,
        llm_names: Optional[List[str]] = None,
        date_range: Optional[Tuple[datetime, datetime]] = None
    ) -> pd.DataFrame:
        """
        Compare scorer performance across runs.
        
        Parameters:
        -----------
        run_ids : Optional[List[str]]
            Specific run IDs to compare. If None, use filters below.
        benchmark_name : Optional[str]
            Filter by benchmark name
        llm_names : Optional[List[str]]
            Filter by LLM names
        date_range : Optional[Tuple[datetime, datetime]]
            Filter by date range (start, end)
        
        Returns:
        --------
        pd.DataFrame
            Comparison dataframe with scorer metrics
        """
        # Get runs to analyze
        runs = self._get_runs(run_ids, benchmark_name, llm_names, None, date_range)
        
        if not runs:
            logger.warning("No runs found matching criteria")
            return pd.DataFrame()
        
        # Build comparison dataframe
        comparison_data = []
        
        for run in runs:
            # Group results by scorer
            scorer_metrics = defaultdict(list)
            
            for result in run.results:
                for scorer_name, score in result.scores.items():
                    scorer_metrics[scorer_name].append(score)
            
            # Compute aggregates for each scorer
            for scorer_name, scores in scorer_metrics.items():
                comparison_data.append({
                    'run_id': run.metadata.run_id,
                    'benchmark_name': run.metadata.config.benchmark_name,
                    'scorer_name': scorer_name,
                    'mean_score': pd.Series(scores).mean(),
                    'std_score': pd.Series(scores).std(),
                    'min_score': pd.Series(scores).min(),
                    'max_score': pd.Series(scores).max(),
                    'num_samples': len(scores),
                    'created_at': run.metadata.created_at
                })
        
        df = pd.DataFrame(comparison_data)
        return df
    
    def compare_llms(
        self,
        run_ids: Optional[List[str]] = None,
        benchmark_name: Optional[str] = None,
        scorer_names: Optional[List[str]] = None,
        date_range: Optional[Tuple[datetime, datetime]] = None
    ) -> pd.DataFrame:
        """
        Compare LLM performance across runs.
        
        Parameters:
        -----------
        run_ids : Optional[List[str]]
            Specific run IDs to compare. If None, use filters below.
        benchmark_name : Optional[str]
            Filter by benchmark name
        scorer_names : Optional[List[str]]
            Filter by scorer names
        date_range : Optional[Tuple[datetime, datetime]]
            Filter by date range (start, end)
        
        Returns:
        --------
        pd.DataFrame
            Comparison dataframe with LLM metrics
        """
        # Get runs to analyze
        runs = self._get_runs(run_ids, benchmark_name, None, scorer_names, date_range)
        
        if not runs:
            logger.warning("No runs found matching criteria")
            return pd.DataFrame()
        
        # Build comparison dataframe
        comparison_data = []
        
        for run in runs:
            # Group results by LLM and scorer
            llm_scorer_metrics = defaultdict(lambda: defaultdict(list))
            
            for result in run.results:
                for scorer_name, score in result.scores.items():
                    llm_scorer_metrics[result.llm_name][scorer_name].append(score)
            
            # Compute aggregates for each LLM/scorer combination
            for llm_name, scorer_dict in llm_scorer_metrics.items():
                for scorer_name, scores in scorer_dict.items():
                    comparison_data.append({
                        'run_id': run.metadata.run_id,
                        'benchmark_name': run.metadata.config.benchmark_name,
                        'llm_name': llm_name,
                        'scorer_name': scorer_name,
                        'mean_score': pd.Series(scores).mean(),
                        'std_score': pd.Series(scores).std(),
                        'min_score': pd.Series(scores).min(),
                        'max_score': pd.Series(scores).max(),
                        'num_samples': len(scores),
                        'created_at': run.metadata.created_at
                    })
        
        df = pd.DataFrame(comparison_data)
        return df
    
    def aggregate_metrics(
        self,
        run_id: str,
        groupby: str = "llm"
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute aggregate statistics for a specific run.
        
        Parameters:
        -----------
        run_id : str
            Run ID to analyze
        groupby : str
            How to group results ("llm" or "scorer")
        
        Returns:
        --------
        Dict[str, Dict[str, float]]
            Nested dict: {group_name: {metric_name: value}}
        """
        run = self.db.load_run(run_id)
        
        if groupby == "llm":
            return self._aggregate_by_llm(run)
        elif groupby == "scorer":
            return self._aggregate_by_scorer(run)
        else:
            raise ValueError(f"Invalid groupby value: {groupby}. Must be 'llm' or 'scorer'")
    
    def plot_results(
        self,
        comparison_df: pd.DataFrame,
        plot_type: str = "bar",
        save_path: Optional[str] = None,
        **plot_kwargs
    ):
        """
        Generate visualizations from comparison dataframes.
        
        Parameters:
        -----------
        comparison_df : pd.DataFrame
            DataFrame from compare_scorers() or compare_llms()
        plot_type : str
            Type of plot to generate ("bar", "box", "violin")
        save_path : Optional[str]
            If provided, save plot to this path
        **plot_kwargs
            Additional arguments passed to plotting function
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
        except ImportError:
            logger.error("matplotlib and seaborn required for plotting. Install with: pip install matplotlib seaborn")
            return
        
        if comparison_df.empty:
            logger.warning("Empty comparison dataframe, nothing to plot")
            return
        
        # Determine what we're comparing
        if 'llm_name' in comparison_df.columns and 'scorer_name' in comparison_df.columns:
            # LLM comparison
            self._plot_llm_comparison(comparison_df, plot_type, save_path, **plot_kwargs)
        elif 'scorer_name' in comparison_df.columns:
            # Scorer comparison
            self._plot_scorer_comparison(comparison_df, plot_type, save_path, **plot_kwargs)
        else:
            logger.error("Unknown comparison dataframe format")
    
    def export_report(
        self,
        run_ids: List[str],
        output_path: str,
        format: str = "html"
    ):
        """
        Generate and export a comprehensive analysis report.
        
        Parameters:
        -----------
        run_ids : List[str]
            Run IDs to include in report
        output_path : str
            Path to save report
        format : str
            Output format ("html" or "markdown")
        """
        logger.info(f"Generating report for {len(run_ids)} runs")
        
        # Load all runs
        runs = [self.db.load_run(run_id) for run_id in run_ids]
        
        # Generate report content
        if format == "markdown":
            content = self._generate_markdown_report(runs)
        elif format == "html":
            content = self._generate_html_report(runs)
        else:
            raise ValueError(f"Invalid format: {format}. Must be 'html' or 'markdown'")
        
        # Write to file
        output_file = Path(output_path)
        output_file.write_text(content)
        logger.info(f"Report saved to {output_path}")
    
    def list_runs(
        self,
        benchmark_name: Optional[str] = None,
        limit: int = 20
    ) -> pd.DataFrame:
        """
        List recent benchmark runs with summary information.
        
        Parameters:
        -----------
        benchmark_name : Optional[str]
            Filter by benchmark name
        limit : int
            Maximum number of runs to return
        
        Returns:
        --------
        pd.DataFrame
            DataFrame with run metadata
        """
        runs_metadata = self.db.query_runs(
            benchmark_name=benchmark_name,
            status="completed",
            limit=limit
        )
        
        if not runs_metadata:
            return pd.DataFrame()
        
        data = []
        for metadata in runs_metadata:
            data.append({
                'run_id': metadata.run_id,
                'benchmark_name': metadata.config.benchmark_name,
                'llm_names': ', '.join(metadata.config.llm_names),
                'scorer_names': ', '.join(metadata.config.scorer_names),
                'dataset_name': metadata.config.dataset_name,
                'num_responses': metadata.config.num_responses,
                'created_at': metadata.created_at,
                'status': metadata.status
            })
        
        return pd.DataFrame(data)
    
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
    
    # Private helper methods
    
    def _get_runs(
        self,
        run_ids: Optional[List[str]],
        benchmark_name: Optional[str],
        llm_names: Optional[List[str]],
        scorer_names: Optional[List[str]],
        date_range: Optional[Tuple[datetime, datetime]]
    ) -> List[BenchmarkRun]:
        """Helper to get runs based on IDs or filters."""
        if run_ids:
            return [self.db.load_run(run_id) for run_id in run_ids]
        else:
            metadata_list = self.db.query_runs(
                benchmark_name=benchmark_name,
                llm_names=llm_names,
                scorer_names=scorer_names,
                date_range=date_range,
                status="completed"
            )
            return [self.db.load_run(m.run_id) for m in metadata_list]
    
    def _aggregate_by_llm(self, run: BenchmarkRun) -> Dict[str, Dict[str, float]]:
        """Aggregate metrics grouped by LLM."""
        llm_metrics = defaultdict(lambda: defaultdict(list))
        
        for result in run.results:
            for scorer_name, score in result.scores.items():
                llm_metrics[result.llm_name][scorer_name].append(score)
        
        # Compute statistics
        aggregated = {}
        for llm_name, scorer_dict in llm_metrics.items():
            aggregated[llm_name] = {}
            for scorer_name, scores in scorer_dict.items():
                series = pd.Series(scores)
                aggregated[llm_name][f"{scorer_name}_mean"] = series.mean()
                aggregated[llm_name][f"{scorer_name}_std"] = series.std()
                aggregated[llm_name][f"{scorer_name}_min"] = series.min()
                aggregated[llm_name][f"{scorer_name}_max"] = series.max()
        
        return aggregated
    
    def _aggregate_by_scorer(self, run: BenchmarkRun) -> Dict[str, Dict[str, float]]:
        """Aggregate metrics grouped by scorer."""
        scorer_metrics = defaultdict(list)
        
        for result in run.results:
            for scorer_name, score in result.scores.items():
                scorer_metrics[scorer_name].append(score)
        
        # Compute statistics
        aggregated = {}
        for scorer_name, scores in scorer_metrics.items():
            series = pd.Series(scores)
            aggregated[scorer_name] = {
                'mean': series.mean(),
                'std': series.std(),
                'min': series.min(),
                'max': series.max(),
                'count': len(scores)
            }
        
        return aggregated
    
    def _plot_llm_comparison(
        self,
        df: pd.DataFrame,
        plot_type: str,
        save_path: Optional[str],
        **kwargs
    ):
        """Plot LLM comparison."""
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        plt.figure(figsize=kwargs.get('figsize', (10, 6)))
        
        if plot_type == "bar":
            # Create grouped bar chart
            if 'scorer_name' in df.columns:
                # Group by scorer
                pivot_df = df.pivot(index='llm_name', columns='scorer_name', values='mean_score')
                pivot_df.plot(kind='bar', ax=plt.gca())
                plt.ylabel('Mean Score')
                plt.xlabel('LLM')
                plt.title('LLM Performance by Scorer')
            else:
                df.plot(x='llm_name', y='mean_score', kind='bar', ax=plt.gca())
                plt.ylabel('Mean Score')
                plt.xlabel('LLM')
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=kwargs.get('dpi', 300), bbox_inches='tight')
            logger.info(f"Plot saved to {save_path}")
        else:
            plt.show()
    
    def _plot_scorer_comparison(
        self,
        df: pd.DataFrame,
        plot_type: str,
        save_path: Optional[str],
        **kwargs
    ):
        """Plot scorer comparison."""
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=kwargs.get('figsize', (10, 6)))
        
        if plot_type == "bar":
            df.plot(x='scorer_name', y='mean_score', kind='bar', ax=plt.gca())
            plt.ylabel('Mean Score')
            plt.xlabel('Scorer')
            plt.title('Scorer Performance Comparison')
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=kwargs.get('dpi', 300), bbox_inches='tight')
            logger.info(f"Plot saved to {save_path}")
        else:
            plt.show()
    
    def _generate_markdown_report(self, runs: List[BenchmarkRun]) -> str:
        """Generate markdown report."""
        lines = ["# Benchmark Report", ""]
        lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"Number of runs: {len(runs)}")
        lines.append("")
        
        for run in runs:
            lines.append(f"## Run: {run.metadata.run_id}")
            lines.append(f"- Benchmark: {run.metadata.config.benchmark_name}")
            lines.append(f"- LLMs: {', '.join(run.metadata.config.llm_names)}")
            lines.append(f"- Scorers: {', '.join(run.metadata.config.scorer_names)}")
            lines.append(f"- Created: {run.metadata.created_at}")
            lines.append("")
        
        return "\n".join(lines)
    
    def _generate_html_report(self, runs: List[BenchmarkRun]) -> str:
        """Generate HTML report."""
        # Convert markdown to HTML for now (simple version)
        md_content = self._generate_markdown_report(runs)
        
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Benchmark Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        h1 {{ color: #333; }}
        h2 {{ color: #666; margin-top: 30px; }}
        ul {{ list-style-type: none; }}
    </style>
</head>
<body>
    <pre>{md_content}</pre>
</body>
</html>
"""
        return html


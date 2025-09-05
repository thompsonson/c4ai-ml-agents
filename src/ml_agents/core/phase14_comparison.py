"""Phase 14 comparison framework for reasoning approaches."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from datasets import Dataset

from ml_agents.config import ExperimentConfig
from ml_agents.core.experiment_runner import ExperimentRunner
from ml_agents.core.results_processor import ResultsProcessor
from ml_agents.utils.logging_config import get_logger

logger = get_logger(__name__)


class ReasoningApproachComparison:
    """Manages comparison of reasoning approaches for Phase 14."""

    def __init__(self, output_dir: Optional[Path] = None):
        """Initialize comparison framework.

        Args:
            output_dir: Directory for output files (defaults to ./outputs/phase14)
        """
        self.output_dir = output_dir or Path("outputs/phase14")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.baseline_results: Optional[Dict[str, Any]] = None
        self.approach_results: Dict[str, Dict[str, Any]] = {}

        logger.info(
            f"Initialized ReasoningApproachComparison with output_dir: {self.output_dir}"
        )

    def run_baseline(
        self,
        config: ExperimentConfig,
        dataset_id: str = "LOCAL_TEST",
        sample_size: int = 25,
    ) -> Dict[str, Any]:
        """Run baseline test with None approach.

        Args:
            config: Base experiment configuration
            dataset_id: Dataset to use
            sample_size: Number of samples to test

        Returns:
            Baseline results dictionary
        """
        logger.info(f"Running baseline test with None approach on {dataset_id}")

        # Create baseline config
        baseline_config = ExperimentConfig(
            benchmark_id=dataset_id,
            sample_count=sample_size,
            provider=config.provider,
            model=config.model,
            api_base_url=config.api_base_url,
            temperature=config.temperature,
            reasoning_approaches=["None"],
        )

        # Run experiment
        runner = ExperimentRunner(baseline_config)
        results = runner.run_single_experiment("None")

        # Extract metrics
        self.baseline_results = self._extract_metrics(results, "None")

        # Save baseline results
        baseline_path = self.output_dir / "baseline_results.json"
        with open(baseline_path, "w") as f:
            json.dump(self.baseline_results, f, indent=2)

        logger.info(f"Baseline results saved to {baseline_path}")
        return self.baseline_results

    def run_approach(
        self,
        config: ExperimentConfig,
        approach: str,
        dataset_id: str = "LOCAL_TEST",
        sample_size: int = 25,
    ) -> Dict[str, Any]:
        """Run test with specific reasoning approach.

        Args:
            config: Base experiment configuration
            approach: Reasoning approach name
            dataset_id: Dataset to use
            sample_size: Number of samples to test

        Returns:
            Approach results dictionary
        """
        logger.info(f"Running {approach} test on {dataset_id}")

        # Create approach config
        approach_config = ExperimentConfig(
            benchmark_id=dataset_id,
            sample_count=sample_size,
            provider=config.provider,
            model=config.model,
            api_base_url=config.api_base_url,
            temperature=config.temperature,
            reasoning_approaches=[approach],
        )

        # Run experiment
        runner = ExperimentRunner(approach_config)
        results = runner.run_single_experiment(approach)

        # Extract metrics
        metrics = self._extract_metrics(results, approach)
        self.approach_results[approach] = metrics

        # Save approach results
        approach_path = self.output_dir / f"{approach.lower()}_results.json"
        with open(approach_path, "w") as f:
            json.dump(metrics, f, indent=2)

        logger.info(f"{approach} results saved to {approach_path}")
        return metrics

    def compare_approaches(
        self,
        approaches: List[str],
        config: ExperimentConfig,
        dataset_id: str = "LOCAL_TEST",
        sample_size: int = 25,
    ) -> pd.DataFrame:
        """Run and compare multiple reasoning approaches.

        Args:
            approaches: List of approach names to compare
            config: Base experiment configuration
            dataset_id: Dataset to use
            sample_size: Number of samples per approach

        Returns:
            DataFrame with comparison results
        """
        logger.info(f"Comparing approaches: {approaches}")

        # Run baseline if not already done
        if not self.baseline_results:
            self.run_baseline(config, dataset_id, sample_size)

        # Run each approach
        for approach in approaches:
            if approach not in self.approach_results:
                self.run_approach(config, approach, dataset_id, sample_size)

        # Create comparison DataFrame
        comparison_data = []

        # Add baseline
        baseline_row = {
            "approach": "None (Baseline)",
            **self.baseline_results["metrics"],
            "time_overhead": 1.0,
            "token_overhead": 1.0,
        }
        comparison_data.append(baseline_row)

        # Add each approach with overhead calculations
        for approach, results in self.approach_results.items():
            # Calculate overheads with division by zero protection
            baseline_time = self.baseline_results["metrics"]["avg_response_time"]
            baseline_tokens = self.baseline_results["metrics"]["avg_tokens"]

            time_overhead = (
                results["metrics"]["avg_response_time"] / baseline_time
                if baseline_time > 0
                else 1.0
            )
            token_overhead = (
                results["metrics"]["avg_tokens"] / baseline_tokens
                if baseline_tokens > 0
                else 1.0
            )

            approach_row = {
                "approach": approach,
                **results["metrics"],
                "time_overhead": time_overhead,
                "token_overhead": token_overhead,
            }
            comparison_data.append(approach_row)

        # Create DataFrame
        df = pd.DataFrame(comparison_data)

        # Save comparison
        comparison_path = self.output_dir / "comparison_results.csv"
        df.to_csv(comparison_path, index=False)
        logger.info(f"Comparison results saved to {comparison_path}")

        # Generate summary report
        self._generate_summary_report(df)

        return df

    def _extract_metrics(self, results, approach: str) -> Dict[str, Any]:
        """Extract key metrics from experiment results.

        Args:
            results: ExperimentSummary object
            approach: Approach name

        Returns:
            Dictionary of extracted metrics
        """
        # Extract data from ExperimentSummary object
        if (
            not hasattr(results, "results_summary")
            or approach not in results.results_summary
        ):
            raise ValueError(f"No results found for approach: {approach}")

        approach_results = results.results_summary[approach]

        # Calculate metrics from the approach results
        total_samples = approach_results.get("total_samples", results.total_samples)
        correct_samples = approach_results.get("correct_samples", 0)

        # Get timing and token information
        avg_response_time = approach_results.get("avg_response_time", 0)
        avg_tokens = approach_results.get("avg_tokens", 0)
        total_tokens = approach_results.get(
            "total_tokens", avg_tokens * total_samples if avg_tokens else 0
        )
        error_count = approach_results.get("error_count", 0)

        metrics = {
            "total_samples": total_samples,
            "correct_samples": correct_samples,
            "correctness_rate": (
                correct_samples / total_samples if total_samples > 0 else 0
            ),
            "avg_response_time": avg_response_time,
            "avg_tokens": avg_tokens,
            "total_tokens": int(total_tokens),
            "error_count": error_count,
        }

        return {
            "approach": approach,
            "timestamp": datetime.now().isoformat(),
            "metrics": metrics,
            "experiment_id": results.experiment_id,
        }

    def _generate_summary_report(self, df: pd.DataFrame) -> None:
        """Generate summary report with analysis and recommendations.

        Args:
            df: Comparison DataFrame
        """
        report_lines = [
            "# Phase 14 Reasoning Approaches Comparison Report",
            f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "\n## Summary Statistics\n",
        ]

        # Summary table in proper markdown format
        report_lines.append(
            "| Approach | Total Samples | Correct | Rate | Avg Time (s) | Avg Tokens | Total Tokens | Errors | Time Overhead | Token Overhead |"
        )
        report_lines.append(
            "|----------|---------------|---------|------|--------------|------------|--------------|--------|---------------|----------------|"
        )

        for _, row in df.iterrows():
            report_lines.append(
                f"| {row['approach']} | {row['total_samples']} | {row['correct_samples']} | "
                f"{row['correctness_rate']:.1%} | {row['avg_response_time']:.2f} | "
                f"{row['avg_tokens']:.1f} | {row['total_tokens']:,} | {row['error_count']} | "
                f"{row['time_overhead']:.1f}x | {row['token_overhead']:.1f}x |"
            )

        # Analysis
        report_lines.extend(
            [
                "\n## Performance Analysis\n",
                "### Correctness Improvements",
            ]
        )

        baseline_correctness = df[df["approach"] == "None (Baseline)"][
            "correctness_rate"
        ].values[0]

        for _, row in df.iterrows():
            if row["approach"] != "None (Baseline)":
                if baseline_correctness > 0:
                    improvement = (
                        (row["correctness_rate"] - baseline_correctness)
                        / baseline_correctness
                    ) * 100
                    report_lines.append(
                        f"- **{row['approach']}**: {improvement:+.1f}% "
                        f"({row['correctness_rate']:.2f} vs {baseline_correctness:.2f})"
                    )
                else:
                    # Handle zero baseline case
                    if row["correctness_rate"] > 0:
                        report_lines.append(
                            f"- **{row['approach']}**: Improved from 0.00 to {row['correctness_rate']:.2f}"
                        )
                    else:
                        report_lines.append(
                            f"- **{row['approach']}**: No improvement (0.00 vs 0.00)"
                        )

        # Token and time analysis
        report_lines.extend(
            [
                "\n### Resource Usage",
            ]
        )

        for _, row in df.iterrows():
            if row["approach"] != "None (Baseline)":
                report_lines.append(
                    f"- **{row['approach']}**: {row['token_overhead']:.1f}x tokens, "
                    f"{row['time_overhead']:.1f}x time"
                )

        # Recommendations
        report_lines.extend(
            [
                "\n## Recommendations\n",
                "Based on the analysis:",
            ]
        )

        # Find best approach by correctness
        best_approach = df[df["approach"] != "None (Baseline)"].nlargest(
            1, "correctness_rate"
        )
        if not best_approach.empty:
            report_lines.append(
                f"- **Best correctness**: {best_approach.iloc[0]['approach']} "
                f"({best_approach.iloc[0]['correctness_rate']:.2f})"
            )

        # Find most efficient approach
        df_filtered = df[df["approach"] != "None (Baseline)"].copy()
        df_filtered["efficiency_score"] = (
            df_filtered["correctness_rate"] / df_filtered["token_overhead"]
        )
        most_efficient = df_filtered.nlargest(1, "efficiency_score")
        if not most_efficient.empty:
            report_lines.append(
                f"- **Most efficient**: {most_efficient.iloc[0]['approach']} "
                f"(efficiency score: {most_efficient.iloc[0]['efficiency_score']:.3f})"
            )

        # Write report
        report_path = self.output_dir / "comparison_report.md"
        with open(report_path, "w") as f:
            f.write("\n".join(report_lines))

        logger.info(f"Summary report saved to {report_path}")

    def generate_visualizations(self, df: pd.DataFrame) -> None:
        """Generate visualization plots for comparison.

        Args:
            df: Comparison DataFrame
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns

            # Set style
            sns.set_theme(style="whitegrid")

            # Create figure with subplots
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))

            # 1. Correctness comparison
            ax = axes[0, 0]
            df_plot = df[df["approach"] != "None (Baseline)"].copy()
            baseline_correctness = df[df["approach"] == "None (Baseline)"][
                "correctness_rate"
            ].values[0]
            ax.axhline(
                y=baseline_correctness, color="red", linestyle="--", label="Baseline"
            )
            ax.bar(df_plot["approach"], df_plot["correctness_rate"])
            ax.set_title("Correctness Rate by Approach")
            ax.set_ylabel("Correctness Rate")
            ax.set_ylim(0, 1)
            ax.legend()

            # 2. Token usage
            ax = axes[0, 1]
            ax.bar(df["approach"], df["avg_tokens"])
            ax.set_title("Average Token Usage")
            ax.set_ylabel("Tokens")
            ax.tick_params(axis="x", rotation=45)

            # 3. Response time
            ax = axes[1, 0]
            ax.bar(df["approach"], df["avg_response_time"])
            ax.set_title("Average Response Time")
            ax.set_ylabel("Time (seconds)")
            ax.tick_params(axis="x", rotation=45)

            # 4. Efficiency scatter
            ax = axes[1, 1]
            df_plot = df[df["approach"] != "None (Baseline)"].copy()
            ax.scatter(df_plot["token_overhead"], df_plot["correctness_rate"], s=100)
            for _, row in df_plot.iterrows():
                ax.annotate(
                    row["approach"],
                    (row["token_overhead"], row["correctness_rate"]),
                    xytext=(5, 5),
                    textcoords="offset points",
                    fontsize=8,
                )
            ax.set_xlabel("Token Usage Overhead")
            ax.set_ylabel("Correctness Rate")
            ax.set_title("Efficiency Trade-off")

            # Adjust layout and save
            plt.tight_layout()
            viz_path = self.output_dir / "comparison_visualizations.png"
            plt.savefig(viz_path, dpi=300)
            plt.close()

            logger.info(f"Visualizations saved to {viz_path}")

        except ImportError:
            logger.warning("Matplotlib/Seaborn not available. Skipping visualizations.")

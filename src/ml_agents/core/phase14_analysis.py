"""Phase 14 comparative analysis module for reasoning approaches."""

import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from ml_agents.core.database_manager import DatabaseManager
from ml_agents.utils.logging_config import get_logger

logger = get_logger(__name__)


class Phase14Analyzer:
    """Provides deep analysis of Phase 14 reasoning approach comparison results."""

    def __init__(self, db_path: Optional[Path] = None):
        """Initialize analyzer.

        Args:
            db_path: Path to SQLite database (uses default if None)
        """
        self.db_manager = DatabaseManager(db_path)
        logger.info("Initialized Phase 14 Analyzer")

    def load_phase14_experiments(self) -> pd.DataFrame:
        """Load all Phase 14 experiments from database.

        Returns:
            DataFrame with experiment results
        """
        query = """
        SELECT
            e.experiment_id,
            e.benchmark_id,
            e.approach,
            e.model,
            e.provider,
            e.sample_count,
            e.correct_samples,
            e.error_samples,
            e.avg_response_time,
            e.avg_tokens_used,
            e.total_cost,
            e.timestamp
        FROM experiments e
        WHERE e.benchmark_id IN ('LOCAL_TEST', 'LOCAL_TEST_LARGE')
        ORDER BY e.timestamp DESC
        """

        return pd.read_sql_query(query, self.db_manager.connection)

    def compare_approach_effectiveness(
        self, experiments_df: pd.DataFrame
    ) -> Dict[str, Any]:
        """Analyze effectiveness of reasoning approaches.

        Args:
            experiments_df: DataFrame with experiment results

        Returns:
            Dictionary with effectiveness analysis
        """
        logger.info("Analyzing approach effectiveness")

        analysis = {
            "correctness_analysis": {},
            "efficiency_analysis": {},
            "statistical_tests": {},
        }

        # Group by approach
        approach_groups = experiments_df.groupby("approach")

        # Correctness analysis
        correctness_stats = {}
        for approach, group in approach_groups:
            success_rates = group["correct_samples"] / group["sample_count"]
            correctness_stats[approach] = {
                "mean_correctness": success_rates.mean(),
                "std_correctness": success_rates.std(),
                "min_correctness": success_rates.min(),
                "max_correctness": success_rates.max(),
                "sample_count": len(group),
            }

        analysis["correctness_analysis"] = correctness_stats

        # Efficiency analysis (correctness per token)
        efficiency_stats = {}
        for approach, group in approach_groups:
            success_rates = group["correct_samples"] / group["sample_count"]
            efficiency = success_rates / group["avg_tokens_used"]
            efficiency_stats[approach] = {
                "mean_efficiency": efficiency.mean(),
                "std_efficiency": efficiency.std(),
                "tokens_per_correct": (
                    group["avg_tokens_used"] / group["correct_samples"]
                ).mean(),
            }

        analysis["efficiency_analysis"] = efficiency_stats

        # Statistical significance tests
        if "None" in correctness_stats:
            baseline_success_rates = None
            for approach, group in approach_groups:
                if approach == "None":
                    baseline_success_rates = (
                        group["correct_samples"] / group["sample_count"]
                    )
                    break

            if baseline_success_rates is not None:
                for approach, group in approach_groups:
                    if approach != "None":
                        approach_success_rates = (
                            group["correct_samples"] / group["sample_count"]
                        )

                        # Perform t-test
                        t_stat, p_value = stats.ttest_ind(
                            approach_success_rates, baseline_success_rates
                        )

                        analysis["statistical_tests"][approach] = {
                            "t_statistic": t_stat,
                            "p_value": p_value,
                            "significant": p_value < 0.05,
                        }

        return analysis

    def analyze_scaling_behavior(self, experiments_df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze how approaches scale with dataset size.

        Args:
            experiments_df: DataFrame with experiment results

        Returns:
            Dictionary with scaling analysis
        """
        logger.info("Analyzing scaling behavior")

        analysis = {}

        # Group by approach and dataset size
        for approach in experiments_df["approach"].unique():
            approach_data = experiments_df[experiments_df["approach"] == approach]

            if len(approach_data) < 2:
                continue

            # Analyze relationship between sample count and performance
            sample_counts = approach_data["sample_count"].values
            correctness_rates = (
                approach_data["correct_samples"] / approach_data["sample_count"]
            ).values
            response_times = approach_data["avg_response_time"].values
            token_usage = approach_data["avg_tokens_used"].values

            analysis[approach] = {
                "correctness_scaling": {
                    "correlation": np.corrcoef(sample_counts, correctness_rates)[0, 1],
                    "slope": np.polyfit(sample_counts, correctness_rates, 1)[0],
                },
                "time_scaling": {
                    "correlation": np.corrcoef(sample_counts, response_times)[0, 1],
                    "slope": np.polyfit(sample_counts, response_times, 1)[0],
                },
                "token_scaling": {
                    "correlation": np.corrcoef(sample_counts, token_usage)[0, 1],
                    "slope": np.polyfit(sample_counts, token_usage, 1)[0],
                },
            }

        return analysis

    def identify_optimal_approaches(
        self, experiments_df: pd.DataFrame, weights: Optional[Dict[str, float]] = None
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Identify optimal approaches based on weighted scoring.

        Args:
            experiments_df: DataFrame with experiment results
            weights: Dictionary of weights for scoring criteria

        Returns:
            List of (approach, score, details) tuples sorted by score
        """
        if weights is None:
            weights = {
                "correctness": 0.4,
                "efficiency": 0.3,
                "speed": 0.2,
                "stability": 0.1,
            }

        logger.info(f"Identifying optimal approaches with weights: {weights}")

        # Calculate metrics for each approach
        approach_scores = []

        for approach in experiments_df["approach"].unique():
            approach_data = experiments_df[experiments_df["approach"] == approach]

            if len(approach_data) == 0:
                continue

            # Calculate metrics
            correctness_rates = (
                approach_data["correct_samples"] / approach_data["sample_count"]
            )
            response_times = approach_data["avg_response_time"]
            token_usage = approach_data["avg_tokens_used"]

            # Normalize metrics (higher is better for all)
            correctness_score = correctness_rates.mean()
            efficiency_score = (correctness_rates / token_usage).mean()
            speed_score = 1 / response_times.mean()  # Inverse of time
            stability_score = 1 - correctness_rates.std()  # Lower std is better

            # Calculate weighted score
            total_score = (
                weights["correctness"] * correctness_score
                + weights["efficiency"] * efficiency_score
                + weights["speed"] * speed_score
                + weights["stability"] * stability_score
            )

            details = {
                "correctness_score": correctness_score,
                "efficiency_score": efficiency_score,
                "speed_score": speed_score,
                "stability_score": stability_score,
                "experiment_count": len(approach_data),
                "avg_correctness": correctness_rates.mean(),
                "avg_response_time": response_times.mean(),
                "avg_tokens": token_usage.mean(),
            }

            approach_scores.append((approach, total_score, details))

        # Sort by score (descending)
        approach_scores.sort(key=lambda x: x[1], reverse=True)

        return approach_scores

    def generate_recommendations(self, experiments_df: pd.DataFrame) -> Dict[str, Any]:
        """Generate recommendations based on analysis.

        Args:
            experiments_df: DataFrame with experiment results

        Returns:
            Dictionary with recommendations
        """
        logger.info("Generating recommendations")

        effectiveness_analysis = self.compare_approach_effectiveness(experiments_df)
        scaling_analysis = self.analyze_scaling_behavior(experiments_df)
        optimal_approaches = self.identify_optimal_approaches(experiments_df)

        recommendations = {
            "best_overall": None,
            "best_for_accuracy": None,
            "best_for_efficiency": None,
            "best_for_speed": None,
            "use_cases": {},
            "warnings": [],
        }

        if optimal_approaches:
            recommendations["best_overall"] = {
                "approach": optimal_approaches[0][0],
                "score": optimal_approaches[0][1],
                "details": optimal_approaches[0][2],
            }

        # Best for specific criteria
        correctness_data = effectiveness_analysis["correctness_analysis"]
        efficiency_data = effectiveness_analysis["efficiency_analysis"]

        if correctness_data:
            best_accuracy = max(
                correctness_data.items(), key=lambda x: x[1]["mean_correctness"]
            )
            recommendations["best_for_accuracy"] = {
                "approach": best_accuracy[0],
                "correctness": best_accuracy[1]["mean_correctness"],
            }

        if efficiency_data:
            best_efficiency = max(
                efficiency_data.items(), key=lambda x: x[1]["mean_efficiency"]
            )
            recommendations["best_for_efficiency"] = {
                "approach": best_efficiency[0],
                "efficiency": best_efficiency[1]["mean_efficiency"],
            }

        # Speed analysis
        speed_data = experiments_df.groupby("approach")["avg_response_time"].mean()
        if not speed_data.empty:
            fastest_approach = speed_data.idxmin()
            recommendations["best_for_speed"] = {
                "approach": fastest_approach,
                "avg_time": speed_data[fastest_approach],
            }

        # Use case recommendations
        recommendations["use_cases"] = {
            "high_accuracy_required": recommendations.get("best_for_accuracy", {}).get(
                "approach"
            ),
            "resource_constrained": recommendations.get("best_for_efficiency", {}).get(
                "approach"
            ),
            "real_time_applications": recommendations.get("best_for_speed", {}).get(
                "approach"
            ),
            "general_purpose": recommendations.get("best_overall", {}).get("approach"),
        }

        # Warnings based on analysis
        statistical_tests = effectiveness_analysis.get("statistical_tests", {})
        non_significant = [
            approach
            for approach, test in statistical_tests.items()
            if not test.get("significant", False)
        ]

        if non_significant:
            recommendations["warnings"].append(
                f"Approaches with no significant improvement over baseline: {', '.join(non_significant)}"
            )

        if len(experiments_df) < 10:
            recommendations["warnings"].append(
                "Limited experiment data - recommendations may not be reliable"
            )

        return recommendations

    def export_analysis_report(
        self, output_path: Path, experiments_df: Optional[pd.DataFrame] = None
    ) -> None:
        """Export comprehensive analysis report.

        Args:
            output_path: Path for output file
            experiments_df: DataFrame with experiments (loads from DB if None)
        """
        if experiments_df is None:
            experiments_df = self.load_phase14_experiments()

        if experiments_df.empty:
            logger.warning("No Phase 14 experiments found in database")
            return

        logger.info(f"Generating analysis report: {output_path}")

        # Run all analyses
        effectiveness = self.compare_approach_effectiveness(experiments_df)
        scaling = self.analyze_scaling_behavior(experiments_df)
        optimal = self.identify_optimal_approaches(experiments_df)
        recommendations = self.generate_recommendations(experiments_df)

        # Create comprehensive report
        report = {
            "metadata": {
                "generated_at": pd.Timestamp.now().isoformat(),
                "experiment_count": len(experiments_df),
                "approaches_tested": list(experiments_df["approach"].unique()),
                "datasets_used": list(experiments_df["benchmark_id"].unique()),
            },
            "effectiveness_analysis": effectiveness,
            "scaling_analysis": scaling,
            "optimal_approaches": [
                {"approach": approach, "score": score, "details": details}
                for approach, score, details in optimal
            ],
            "recommendations": recommendations,
            "raw_data_summary": experiments_df.describe().to_dict(),
        }

        # Save report
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2, default=str)

        logger.info(f"Analysis report saved to {output_path}")


def main():
    """Run Phase 14 analysis on existing experiment data."""
    analyzer = Phase14Analyzer()

    # Load experiments
    experiments_df = analyzer.load_phase14_experiments()

    if experiments_df.empty:
        print("No Phase 14 experiments found in database.")
        print("Run some experiments first using the test scripts.")
        return

    print(f"Found {len(experiments_df)} Phase 14 experiments")
    print(f"Approaches: {', '.join(experiments_df['approach'].unique())}")

    # Generate recommendations
    recommendations = analyzer.generate_recommendations(experiments_df)

    print("\n" + "=" * 60)
    print("PHASE 14 ANALYSIS RECOMMENDATIONS")
    print("=" * 60)

    if recommendations["best_overall"]:
        best = recommendations["best_overall"]
        print(f"Best Overall: {best['approach']} (score: {best['score']:.3f})")
        print(f"  - Accuracy: {best['details']['avg_correctness']:.2%}")
        print(f"  - Avg Time: {best['details']['avg_response_time']:.2f}s")
        print(f"  - Avg Tokens: {best['details']['avg_tokens']:.0f}")

    print("\nUse Case Recommendations:")
    for use_case, approach in recommendations["use_cases"].items():
        if approach:
            print(f"  - {use_case.replace('_', ' ').title()}: {approach}")

    if recommendations["warnings"]:
        print("\nWarnings:")
        for warning in recommendations["warnings"]:
            print(f"  - {warning}")

    # Export full report
    output_path = Path("outputs/phase14/analysis_report.json")
    analyzer.export_analysis_report(output_path, experiments_df)
    print(f"\nFull analysis report saved to: {output_path}")


if __name__ == "__main__":
    main()

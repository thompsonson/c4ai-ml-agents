#!/usr/bin/env python3
"""
ML Agents: None vs ChainOfThought Reasoning Analysis

This script performs comprehensive analysis comparing None and ChainOfThought
reasoning approaches using data from the ML Agents database.
"""

import json
import re
import sqlite3
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import chi2_contingency, ttest_ind

# Set style for better-looking plots
plt.style.use("default")
sns.set_palette("husl")


class ReasoningAnalyzer:
    """Comprehensive analyzer for None vs ChainOfThought reasoning approaches."""

    def __init__(self, db_path: str):
        """Initialize analyzer with database connection."""
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.data = None

    def load_data(self) -> pd.DataFrame:
        """Load comparative data from database."""
        query = """
        SELECT
            r.approach_name,
            r.provider,
            r.model,
            r.sample_index,
            r.input_text,
            r.expected_answer,
            r.raw_output,
            r.parsed_answer,
            r.parsing_method,
            r.parsing_confidence,
            r.is_correct,
            r.execution_time_ms,
            r.cost_estimate,
            r.created_at,
            r.metadata_json,
            e.dataset_name,
            e.created_at as experiment_date
        FROM runs r
        JOIN experiments e ON r.experiment_id = e.id
        WHERE r.approach_name IN ('None', 'ChainOfThought')
        ORDER BY r.created_at
        """

        self.data = pd.read_sql_query(query, self.conn)
        print(f"Loaded {len(self.data)} runs for analysis")
        return self.data

    def basic_statistics(self) -> Dict:
        """Generate basic statistics for both approaches."""
        stats = {}

        for approach in ["None", "ChainOfThought"]:
            approach_data = self.data[self.data["approach_name"] == approach]

            stats[approach] = {
                "total_runs": len(approach_data),
                "accuracy": approach_data["is_correct"].mean(),
                "correct_count": approach_data["is_correct"].sum(),
                "incorrect_count": (approach_data["is_correct"] == 0).sum(),
                "avg_execution_time_ms": approach_data["execution_time_ms"].mean(),
                "median_execution_time_ms": approach_data["execution_time_ms"].median(),
                "avg_cost": approach_data["cost_estimate"].mean(),
                "total_cost": approach_data["cost_estimate"].sum(),
                "parsing_methods": approach_data["parsing_method"]
                .value_counts()
                .to_dict(),
            }

        return stats

    def statistical_significance_test(self) -> Dict:
        """Perform chi-square test for accuracy difference significance."""
        # Create contingency table
        contingency_table = pd.crosstab(
            self.data["approach_name"], self.data["is_correct"]
        )

        # Perform chi-square test
        chi2, p_value, dof, expected = chi2_contingency(contingency_table)

        return {
            "contingency_table": contingency_table,
            "chi2_statistic": chi2,
            "p_value": p_value,
            "degrees_of_freedom": dof,
            "expected_frequencies": expected,
            "significant": p_value < 0.05,
        }

    def execution_time_analysis(self) -> Dict:
        """Analyze execution time differences."""
        none_times = self.data[self.data["approach_name"] == "None"][
            "execution_time_ms"
        ].dropna()
        cot_times = self.data[self.data["approach_name"] == "ChainOfThought"][
            "execution_time_ms"
        ].dropna()

        # Perform t-test
        t_stat, p_value = ttest_ind(none_times, cot_times)

        return {
            "none_mean": none_times.mean(),
            "none_median": none_times.median(),
            "none_std": none_times.std(),
            "cot_mean": cot_times.mean(),
            "cot_median": cot_times.median(),
            "cot_std": cot_times.std(),
            "t_statistic": t_stat,
            "p_value": p_value,
            "significant": p_value < 0.05,
            "effect_size_ms": cot_times.mean() - none_times.mean(),
            "effect_size_percent": (
                (cot_times.mean() - none_times.mean()) / none_times.mean()
            )
            * 100,
        }

    def categorize_tasks(self) -> pd.DataFrame:
        """Categorize tasks based on input text patterns."""

        def categorize_input(input_text: str) -> str:
            if pd.isna(input_text):
                return "unknown"

            text_lower = input_text.lower()

            # Creative writing tasks
            if any(
                keyword in text_lower
                for keyword in ["write a joke", "story", "creative", "poem"]
            ):
                return "creative_writing"

            # Financial analysis
            elif any(
                keyword in text_lower
                for keyword in ["hawkish", "dovish", "monetary policy", "central bank"]
            ):
                return "financial_analysis"

            # Content moderation
            elif any(
                keyword in text_lower
                for keyword in ["moderation", "toxicity", "obscene", "threat"]
            ):
                return "content_moderation"

            # Math problems
            elif any(
                keyword in text_lower
                for keyword in ["calculate", "solve", "math", "equation"]
            ):
                return "mathematics"

            # Question answering
            elif any(
                keyword in text_lower
                for keyword in ["what is", "how to", "explain", "define"]
            ):
                return "question_answering"

            # Reasoning/logic
            elif any(
                keyword in text_lower
                for keyword in ["logic", "reasoning", "analyze", "consider"]
            ):
                return "logical_reasoning"

            else:
                return "other"

        # Add task category column
        task_data = self.data.copy()
        task_data["task_category"] = task_data["input_text"].apply(categorize_input)

        return task_data

    def task_specific_analysis(self) -> Dict:
        """Analyze performance by task category."""
        task_data = self.categorize_tasks()

        # Group by task category and approach
        task_performance = (
            task_data.groupby(["task_category", "approach_name"])
            .agg(
                {
                    "is_correct": ["count", "sum", "mean"],
                    "execution_time_ms": "mean",
                    "cost_estimate": "mean",
                }
            )
            .round(4)
        )

        # Flatten column names
        task_performance.columns = [
            "_".join(col).strip() for col in task_performance.columns
        ]
        task_performance = task_performance.reset_index()

        # Calculate accuracy for readability
        task_performance["accuracy"] = task_performance["is_correct_mean"]

        return {
            "task_distribution": task_data["task_category"].value_counts().to_dict(),
            "performance_by_task": task_performance,
        }

    def model_comparison(self) -> Dict:
        """Compare performance across different models."""
        model_stats = (
            self.data.groupby(["provider", "model", "approach_name"])
            .agg(
                {
                    "is_correct": ["count", "sum", "mean"],
                    "execution_time_ms": "mean",
                    "cost_estimate": "mean",
                }
            )
            .round(4)
        )

        model_stats.columns = ["_".join(col).strip() for col in model_stats.columns]
        model_stats = model_stats.reset_index()
        model_stats["accuracy"] = model_stats["is_correct_mean"]

        return {
            "model_distribution": self.data.groupby(["provider", "model"])
            .size()
            .to_dict(),
            "performance_by_model": model_stats,
        }

    def temporal_analysis(self) -> Dict:
        """Analyze performance trends over time."""
        # Convert timestamps
        temporal_data = self.data.copy()
        temporal_data["created_at"] = pd.to_datetime(temporal_data["created_at"])
        temporal_data["date"] = temporal_data["created_at"].dt.date

        # Daily performance
        daily_stats = (
            temporal_data.groupby(["date", "approach_name"])
            .agg({"is_correct": ["count", "sum", "mean"], "execution_time_ms": "mean"})
            .round(4)
        )

        daily_stats.columns = ["_".join(col).strip() for col in daily_stats.columns]
        daily_stats = daily_stats.reset_index()
        daily_stats["accuracy"] = daily_stats["is_correct_mean"]

        return {
            "date_range": {
                "start": temporal_data["created_at"].min(),
                "end": temporal_data["created_at"].max(),
            },
            "daily_performance": daily_stats,
        }

    def generate_visualizations(self, output_dir: str = "./analysis_output"):
        """Generate comprehensive visualizations."""
        import os

        os.makedirs(output_dir, exist_ok=True)

        # 1. Accuracy Comparison Bar Chart
        stats = self.basic_statistics()
        approaches = list(stats.keys())
        accuracies = [stats[approach]["accuracy"] for approach in approaches]

        plt.figure(figsize=(10, 6))
        bars = plt.bar(approaches, accuracies, color=["#3498db", "#e74c3c"])
        plt.title(
            "Accuracy Comparison: None vs ChainOfThought",
            fontsize=16,
            fontweight="bold",
        )
        plt.ylabel("Accuracy Rate", fontsize=12)
        plt.ylim(0, max(accuracies) * 1.2)

        # Add value labels on bars
        for bar, acc in zip(bars, accuracies):
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{acc:.3f}",
                ha="center",
                va="bottom",
                fontweight="bold",
            )

        plt.tight_layout()
        plt.savefig(
            f"{output_dir}/accuracy_comparison.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

        # 2. Execution Time Distribution
        plt.figure(figsize=(12, 6))
        none_times = self.data[self.data["approach_name"] == "None"][
            "execution_time_ms"
        ].dropna()
        cot_times = self.data[self.data["approach_name"] == "ChainOfThought"][
            "execution_time_ms"
        ].dropna()

        plt.subplot(1, 2, 1)
        plt.hist(none_times, bins=50, alpha=0.7, label="None", color="#3498db")
        plt.hist(cot_times, bins=50, alpha=0.7, label="ChainOfThought", color="#e74c3c")
        plt.xlabel("Execution Time (ms)")
        plt.ylabel("Frequency")
        plt.title("Execution Time Distribution")
        plt.legend()
        plt.xlim(0, np.percentile(np.concatenate([none_times, cot_times]), 95))

        plt.subplot(1, 2, 2)
        plt.boxplot([none_times, cot_times], labels=["None", "ChainOfThought"])
        plt.ylabel("Execution Time (ms)")
        plt.title("Execution Time Box Plot")
        plt.yscale("log")

        plt.tight_layout()
        plt.savefig(
            f"{output_dir}/execution_time_analysis.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

        # 3. Task Category Performance
        task_analysis = self.task_specific_analysis()
        task_perf = task_analysis["performance_by_task"]

        if len(task_perf) > 0:
            # Pivot for better visualization
            pivot_data = task_perf.pivot(
                index="task_category", columns="approach_name", values="accuracy"
            )

            plt.figure(figsize=(12, 8))
            pivot_data.plot(kind="bar", width=0.8)
            plt.title("Accuracy by Task Category", fontsize=16, fontweight="bold")
            plt.ylabel("Accuracy Rate", fontsize=12)
            plt.xlabel("Task Category", fontsize=12)
            plt.xticks(rotation=45, ha="right")
            plt.legend(title="Approach")
            plt.tight_layout()
            plt.savefig(
                f"{output_dir}/task_category_performance.png",
                dpi=300,
                bbox_inches="tight",
            )
            plt.close()

    def generate_comprehensive_report(
        self, output_file: str = "none_vs_cot_analysis_report.md"
    ):
        """Generate a comprehensive markdown report."""
        stats = self.basic_statistics()
        significance = self.statistical_significance_test()
        time_analysis = self.execution_time_analysis()
        task_analysis = self.task_specific_analysis()
        model_analysis = self.model_comparison()
        temporal_analysis = self.temporal_analysis()

        report = f"""# ML Agents: None vs ChainOfThought Reasoning Analysis Report

Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Executive Summary

This analysis compares the performance of **None** (direct prompting) vs **ChainOfThought** reasoning approaches using {stats['None']['total_runs'] + stats['ChainOfThought']['total_runs']} total experimental runs from the ML Agents database.

### Key Findings

- **None approach accuracy**: {stats['None']['accuracy']:.3f} ({stats['None']['correct_count']:,} correct out of {stats['None']['total_runs']:,} runs)
- **ChainOfThought accuracy**: {stats['ChainOfThought']['accuracy']:.3f} ({stats['ChainOfThought']['correct_count']:,} correct out of {stats['ChainOfThought']['total_runs']:,} runs)
- **Performance difference**: {(stats['ChainOfThought']['accuracy'] - stats['None']['accuracy']) * 100:+.2f} percentage points
- **Statistical significance**: {'Significant' if significance['significant'] else 'Not significant'} (p={significance['p_value']:.6f})

## 1. Basic Statistics

### None Approach
- **Total runs**: {stats['None']['total_runs']:,}
- **Accuracy**: {stats['None']['accuracy']:.4f} ({stats['None']['accuracy']*100:.2f}%)
- **Correct answers**: {stats['None']['correct_count']:,}
- **Incorrect answers**: {stats['None']['incorrect_count']:,}
- **Average execution time**: {stats['None']['avg_execution_time_ms']:.1f} ms
- **Median execution time**: {stats['None']['median_execution_time_ms']:.1f} ms
- **Average cost per run**: ${stats['None']['avg_cost']:.6f}
- **Total estimated cost**: ${stats['None']['total_cost']:.4f}

### ChainOfThought Approach
- **Total runs**: {stats['ChainOfThought']['total_runs']:,}
- **Accuracy**: {stats['ChainOfThought']['accuracy']:.4f} ({stats['ChainOfThought']['accuracy']*100:.2f}%)
- **Correct answers**: {stats['ChainOfThought']['correct_count']:,}
- **Incorrect answers**: {stats['ChainOfThought']['incorrect_count']:,}
- **Average execution time**: {stats['ChainOfThought']['avg_execution_time_ms']:.1f} ms
- **Median execution time**: {stats['ChainOfThought']['median_execution_time_ms']:.1f} ms
- **Average cost per run**: ${stats['ChainOfThought']['avg_cost']:.6f}
- **Total estimated cost**: ${stats['ChainOfThought']['total_cost']:.4f}

## 2. Statistical Significance Analysis

**Chi-Square Test Results:**
- **Chi-square statistic**: {significance['chi2_statistic']:.4f}
- **P-value**: {significance['p_value']:.6f}
- **Degrees of freedom**: {significance['degrees_of_freedom']}
- **Result**: {'The accuracy difference is statistically significant' if significance['significant'] else 'The accuracy difference is not statistically significant'} (α = 0.05)

**Contingency Table:**
```
{significance['contingency_table'].to_string()}
```

## 3. Execution Time Analysis

**Performance Impact:**
- **None mean time**: {time_analysis['none_mean']:.1f} ms
- **ChainOfThought mean time**: {time_analysis['cot_mean']:.1f} ms
- **Difference**: {time_analysis['effect_size_ms']:+.1f} ms ({time_analysis['effect_size_percent']:+.1f}% {"slower" if time_analysis['effect_size_ms'] > 0 else "faster"})
- **Statistical significance**: {'Significant' if time_analysis['significant'] else 'Not significant'} (p={time_analysis['p_value']:.6f})

## 4. Task Category Analysis

**Task Distribution:**
"""

        # Add task distribution
        for task, count in task_analysis["task_distribution"].items():
            report += f"- **{task.replace('_', ' ').title()}**: {count:,} runs\n"

        report += f"""
**Performance by Task Category:**

{task_analysis['performance_by_task'].to_string(index=False) if len(task_analysis['performance_by_task']) > 0 else 'No task categorization data available'}

## 5. Model Comparison

**Model Distribution:**
"""

        # Add model distribution
        for (provider, model), count in model_analysis["model_distribution"].items():
            report += f"- **{provider}/{model}**: {count:,} runs\n"

        report += f"""
**Performance by Model:**

{model_analysis['performance_by_model'].to_string(index=False)}

## 6. Recommendations

Based on this analysis:

### Performance Recommendations
"""

        if stats["None"]["accuracy"] > stats["ChainOfThought"]["accuracy"]:
            report += f"""
1. **Direct prompting (None) performs better** with {stats['None']['accuracy']*100:.2f}% accuracy vs {stats['ChainOfThought']['accuracy']*100:.2f}%
2. **Faster execution**: None approach is {abs(time_analysis['effect_size_percent']):.1f}% faster
3. **Cost efficiency**: None approach may be more cost-effective for these task types
"""
        else:
            report += f"""
1. **ChainOfThought reasoning shows benefits** with {stats['ChainOfThought']['accuracy']*100:.2f}% accuracy vs {stats['None']['accuracy']*100:.2f}%
2. **Time tradeoff**: ChainOfThought takes {time_analysis['effect_size_percent']:.1f}% more time but may provide better results
3. **Consider task complexity**: More complex tasks may benefit more from structured reasoning
"""

        report += f"""
### Cost-Benefit Analysis
- **Time overhead**: ChainOfThought adds ~{time_analysis['effect_size_ms']:.0f}ms per query
- **Accuracy gain/loss**: {(stats['ChainOfThought']['accuracy'] - stats['None']['accuracy']) * 100:+.2f} percentage points
- **Cost efficiency**: {'None approach is more cost-effective' if stats['None']['avg_cost'] < stats['ChainOfThought']['avg_cost'] else 'ChainOfThought provides better value per correct answer'}

### Next Steps
1. Analyze specific failure cases to understand reasoning limitations
2. Test approaches on more complex reasoning tasks
3. Consider task-specific approach selection
4. Optimize prompt engineering for both approaches

---
*Analysis conducted using ML Agents reasoning evaluation framework*
*Database: {self.db_path}*
"""

        # Write report
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(report)

        return report


def main():
    """Main analysis execution."""
    print("🔍 Starting None vs ChainOfThought Analysis...")

    # Initialize analyzer
    analyzer = ReasoningAnalyzer(
        "/Users/mthompson/Projects/c4ai/ml-agents/ml_agents_results.db"
    )

    # Load data
    print("📊 Loading experimental data...")
    analyzer.load_data()

    # Generate analysis
    print("📈 Generating visualizations...")
    analyzer.generate_visualizations()

    print("📝 Creating comprehensive report...")
    report = analyzer.generate_comprehensive_report()

    print("✅ Analysis complete!")
    print("\nGenerated files:")
    print("- none_vs_cot_analysis_report.md")
    print("- analysis_output/accuracy_comparison.png")
    print("- analysis_output/execution_time_analysis.png")
    print("- analysis_output/task_category_performance.png")

    # Print key findings
    stats = analyzer.basic_statistics()
    print(f"\n🎯 Key Findings:")
    print(f"   • None accuracy: {stats['None']['accuracy']:.3f}")
    print(f"   • ChainOfThought accuracy: {stats['ChainOfThought']['accuracy']:.3f}")
    print(
        f"   • Difference: {(stats['ChainOfThought']['accuracy'] - stats['None']['accuracy']) * 100:+.2f} pp"
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Generate comprehensive Phase 14 report with visualizations."""

import argparse
import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ml_agents.core.phase14_analysis import Phase14Analyzer
from ml_agents.utils.logging_config import get_logger, setup_logging


def generate_markdown_report(analysis_data: dict, output_path: Path) -> None:
    """Generate markdown report from analysis data.

    Args:
        analysis_data: Analysis data dictionary
        output_path: Path for markdown report
    """
    logger = get_logger(__name__)
    logger.info(f"Generating markdown report: {output_path}")

    report_lines = [
        "# Phase 14 Reasoning Approaches Comparison Report",
        "",
        f"**Generated:** {analysis_data['metadata']['generated_at']}",
        f"**Experiments Analyzed:** {analysis_data['metadata']['experiment_count']}",
        f"**Approaches Tested:** {', '.join(analysis_data['metadata']['approaches_tested'])}",
        f"**Datasets Used:** {', '.join(analysis_data['metadata']['datasets_used'])}",
        "",
        "## Executive Summary",
        "",
    ]

    recommendations = analysis_data.get("recommendations", {})

    if recommendations.get("best_overall"):
        best = recommendations["best_overall"]
        report_lines.extend(
            [
                f"**Best Overall Approach:** {best['approach']}",
                f"- Overall Score: {best['score']:.3f}",
                f"- Average Correctness: {best['details']['avg_correctness']:.1%}",
                f"- Average Response Time: {best['details']['avg_response_time']:.2f}s",
                f"- Average Token Usage: {best['details']['avg_tokens']:.0f}",
                "",
            ]
        )

    # Use case recommendations
    use_cases = recommendations.get("use_cases", {})
    if any(use_cases.values()):
        report_lines.extend(
            [
                "### Recommended Use Cases",
                "",
            ]
        )

        for use_case, approach in use_cases.items():
            if approach:
                formatted_use_case = use_case.replace("_", " ").title()
                report_lines.append(f"- **{formatted_use_case}:** {approach}")

        report_lines.append("")

    # Warnings
    warnings = recommendations.get("warnings", [])
    if warnings:
        report_lines.extend(
            [
                "### ⚠️ Important Notes",
                "",
            ]
        )
        for warning in warnings:
            report_lines.append(f"- {warning}")
        report_lines.append("")

    # Detailed Analysis
    report_lines.extend(
        [
            "## Detailed Analysis",
            "",
            "### Correctness Analysis",
            "",
        ]
    )

    correctness_analysis = analysis_data.get("effectiveness_analysis", {}).get(
        "correctness_analysis", {}
    )
    if correctness_analysis:
        for approach, stats in correctness_analysis.items():
            report_lines.extend(
                [
                    f"**{approach}:**",
                    f"- Mean Correctness: {stats['mean_correctness']:.1%}",
                    f"- Standard Deviation: {stats['std_correctness']:.3f}",
                    f"- Range: {stats['min_correctness']:.1%} - {stats['max_correctness']:.1%}",
                    f"- Sample Count: {stats['sample_count']}",
                    "",
                ]
            )

    # Efficiency Analysis
    report_lines.extend(
        [
            "### Efficiency Analysis",
            "",
        ]
    )

    efficiency_analysis = analysis_data.get("effectiveness_analysis", {}).get(
        "efficiency_analysis", {}
    )
    if efficiency_analysis:
        for approach, stats in efficiency_analysis.items():
            report_lines.extend(
                [
                    f"**{approach}:**",
                    f"- Mean Efficiency: {stats['mean_efficiency']:.6f} correct/token",
                    f"- Tokens per Correct Answer: {stats['tokens_per_correct']:.1f}",
                    "",
                ]
            )

    # Statistical Tests
    statistical_tests = analysis_data.get("effectiveness_analysis", {}).get(
        "statistical_tests", {}
    )
    if statistical_tests:
        report_lines.extend(
            [
                "### Statistical Significance Tests",
                "",
                "Comparison against baseline (None approach):",
                "",
            ]
        )

        for approach, test in statistical_tests.items():
            significance = (
                "✓ Significant"
                if test.get("significant", False)
                else "✗ Not Significant"
            )
            report_lines.extend(
                [
                    f"**{approach}:**",
                    f"- t-statistic: {test.get('t_statistic', 0):.3f}",
                    f"- p-value: {test.get('p_value', 1):.3f}",
                    f"- Result: {significance}",
                    "",
                ]
            )

    # Scaling Analysis
    scaling_analysis = analysis_data.get("scaling_analysis", {})
    if scaling_analysis:
        report_lines.extend(
            [
                "### Scaling Behavior",
                "",
            ]
        )

        for approach, scaling_data in scaling_analysis.items():
            correctness_scaling = scaling_data.get("correctness_scaling", {})
            time_scaling = scaling_data.get("time_scaling", {})

            report_lines.extend(
                [
                    f"**{approach}:**",
                    f"- Correctness vs Size Correlation: {correctness_scaling.get('correlation', 0):.3f}",
                    f"- Response Time vs Size Correlation: {time_scaling.get('correlation', 0):.3f}",
                    "",
                ]
            )

    # Optimal Approaches Ranking
    optimal_approaches = analysis_data.get("optimal_approaches", [])
    if optimal_approaches:
        report_lines.extend(
            [
                "### Approach Ranking (Overall Score)",
                "",
                "| Rank | Approach | Score | Correctness | Avg Time | Avg Tokens |",
                "|------|----------|-------|-------------|----------|------------|",
            ]
        )

        for i, approach_data in enumerate(optimal_approaches, 1):
            approach = approach_data["approach"]
            score = approach_data["score"]
            details = approach_data["details"]

            report_lines.append(
                f"| {i} | {approach} | {score:.3f} | {details['avg_correctness']:.1%} | "
                f"{details['avg_response_time']:.2f}s | {details['avg_tokens']:.0f} |"
            )

        report_lines.append("")

    # Raw Data Summary
    report_lines.extend(
        [
            "## Appendix: Raw Data Summary",
            "",
            "```",
            json.dumps(analysis_data.get("raw_data_summary", {}), indent=2),
            "```",
        ]
    )

    # Write report
    with open(output_path, "w") as f:
        f.write("\n".join(report_lines))

    logger.info(f"Markdown report saved to {output_path}")


def generate_csv_summary(analysis_data: dict, output_path: Path) -> None:
    """Generate CSV summary of results.

    Args:
        analysis_data: Analysis data dictionary
        output_path: Path for CSV file
    """
    logger = get_logger(__name__)
    logger.info(f"Generating CSV summary: {output_path}")

    try:
        import pandas as pd

        # Extract approach data
        optimal_approaches = analysis_data.get("optimal_approaches", [])
        correctness_data = analysis_data.get("effectiveness_analysis", {}).get(
            "correctness_analysis", {}
        )
        efficiency_data = analysis_data.get("effectiveness_analysis", {}).get(
            "efficiency_analysis", {}
        )

        # Create summary data
        summary_data = []

        for approach_info in optimal_approaches:
            approach = approach_info["approach"]
            score = approach_info["score"]
            details = approach_info["details"]

            # Get additional data if available
            correctness_stats = correctness_data.get(approach, {})
            efficiency_stats = efficiency_data.get(approach, {})

            summary_data.append(
                {
                    "approach": approach,
                    "overall_score": score,
                    "avg_correctness": details["avg_correctness"],
                    "correctness_std": correctness_stats.get("std_correctness", 0),
                    "avg_response_time": details["avg_response_time"],
                    "avg_tokens": details["avg_tokens"],
                    "efficiency": efficiency_stats.get("mean_efficiency", 0),
                    "tokens_per_correct": efficiency_stats.get("tokens_per_correct", 0),
                    "experiment_count": details["experiment_count"],
                }
            )

        # Create DataFrame and save
        df = pd.DataFrame(summary_data)
        df.to_csv(output_path, index=False)

        logger.info(f"CSV summary saved to {output_path}")

    except ImportError:
        logger.warning("Pandas not available, skipping CSV generation")


def generate_html_report(analysis_data: dict, output_path: Path) -> None:
    """Generate HTML report with embedded visualizations.

    Args:
        analysis_data: Analysis data dictionary
        output_path: Path for HTML file
    """
    logger = get_logger(__name__)
    logger.info(f"Generating HTML report: {output_path}")

    try:
        import base64
        from io import BytesIO

        import matplotlib.pyplot as plt

        # Generate plots
        plots = []

        # Correctness comparison plot
        correctness_data = analysis_data.get("effectiveness_analysis", {}).get(
            "correctness_analysis", {}
        )
        if correctness_data:
            approaches = list(correctness_data.keys())
            correctness_values = [
                correctness_data[a]["mean_correctness"] for a in approaches
            ]

            plt.figure(figsize=(10, 6))
            bars = plt.bar(approaches, correctness_values)
            plt.title("Correctness Rate by Reasoning Approach")
            plt.ylabel("Correctness Rate")
            plt.ylim(0, 1)
            plt.xticks(rotation=45)

            # Color bars based on performance
            for i, bar in enumerate(bars):
                if correctness_values[i] > 0.8:
                    bar.set_color("green")
                elif correctness_values[i] > 0.6:
                    bar.set_color("orange")
                else:
                    bar.set_color("red")

            # Save plot to base64
            buffer = BytesIO()
            plt.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
            buffer.seek(0)
            plot_data = base64.b64encode(buffer.getvalue()).decode()
            plots.append(("Correctness Comparison", plot_data))
            plt.close()

        # Create HTML content
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Phase 14 Reasoning Approaches Comparison Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        .header {{ background-color: #f5f5f5; padding: 20px; border-radius: 5px; }}
        .section {{ margin: 30px 0; }}
        .plot {{ text-align: center; margin: 20px 0; }}
        .plot img {{ max-width: 100%; height: auto; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
        .warning {{ background-color: #fff3cd; border-left: 4px solid #ffc107; padding: 10px; }}
        .success {{ background-color: #d4edda; border-left: 4px solid #28a745; padding: 10px; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Phase 14 Reasoning Approaches Comparison Report</h1>
        <p><strong>Generated:</strong> {analysis_data['metadata']['generated_at']}</p>
        <p><strong>Experiments:</strong> {analysis_data['metadata']['experiment_count']}</p>
        <p><strong>Approaches:</strong> {', '.join(analysis_data['metadata']['approaches_tested'])}</p>
    </div>
"""

        # Add best approach section
        recommendations = analysis_data.get("recommendations", {})
        if recommendations.get("best_overall"):
            best = recommendations["best_overall"]
            html_content += f"""
    <div class="section success">
        <h2>🏆 Best Overall Approach</h2>
        <h3>{best['approach']}</h3>
        <ul>
            <li>Overall Score: {best['score']:.3f}</li>
            <li>Average Correctness: {best['details']['avg_correctness']:.1%}</li>
            <li>Average Response Time: {best['details']['avg_response_time']:.2f}s</li>
            <li>Average Token Usage: {best['details']['avg_tokens']:.0f}</li>
        </ul>
    </div>
"""

        # Add warnings
        warnings = recommendations.get("warnings", [])
        if warnings:
            html_content += (
                '<div class="section warning"><h2>⚠️ Important Notes</h2><ul>'
            )
            for warning in warnings:
                html_content += f"<li>{warning}</li>"
            html_content += "</ul></div>"

        # Add visualizations
        if plots:
            html_content += '<div class="section"><h2>📊 Visualizations</h2>'
            for plot_title, plot_data in plots:
                html_content += f"""
    <div class="plot">
        <h3>{plot_title}</h3>
        <img src="data:image/png;base64,{plot_data}" alt="{plot_title}">
    </div>
"""
            html_content += "</div>"

        # Add ranking table
        optimal_approaches = analysis_data.get("optimal_approaches", [])
        if optimal_approaches:
            html_content += """
    <div class="section">
        <h2>📈 Approach Rankings</h2>
        <table>
            <tr>
                <th>Rank</th>
                <th>Approach</th>
                <th>Overall Score</th>
                <th>Correctness</th>
                <th>Avg Time (s)</th>
                <th>Avg Tokens</th>
            </tr>
"""

            for i, approach_data in enumerate(optimal_approaches, 1):
                approach = approach_data["approach"]
                score = approach_data["score"]
                details = approach_data["details"]

                html_content += f"""
            <tr>
                <td>{i}</td>
                <td><strong>{approach}</strong></td>
                <td>{score:.3f}</td>
                <td>{details['avg_correctness']:.1%}</td>
                <td>{details['avg_response_time']:.2f}</td>
                <td>{details['avg_tokens']:.0f}</td>
            </tr>
"""

            html_content += "</table></div>"

        html_content += """
</body>
</html>
"""

        # Write HTML file
        with open(output_path, "w") as f:
            f.write(html_content)

        logger.info(f"HTML report saved to {output_path}")

    except ImportError as e:
        logger.warning(f"Required packages not available for HTML generation: {e}")


def main():
    """Generate comprehensive Phase 14 report."""
    parser = argparse.ArgumentParser(description="Generate Phase 14 comparison report")

    parser.add_argument(
        "--output-dir",
        default="outputs/phase14/reports",
        help="Output directory for reports (default: outputs/phase14/reports)",
    )

    parser.add_argument(
        "--format",
        nargs="+",
        choices=["markdown", "html", "csv", "json"],
        default=["markdown", "html", "csv"],
        help="Report formats to generate (default: markdown html csv)",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    # Setup logging
    level = "DEBUG" if args.verbose else "ERROR"
    setup_logging(level=level)
    logger = get_logger(__name__)

    logger.info("Starting Phase 14 report generation")

    # Create analyzer and load data
    analyzer = Phase14Analyzer()
    experiments_df = analyzer.load_phase14_experiments()

    if experiments_df.empty:
        logger.error("No Phase 14 experiments found in database")
        logger.info("Run experiments first using the test scripts")
        sys.exit(1)

    logger.info(f"Found {len(experiments_df)} experiments")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate analysis
    logger.info("Performing analysis...")
    analysis_path = output_dir / "analysis_data.json"
    analyzer.export_analysis_report(analysis_path, experiments_df)

    # Load analysis data
    with open(analysis_path, "r") as f:
        analysis_data = json.load(f)

    # Generate requested formats
    if "markdown" in args.format:
        generate_markdown_report(analysis_data, output_dir / "phase14_report.md")

    if "html" in args.format:
        generate_html_report(analysis_data, output_dir / "phase14_report.html")

    if "csv" in args.format:
        generate_csv_summary(analysis_data, output_dir / "phase14_summary.csv")

    if "json" in args.format:
        # JSON already generated by analyzer
        logger.info(f"JSON report available at: {analysis_path}")

    logger.info(f"Reports generated in: {output_dir}")
    logger.info("Phase 14 report generation completed successfully!")


if __name__ == "__main__":
    main()

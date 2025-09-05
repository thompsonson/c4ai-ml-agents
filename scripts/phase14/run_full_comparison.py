#!/usr/bin/env python3
"""Run comprehensive comparison of all reasoning approaches for Phase 14."""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from base_test import setup_local_api_config

from ml_agents.config import ExperimentConfig
from ml_agents.core.phase14_comparison import ReasoningApproachComparison
from ml_agents.utils.logging_config import get_logger, setup_logging


def main():
    """Run full comparison of reasoning approaches."""
    parser = argparse.ArgumentParser(
        description="Run comprehensive comparison of reasoning approaches for Phase 14"
    )

    parser.add_argument(
        "--approaches",
        nargs="+",
        default=["ChainOfThought", "TreeOfThought", "AsPlanning"],
        help="Approaches to test (default: ChainOfThought TreeOfThought AsPlanning)",
    )

    parser.add_argument(
        "--dataset",
        default="LOCAL_TEST",
        help="Dataset to use (default: LOCAL_TEST)",
    )

    parser.add_argument(
        "--samples",
        type=int,
        default=25,
        help="Number of samples per approach (default: 25)",
    )

    parser.add_argument(
        "--api-base",
        help="API base URL (default: http://pop-os:8000/v1)",
    )

    parser.add_argument(
        "--model",
        help="Model to use (default: Qwen/Qwen2.5-1.5B-Instruct)",
    )

    parser.add_argument(
        "--output-dir",
        default="outputs/phase14/full_comparison",
        help="Output directory (default: outputs/phase14/full_comparison)",
    )

    parser.add_argument(
        "--generate-viz",
        action="store_true",
        help="Generate visualization plots",
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

    logger.info("Starting Phase 14 comprehensive comparison")
    logger.info(f"Approaches: {', '.join(args.approaches)}")
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Samples per approach: {args.samples}")

    # Get configuration
    config_params = setup_local_api_config()

    # Override with command line arguments
    if args.api_base:
        config_params["api_base_url"] = args.api_base
    if args.model:
        config_params["model"] = args.model

    # Create experiment config
    config = ExperimentConfig(
        benchmark_id=args.dataset,
        sample_count=args.samples,
        provider=config_params["provider"],
        model=config_params["model"],
        api_base_url=config_params["api_base_url"],
        temperature=config_params["temperature"],
        max_reasoning_calls=20,  # Increased from default 5 to 20
    )

    # Create comparison framework
    comparison = ReasoningApproachComparison(output_dir=Path(args.output_dir))

    try:
        # Run comparison
        logger.info("Running comparison across all approaches...")
        results_df = comparison.compare_approaches(
            args.approaches, config, args.dataset, args.samples
        )

        # Display results
        logger.info(f"\n{'='*80}")
        logger.info("PHASE 14 COMPARISON RESULTS")
        logger.info(f"{'='*80}")
        logger.info("\n" + results_df.to_string(index=False))

        # Generate visualizations if requested
        if args.generate_viz:
            logger.info("\nGenerating visualizations...")
            comparison.generate_visualizations(results_df)

        # Summary insights
        logger.info(f"\n{'='*80}")
        logger.info("KEY INSIGHTS")
        logger.info(f"{'='*80}")

        baseline_correctness = results_df[results_df["approach"] == "None (Baseline)"][
            "correctness_rate"
        ].values[0]
        approach_results = results_df[results_df["approach"] != "None (Baseline)"]

        if not approach_results.empty:
            best_accuracy = approach_results.nlargest(1, "correctness_rate").iloc[0]
            logger.info(
                f"• Best accuracy: {best_accuracy['approach']} ({best_accuracy['correctness_rate']:.2%})"
            )

            most_efficient = approach_results.copy()
            most_efficient["efficiency"] = (
                most_efficient["correctness_rate"] / most_efficient["token_overhead"]
            )
            best_efficiency = most_efficient.nlargest(1, "efficiency").iloc[0]
            logger.info(
                f"• Most efficient: {best_efficiency['approach']} (efficiency: {best_efficiency['efficiency']:.3f})"
            )

            fastest = approach_results.nsmallest(1, "time_overhead").iloc[0]
            logger.info(
                f"• Fastest: {fastest['approach']} ({fastest['time_overhead']:.1f}x baseline time)"
            )

            # Check if any approach meets success criteria
            successful_approaches = approach_results[
                (approach_results["correctness_rate"] - baseline_correctness)
                / baseline_correctness
                > 0.25
            ]

            if not successful_approaches.empty:
                logger.info(
                    f"• Approaches meeting >25% improvement criteria: {len(successful_approaches)}"
                )
                for _, row in successful_approaches.iterrows():
                    improvement = (
                        (row["correctness_rate"] - baseline_correctness)
                        / baseline_correctness
                    ) * 100
                    logger.info(
                        f"  - {row['approach']}: {improvement:.1f}% improvement"
                    )
            else:
                logger.warning(
                    "• No approaches met the >25% improvement success criteria"
                )

        logger.info(f"\nDetailed results saved to: {args.output_dir}")
        logger.info("Phase 14 comparison completed successfully!")

        return results_df

    except Exception as e:
        logger.error(f"Error during comparison: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()

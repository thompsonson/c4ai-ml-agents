#!/usr/bin/env python3
"""Base script for Phase 14 reasoning approach testing."""

import argparse
import json
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from ml_agents.config import ExperimentConfig
from ml_agents.core.phase14_comparison import ReasoningApproachComparison
from ml_agents.utils.logging_config import get_logger, setup_logging


def setup_local_api_config():
    """Set up configuration for local API testing."""
    return {
        "provider": "local-openai",
        "model": "Qwen/Qwen2.5-1.5B-Instruct",
        "api_base_url": os.getenv("ML_AGENTS_API_BASE_URL", "http://pop-os:8000/v1"),
        "api_key": "",  # Not required for local
        "temperature": 0.3,
    }


def run_approach_test(approach_name: str, args):
    """Run test for specific reasoning approach.

    Args:
        approach_name: Name of the reasoning approach
        args: Command line arguments
    """
    # Setup logging
    level = "DEBUG" if args.verbose else "ERROR"
    setup_logging(level=level)
    logger = get_logger(__name__)

    logger.info(f"Starting Phase 14 test for {approach_name}")

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
    output_dir = Path(args.output_dir) / approach_name.lower()
    comparison = ReasoningApproachComparison(output_dir=output_dir)

    # Run baseline if requested
    if args.run_baseline:
        logger.info("Running baseline test with None approach")
        baseline_results = comparison.run_baseline(config, args.dataset, args.samples)
        logger.info(
            f"Baseline correctness: {baseline_results['metrics']['correctness_rate']:.2%}"
        )
        logger.info(
            f"Baseline avg tokens: {baseline_results['metrics']['avg_tokens']:.1f}"
        )
        logger.info(
            f"Baseline avg time: {baseline_results['metrics']['avg_response_time']:.2f}s"
        )

    # Run approach test
    logger.info(f"Running {approach_name} approach test")
    results = comparison.run_approach(config, approach_name, args.dataset, args.samples)

    # Display results
    logger.info(f"\n{'='*60}")
    logger.info(f"{approach_name} Test Results:")
    logger.info(f"{'='*60}")
    logger.info(f"Correctness Rate: {results['metrics']['correctness_rate']:.2%}")
    logger.info(
        f"Average Response Time: {results['metrics']['avg_response_time']:.2f}s"
    )
    logger.info(f"Average Token Usage: {results['metrics']['avg_tokens']:.1f}")
    logger.info(f"Total Tokens Used: {results['metrics']['total_tokens']:,}")
    logger.info(f"Error Count: {results['metrics']['error_count']}")

    # Compare to baseline if available
    if comparison.baseline_results:
        baseline_metrics = comparison.baseline_results["metrics"]
        correctness_improvement = (
            (
                results["metrics"]["correctness_rate"]
                - baseline_metrics["correctness_rate"]
            )
            / baseline_metrics["correctness_rate"]
            * 100
        )
        token_overhead = (
            results["metrics"]["avg_tokens"] / baseline_metrics["avg_tokens"]
        )
        time_overhead = (
            results["metrics"]["avg_response_time"]
            / baseline_metrics["avg_response_time"]
        )

        logger.info(f"\nComparison to Baseline:")
        logger.info(f"Correctness Improvement: {correctness_improvement:+.1f}%")
        logger.info(f"Token Usage Overhead: {token_overhead:.1f}x")
        logger.info(f"Response Time Overhead: {time_overhead:.1f}x")

    logger.info(f"\nResults saved to: {output_dir}")

    return results


def create_argument_parser(description: str):
    """Create argument parser for approach test scripts."""
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument(
        "--dataset",
        default="LOCAL_TEST",
        help="Dataset to use (default: LOCAL_TEST)",
    )

    parser.add_argument(
        "--samples",
        type=int,
        default=25,
        help="Number of samples to test (default: 25)",
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
        default="outputs/phase14",
        help="Output directory (default: outputs/phase14)",
    )

    parser.add_argument(
        "--run-baseline",
        action="store_true",
        help="Run baseline test before approach test",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    return parser


if __name__ == "__main__":
    # This script should not be run directly
    print("This is a base script. Run one of the specific approach test scripts.")
    sys.exit(1)

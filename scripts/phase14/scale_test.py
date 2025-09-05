#!/usr/bin/env python3
"""Scale testing script for Phase 14 reasoning approaches."""

import argparse
import resource
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from base_test import setup_local_api_config

from ml_agents.config import ExperimentConfig
from ml_agents.core.phase14_comparison import ReasoningApproachComparison
from ml_agents.utils.logging_config import get_logger, setup_logging


def get_memory_usage():
    """Get current memory usage in MB."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def run_scale_test(
    approach: str,
    dataset_sizes: list[int],
    config: ExperimentConfig,
    output_dir: Path,
) -> dict:
    """Run scale test for specific approach across different dataset sizes.

    Args:
        approach: Reasoning approach to test
        dataset_sizes: List of dataset sizes to test
        config: Base experiment configuration
        output_dir: Output directory for results

    Returns:
        Dictionary with scale test results
    """
    logger = get_logger(__name__)
    logger.info(f"Running scale test for {approach}")

    comparison = ReasoningApproachComparison(output_dir=output_dir)
    scale_results = {
        "approach": approach,
        "dataset_sizes": dataset_sizes,
        "results": [],
        "performance_metrics": {
            "memory_usage": [],
            "processing_time": [],
            "throughput": [],
        },
    }

    for size in dataset_sizes:
        logger.info(f"Testing {approach} with {size} samples")

        # Record starting memory
        start_memory = get_memory_usage()
        start_time = time.time()

        # Create config for this size
        size_config = ExperimentConfig(
            benchmark_id="LOCAL_TEST_LARGE" if size > 100 else "LOCAL_TEST",
            sample_count=size,
            provider=config.provider,
            model=config.model,
            api_base_url=config.api_base_url,
            temperature=config.temperature,
            max_reasoning_calls=20,  # Increased from default 5 to 20
        )

        try:
            # Run test
            results = comparison.run_approach(
                size_config, approach, size_config.benchmark_id, size
            )

            # Record ending metrics
            end_time = time.time()
            end_memory = get_memory_usage()

            processing_time = end_time - start_time
            memory_usage = end_memory - start_memory
            throughput = size / processing_time if processing_time > 0 else 0

            scale_results["results"].append(
                {
                    "size": size,
                    "correctness_rate": results["metrics"]["correctness_rate"],
                    "avg_response_time": results["metrics"]["avg_response_time"],
                    "avg_tokens": results["metrics"]["avg_tokens"],
                    "total_tokens": results["metrics"]["total_tokens"],
                    "error_count": results["metrics"]["error_count"],
                    "processing_time": processing_time,
                    "memory_usage_mb": memory_usage,
                    "throughput_samples_per_sec": throughput,
                }
            )

            scale_results["performance_metrics"]["memory_usage"].append(memory_usage)
            scale_results["performance_metrics"]["processing_time"].append(
                processing_time
            )
            scale_results["performance_metrics"]["throughput"].append(throughput)

            logger.info(
                f"Size {size}: {processing_time:.1f}s, {memory_usage:.1f}MB, "
                f"{throughput:.2f} samples/sec"
            )

        except Exception as e:
            logger.error(f"Error testing size {size}: {str(e)}")
            scale_results["results"].append(
                {
                    "size": size,
                    "error": str(e),
                    "processing_time": time.time() - start_time,
                    "memory_usage_mb": get_memory_usage() - start_memory,
                }
            )

    return scale_results


def analyze_scaling_performance(scale_results: dict) -> dict:
    """Analyze scaling performance characteristics.

    Args:
        scale_results: Results from scale test

    Returns:
        Dictionary with performance analysis
    """
    logger = get_logger(__name__)
    logger.info("Analyzing scaling performance")

    successful_results = [r for r in scale_results["results"] if "error" not in r]

    if len(successful_results) < 2:
        return {"error": "Insufficient successful results for analysis"}

    sizes = [r["size"] for r in successful_results]
    times = [r["processing_time"] for r in successful_results]
    memory_usage = [r["memory_usage_mb"] for r in successful_results]
    throughput = [r["throughput_samples_per_sec"] for r in successful_results]
    correctness = [r["correctness_rate"] for r in successful_results]

    analysis = {
        "linear_scaling": {
            "time_vs_size": {
                "correlation": 0,
                "slope": 0,
                "is_linear": False,
            },
            "memory_vs_size": {
                "correlation": 0,
                "slope": 0,
                "is_linear": False,
            },
        },
        "performance_degradation": {
            "throughput_decline": (
                (throughput[0] - throughput[-1]) / throughput[0]
                if throughput[0] > 0
                else 0
            ),
            "correctness_stability": max(correctness) - min(correctness),
        },
        "resource_efficiency": {
            "memory_per_sample": [m / s for m, s in zip(memory_usage, sizes)],
            "time_per_sample": [t / s for t, s in zip(times, sizes)],
        },
        "scalability_score": 0,
    }

    # Calculate correlations and slopes
    try:
        import numpy as np

        # Time vs size analysis
        time_corr = np.corrcoef(sizes, times)[0, 1]
        time_slope = np.polyfit(sizes, times, 1)[0]
        analysis["linear_scaling"]["time_vs_size"]["correlation"] = time_corr
        analysis["linear_scaling"]["time_vs_size"]["slope"] = time_slope
        analysis["linear_scaling"]["time_vs_size"]["is_linear"] = abs(time_corr) > 0.8

        # Memory vs size analysis
        mem_corr = np.corrcoef(sizes, memory_usage)[0, 1]
        mem_slope = np.polyfit(sizes, memory_usage, 1)[0]
        analysis["linear_scaling"]["memory_vs_size"]["correlation"] = mem_corr
        analysis["linear_scaling"]["memory_vs_size"]["slope"] = mem_slope
        analysis["linear_scaling"]["memory_vs_size"]["is_linear"] = abs(mem_corr) > 0.8

        # Calculate scalability score (0-1, higher is better)
        # Based on linear scaling, low degradation, and efficiency
        time_score = 1 - min(1, abs(1 - time_corr))  # Penalize non-linear time scaling
        memory_score = 1 - min(
            1, analysis["performance_degradation"]["throughput_decline"]
        )
        correctness_score = (
            1 - analysis["performance_degradation"]["correctness_stability"]
        )

        analysis["scalability_score"] = (
            time_score + memory_score + correctness_score
        ) / 3

    except ImportError:
        logger.warning("NumPy not available for advanced analysis")

    return analysis


def main():
    """Run scale testing for reasoning approaches."""
    parser = argparse.ArgumentParser(
        description="Scale testing for Phase 14 reasoning approaches"
    )

    parser.add_argument(
        "approach",
        choices=["ChainOfThought", "TreeOfThought", "AsPlanning", "None"],
        help="Reasoning approach to test",
    )

    parser.add_argument(
        "--sizes",
        nargs="+",
        type=int,
        default=[25, 50, 100, 200],
        help="Dataset sizes to test (default: 25 50 100 200)",
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
        default="outputs/phase14/scale_test",
        help="Output directory (default: outputs/phase14/scale_test)",
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

    logger.info(f"Starting scale test for {args.approach}")
    logger.info(f"Testing sizes: {args.sizes}")

    # Get configuration
    config_params = setup_local_api_config()

    # Override with command line arguments
    if args.api_base:
        config_params["api_base_url"] = args.api_base
    if args.model:
        config_params["model"] = args.model

    # Create base config
    config = ExperimentConfig(
        benchmark_id="LOCAL_TEST",
        sample_count=args.sizes[0],
        provider=config_params["provider"],
        model=config_params["model"],
        api_base_url=config_params["api_base_url"],
        temperature=config_params["temperature"],
        max_reasoning_calls=20,  # Increased from default 5 to 20
    )

    # Create output directory
    output_dir = Path(args.output_dir) / args.approach.lower()
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Run scale test
        scale_results = run_scale_test(args.approach, args.sizes, config, output_dir)

        # Analyze results
        performance_analysis = analyze_scaling_performance(scale_results)

        # Combine results
        full_results = {
            "scale_test_results": scale_results,
            "performance_analysis": performance_analysis,
            "test_metadata": {
                "approach": args.approach,
                "sizes_tested": args.sizes,
                "model": config_params["model"],
                "provider": config_params["provider"],
                "api_base": config_params["api_base_url"],
                "timestamp": time.time(),
            },
        }

        # Save results
        import json

        results_path = output_dir / "scale_test_results.json"
        with open(results_path, "w") as f:
            json.dump(full_results, f, indent=2, default=str)

        # Display summary
        logger.info(f"\n{'='*60}")
        logger.info(f"SCALE TEST SUMMARY - {args.approach}")
        logger.info(f"{'='*60}")

        successful_results = [r for r in scale_results["results"] if "error" not in r]

        if successful_results:
            logger.info(
                f"Successful tests: {len(successful_results)}/{len(args.sizes)}"
            )

            # Performance summary
            first_result = successful_results[0]
            last_result = successful_results[-1]

            logger.info(f"Size {first_result['size']} -> {last_result['size']}:")
            logger.info(
                f"  Correctness: {first_result['correctness_rate']:.2%} -> {last_result['correctness_rate']:.2%}"
            )
            logger.info(
                f"  Throughput: {first_result['throughput_samples_per_sec']:.1f} -> {last_result['throughput_samples_per_sec']:.1f} samples/sec"
            )
            logger.info(
                f"  Memory/sample: {first_result['memory_usage_mb']/first_result['size']:.2f} -> {last_result['memory_usage_mb']/last_result['size']:.2f} MB"
            )

            if "scalability_score" in performance_analysis:
                logger.info(
                    f"Scalability Score: {performance_analysis['scalability_score']:.2f}/1.0"
                )

                if performance_analysis["scalability_score"] > 0.8:
                    logger.info("✓ Excellent scalability")
                elif performance_analysis["scalability_score"] > 0.6:
                    logger.info("⚠ Good scalability")
                else:
                    logger.info("⚠ Limited scalability")

        failed_results = [r for r in scale_results["results"] if "error" in r]
        if failed_results:
            logger.warning(f"Failed tests: {len(failed_results)}")
            for result in failed_results:
                logger.warning(f"  Size {result['size']}: {result['error']}")

        logger.info(f"\nDetailed results saved to: {results_path}")

    except Exception as e:
        logger.error(f"Scale test failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Extended test script for local API performance testing."""

import json

# Add the src directory to Python path
import sys
import time
from pathlib import Path
from statistics import mean, stdev

sys.path.insert(0, str(Path(__file__).parent / "src"))

from ml_agents.config import ExperimentConfig, ParsingConfig
from ml_agents.reasoning.none import NoneReasoning
from ml_agents.utils.logging_config import setup_logging


def test_local_api_extended():
    """Test the local API with None approach using more samples."""
    # Setup logging
    setup_logging(level="INFO")

    # Create test configuration
    config = ExperimentConfig(
        provider="local-openai",
        model="Qwen/Qwen2.5-1.5B-Instruct",
        api_base_url="http://pop-os:8000/v1",
        temperature=0.3,
        max_tokens=100,
        sample_count=25,
        reasoning_approaches=["None"],
        parsing=ParsingConfig(use_structured_parsing=False),
    )

    # Create None reasoning instance
    none_reasoning = NoneReasoning(config)

    # Extended test questions (25 samples)
    test_questions = [
        # Basic arithmetic
        "What is 2 + 2?",
        "What is 10 - 5?",
        "What is 3 × 4?",
        "What is 20 ÷ 4?",
        "What is 7 + 8?",
        # General knowledge
        "What is the capital of France?",
        "What is the capital of Japan?",
        "What is the capital of Brazil?",
        "What is the largest planet in our solar system?",
        "Who wrote Romeo and Juliet?",
        # Sequences and patterns
        "Complete the sequence: 2, 4, 6, 8, ?",
        "Complete the sequence: 1, 3, 5, 7, ?",
        "Complete the sequence: 10, 20, 30, 40, ?",
        "What comes next: A, B, C, D, ?",
        "Complete: Monday, Tuesday, Wednesday, ?",
        # Colors and nature
        "What color is the sky on a clear day?",
        "What color is grass?",
        "What color is a typical banana when ripe?",
        "How many colors are in a rainbow?",
        "What is the color of fresh snow?",
        # Time and counting
        "How many days are in a week?",
        "How many months are in a year?",
        "How many hours are in a day?",
        "How many minutes are in an hour?",
        "How many seconds are in a minute?",
    ]

    print("\n🚀 Extended Local API Test with None Approach")
    print(f"Model: {config.model}")
    print(f"API Base: {config.api_base_url}")
    print(f"Provider: {config.provider}")
    print(f"Total Tests: {len(test_questions)}")
    print("-" * 60)

    results = []
    response_times = []
    token_counts = []

    start_total_time = time.time()

    for i, question in enumerate(test_questions, 1):
        print(f"\n[Test {i}/{len(test_questions)}] ", end="", flush=True)

        try:
            start_time = time.time()
            response = none_reasoning.execute(question)
            end_time = time.time()

            response_time = end_time - start_time
            response_times.append(response_time)
            token_counts.append(response.total_tokens)

            print(f"✓ ({response_time:.2f}s, {response.total_tokens} tokens)")

            results.append(
                {
                    "question": question,
                    "response": (
                        response.text[:200] + "..."
                        if len(response.text) > 200
                        else response.text
                    ),
                    "extracted_answer": response.extracted_answer,
                    "time": response_time,
                    "tokens": response.total_tokens,
                    "prompt_tokens": response.prompt_tokens,
                    "completion_tokens": response.completion_tokens,
                }
            )

        except Exception as e:
            print(f"✗ Error: {str(e)[:50]}...")
            results.append({"question": question, "error": str(e)})

    end_total_time = time.time()
    total_time = end_total_time - start_total_time

    # Summary
    print("\n" + "=" * 60)
    print("📊 Extended Test Summary")
    print("=" * 60)

    successful_tests = [r for r in results if "error" not in r]
    failed_tests = [r for r in results if "error" in r]

    print(f"\n✅ Successful: {len(successful_tests)}/{len(test_questions)}")
    print(f"❌ Failed: {len(failed_tests)}/{len(test_questions)}")

    if successful_tests:
        print(f"\n⏱️  Performance Metrics:")
        print(f"   Total Time: {total_time:.2f}s")
        print(
            f"   Avg Response Time: {mean(response_times):.2f}s (±{stdev(response_times):.2f}s)"
        )
        print(
            f"   Min/Max Response Time: {min(response_times):.2f}s / {max(response_times):.2f}s"
        )
        print(
            f"   Throughput: {len(successful_tests) / total_time:.1f} requests/second"
        )

        print(f"\n📝 Token Usage:")
        print(f"   Total Tokens: {sum(token_counts):,}")
        print(
            f"   Avg Tokens/Request: {mean(token_counts):.0f} (±{stdev(token_counts):.0f})"
        )
        print(f"   Min/Max Tokens: {min(token_counts)} / {max(token_counts)}")

    if failed_tests:
        print(f"\n⚠️  Failed Tests:")
        for test in failed_tests[:5]:  # Show first 5 failures
            print(f"   - {test['question']}: {test['error'][:50]}...")

    # Save results
    output_file = Path("test_local_api_extended_results.json")
    with open(output_file, "w") as f:
        json.dump(
            {
                "summary": {
                    "total_tests": len(test_questions),
                    "successful": len(successful_tests),
                    "failed": len(failed_tests),
                    "total_time": total_time,
                    "avg_response_time": mean(response_times) if response_times else 0,
                    "throughput": (
                        len(successful_tests) / total_time if total_time > 0 else 0
                    ),
                    "total_tokens": sum(token_counts) if token_counts else 0,
                },
                "results": results,
            },
            f,
            indent=2,
        )

    print(f"\n💾 Detailed results saved to: {output_file}")


if __name__ == "__main__":
    test_local_api_extended()

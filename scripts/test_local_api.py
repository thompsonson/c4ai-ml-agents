#!/usr/bin/env python3
"""Test script for local API with None approach."""

import json

# Add the src directory to Python path
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from ml_agents.config import ExperimentConfig, ParsingConfig
from ml_agents.reasoning.none import NoneReasoning
from ml_agents.utils.logging_config import setup_logging


def test_local_api():
    """Test the local API with None approach."""
    # Setup logging
    setup_logging(level="INFO")

    # Create test configuration
    config = ExperimentConfig(
        provider="local-openai",
        model="Qwen/Qwen2.5-1.5B-Instruct",
        api_base_url="http://pop-os:8000/v1",
        temperature=0.3,
        max_tokens=100,
        sample_count=5,
        reasoning_approaches=["None"],
        parsing=ParsingConfig(use_structured_parsing=False),
    )

    # Create None reasoning instance
    none_reasoning = NoneReasoning(config)

    # Test questions
    test_questions = [
        "What is 2 + 2?",
        "What is the capital of France?",
        "Complete the sequence: 2, 4, 6, 8, ?",
        "What color is the sky on a clear day?",
        "How many days are in a week?",
    ]

    print("\n🚀 Testing Local API with None Approach")
    print(f"Model: {config.model}")
    print(f"API Base: {config.api_base_url}")
    print(f"Provider: {config.provider}")
    print("-" * 50)

    results = []

    for i, question in enumerate(test_questions, 1):
        print(f"\n[Test {i}/{len(test_questions)}]")
        print(f"Question: {question}")

        try:
            start_time = time.time()
            response = none_reasoning.execute(question)
            end_time = time.time()

            print(f"Response: {response.text[:100]}...")
            print(f"Extracted Answer: {response.extracted_answer}")
            print(f"Time: {end_time - start_time:.2f}s")
            print(
                f"Tokens: {response.total_tokens} (prompt: {response.prompt_tokens}, completion: {response.completion_tokens})"
            )

            results.append(
                {
                    "question": question,
                    "response": response.text,
                    "extracted_answer": response.extracted_answer,
                    "time": end_time - start_time,
                    "tokens": response.total_tokens,
                }
            )

        except Exception as e:
            print(f"❌ Error: {e}")
            results.append({"question": question, "error": str(e)})

    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Summary")
    print("=" * 50)

    successful_tests = [r for r in results if "error" not in r]
    print(f"Successful: {len(successful_tests)}/{len(test_questions)}")

    if successful_tests:
        avg_time = sum(r["time"] for r in successful_tests) / len(successful_tests)
        avg_tokens = sum(r["tokens"] for r in successful_tests) / len(successful_tests)
        print(f"Avg Response Time: {avg_time:.2f}s")
        print(f"Avg Tokens Used: {avg_tokens:.0f}")

    # Save results
    output_file = Path("test_local_api_results.json")
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n💾 Results saved to: {output_file}")


if __name__ == "__main__":
    test_local_api()

#!/usr/bin/env python3
"""Test script for Instructor async compatibility with Local-OpenAI provider."""

import asyncio
import json
import sys
import time
from pathlib import Path

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ml_agents.config import ExperimentConfig
from ml_agents.utils.api_clients import create_api_client
from ml_agents.utils.instructor_clients import InstructorClientManager
from ml_agents.utils.logging_config import setup_logging
from ml_agents.utils.reasoning_extraction import NoneReasoningExtraction


async def test_single_async_extraction():
    """Test single async structured extraction."""
    print("\n🔄 Testing Single Async Extraction")
    print("-" * 40)

    # Create test configuration for Local-OpenAI
    config = ExperimentConfig(
        provider="local-openai",
        model="Qwen/Qwen2.5-1.5B-Instruct",
        api_base_url="http://pop-os:8000/v1",
        temperature=0.3,
        max_tokens=100,
    )

    # Create API client and InstructorClientManager
    api_client = create_api_client(config)
    instructor_manager = InstructorClientManager(api_client)

    # Test message
    messages = [
        {"role": "user", "content": "What is 2 + 2? Please provide a clear answer."}
    ]

    try:
        start_time = time.time()

        # Test async structured extraction
        extraction = await instructor_manager.extract_structured_response_async(
            messages=messages, response_model=NoneReasoningExtraction
        )

        end_time = time.time()

        print(f"✅ Success!")
        print(f"Answer: {extraction.answer_value}")
        print(f"Reasoning: {extraction.full_reasoning_text[:100]}...")
        print(f"Confidence: {extraction.confidence}")
        print(f"Time: {end_time - start_time:.2f}s")

        return True, extraction

    except Exception as e:
        print(f"❌ Failed: {e}")
        return False, None


async def test_concurrent_extractions():
    """Test concurrent structured extractions."""
    print("\n🚀 Testing Concurrent Extractions")
    print("-" * 40)

    # Create test configuration
    config = ExperimentConfig(
        provider="local-openai",
        model="Qwen/Qwen2.5-1.5B-Instruct",
        api_base_url="http://pop-os:8000/v1",
        temperature=0.3,
        max_tokens=100,
    )

    # Create API client and InstructorClientManager
    api_client = create_api_client(config)
    instructor_manager = InstructorClientManager(api_client)

    # Test questions
    test_questions = [
        "What is 5 + 3?",
        "What is the capital of Germany?",
        "What color is grass?",
        "How many hours are in a day?",
        "What is 10 - 4?",
    ]

    # Prepare messages list
    messages_list = [
        [{"role": "user", "content": f"{question} Please provide a clear answer."}]
        for question in test_questions
    ]

    try:
        start_time = time.time()

        # Test concurrent structured extractions
        extractions = await instructor_manager.extract_concurrent_responses(
            messages_list=messages_list,
            response_model=NoneReasoningExtraction,
            concurrency_limit=3,  # Test with limit of 3
        )

        end_time = time.time()

        # Count successful extractions
        successful = [
            ext
            for ext in extractions
            if ext is not None and not isinstance(ext, Exception)
        ]
        failed = len(extractions) - len(successful)

        print(f"✅ Completed!")
        print(f"Successful: {len(successful)}/{len(test_questions)}")
        print(f"Failed: {failed}")
        print(f"Total Time: {end_time - start_time:.2f}s")
        print(
            f"Avg Time per Request: {(end_time - start_time) / len(test_questions):.2f}s"
        )

        # Show results
        for i, (question, extraction) in enumerate(zip(test_questions, extractions)):
            if extraction is not None and not isinstance(extraction, Exception):
                print(f"Q{i+1}: {question} → {extraction.answer_value}")
            else:
                print(f"Q{i+1}: {question} → FAILED")

        return True, extractions

    except Exception as e:
        print(f"❌ Failed: {e}")
        return False, None


async def test_performance_comparison():
    """Compare sequential vs concurrent performance."""
    print("\n⚡ Testing Performance Comparison")
    print("-" * 40)

    # Create test configuration
    config = ExperimentConfig(
        provider="local-openai",
        model="Qwen/Qwen2.5-1.5B-Instruct",
        api_base_url="http://pop-os:8000/v1",
        temperature=0.3,
        max_tokens=50,  # Shorter for faster testing
    )

    # Create API client and InstructorClientManager
    api_client = create_api_client(config)
    instructor_manager = InstructorClientManager(api_client)

    # Test questions for performance test
    questions = [
        f"What is {i} + {i+1}?" for i in range(1, 11)
    ]  # 10 simple math questions

    messages_list = [
        [{"role": "user", "content": f"{question} Give just the number."}]
        for question in questions
    ]

    # Test sequential execution
    print("🔄 Sequential execution...")
    sequential_start = time.time()
    sequential_results = []

    for messages in messages_list:
        try:
            result = await instructor_manager.extract_structured_response_async(
                messages=messages, response_model=NoneReasoningExtraction
            )
            sequential_results.append(result)
        except Exception as e:
            sequential_results.append(None)

    sequential_end = time.time()
    sequential_time = sequential_end - sequential_start

    # Test concurrent execution
    print("🚀 Concurrent execution...")
    concurrent_start = time.time()

    concurrent_results = await instructor_manager.extract_concurrent_responses(
        messages_list=messages_list,
        response_model=NoneReasoningExtraction,
        concurrency_limit=5,
    )

    concurrent_end = time.time()
    concurrent_time = concurrent_end - concurrent_start

    # Results
    speedup = sequential_time / concurrent_time if concurrent_time > 0 else 0

    print(f"Sequential Time: {sequential_time:.2f}s")
    print(f"Concurrent Time: {concurrent_time:.2f}s")
    print(f"Speedup: {speedup:.2f}x")

    if speedup > 1.5:
        print("🎉 Significant performance improvement achieved!")
    elif speedup > 1.0:
        print("✅ Performance improvement detected")
    else:
        print("⚠️  No significant performance improvement")

    return speedup


async def main():
    """Main test function."""
    print("🧪 Instructor Async Compatibility Test")
    print("=" * 50)

    # Setup logging
    setup_logging(level="INFO")

    results = {}

    # Test 1: Single async extraction
    success1, extraction1 = await test_single_async_extraction()
    results["single_async"] = success1

    if not success1:
        print("\n❌ Single async test failed. Stopping here.")
        return

    # Test 2: Concurrent extractions
    success2, extractions2 = await test_concurrent_extractions()
    results["concurrent"] = success2

    # Test 3: Performance comparison (only if concurrent works)
    if success2:
        speedup = await test_performance_comparison()
        results["speedup"] = speedup

    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Summary")
    print("=" * 50)

    print(f"Single Async: {'✅ PASS' if results['single_async'] else '❌ FAIL'}")
    print(f"Concurrent: {'✅ PASS' if results.get('concurrent', False) else '❌ FAIL'}")

    if "speedup" in results:
        print(f"Performance: {results['speedup']:.2f}x speedup")

    # Save results
    output_file = Path("test_instructor_async_results.json")
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n💾 Results saved to: {output_file}")

    if all([results.get("single_async"), results.get("concurrent")]):
        print(
            "\n🎉 All tests passed! Instructor async is compatible with Local-OpenAI."
        )
    else:
        print("\n❌ Some tests failed. Check the results for details.")


if __name__ == "__main__":
    asyncio.run(main())

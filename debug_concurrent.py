#!/usr/bin/env python3
"""Debug script to test concurrent processing and identify NoneType issues."""

import asyncio
import json

from ml_agents.config import ExperimentConfig
from ml_agents.reasoning.chain_of_thought import ChainOfThoughtReasoning


def create_test_config():
    """Create a minimal test configuration."""
    return ExperimentConfig(
        dataset_name="test",
        sample_count=3,
        provider="local-openai",
        model="qwen2.5:1.5b",
        reasoning_approaches=["ChainOfThought"],
        output_dir="./outputs/debug",
        temperature=0.1,
        max_tokens=100,
        parallel_requests=3,
    )


async def test_concurrent():
    """Test concurrent processing directly."""
    print("🔧 Creating test configuration...")
    config = create_test_config()

    print("🔧 Initializing reasoning approach...")
    reasoning = ChainOfThoughtReasoning(config)

    # Create simple test prompts
    test_prompts = [
        "What is 2 + 2?",
        "What is the capital of France?",
        "What color is the sky?",
    ]

    print(f"🔧 Testing concurrent processing with {len(test_prompts)} prompts...")
    print(f"🔧 Test prompts: {test_prompts}")

    try:
        # Test the concurrent execution
        results = await reasoning.execute_concurrent(
            prompts=test_prompts, concurrency_limit=3
        )

        print(f"🎉 Success! Got {len(results)} results:")
        for i, result in enumerate(results):
            print(f"  Result {i}: {type(result).__name__} - {result}")

    except Exception as e:
        print(f"❌ Error in concurrent execution: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_concurrent())

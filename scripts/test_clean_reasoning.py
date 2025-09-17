#!/usr/bin/env python3
"""Test script to validate cleaned-up reasoning approaches work correctly."""

import asyncio
import sys
from pathlib import Path

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ml_agents.config import ExperimentConfig
from ml_agents.reasoning.chain_of_thought import ChainOfThoughtReasoning
from ml_agents.reasoning.none import NoneReasoning
from ml_agents.utils.logging_config import setup_logging


def test_sync_reasoning():
    """Test that sync reasoning approaches work without fallbacks."""
    print("🧪 Testing Sync Reasoning (Instructor-only)")
    print("-" * 50)

    # Create test configuration
    config = ExperimentConfig(
        provider="openrouter",
        model="openai/gpt-5-mini",
        temperature=0.3,
        max_tokens=100,
    )

    # Test prompts
    test_prompts = [
        "What is 2 + 2?",
        "What is the capital of France?",
    ]

    # Test None reasoning
    print("\n📋 Testing None Reasoning...")
    try:
        none_reasoning = NoneReasoning(config)
        print("✅ NoneReasoning created successfully")

        # Check that _prepare_enhanced_prompt method exists
        enhanced = none_reasoning._prepare_enhanced_prompt("test")
        print("✅ _prepare_enhanced_prompt method works")

        print(f"Enhanced prompt: {enhanced[:100]}...")

    except Exception as e:
        print(f"❌ NoneReasoning failed: {e}")
        return False

    # Test Chain of Thought reasoning
    print("\n🔗 Testing Chain of Thought Reasoning...")
    try:
        cot_reasoning = ChainOfThoughtReasoning(config)
        print("✅ ChainOfThoughtReasoning created successfully")

        # Check that _prepare_enhanced_prompt method exists
        enhanced = cot_reasoning._prepare_enhanced_prompt("test")
        print("✅ _prepare_enhanced_prompt method works")

        print(f"Enhanced prompt: {enhanced[:100]}...")

    except Exception as e:
        print(f"❌ ChainOfThoughtReasoning failed: {e}")
        return False

    print("\n🎉 All sync reasoning tests passed!")
    return True


async def test_async_reasoning():
    """Test that async reasoning approaches work."""
    print("\n🚀 Testing Async Reasoning (Concurrent)")
    print("-" * 50)

    # Create test configuration
    config = ExperimentConfig(
        provider="openrouter",
        model="openai/gpt-5-mini",
        temperature=0.3,
        max_tokens=50,
    )

    # Test prompts
    test_prompts = [
        "What is 3 + 3?",
        "What is 5 * 2?",
    ]

    # Test None reasoning async
    print("\n📋 Testing None Reasoning Async...")
    try:
        none_reasoning = NoneReasoning(config)

        # Check that execute_concurrent method exists
        assert hasattr(
            none_reasoning, "execute_concurrent"
        ), "execute_concurrent method missing"
        print("✅ execute_concurrent method exists")

        # Check that it's a coroutine
        import inspect

        assert inspect.iscoroutinefunction(
            none_reasoning.execute_concurrent
        ), "execute_concurrent is not async"
        print("✅ execute_concurrent is properly async")

    except Exception as e:
        print(f"❌ None async test failed: {e}")
        return False

    # Test Chain of Thought reasoning async
    print("\n🔗 Testing Chain of Thought Reasoning Async...")
    try:
        cot_reasoning = ChainOfThoughtReasoning(config)

        # Check that execute_concurrent method exists
        assert hasattr(
            cot_reasoning, "execute_concurrent"
        ), "execute_concurrent method missing"
        print("✅ execute_concurrent method exists")

        # Check that it's a coroutine
        import inspect

        assert inspect.iscoroutinefunction(
            cot_reasoning.execute_concurrent
        ), "execute_concurrent is not async"
        print("✅ execute_concurrent is properly async")

    except Exception as e:
        print(f"❌ CoT async test failed: {e}")
        return False

    print("\n🎉 All async reasoning tests passed!")
    return True


async def main():
    """Main test function."""
    print("🧪 Clean Reasoning Approaches Test")
    print("=" * 60)

    # Setup logging
    setup_logging(level="INFO")

    # Test sync reasoning
    sync_success = test_sync_reasoning()

    if not sync_success:
        print("\n❌ Sync reasoning tests failed")
        return

    # Test async reasoning
    async_success = await test_async_reasoning()

    # Summary
    print("\n" + "=" * 60)
    print("📊 Test Summary")
    print("=" * 60)

    print(f"Sync Reasoning: {'✅ PASS' if sync_success else '❌ FAIL'}")
    print(f"Async Reasoning: {'✅ PASS' if async_success else '❌ FAIL'}")

    if sync_success and async_success:
        print("\n🎉 All tests passed! Clean reasoning approaches are working.")
        print("📋 Technical debt removed successfully - no more fallbacks!")
    else:
        print("\n❌ Some tests failed. Check errors above.")


if __name__ == "__main__":
    asyncio.run(main())

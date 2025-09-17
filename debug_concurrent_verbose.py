#!/usr/bin/env python3
"""Debug script with verbose logging to diagnose concurrent processing issues."""

import asyncio
import json
import logging

from ml_agents.config import ExperimentConfig
from ml_agents.reasoning.chain_of_thought import ChainOfThoughtReasoning

# Set verbose logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def create_test_config():
    """Create a minimal test configuration."""
    return ExperimentConfig(
        dataset_name="test",
        sample_count=1,  # Start with just 1 to make debugging easier
        provider="local-openai",
        model="RedHatAI/Qwen2-1.5B-Instruct-quantized.w4a16",
        reasoning_approaches=["ChainOfThought"],
        output_dir="./outputs/debug",
        temperature=0.1,
        max_tokens=100,
        parallel_requests=1,
    )


async def test_concurrent():
    """Test concurrent processing directly with verbose logging."""
    print("🔧 Creating test configuration...")
    config = create_test_config()

    print("🔧 Initializing reasoning approach...")
    reasoning = ChainOfThoughtReasoning(config)

    # Test a single simple prompt
    test_prompts = ["What is 2 + 2?"]

    print(f"🔧 Testing with single prompt: {test_prompts[0]}")

    # First, test the prompt enhancement
    try:
        enhanced_prompt = reasoning._prepare_enhanced_prompt(test_prompts[0])
        print(f"✅ Enhanced prompt created successfully:")
        print(f"   Length: {len(enhanced_prompt)} characters")
        print(f"   First 200 chars: {enhanced_prompt[:200]}...")
    except Exception as e:
        print(f"❌ Error in prompt enhancement: {e}")
        return

    # Test direct API call first
    try:
        print(f"🔧 Testing direct API call...")
        client_manager = reasoning.instructor_manager

        messages = [{"role": "user", "content": enhanced_prompt}]
        print(f"🔧 Messages prepared: {len(messages)} message(s)")

        # Get extraction model
        from ml_agents.utils.reasoning_extraction import REASONING_EXTRACTION_MODELS

        approach_key = reasoning.approach_name.lower()
        extraction_model = REASONING_EXTRACTION_MODELS.get(approach_key)
        if not extraction_model:
            from ml_agents.utils.reasoning_extraction import NoneReasoningExtraction

            extraction_model = NoneReasoningExtraction

        print(f"🔧 Using extraction model: {extraction_model}")

        # Try direct extraction
        direct_result = await client_manager.extract_structured_response_async(
            messages=messages,
            response_model=extraction_model,
            temperature=0.1,
            max_tokens=100,
        )
        print(f"✅ Direct API call successful: {type(direct_result)} - {direct_result}")

    except Exception as e:
        print(f"❌ Direct API call failed: {e}")
        import traceback

        traceback.print_exc()
        return

    # Now test concurrent execution
    try:
        print(f"🔧 Testing concurrent processing...")
        results = await reasoning.execute_concurrent(
            prompts=test_prompts, concurrency_limit=1
        )

        print(f"🎉 Got result: {type(results[0])} - {results[0]}")

    except Exception as e:
        print(f"❌ Error in concurrent execution: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_concurrent())

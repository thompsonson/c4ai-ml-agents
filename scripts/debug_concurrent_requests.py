#!/usr/bin/env python3
"""Debug script to test if concurrent requests are actually being sent to vLLM."""

import asyncio
import sys
import time
from pathlib import Path

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ml_agents.config import ExperimentConfig
from ml_agents.utils.api_clients import create_api_client
from ml_agents.utils.instructor_clients import InstructorClientManager
from ml_agents.utils.reasoning_extraction import NoneReasoningExtraction


async def test_direct_openai_async():
    """Test direct OpenAI async client to see if requests are concurrent."""
    print("🧪 Testing Direct OpenAI Async Client")
    print("-" * 50)

    try:
        import openai

        # Create OpenAI client with local-openai settings
        client = openai.AsyncOpenAI(
            base_url="http://pop-os:8000/v1", api_key="not-needed"
        )

        # Test concurrent requests directly
        messages_list = [
            [{"role": "user", "content": f"What is {i} + {i}? Give just the number."}]
            for i in range(1, 4)
        ]

        print(f"🚀 Sending {len(messages_list)} concurrent requests to vLLM...")
        start_time = time.time()

        # Create tasks for concurrent execution
        tasks = []
        for i, messages in enumerate(messages_list):
            task = client.chat.completions.create(
                model="RedHatAI/Qwen2-1.5B-Instruct-quantized.w4a16",
                messages=messages,
                max_tokens=50,
                temperature=0.3,
            )
            tasks.append(task)
            print(f"   📤 Created task {i+1}")

        # Execute all tasks concurrently
        print("⏳ Waiting for all responses...")
        responses = await asyncio.gather(*tasks)

        end_time = time.time()
        duration = end_time - start_time

        print(f"✅ Received {len(responses)} responses in {duration:.2f}s")
        for i, response in enumerate(responses):
            content = response.choices[0].message.content
            print(f"   📥 Response {i+1}: {content[:50]}...")

        # If this took much less time than 3 sequential requests would take,
        # then the requests were truly concurrent
        estimated_sequential_time = len(messages_list) * 2.0  # Assume ~2s per request
        if duration < estimated_sequential_time * 0.7:
            print(
                f"🎉 Requests appear to be concurrent! ({duration:.2f}s < {estimated_sequential_time:.2f}s)"
            )
            return True
        else:
            print(
                f"⚠️  Requests may have been sequential ({duration:.2f}s ≈ {estimated_sequential_time:.2f}s)"
            )
            return False

    except Exception as e:
        print(f"❌ Direct async test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


async def test_instructor_async():
    """Test Instructor async to see if it preserves concurrency."""
    print("\n🎓 Testing Instructor Async Client")
    print("-" * 50)

    try:
        config = ExperimentConfig(
            provider="local-openai",
            model="RedHatAI/Qwen2-1.5B-Instruct-quantized.w4a16",
            api_base_url="http://pop-os:8000/v1",
            temperature=0.3,
            max_tokens=50,
        )

        api_client = create_api_client(config)
        instructor_manager = InstructorClientManager(api_client)

        # Test concurrent structured extraction
        messages_list = [
            [{"role": "user", "content": f"What is {i} + {i}? Give just the number."}]
            for i in range(1, 4)
        ]

        print(f"🚀 Sending {len(messages_list)} concurrent structured extractions...")
        start_time = time.time()

        results = await instructor_manager.extract_concurrent_responses(
            messages_list=messages_list,
            response_model=NoneReasoningExtraction,
            concurrency_limit=3,
            temperature=0.3,
            max_tokens=50,
        )

        end_time = time.time()
        duration = end_time - start_time

        print(f"✅ Received {len(results)} structured results in {duration:.2f}s")
        successful = len(
            [r for r in results if r is not None and not isinstance(r, Exception)]
        )
        print(f"📊 Successful extractions: {successful}/{len(results)}")

        # Check timing
        estimated_sequential_time = len(messages_list) * 2.0
        if duration < estimated_sequential_time * 0.7:
            print(
                f"🎉 Instructor requests appear to be concurrent! ({duration:.2f}s < {estimated_sequential_time:.2f}s)"
            )
            return True
        else:
            print(
                f"⚠️  Instructor requests may have been sequential ({duration:.2f}s ≈ {estimated_sequential_time:.2f}s)"
            )
            return False

    except Exception as e:
        print(f"❌ Instructor async test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


async def monitor_request_timing():
    """Monitor request timing to detect concurrency."""
    print("\n⏱️  Monitoring Request Timing")
    print("-" * 50)

    # Simple timing test with logging
    config = ExperimentConfig(
        provider="local-openai",
        model="RedHatAI/Qwen2-1.5B-Instruct-quantized.w4a16",
        api_base_url="http://pop-os:8000/v1",
        temperature=0.3,
        max_tokens=30,
    )

    api_client = create_api_client(config)
    instructor_manager = InstructorClientManager(api_client)

    # Simple requests
    messages_list = [
        [{"role": "user", "content": "Say 'Hello 1'"}],
        [{"role": "user", "content": "Say 'Hello 2'"}],
        [{"role": "user", "content": "Say 'Hello 3'"}],
    ]

    print("🚀 Starting timed concurrent test...")

    # Record start times
    request_times = []

    async def timed_request(messages, index):
        start = time.time()
        print(f"   📤 Request {index} starting at {start:.3f}")

        try:
            result = await instructor_manager.extract_structured_response_async(
                messages=messages,
                response_model=NoneReasoningExtraction,
                temperature=0.3,
                max_tokens=30,
            )
            end = time.time()
            duration = end - start
            print(
                f"   📥 Request {index} completed at {end:.3f} (took {duration:.3f}s)"
            )
            return result
        except Exception as e:
            end = time.time()
            duration = end - start
            print(
                f"   ❌ Request {index} failed at {end:.3f} (took {duration:.3f}s): {e}"
            )
            return None

    # Execute with asyncio.gather to ensure true concurrency
    start_time = time.time()
    tasks = [timed_request(messages, i + 1) for i, messages in enumerate(messages_list)]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    end_time = time.time()

    total_duration = end_time - start_time
    print(f"🏁 All requests completed in {total_duration:.3f}s")

    return True


async def main():
    """Run concurrent request debugging."""
    print("🔍 Debugging Concurrent Request Implementation")
    print("=" * 60)

    # Test 1: Direct OpenAI async
    direct_success = await test_direct_openai_async()

    # Test 2: Instructor async
    instructor_success = await test_instructor_async()

    # Test 3: Timing monitoring
    await monitor_request_timing()

    # Summary
    print("\n" + "=" * 60)
    print("📊 Concurrency Debug Results")
    print("=" * 60)

    print(
        f"Direct OpenAI Async: {'✅ CONCURRENT' if direct_success else '⚠️  SEQUENTIAL'}"
    )
    print(
        f"Instructor Async: {'✅ CONCURRENT' if instructor_success else '⚠️  SEQUENTIAL'}"
    )

    if direct_success and not instructor_success:
        print(
            "\n🔍 Issue identified: Instructor library may not be preserving async concurrency"
        )
        print("💡 Possible solutions:")
        print("   • Check Instructor library version and async support")
        print("   • Verify Instructor is using async OpenAI client properly")
        print("   • Consider bypassing Instructor for concurrent requests")
    elif not direct_success:
        print(
            "\n🔍 Issue identified: OpenAI async client not working with local-openai"
        )
        print("💡 Possible solutions:")
        print("   • Check vLLM server configuration")
        print("   • Verify OpenAI client async implementation")
        print("   • Test with different async HTTP client")
    else:
        print("\n✅ Both direct and Instructor async appear to work!")
        print("🔍 If vLLM still shows sequential requests, check server logs")


if __name__ == "__main__":
    asyncio.run(main())

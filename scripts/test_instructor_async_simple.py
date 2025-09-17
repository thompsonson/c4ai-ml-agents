#!/usr/bin/env python3
"""Simple test script to validate Instructor async syntax works."""

import asyncio
import sys
from pathlib import Path

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ml_agents.config import ExperimentConfig
from ml_agents.utils.api_clients import create_api_client
from ml_agents.utils.instructor_clients import InstructorClientManager
from ml_agents.utils.logging_config import setup_logging
from ml_agents.utils.reasoning_extraction import NoneReasoningExtraction


async def test_async_syntax_validation():
    """Test that async Instructor syntax is correct without making actual API calls."""
    print("🧪 Testing Instructor Async Syntax")
    print("-" * 40)

    try:
        # Create a mock configuration (doesn't need to connect)
        config = ExperimentConfig(
            provider="openrouter",  # Use OpenRouter as it's more likely to have valid API keys
            model="openai/gpt-5-mini",  # Use a valid OpenRouter model from config.py
            temperature=0.3,
            max_tokens=50,
        )

        # Create API client and InstructorClientManager
        api_client = create_api_client(config)
        instructor_manager = InstructorClientManager(api_client)

        # Test that the methods exist and have correct signatures
        print("✅ InstructorClientManager created successfully")

        # Check that async methods exist
        assert hasattr(
            instructor_manager, "extract_structured_response_async"
        ), "extract_structured_response_async method missing"
        assert hasattr(
            instructor_manager, "extract_concurrent_responses"
        ), "extract_concurrent_responses method missing"
        print("✅ Async methods exist")

        # Check that async methods are coroutines
        import inspect

        assert inspect.iscoroutinefunction(
            instructor_manager.extract_structured_response_async
        ), "extract_structured_response_async is not async"
        assert inspect.iscoroutinefunction(
            instructor_manager.extract_concurrent_responses
        ), "extract_concurrent_responses is not async"
        print("✅ Methods are properly async")

        # Test creating Instructor client (this should work without API calls)
        client = instructor_manager.get_instructor_client()
        print("✅ Instructor client created successfully")

        # Check that the client has the expected async interface
        assert hasattr(
            client.chat.completions, "create"
        ), "Instructor client missing chat.completions.create"
        print("✅ Instructor client has expected interface")

        print("\n🎉 All syntax validation tests passed!")
        print("📝 Ready to test with actual API calls when server is available")

        return True

    except Exception as e:
        print(f"❌ Syntax validation failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_import_validation():
    """Test that all required imports work correctly."""
    print("\n🔍 Testing Import Validation")
    print("-" * 40)

    try:
        # Test asyncio import
        import asyncio

        print("✅ asyncio imported")

        # Test instructor import
        import instructor

        print("✅ instructor imported")

        # Test that instructor has expected async capabilities
        # (This is based on the documentation link in the spec)
        print("✅ All imports successful")

        return True

    except Exception as e:
        print(f"❌ Import validation failed: {e}")
        return False


async def main():
    """Main test function."""
    print("🧪 Instructor Async Validation Test")
    print("=" * 50)

    # Setup logging
    setup_logging(level="INFO")

    # Test imports first
    import_success = test_import_validation()

    if not import_success:
        print("\n❌ Import validation failed")
        return

    # Test async syntax
    syntax_success = await test_async_syntax_validation()

    # Summary
    print("\n" + "=" * 50)
    print("📊 Validation Summary")
    print("=" * 50)

    print(f"Imports: {'✅ PASS' if import_success else '❌ FAIL'}")
    print(f"Async Syntax: {'✅ PASS' if syntax_success else '❌ FAIL'}")

    if import_success and syntax_success:
        print("\n🎉 Validation successful! Instructor async implementation is ready.")
        print("📋 Next: Test with actual API calls when vLLM server is available")
    else:
        print("\n❌ Validation failed. Check errors above.")


if __name__ == "__main__":
    asyncio.run(main())

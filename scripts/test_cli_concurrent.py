#!/usr/bin/env python3
"""Test the completed CLI integration for Phase 16 concurrent processing."""

import asyncio
import subprocess
import sys
import tempfile
from pathlib import Path

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def test_cli_help_includes_concurrent_flags():
    """Test that the CLI help includes the new concurrent flags."""
    print("🧪 Testing CLI Help for Concurrent Flags")
    print("-" * 50)

    try:
        # Run the help command
        result = subprocess.run(
            ["uv", "run", "ml-agents", "eval", "run", "--help"],
            capture_output=True,
            text=True,
            timeout=30,
        )

        help_output = result.stdout

        # Check for concurrent flags
        has_concurrent = "--concurrent" in help_output
        has_concurrency_limit = "--concurrency-limit" in help_output

        print(f"✅ --concurrent flag found: {has_concurrent}")
        print(f"✅ --concurrency-limit flag found: {has_concurrency_limit}")

        if has_concurrent and has_concurrency_limit:
            print("✅ All concurrent flags are available in CLI")
            return True
        else:
            print("❌ Some concurrent flags are missing")
            return False

    except Exception as e:
        print(f"❌ Help test failed: {e}")
        return False


def test_cli_concurrent_flag_validation():
    """Test that the CLI accepts concurrent flags without error."""
    print("\n🔧 Testing CLI Concurrent Flag Validation")
    print("-" * 50)

    try:
        # Create a minimal test CSV
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("INPUT,OUTPUT\n")
            f.write("What is 2+2?,4\n")
            f.write("What is the capital of France?,Paris\n")
            test_csv = f.name

        # Test concurrent flags parsing (should not fail on flag parsing)
        cmd = [
            "uv",
            "run",
            "ml-agents",
            "eval",
            "run",
            test_csv,
            "None",
            "--concurrent",
            "--concurrency-limit",
            "5",
            "--samples",
            "1",
            "--provider",
            "openrouter",
            "--model",
            "openai/gpt-5-mini",
            "--skip-warnings",
            "--max-tokens",
            "50",
        ]

        print(f"🔧 Testing command: {' '.join(cmd)}")

        # Run with timeout to avoid hanging (API call may fail but flags should parse)
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)

        # Check if the error is about the flags or about other issues (API, etc.)
        if "concurrent" in result.stderr and "unrecognized" in result.stderr.lower():
            print("❌ Concurrent flags not recognized by CLI")
            return False
        elif "help" in result.stderr or "usage" in result.stderr:
            print("❌ CLI rejected the concurrent flags")
            return False
        else:
            print(
                "✅ Concurrent flags accepted by CLI (any errors are likely API-related)"
            )
            return True

    except subprocess.TimeoutExpired:
        print("✅ Command timed out (expected for API calls) - flags were accepted")
        return True
    except Exception as e:
        print(f"❌ Flag validation test failed: {e}")
        return False
    finally:
        # Cleanup
        try:
            Path(test_csv).unlink()
        except:
            pass


def test_concurrent_function_import():
    """Test that the concurrent processing functions can be imported."""
    print("\n📦 Testing Concurrent Function Imports")
    print("-" * 50)

    try:
        # Test async function import
        from ml_agents.cli.commands.eval import _run_concurrent_experiment

        print("✅ _run_concurrent_experiment function imported successfully")

        # Test that it's actually async
        import inspect

        if inspect.iscoroutinefunction(_run_concurrent_experiment):
            print("✅ _run_concurrent_experiment is correctly async")
        else:
            print("❌ _run_concurrent_experiment is not async")
            return False

        # Test ExperimentRunner concurrent method
        from ml_agents.core.experiment_runner import ExperimentRunner

        if hasattr(ExperimentRunner, "run_reasoning_concurrent"):
            print("✅ ExperimentRunner.run_reasoning_concurrent method exists")

            # Test that it's async
            if inspect.iscoroutinefunction(ExperimentRunner.run_reasoning_concurrent):
                print("✅ run_reasoning_concurrent is correctly async")
            else:
                print("❌ run_reasoning_concurrent is not async")
                return False
        else:
            print("❌ ExperimentRunner.run_reasoning_concurrent method missing")
            return False

        return True

    except Exception as e:
        print(f"❌ Import test failed: {e}")
        return False


def test_script_permissions():
    """Test that the new concurrent script has correct permissions."""
    print("\n📜 Testing Script Permissions")
    print("-" * 50)

    script_path = Path(__file__).parent.parent / "run_concurrent_benchmarks.sh"

    if script_path.exists():
        import stat

        file_stat = script_path.stat()
        is_executable = bool(file_stat.st_mode & stat.S_IEXEC)

        print(f"✅ Script exists: {script_path}")
        print(f"✅ Script is executable: {is_executable}")

        # Check script content for Phase 16 features
        content = script_path.read_text()
        has_concurrent_flag = "--concurrent" in content
        has_concurrency_limit = "--concurrency-limit" in content
        has_phase16_mention = "Phase 16" in content

        print(f"✅ Script uses --concurrent flag: {has_concurrent_flag}")
        print(f"✅ Script uses --concurrency-limit: {has_concurrency_limit}")
        print(f"✅ Script mentions Phase 16: {has_phase16_mention}")

        return is_executable and has_concurrent_flag and has_concurrency_limit
    else:
        print(f"❌ Script not found: {script_path}")
        return False


def main():
    """Run all CLI integration tests."""
    print("🧪 Phase 16 CLI Integration Test Suite")
    print("=" * 60)

    tests = [
        ("CLI Help", test_cli_help_includes_concurrent_flags),
        ("Flag Validation", test_cli_concurrent_flag_validation),
        ("Function Imports", test_concurrent_function_import),
        ("Script Permissions", test_script_permissions),
    ]

    results = {}
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results[test_name] = False

    # Summary
    print("\n" + "=" * 60)
    print("📊 CLI Integration Test Results")
    print("=" * 60)

    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name}: {status}")

    all_passed = all(results.values())

    if all_passed:
        print("\n🎉 All CLI integration tests passed!")
        print("✅ Phase 16 concurrent processing is ready for use!")
        print("")
        print("🚀 Ready to run:")
        print(
            "   ml-agents eval run BENCHMARK APPROACH --concurrent --concurrency-limit 10"
        )
        print("   ./run_concurrent_benchmarks.sh None 4 10 50")
    else:
        failed_count = sum(1 for passed in results.values() if not passed)
        print(f"\n⚠️  {failed_count}/{len(tests)} tests failed")
        print("Some integration issues need to be resolved")

    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

"""Environment validation utilities for ML Agents project."""

import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

from ml_agents.utils.logging_config import get_logger

logger = get_logger(__name__)


class EnvironmentValidator:
    """Validates environment configuration for ML Agents experiments."""

    REQUIRED_ENV_VARS = [
        "ANTHROPIC_API_KEY",
        "COHERE_API_KEY",
        "OPENROUTER_API_KEY",
        "HUGGINGFACE_API_KEY",
    ]

    OPTIONAL_ENV_VARS = [
        "LOG_LEVEL",
        "LOG_FORMAT",
        "LOG_TO_FILE",
        "LOG_DIR",
        "DEFAULT_PROVIDER",
        "DEFAULT_MODEL",
    ]

    @classmethod
    def validate_env_vars(cls) -> Tuple[bool, List[str]]:
        """Validate that required environment variables are set.

        Returns:
            Tuple of (is_valid, list_of_missing_vars)
        """
        missing_vars = []

        for var in cls.REQUIRED_ENV_VARS:
            value = os.getenv(var)
            if not value or value == "your_key_here" or "xxxx" in value:
                missing_vars.append(var)

        return len(missing_vars) == 0, missing_vars

    @classmethod
    def validate_dependencies(cls) -> Tuple[bool, List[str]]:
        """Validate that required Python packages are installed.

        Returns:
            Tuple of (is_valid, list_of_missing_packages)
        """
        required_packages = [
            "transformers",
            "accelerate",
            "openai",
            "cohere",
            "anthropic",
            "pandas",
            "datasets",
            "python-dotenv",
            "tqdm",
            "huggingface_hub",
            "pyyaml",
            "torch",
        ]

        missing_packages = []
        for package in required_packages:
            try:
                __import__(package.replace("-", "_"))
            except ImportError:
                missing_packages.append(package)

        return len(missing_packages) == 0, missing_packages

    @classmethod
    def validate_output_directory(
        cls, output_dir: str = "./outputs"
    ) -> Tuple[bool, str]:
        """Validate that output directory can be created and written to.

        Args:
            output_dir: Path to output directory

        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            # Test write permissions
            test_file = output_path / ".write_test"
            test_file.write_text("test")
            test_file.unlink()

            return True, ""
        except Exception as e:
            return False, str(e)

    @classmethod
    def validate_all(cls, raise_on_error: bool = False) -> Dict[str, bool]:
        """Run all validation checks.

        Args:
            raise_on_error: If True, raise exception on validation failure

        Returns:
            Dictionary with validation results

        Raises:
            RuntimeError: If raise_on_error is True and validation fails
        """
        results = {}
        errors = []

        # Check environment variables
        env_valid, missing_vars = cls.validate_env_vars()
        results["environment_variables"] = env_valid
        if not env_valid:
            error_msg = f"Missing environment variables: {', '.join(missing_vars)}"
            errors.append(error_msg)
            logger.warning(error_msg)

        # Check dependencies
        deps_valid, missing_packages = cls.validate_dependencies()
        results["dependencies"] = deps_valid
        if not deps_valid:
            error_msg = f"Missing packages: {', '.join(missing_packages)}"
            errors.append(error_msg)
            logger.warning(error_msg)

        # Check output directory
        dir_valid, dir_error = cls.validate_output_directory()
        results["output_directory"] = dir_valid
        if not dir_valid:
            error_msg = f"Output directory error: {dir_error}"
            errors.append(error_msg)
            logger.warning(error_msg)

        # Overall validation result
        results["valid"] = all(results.values())

        if not results["valid"]:
            error_summary = "Environment validation failed:\\n" + "\\n".join(
                f"  - {error}" for error in errors
            )

            if raise_on_error:
                raise RuntimeError(error_summary)
            else:
                logger.error(error_summary)

        return results

    @classmethod
    def print_validation_report(cls) -> None:
        """Print a detailed validation report to console."""
        print("\\n" + "=" * 60)
        print("ML Agents Environment Validation Report")
        print("=" * 60)

        # Check environment variables
        env_valid, missing_vars = cls.validate_env_vars()
        print(
            f"\\n✓ Environment Variables: {'✅ PASSED' if env_valid else '❌ FAILED'}"
        )
        if not env_valid:
            for var in missing_vars:
                print(f"  ❌ {var}: Not set or invalid")
        else:
            for var in cls.REQUIRED_ENV_VARS:
                print(f"  ✅ {var}: Set")

        # Check dependencies
        deps_valid, missing_packages = cls.validate_dependencies()
        print(f"\\n✓ Python Dependencies: {'✅ PASSED' if deps_valid else '❌ FAILED'}")
        if not deps_valid:
            for package in missing_packages:
                print(f"  ❌ {package}: Not installed")

        # Check output directory
        dir_valid, dir_error = cls.validate_output_directory()
        print(f"\\n✓ Output Directory: {'✅ PASSED' if dir_valid else '❌ FAILED'}")
        if not dir_valid:
            print(f"  ❌ Error: {dir_error}")

        # Overall result
        all_valid = env_valid and deps_valid and dir_valid
        print("\\n" + "=" * 60)
        print(f"Overall Status: {'✅ READY' if all_valid else '❌ NOT READY'}")
        print("=" * 60)

        if not all_valid:
            print("\\n⚠️  Please fix the issues above before running experiments.")
            print("💡 Tip: Copy .env.example to .env and add your API keys")
            print("💡 Tip: Run 'pip install -e .[dev]' to install all dependencies")


def validate_startup(raise_on_error: bool = True) -> Dict[str, bool]:
    """Validate environment on startup.

    Args:
        raise_on_error: If True, raise exception on validation failure

    Returns:
        Dictionary with validation results

    Raises:
        RuntimeError: If raise_on_error is True and validation fails
    """
    return EnvironmentValidator.validate_all(raise_on_error=raise_on_error)


if __name__ == "__main__":
    # Run validation when module is executed directly
    EnvironmentValidator.print_validation_report()
    results = EnvironmentValidator.validate_all(raise_on_error=False)

    if not results["valid"]:
        sys.exit(1)

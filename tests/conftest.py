"""Pytest configuration and fixtures for ML Agents tests."""

import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Generator
from unittest.mock import Mock, patch

import pytest
from dotenv import load_dotenv

# Load test environment variables
load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env.test")


@pytest.fixture(scope="session")
def test_data_dir() -> Path:
    """Return path to test data directory."""
    return Path(__file__).parent / "data"


@pytest.fixture(scope="session")
def sample_dataset() -> Dict[str, str]:
    """Sample dataset for testing."""
    return {
        "input": "What is 2 + 2?",
        "answer": "4",
        "task": "arithmetic",
    }


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def mock_api_keys() -> Dict[str, str]:
    """Mock API keys for testing."""
    return {
        "anthropic": "test-anthropic-key",
        "cohere": "test-cohere-key",
        "openrouter": "test-openrouter-key",
        "huggingface": "test-hf-key",
    }


@pytest.fixture
def mock_env_vars(mock_api_keys: Dict[str, str]) -> Generator[None, None, None]:
    """Mock environment variables."""
    env_patches = {
        "ANTHROPIC_API_KEY": mock_api_keys["anthropic"],
        "COHERE_API_KEY": mock_api_keys["cohere"],
        "OPENROUTER_API_KEY": mock_api_keys["openrouter"],
        "HUGGINGFACE_API_KEY": mock_api_keys["huggingface"],
        "LOG_LEVEL": "DEBUG",
        "LOG_FORMAT": "human",
        "LOG_TO_FILE": "false",
    }

    with patch.dict(os.environ, env_patches):
        yield


@pytest.fixture
def mock_anthropic_client() -> Mock:
    """Mock Anthropic client."""
    mock_client = Mock()
    mock_response = Mock()
    mock_response.content = [Mock(text="This is a test response.")]
    mock_client.messages.create.return_value = mock_response
    return mock_client


@pytest.fixture
def mock_cohere_client() -> Mock:
    """Mock Cohere client."""
    mock_client = Mock()
    mock_response = Mock()
    mock_response.text = "This is a test response."
    mock_client.chat.return_value = mock_response
    return mock_client


@pytest.fixture
def mock_openai_client() -> Mock:
    """Mock OpenAI/OpenRouter client."""
    mock_client = Mock()
    mock_response = Mock()
    mock_choice = Mock()
    mock_choice.message.content = "This is a test response."
    mock_response.choices = [mock_choice]
    mock_client.chat.completions.create.return_value = mock_response
    return mock_client


@pytest.fixture
def mock_hf_tokenizer() -> Mock:
    """Mock HuggingFace tokenizer."""
    mock_tokenizer = Mock()
    mock_tokenizer.decode.return_value = "This is a test response."
    return mock_tokenizer


@pytest.fixture
def mock_hf_model() -> Mock:
    """Mock HuggingFace model."""
    mock_model = Mock()
    mock_model.generate.return_value = [[1, 2, 3, 4]]  # Mock token IDs
    return mock_model


@pytest.fixture
def mock_dataset() -> Mock:
    """Mock dataset for testing."""
    mock_ds = Mock()
    mock_ds.to_pandas.return_value = pytest.sample_df
    return mock_ds


@pytest.fixture(scope="session", autouse=True)
def setup_test_environment() -> None:
    """Set up test environment."""
    # Create test data if needed
    test_data_dir = Path(__file__).parent / "data"
    test_data_dir.mkdir(exist_ok=True)

    # Create sample test dataset
    sample_data = [
        {"input": "What is 2 + 2?", "answer": "4", "task": "arithmetic"},
        {
            "input": "What is the capital of France?",
            "answer": "Paris",
            "task": "geography",
        },
        {
            "input": "Who wrote Romeo and Juliet?",
            "answer": "Shakespeare",
            "task": "literature",
        },
    ]

    # Store sample dataframe for tests
    import pandas as pd

    pytest.sample_df = pd.DataFrame(sample_data)


@pytest.fixture
def experiment_config_data() -> Dict[str, Any]:
    """Sample experiment configuration data."""
    return {
        "dataset_name": "test/dataset",
        "sample_count": 10,
        "provider": "openrouter",
        "model": "openai/gpt-oss-120b",  # Use a valid model for openrouter
        "temperature": 0.5,
        "max_tokens": 256,
        "top_p": 0.9,
        "reasoning_approaches": ["None", "ChainOfThought"],
        "output_dir": "./test_outputs",
    }


@pytest.fixture
def invalid_config_data() -> Dict[str, Any]:
    """Invalid configuration data for testing validation."""
    return {
        "dataset_name": "",
        "sample_count": -1,
        "provider": "invalid_provider",
        "model": "",
        "temperature": 3.0,  # Invalid: > 2.0
        "max_tokens": 0,  # Invalid: < 1
        "top_p": 1.5,  # Invalid: > 1.0
        "reasoning_approaches": [],
        "output_dir": "",
    }


# Pytest markers for test organization
pytest.mark.unit = pytest.mark.unit
pytest.mark.integration = pytest.mark.integration
pytest.mark.slow = pytest.mark.slow


# Integration test utilities
@pytest.fixture
def real_api_config() -> Dict[str, Any]:
    """Configuration for real API integration tests."""
    return {
        "provider": "openrouter",
        "model": "openai/gpt-3.5-turbo",
        "temperature": 0.3,
        "max_tokens": 100,  # Keep low for cost control
        "top_p": 0.9,
    }


@pytest.fixture
def integration_test_prompts() -> list[str]:
    """Simple prompts for integration testing with minimal token usage."""
    return ["What is 1+1?", "Name one color.", "Say hello."]


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests (may call real APIs)"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests (no external dependencies)"
    )
    config.addinivalue_line("markers", "slow: marks tests as slow running")


def skip_if_no_api_key(provider: str) -> bool:
    """Skip test if API key for provider is not available."""
    key_env_vars = {
        "anthropic": "ANTHROPIC_API_KEY",
        "cohere": "COHERE_API_KEY",
        "openrouter": "OPENROUTER_API_KEY",
        "huggingface": "HUGGINGFACE_API_KEY",
    }

    env_var = key_env_vars.get(provider.lower())
    if not env_var or not os.getenv(env_var):
        return True
    return False


@pytest.fixture
def skip_integration_if_no_keys():
    """Fixture to skip integration tests if no API keys available."""

    def _skip_for_provider(provider: str):
        if skip_if_no_api_key(provider):
            pytest.skip(f"No API key available for {provider} integration tests")

    return _skip_for_provider

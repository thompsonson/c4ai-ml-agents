"""Tests for API clients."""

import time
from typing import Any, Dict
from unittest.mock import MagicMock, Mock, patch

import pytest

from ml_agents.config import ExperimentConfig
from ml_agents.utils.api_clients import (
    AnthropicClient,
    APIClient,
    APIClientError,
    AuthenticationError,
    CohereClient,
    ModelNotFoundError,
    OpenRouterClient,
    RateLimitError,
    create_api_client,
)


class TestAPIClient:
    """Test abstract API client base class."""

    class ConcreteAPIClient(APIClient):
        """Concrete implementation for testing."""

        def generate(self, prompt: str, **kwargs) -> Dict[str, Any]:
            return {"text": "test response", "provider": self.provider}

        def validate_connection(self) -> bool:
            return True

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return ExperimentConfig(
            provider="openrouter",
            model="openai/gpt-oss-120b",
            temperature=0.7,
            max_tokens=100,
            top_p=0.9,
            request_timeout=30,
        )

    @pytest.fixture
    def client(self, config):
        """Create concrete client instance."""
        return self.ConcreteAPIClient(config)

    def test_init(self, config):
        """Test client initialization."""
        client = self.ConcreteAPIClient(config)

        assert client.config == config
        assert client.provider == "openrouter"
        assert client.model == "openai/gpt-oss-120b"
        assert client.temperature == 0.7
        assert client.max_tokens == 100
        assert client.top_p == 0.9
        assert client.timeout == 30

    def test_get_generation_params_defaults(self, client):
        """Test getting default generation parameters."""
        params = client.get_generation_params()

        assert params["temperature"] == 0.7
        assert params["max_tokens"] == 100
        assert params["top_p"] == 0.9

    def test_get_generation_params_with_overrides(self, client):
        """Test generation parameters with overrides."""
        params = client.get_generation_params(temperature=0.5, max_tokens=200)

        assert params["temperature"] == 0.5
        assert params["max_tokens"] == 200
        assert params["top_p"] == 0.9  # Not overridden

    def test_handle_api_error_rate_limit(self, client):
        """Test handling rate limit errors."""
        error = Exception("Rate limit exceeded")

        with pytest.raises(RateLimitError, match="Rate limit exceeded"):
            client.handle_api_error(error)

    def test_handle_api_error_authentication(self, client):
        """Test handling authentication errors."""
        error = Exception("Authentication failed with 401")

        with pytest.raises(AuthenticationError, match="Authentication failed"):
            client.handle_api_error(error)

    def test_handle_api_error_model_not_found(self, client):
        """Test handling model not found errors."""
        error = Exception("Model not found 404")

        with pytest.raises(ModelNotFoundError, match="Model not found"):
            client.handle_api_error(error)

    def test_handle_api_error_generic(self, client):
        """Test handling generic API errors."""
        error = Exception("Generic API error")

        with pytest.raises(APIClientError, match="API request failed"):
            client.handle_api_error(error)


class TestAnthropicClient:
    """Test Anthropic client."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return ExperimentConfig(
            provider="anthropic",
            model="claude-sonnet-4-20250514",
            temperature=0.3,
            max_tokens=100,
        )

    @patch("ml_agents.utils.api_clients.get_api_key", return_value="test_key")
    @patch("ml_agents.utils.api_clients.anthropic.Anthropic")
    def test_init_success(self, mock_anthropic, mock_get_key, config):
        """Test successful Anthropic client initialization."""
        client = AnthropicClient(config)

        assert client.provider == "anthropic"
        assert client.model == "claude-sonnet-4-20250514"
        mock_anthropic.assert_called_once_with(api_key="test_key")

    @patch("ml_agents.utils.api_clients.get_api_key", return_value=None)
    def test_init_no_api_key(self, mock_get_key, config):
        """Test initialization without API key."""
        with pytest.raises(AuthenticationError, match="Anthropic API key not found"):
            AnthropicClient(config)

    @patch("ml_agents.utils.api_clients.get_api_key", return_value="test_key")
    @patch("ml_agents.utils.api_clients.anthropic.Anthropic")
    def test_generate_success(self, mock_anthropic, mock_get_key, config):
        """Test successful text generation."""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.content = [Mock(text="Generated response")]
        mock_response.usage.input_tokens = 10
        mock_response.usage.output_tokens = 20
        mock_response.id = "response_123"

        mock_client.messages.create.return_value = mock_response
        mock_anthropic.return_value = mock_client

        client = AnthropicClient(config)
        result = client.generate("Test prompt")

        assert result.text == "Generated response"
        assert result.provider == "anthropic"
        assert result.model == "claude-sonnet-4-20250514"
        assert result.prompt_tokens == 10
        assert result.completion_tokens == 20
        assert result.total_tokens == 30
        assert result.response_id == "response_123"

    @patch("ml_agents.utils.api_clients.get_api_key", return_value="test_key")
    @patch("ml_agents.utils.api_clients.anthropic.Anthropic")
    def test_validate_connection_success(self, mock_anthropic, mock_get_key, config):
        """Test successful connection validation."""
        mock_client = Mock()
        mock_client.messages.create.return_value = Mock()
        mock_anthropic.return_value = mock_client

        client = AnthropicClient(config)
        result = client.validate_connection()

        assert result is True


class TestCohereClient:
    """Test Cohere client."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return ExperimentConfig(
            provider="cohere", model="command-r", temperature=0.4, max_tokens=150
        )

    @patch("ml_agents.utils.api_clients.get_api_key", return_value="test_key")
    @patch("ml_agents.utils.api_clients.cohere.Client")
    def test_init_success(self, mock_cohere, mock_get_key, config):
        """Test successful Cohere client initialization."""
        client = CohereClient(config)

        assert client.provider == "cohere"
        assert client.model == "command-r"
        mock_cohere.assert_called_once_with("test_key")

    @patch("ml_agents.utils.api_clients.get_api_key", return_value=None)
    def test_init_no_api_key(self, mock_get_key, config):
        """Test initialization without API key."""
        with pytest.raises(AuthenticationError, match="Cohere API key not found"):
            CohereClient(config)

    @patch("ml_agents.utils.api_clients.get_api_key", return_value="test_key")
    @patch("ml_agents.utils.api_clients.cohere.Client")
    def test_generate_success(self, mock_cohere, mock_get_key, config):
        """Test successful text generation."""
        mock_client = Mock()
        mock_generation = Mock()
        mock_generation.text = "  Generated response  "
        mock_generation.id = "gen_123"

        mock_response = Mock()
        mock_response.generations = [mock_generation]

        mock_client.generate.return_value = mock_response
        mock_cohere.return_value = mock_client

        client = CohereClient(config)
        result = client.generate("Test prompt")

        assert result.text == "Generated response"
        assert result.provider == "cohere"
        assert result.model == "command-r"
        assert result.response_id == "gen_123"
        assert result.generation_time is not None

    @patch("ml_agents.utils.api_clients.get_api_key", return_value="test_key")
    @patch("ml_agents.utils.api_clients.cohere.Client")
    def test_validate_connection_success(self, mock_cohere, mock_get_key, config):
        """Test successful connection validation."""
        mock_client = Mock()
        mock_client.generate.return_value = Mock()
        mock_cohere.return_value = mock_client

        client = CohereClient(config)
        result = client.validate_connection()

        assert result is True


class TestOpenRouterClient:
    """Test OpenRouter client."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return ExperimentConfig(
            provider="openrouter",
            model="openai/gpt-oss-120b",
            temperature=0.6,
            max_tokens=200,
        )

    @patch("ml_agents.utils.api_clients.get_api_key", return_value="test_key")
    @patch("ml_agents.utils.api_clients.openai.OpenAI")
    def test_init_success(self, mock_openai, mock_get_key, config):
        """Test successful OpenRouter client initialization."""
        client = OpenRouterClient(config)

        assert client.provider == "openrouter"
        assert client.model == "openai/gpt-oss-120b"
        mock_openai.assert_called_once_with(
            base_url="https://openrouter.ai/api/v1", api_key="test_key"
        )

    @patch("ml_agents.utils.api_clients.get_api_key", return_value=None)
    def test_init_no_api_key(self, mock_get_key, config):
        """Test initialization without API key."""
        with pytest.raises(AuthenticationError, match="OpenRouter API key not found"):
            OpenRouterClient(config)

    @patch("ml_agents.utils.api_clients.get_api_key", return_value="test_key")
    @patch("ml_agents.utils.api_clients.openai.OpenAI")
    def test_generate_success(self, mock_openai, mock_get_key, config):
        """Test successful text generation."""
        mock_client = Mock()

        mock_choice = Mock()
        mock_choice.message.content = "  Generated response  "

        mock_usage = Mock()
        mock_usage.prompt_tokens = 15
        mock_usage.completion_tokens = 25
        mock_usage.total_tokens = 40

        mock_response = Mock()
        mock_response.choices = [mock_choice]
        mock_response.usage = mock_usage
        mock_response.id = "chatcmpl_123"

        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client

        client = OpenRouterClient(config)
        result = client.generate("Test prompt")

        assert result.text == "Generated response"
        assert result.provider == "openrouter"
        assert result.model == "openai/gpt-oss-120b"
        assert result.prompt_tokens == 15
        assert result.completion_tokens == 25
        assert result.total_tokens == 40
        assert result.response_id == "chatcmpl_123"

    @patch("ml_agents.utils.api_clients.get_api_key", return_value="test_key")
    @patch("ml_agents.utils.api_clients.openai.OpenAI")
    def test_validate_connection_success(self, mock_openai, mock_get_key, config):
        """Test successful connection validation."""
        mock_client = Mock()
        mock_client.chat.completions.create.return_value = Mock()
        mock_openai.return_value = mock_client

        client = OpenRouterClient(config)
        result = client.validate_connection()

        assert result is True


class TestCreateAPIClient:
    """Test API client factory function."""

    @patch("ml_agents.utils.api_clients.get_api_key", return_value="test_key")
    def test_create_anthropic_client(self, mock_get_key):
        """Test creating Anthropic client."""
        config = ExperimentConfig(
            provider="anthropic", model="claude-sonnet-4-20250514"
        )

        with patch("ml_agents.utils.api_clients.anthropic.Anthropic"):
            client = create_api_client(config)
            assert isinstance(client, AnthropicClient)

    @patch("ml_agents.utils.api_clients.get_api_key", return_value="test_key")
    def test_create_cohere_client(self, mock_get_key):
        """Test creating Cohere client."""
        config = ExperimentConfig(provider="cohere", model="command-r")

        with patch("ml_agents.utils.api_clients.cohere.Client"):
            client = create_api_client(config)
            assert isinstance(client, CohereClient)

    @patch("ml_agents.utils.api_clients.get_api_key", return_value="test_key")
    def test_create_openrouter_client(self, mock_get_key):
        """Test creating OpenRouter client."""
        config = ExperimentConfig(provider="openrouter", model="openai/gpt-oss-120b")

        with patch("ml_agents.utils.api_clients.openai.OpenAI"):
            client = create_api_client(config)
            assert isinstance(client, OpenRouterClient)

    def test_create_unsupported_client(self):
        """Test creating client for unsupported provider."""
        # Create a config with valid values first, then modify the provider
        config = ExperimentConfig(provider="openrouter", model="openai/gpt-oss-120b")
        config.provider = "unsupported"  # Bypass validation

        with pytest.raises(ValueError, match="Unsupported provider: unsupported"):
            create_api_client(config)

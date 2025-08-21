"""API client wrappers for different model providers."""

import asyncio
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import anthropic
import cohere
import openai
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.config import ExperimentConfig, get_api_key
from src.utils.logging_config import get_logger
from src.utils.rate_limiter import rate_limiter_manager

logger = get_logger(__name__)


@dataclass
class StandardResponse:
    """Standardized response format for all API clients."""

    # Core response data
    text: str
    provider: str
    model: str

    # Token usage
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

    # Timing and performance
    generation_time: float

    # Generation parameters used
    parameters: Dict[str, Any]

    # Provider-specific metadata (optional)
    response_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert response to dictionary."""
        return {
            "text": self.text,
            "provider": self.provider,
            "model": self.model,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "generation_time": self.generation_time,
            "parameters": self.parameters,
            "response_id": self.response_id,
            "metadata": self.metadata,
        }


class APIClientError(Exception):
    """Base exception for API client errors."""

    pass


class RateLimitError(APIClientError):
    """Exception raised when rate limits are exceeded."""

    pass


class AuthenticationError(APIClientError):
    """Exception raised when authentication fails."""

    pass


class ModelNotFoundError(APIClientError):
    """Exception raised when model is not found."""

    pass


class APIClient(ABC):
    """Abstract base class for API clients."""

    def __init__(self, config: ExperimentConfig) -> None:
        """Initialize API client with configuration.

        Args:
            config: Experiment configuration
        """
        self.config = config
        self.provider = config.provider
        self.model = config.model
        self.temperature = config.temperature
        self.max_tokens = config.max_tokens
        self.top_p = config.top_p
        self.timeout = config.request_timeout

        # Initialize rate limiter for this provider
        self.rate_limiter = rate_limiter_manager.get_limiter(self.provider)

        logger.info(f"Initialized {self.__class__.__name__} for model: {self.model}")

    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> StandardResponse:
        """Generate response from model.

        Args:
            prompt: Input prompt
            **kwargs: Additional generation parameters

        Returns:
            Standardized response object
        """
        pass

    @abstractmethod
    def validate_connection(self) -> bool:
        """Validate that the client can connect to the API.

        Returns:
            True if connection is valid

        Raises:
            AuthenticationError: If authentication fails
            APIClientError: If connection fails
        """
        pass

    def get_generation_params(self, **overrides) -> Dict[str, Any]:
        """Get generation parameters with optional overrides.

        Args:
            **overrides: Parameters to override defaults

        Returns:
            Dictionary of generation parameters
        """
        params = {
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
        }
        params.update(overrides)
        return params

    def handle_api_error(self, error: Exception) -> None:
        """Handle and re-raise API errors with appropriate types.

        Args:
            error: Original exception

        Raises:
            APIClientError: Appropriate error type based on original error
        """
        error_msg = str(error)

        if "rate limit" in error_msg.lower() or "429" in error_msg:
            raise RateLimitError(f"Rate limit exceeded: {error_msg}") from error
        elif "authentication" in error_msg.lower() or "401" in error_msg:
            raise AuthenticationError(f"Authentication failed: {error_msg}") from error
        elif "not found" in error_msg.lower() or "404" in error_msg:
            raise ModelNotFoundError(f"Model not found: {error_msg}") from error
        else:
            raise APIClientError(f"API request failed: {error_msg}") from error


class HuggingFaceClient(APIClient):
    """HuggingFace Transformers client for local inference."""

    def __init__(self, config: ExperimentConfig) -> None:
        """Initialize HuggingFace client."""
        super().__init__(config)
        self.tokenizer: Optional[Any] = None
        self.model_instance: Optional[Any] = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load model and tokenizer lazily
        self._model_loaded = False

    def _load_model(self) -> None:
        """Load model and tokenizer."""
        if self._model_loaded:
            return

        try:
            logger.info(f"Loading HuggingFace model: {self.model}")

            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model, trust_remote_code=True, token=get_api_key("huggingface")
            )

            self.model_instance = AutoModelForCausalLM.from_pretrained(
                self.model,
                trust_remote_code=True,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
                token=get_api_key("huggingface"),
            )

            # Add pad token if missing
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            self._model_loaded = True
            logger.info(f"Successfully loaded model on {self.device}")

        except Exception as e:
            self.handle_api_error(e)

    def generate(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Generate response using HuggingFace model.

        Args:
            prompt: Input prompt
            **kwargs: Additional generation parameters

        Returns:
            Dictionary with generated text and metadata
        """
        self._load_model()

        # Apply rate limiting
        self.rate_limiter.acquire()

        start_time = time.time()

        try:
            # Tokenize input
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048,
            )

            if self.device == "cuda":
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Generate parameters
            gen_params = self.get_generation_params(**kwargs)

            # Generate response
            with torch.no_grad():
                outputs = self.model_instance.generate(
                    **inputs,
                    max_new_tokens=gen_params["max_tokens"],
                    temperature=gen_params["temperature"],
                    top_p=gen_params["top_p"],
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )

            # Decode response
            input_length = inputs["input_ids"].shape[1]
            generated_tokens = outputs[0][input_length:]
            response_text = self.tokenizer.decode(
                generated_tokens, skip_special_tokens=True
            )

            end_time = time.time()

            return StandardResponse(
                text=response_text.strip(),
                provider=self.provider,
                model=self.model,
                prompt_tokens=input_length,
                completion_tokens=len(generated_tokens),
                total_tokens=input_length + len(generated_tokens),
                generation_time=end_time - start_time,
                parameters=gen_params,
            )

        except Exception as e:
            self.handle_api_error(e)

    def cleanup_model(self) -> None:
        """Clean up model resources and free GPU memory."""
        if self._model_loaded:
            logger.info("Cleaning up HuggingFace model resources")

            # Delete model and tokenizer
            if self.model_instance is not None:
                del self.model_instance
                self.model_instance = None

            if self.tokenizer is not None:
                del self.tokenizer
                self.tokenizer = None

            # Clear GPU cache if using CUDA
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.debug("Cleared CUDA cache")

            self._model_loaded = False
            logger.info("Model cleanup completed")

    def __del__(self) -> None:
        """Destructor to ensure proper cleanup."""
        try:
            self.cleanup_model()
        except Exception:
            pass  # Ignore errors during cleanup  # nosec B110

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleanup model."""
        self.cleanup_model()

    def validate_connection(self) -> bool:
        """Validate HuggingFace model access."""
        try:
            self._load_model()
            return True
        except Exception as e:
            logger.error(f"HuggingFace validation failed: {e}")
            raise


class AnthropicClient(APIClient):
    """Anthropic Claude API client."""

    def __init__(self, config: ExperimentConfig) -> None:
        """Initialize Anthropic client."""
        super().__init__(config)
        api_key = get_api_key("anthropic")

        if not api_key:
            raise AuthenticationError("Anthropic API key not found")

        self.client = anthropic.Anthropic(api_key=api_key)

    def generate(self, prompt: str, **kwargs) -> StandardResponse:
        """Generate response using Anthropic Claude.

        Args:
            prompt: Input prompt
            **kwargs: Additional generation parameters

        Returns:
            Dictionary with generated text and metadata
        """
        # Apply rate limiting
        self.rate_limiter.acquire()

        start_time = time.time()

        try:
            gen_params = self.get_generation_params(**kwargs)

            response = self.client.messages.create(
                model=self.model,
                max_tokens=gen_params["max_tokens"],
                temperature=gen_params["temperature"],
                top_p=gen_params["top_p"],
                messages=[{"role": "user", "content": prompt}],
            )

            end_time = time.time()

            return StandardResponse(
                text=response.content[0].text,
                provider=self.provider,
                model=self.model,
                prompt_tokens=response.usage.input_tokens,
                completion_tokens=response.usage.output_tokens,
                total_tokens=response.usage.input_tokens + response.usage.output_tokens,
                generation_time=end_time - start_time,
                parameters=gen_params,
                response_id=response.id,
            )

        except Exception as e:
            self.handle_api_error(e)

    def validate_connection(self) -> bool:
        """Validate Anthropic API connection."""
        try:
            # Test with a minimal request
            response = self.client.messages.create(
                model=self.model,
                max_tokens=1,
                messages=[{"role": "user", "content": "Hi"}],
            )
            return response is not None
        except Exception as e:
            logger.error(f"Anthropic validation failed: {e}")
            raise


class CohereClient(APIClient):
    """Cohere API client."""

    def __init__(self, config: ExperimentConfig) -> None:
        """Initialize Cohere client."""
        super().__init__(config)
        api_key = get_api_key("cohere")

        if not api_key:
            raise AuthenticationError("Cohere API key not found")

        self.client = cohere.Client(api_key)

    def generate(self, prompt: str, **kwargs) -> StandardResponse:
        """Generate response using Cohere.

        Args:
            prompt: Input prompt
            **kwargs: Additional generation parameters

        Returns:
            Dictionary with generated text and metadata
        """
        # Apply rate limiting
        self.rate_limiter.acquire()

        start_time = time.time()

        try:
            gen_params = self.get_generation_params(**kwargs)

            response = self.client.generate(
                model=self.model,
                prompt=prompt,
                max_tokens=gen_params["max_tokens"],
                temperature=gen_params["temperature"],
                p=gen_params["top_p"],
                stop_sequences=None,
                return_likelihoods="NONE",
            )

            end_time = time.time()

            return StandardResponse(
                text=response.generations[0].text.strip(),
                provider=self.provider,
                model=self.model,
                prompt_tokens=len(prompt.split()),  # Approximation
                completion_tokens=len(response.generations[0].text.split()),
                total_tokens=len(prompt.split())
                + len(response.generations[0].text.split()),
                generation_time=end_time - start_time,
                parameters=gen_params,
                response_id=response.generations[0].id,
            )

        except Exception as e:
            self.handle_api_error(e)

    def validate_connection(self) -> bool:
        """Validate Cohere API connection."""
        try:
            # Test with a minimal request
            response = self.client.generate(model=self.model, prompt="Hi", max_tokens=1)
            return response is not None
        except Exception as e:
            logger.error(f"Cohere validation failed: {e}")
            raise


class OpenRouterClient(APIClient):
    """OpenRouter API client."""

    def __init__(self, config: ExperimentConfig) -> None:
        """Initialize OpenRouter client."""
        super().__init__(config)
        api_key = get_api_key("openrouter")

        if not api_key:
            raise AuthenticationError("OpenRouter API key not found")

        self.client = openai.OpenAI(
            base_url="https://openrouter.ai/api/v1", api_key=api_key
        )

    def generate(self, prompt: str, **kwargs) -> StandardResponse:
        """Generate response using OpenRouter.

        Args:
            prompt: Input prompt
            **kwargs: Additional generation parameters

        Returns:
            Dictionary with generated text and metadata
        """
        # Apply rate limiting
        self.rate_limiter.acquire()

        start_time = time.time()

        try:
            gen_params = self.get_generation_params(**kwargs)

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=gen_params["max_tokens"],
                temperature=gen_params["temperature"],
                top_p=gen_params["top_p"],
            )

            end_time = time.time()

            response_text = (
                response.choices[0].message.content.strip()
                if response.choices[0].message.content
                else ""
            )

            # Handle empty responses due to token limits
            if not response_text and response.choices[0].finish_reason == "length":
                response_text = "[RESPONSE_TRUNCATED_DUE_TO_TOKEN_LIMIT]"
                logger.warning(
                    f"Response truncated due to token limit (max_tokens={self.max_tokens})"
                )
            elif not response_text:
                response_text = "[EMPTY_RESPONSE]"
                logger.warning("Received empty response from API")

            return StandardResponse(
                text=response_text,
                provider=self.provider,
                model=self.model,
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
                total_tokens=response.usage.total_tokens,
                generation_time=end_time - start_time,
                parameters=gen_params,
                response_id=response.id,
            )

        except Exception as e:
            self.handle_api_error(e)

    def validate_connection(self) -> bool:
        """Validate OpenRouter API connection."""
        try:
            # Test with a minimal request
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": "Hi"}],
                max_tokens=1,
            )
            return response is not None
        except Exception as e:
            logger.error(f"OpenRouter validation failed: {e}")
            raise


def create_api_client(config: ExperimentConfig) -> APIClient:
    """Factory function to create appropriate API client.

    Args:
        config: Experiment configuration

    Returns:
        Appropriate API client instance

    Raises:
        ValueError: If provider is not supported
    """
    client_map = {
        "huggingface": HuggingFaceClient,
        "anthropic": AnthropicClient,
        "cohere": CohereClient,
        "openrouter": OpenRouterClient,
    }

    if config.provider not in client_map:
        raise ValueError(f"Unsupported provider: {config.provider}")

    client_class = client_map[config.provider]
    return client_class(config)

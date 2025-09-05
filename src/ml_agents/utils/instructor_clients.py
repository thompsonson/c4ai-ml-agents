"""Instructor client manager for provider-aware structured output extraction.

This module provides the InstructorClientManager class that handles provider-specific
Instructor client configuration with automatic fallback mechanisms.
"""

from typing import Any, Type

import instructor
from pydantic import BaseModel

from ml_agents.utils.logging_config import get_logger

logger = get_logger(__name__)


class InstructorClientManager:
    """Manages Instructor client configuration with provider-aware mode selection.

    This class handles the complexity of different provider capabilities and
    automatically selects the optimal Instructor mode (TOOLS vs JSON) based
    on provider support, with graceful fallback to secondary modes.

    Args:
        api_client: The API client instance (contains .client and .provider attributes)
    """

    def __init__(self, api_client) -> None:
        """Initialize the Instructor client manager.

        Args:
            api_client: API client with .client and .provider attributes
        """
        self.api_client = api_client
        self.provider = api_client.provider
        self._instructor_client_cache = {}

        logger.debug(
            f"Initialized InstructorClientManager for provider: {self.provider}"
        )

    def get_instructor_client(self, mode: str = None):
        """Get Instructor client for provider with specified mode.

        Args:
            mode: Specific mode to use, or None for automatic selection

        Returns:
            Configured Instructor client

        Raises:
            ValueError: If provider is not supported
        """
        if mode is None:
            mode = self.get_primary_mode()

        # Use cache to avoid recreating clients
        cache_key = f"{self.provider}_{mode}"
        if cache_key in self._instructor_client_cache:
            return self._instructor_client_cache[cache_key]

        # Create provider-specific Instructor client
        try:
            if self.provider == "anthropic":
                client = instructor.from_anthropic(
                    self.api_client.client, mode=getattr(instructor.Mode, mode)
                )
            elif self.provider == "openrouter":
                client = instructor.from_openai(
                    self.api_client.client, mode=getattr(instructor.Mode, mode)
                )
            elif self.provider == "cohere":
                client = instructor.from_cohere(
                    self.api_client.client, mode=getattr(instructor.Mode, mode)
                )
            elif self.provider == "local-openai":
                client = instructor.from_openai(
                    self.api_client.client, mode=getattr(instructor.Mode, mode)
                )
            else:
                raise ValueError(f"Unsupported provider: {self.provider}")

            # Cache the client
            self._instructor_client_cache[cache_key] = client
            logger.debug(
                f"Created Instructor client for {self.provider} with {mode} mode"
            )
            return client

        except Exception as e:
            logger.error(
                f"Failed to create Instructor client for {self.provider} with {mode} mode: {e}"
            )
            raise

    def get_primary_mode(self) -> str:
        """Get primary mode for provider.

        Returns:
            Primary Instructor mode for the provider
        """
        mode_map = {
            "anthropic": "ANTHROPIC_TOOLS",
            "openrouter": "TOOLS",
            "cohere": "JSON",
            "local-openai": "TOOLS",
        }
        return mode_map[self.provider]

    def get_fallback_mode(self) -> str:
        """Get fallback mode for provider.

        Returns:
            Fallback Instructor mode for the provider
        """
        fallback_map = {
            "anthropic": "ANTHROPIC_JSON",
            "openrouter": "JSON",
            "cohere": "JSON",  # Same as primary for Cohere
            "local-openai": "JSON",
        }
        return fallback_map[self.provider]

    def extract_structured_response(
        self, messages: list, response_model: Type[BaseModel], **kwargs
    ) -> Any:
        """Extract structured response with provider-aware fallback.

        Args:
            messages: List of messages for the chat completion
            response_model: Pydantic model class for structured output
            **kwargs: Additional arguments for the completion call

        Returns:
            Instance of response_model with extracted structured data

        Raises:
            Exception: If both primary and fallback modes fail
        """
        primary_mode = self.get_primary_mode()
        fallback_mode = self.get_fallback_mode()

        # Try primary mode first
        try:
            logger.debug(f"Attempting structured extraction with {primary_mode} mode")
            client = self.get_instructor_client(primary_mode)

            result = client.chat.completions.create(
                model=self.api_client.model,
                response_model=response_model,
                messages=messages,
                temperature=kwargs.get(
                    "temperature", getattr(self.api_client, "temperature", 0.3)
                ),
                max_tokens=kwargs.get(
                    "max_tokens", getattr(self.api_client, "max_tokens", 2000)
                ),
                **{
                    k: v
                    for k, v in kwargs.items()
                    if k not in ["temperature", "max_tokens"]
                },
            )

            logger.debug(f"Structured extraction successful with {primary_mode} mode")
            return result

        except Exception as e:
            logger.warning(f"Primary mode {primary_mode} failed: {e}")

            # Try fallback mode if different from primary
            if primary_mode != fallback_mode:
                try:
                    logger.debug(
                        f"Attempting fallback extraction with {fallback_mode} mode"
                    )
                    client = self.get_instructor_client(fallback_mode)

                    result = client.chat.completions.create(
                        model=self.api_client.model,
                        response_model=response_model,
                        messages=messages,
                        temperature=kwargs.get(
                            "temperature", getattr(self.api_client, "temperature", 0.3)
                        ),
                        max_tokens=kwargs.get(
                            "max_tokens", getattr(self.api_client, "max_tokens", 2000)
                        ),
                        **{
                            k: v
                            for k, v in kwargs.items()
                            if k not in ["temperature", "max_tokens"]
                        },
                    )

                    logger.info(
                        f"Fallback extraction successful with {fallback_mode} mode"
                    )
                    return result

                except Exception as fallback_error:
                    logger.error(
                        f"Fallback mode {fallback_mode} also failed: {fallback_error}"
                    )
                    raise fallback_error
            else:
                # No fallback available, re-raise original error
                raise e

    def is_provider_supported(self) -> bool:
        """Check if the current provider is supported.

        Returns:
            True if provider is supported, False otherwise
        """
        supported_providers = ["anthropic", "openrouter", "cohere", "local-openai"]
        return self.provider in supported_providers

    def get_provider_capabilities(self) -> dict:
        """Get provider capabilities and supported modes.

        Returns:
            Dictionary with provider capability information
        """
        return {
            "provider": self.provider,
            "primary_mode": self.get_primary_mode(),
            "fallback_mode": self.get_fallback_mode(),
            "has_fallback": self.get_primary_mode() != self.get_fallback_mode(),
            "supported": self.is_provider_supported(),
        }

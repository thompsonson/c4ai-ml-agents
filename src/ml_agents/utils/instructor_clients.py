"""Instructor client manager for provider-aware structured output extraction.

This module provides the InstructorClientManager class that handles provider-specific
Instructor client configuration with direct integration with native client libraries.
"""

import asyncio
import os
from typing import Any, List, Type

import instructor
from pydantic import BaseModel

from ml_agents.config import ExperimentConfig, get_api_key
from ml_agents.utils.logging_config import get_logger

logger = get_logger(__name__)


class InstructorClientManager:
    """Manages Instructor client configuration with provider-aware mode selection.

    This class handles the complexity of different provider capabilities and
    automatically selects the optimal Instructor mode (TOOLS vs JSON) based
    on provider support. Creates native clients directly from configuration.

    Args:
        config: ExperimentConfig with provider, model, and API settings
    """

    def __init__(self, config: ExperimentConfig) -> None:
        """Initialize the Instructor client manager.

        Args:
            config: ExperimentConfig with provider and model settings
        """
        self.config = config
        self.provider = config.provider
        self.model = config.model
        self.temperature = config.temperature
        self.max_tokens = config.max_tokens
        self._instructor_client_cache = {}
        self._sync_client = None
        self._async_client = None

        logger.debug(
            f"Initialized InstructorClientManager for provider: {self.provider}, model: {self.model}"
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

        # Create provider-specific Instructor client from native client
        try:
            sync_client = self._get_sync_client()

            if self.provider == "anthropic":
                client = instructor.from_anthropic(
                    sync_client, mode=getattr(instructor.Mode, mode)
                )
            elif self.provider in ["openrouter", "local-openai"]:
                client = instructor.from_openai(
                    sync_client, mode=getattr(instructor.Mode, mode)
                )
            elif self.provider == "cohere":
                client = instructor.from_cohere(
                    sync_client, mode=getattr(instructor.Mode, mode)
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
                model=self.model,
                response_model=response_model,
                messages=messages,
                temperature=kwargs.get("temperature", self.temperature),
                max_tokens=kwargs.get("max_tokens", self.max_tokens),
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
                        model=self.model,
                        response_model=response_model,
                        messages=messages,
                        temperature=kwargs.get("temperature", self.temperature),
                        max_tokens=kwargs.get("max_tokens", self.max_tokens),
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

    def _get_sync_client(self):
        """Get or create synchronous native client."""
        if self._sync_client is None:
            self._sync_client = self._create_native_client(async_client=False)
        return self._sync_client

    def _get_async_client(self):
        """Get or create asynchronous native client for concurrent operations."""
        if self._async_client is None:
            self._async_client = self._create_native_client(async_client=True)
        return self._async_client

    def _create_native_client(self, async_client: bool = False):
        """Create native client (sync or async) based on provider.

        Args:
            async_client: If True, create async client; otherwise sync client

        Returns:
            Native client instance (OpenAI, Anthropic, Cohere)
        """
        if self.provider == "local-openai":
            import openai

            base_url = getattr(self.config, "api_base_url", "http://pop-os:8000/v1")
            client_class = openai.AsyncOpenAI if async_client else openai.OpenAI
            # vLLM server doesn't require authentication when no --api-key is set
            # We need to avoid sending any Authorization header
            try:
                # Try without api_key parameter - this might avoid sending auth header
                return client_class(base_url=base_url)
            except Exception:
                # Fallback to minimal key if required by OpenAI client
                return client_class(base_url=base_url, api_key="dummy")
        elif self.provider == "openrouter":
            import openai

            api_key = get_api_key("openrouter")
            if not api_key:
                raise ValueError("OpenRouter API key not found")
            client_class = openai.AsyncOpenAI if async_client else openai.OpenAI
            return client_class(
                base_url="https://openrouter.ai/api/v1", api_key=api_key
            )
        elif self.provider == "anthropic":
            import anthropic

            api_key = get_api_key("anthropic")
            if not api_key:
                raise ValueError("Anthropic API key not found")
            client_class = (
                anthropic.AsyncAnthropic if async_client else anthropic.Anthropic
            )
            return client_class(api_key=api_key)
        elif self.provider == "cohere":
            import cohere

            api_key = get_api_key("cohere")
            if not api_key:
                raise ValueError("Cohere API key not found")
            if async_client:
                return cohere.AsyncClient(api_key=api_key)
            else:
                return cohere.Client(api_key=api_key)
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    def _get_async_instructor_client(self, mode: str = None):
        """Get async Instructor client for concurrent operations."""
        if mode is None:
            mode = self.get_primary_mode()

        cache_key = f"async_{self.provider}_{mode}"
        if cache_key in self._instructor_client_cache:
            return self._instructor_client_cache[cache_key]

        # Create async Instructor client from native async client
        async_client = self._get_async_client()

        try:
            if self.provider in ["local-openai", "openrouter"]:
                client = instructor.from_openai(
                    async_client, mode=getattr(instructor.Mode, mode)
                )
            elif self.provider == "anthropic":
                client = instructor.from_anthropic(
                    async_client, mode=getattr(instructor.Mode, mode)
                )
            elif self.provider == "cohere":
                client = instructor.from_cohere(
                    async_client, mode=getattr(instructor.Mode, mode)
                )
            else:
                raise ValueError(f"Unsupported provider for async: {self.provider}")

            self._instructor_client_cache[cache_key] = client
            logger.debug(
                f"Created async Instructor client for {self.provider} with {mode} mode"
            )
            return client

        except Exception as e:
            logger.error(
                f"Failed to create async Instructor client for {self.provider}: {e}"
            )
            raise

    async def extract_structured_response_async(
        self, messages: list, response_model: Type[BaseModel], **kwargs
    ) -> Any:
        """Extract structured response asynchronously with Instructor.

        Args:
            messages: List of messages for the chat completion
            response_model: Pydantic model class for structured output
            **kwargs: Additional arguments for the completion call

        Returns:
            Instance of response_model with extracted structured data

        Raises:
            Exception: If async structured extraction fails
        """
        # Get primary mode for async operation (no fallbacks in async)
        primary_mode = self.get_primary_mode()

        import time

        api_start_time = time.time()
        logger.debug(
            f"📞 Attempting async API call to {self.provider} with {primary_mode} mode (model: {self.model})"
        )
        client = self._get_async_instructor_client(primary_mode)

        # Pure Instructor async implementation - NO FALLBACKS
        result = await client.chat.completions.create(
            model=self.model,
            response_model=response_model,
            messages=messages,
            temperature=kwargs.get("temperature", self.temperature),
            max_tokens=kwargs.get("max_tokens", self.max_tokens),
            **{
                k: v
                for k, v in kwargs.items()
                if k not in ["temperature", "max_tokens"]
            },
        )

        api_end_time = time.time()
        logger.debug(
            f"✅ API call to {self.provider} successful in {api_end_time - api_start_time:.2f}s with {primary_mode} mode"
        )
        return result

    async def extract_concurrent_responses(
        self,
        messages_list: List[List[dict]],
        response_model: Type[BaseModel],
        concurrency_limit: int = 10,
        **kwargs,
    ) -> List[BaseModel]:
        """Extract multiple structured responses concurrently.

        Args:
            messages_list: List of message lists for concurrent chat completions
            response_model: Pydantic model class for structured output
            concurrency_limit: Maximum number of concurrent operations
            **kwargs: Additional arguments for the completion calls

        Returns:
            List of response_model instances with extracted structured data

        Raises:
            Exception: If concurrent extraction fails
        """
        logger.info(
            f"🚀 Starting concurrent extraction of {len(messages_list)} requests with semaphore limit {concurrency_limit} to {self.provider}"
        )
        logger.debug(
            f"📋 First few messages_list items: {messages_list[:3] if len(messages_list) >= 3 else messages_list}"
        )

        # Create semaphore to limit concurrent operations
        semaphore = asyncio.Semaphore(concurrency_limit)

        async def bounded_extract(messages, request_id):
            """Extract single response with semaphore limiting."""
            logger.debug(
                f"🔄 Request {request_id}: messages={messages}, type={type(messages)}"
            )

            # Check if messages is None or empty
            if messages is None:
                logger.error(f"❌ Request {request_id}: messages is None!")
                return None
            if not messages:
                logger.error(f"❌ Request {request_id}: messages is empty!")
                return None

            # Safer semaphore status logging
            active_count = (
                concurrency_limit - semaphore._value
                if hasattr(semaphore, "_value")
                else "unknown"
            )
            waiting_count = "unknown"
            if hasattr(semaphore, "_waiters") and semaphore._waiters is not None:
                waiting_count = len(semaphore._waiters)
            logger.debug(
                f"🔄 Request {request_id}: Waiting for semaphore (active: {active_count}, waiting: {waiting_count})"
            )

            async with semaphore:
                import time

                start_time = time.time()
                logger.info(
                    f"🚀 Request {request_id}: Starting API call to {self.provider} (semaphore acquired)"
                )

                try:
                    result = await self.extract_structured_response_async(
                        messages, response_model, **kwargs
                    )
                    end_time = time.time()
                    logger.info(
                        f"✅ Request {request_id}: Completed in {end_time - start_time:.2f}s"
                    )
                    return result
                except Exception as e:
                    end_time = time.time()
                    logger.warning(
                        f"❌ Request {request_id}: Failed after {end_time - start_time:.2f}s - {e}"
                    )
                    # Return None for failed extractions to support partial results
                    return None

        # Create tasks for all extractions with request IDs
        tasks = [bounded_extract(msgs, i + 1) for i, msgs in enumerate(messages_list)]

        # Execute all tasks concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter out None results and exceptions for partial results support
        successful_results = []
        failed_count = 0

        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Extraction {i} failed with exception: {result}")
                failed_count += 1
            elif result is None:
                logger.warning(f"Extraction {i} returned None (failed)")
                failed_count += 1
            else:
                successful_results.append(result)

        logger.info(
            f"🏁 Concurrent extraction completed: {len(successful_results)} successful, {failed_count} failed (total: {len(results)} requests to {self.provider})"
        )

        return (
            results  # Return all results including None/exceptions for caller to handle
        )

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

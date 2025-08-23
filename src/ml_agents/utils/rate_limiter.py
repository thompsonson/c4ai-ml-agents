"""Rate limiting utilities for API clients."""

import asyncio
import math
import random
import threading
import time
from dataclasses import dataclass
from typing import Dict, Optional

from ml_agents.utils.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class RateLimitConfig:
    """Configuration for rate limiting."""

    # Tokens per second
    tokens_per_second: float = 1.0

    # Maximum tokens in bucket
    max_tokens: int = 10

    # Maximum retry attempts
    max_retries: int = 5

    # Base backoff time in seconds
    base_backoff: float = 1.0

    # Maximum backoff time in seconds
    max_backoff: float = 60.0

    # Jitter factor (0.0 to 1.0)
    jitter: float = 0.1


# Provider-specific rate limits (tokens per second)
PROVIDER_RATE_LIMITS: Dict[str, RateLimitConfig] = {
    "anthropic": RateLimitConfig(
        tokens_per_second=5.0,  # Conservative for Claude
        max_tokens=20,
        max_retries=3,
        base_backoff=2.0,
        max_backoff=30.0,
    ),
    "cohere": RateLimitConfig(
        tokens_per_second=10.0,  # Cohere is more permissive
        max_tokens=50,
        max_retries=3,
        base_backoff=1.0,
        max_backoff=15.0,
    ),
    "openrouter": RateLimitConfig(
        tokens_per_second=2.0,  # Conservative for OpenRouter
        max_tokens=10,
        max_retries=5,
        base_backoff=1.5,
        max_backoff=45.0,
    ),
}


class TokenBucket:
    """Token bucket rate limiter implementation."""

    def __init__(self, config: RateLimitConfig) -> None:
        """Initialize token bucket.

        Args:
            config: Rate limiting configuration
        """
        self.config = config
        self.tokens_per_second = config.tokens_per_second
        self.max_tokens = config.max_tokens

        # Current token count
        self._tokens = float(config.max_tokens)
        self._last_update = time.time()
        self._lock = threading.Lock()

        logger.debug(
            f"Initialized TokenBucket: {self.tokens_per_second} tokens/sec, "
            f"max {self.max_tokens} tokens"
        )

    def acquire(self, tokens: int = 1) -> bool:
        """Attempt to acquire tokens from bucket.

        Args:
            tokens: Number of tokens to acquire

        Returns:
            True if tokens were acquired, False otherwise
        """
        with self._lock:
            now = time.time()

            # Add tokens based on elapsed time
            elapsed = now - self._last_update
            self._tokens = min(
                self.max_tokens, self._tokens + elapsed * self.tokens_per_second
            )
            self._last_update = now

            # Check if we have enough tokens
            if self._tokens >= tokens:
                self._tokens -= tokens
                return True

            return False

    def wait_time(self, tokens: int = 1) -> float:
        """Calculate wait time for acquiring tokens.

        Args:
            tokens: Number of tokens needed

        Returns:
            Time to wait in seconds
        """
        with self._lock:
            if self._tokens >= tokens:
                return 0.0

            tokens_needed = tokens - self._tokens
            return tokens_needed / self.tokens_per_second

    def get_status(self) -> Dict[str, float]:
        """Get current bucket status.

        Returns:
            Dictionary with current tokens and rate info
        """
        with self._lock:
            return {
                "current_tokens": self._tokens,
                "max_tokens": self.max_tokens,
                "tokens_per_second": self.tokens_per_second,
            }


class ExponentialBackoff:
    """Exponential backoff calculator with jitter."""

    def __init__(self, config: RateLimitConfig) -> None:
        """Initialize backoff calculator.

        Args:
            config: Rate limiting configuration
        """
        self.config = config
        self.base_backoff = config.base_backoff
        self.max_backoff = config.max_backoff
        self.max_retries = config.max_retries
        self.jitter = config.jitter

    def calculate_backoff(self, attempt: int) -> float:
        """Calculate backoff time for retry attempt.

        Args:
            attempt: Retry attempt number (starting from 0)

        Returns:
            Backoff time in seconds
        """
        if attempt >= self.max_retries:
            return self.max_backoff

        # Exponential backoff: base * 2^attempt
        backoff = self.base_backoff * (2**attempt)
        backoff = min(backoff, self.max_backoff)

        # Add jitter to avoid thundering herd
        jitter_amount = backoff * self.jitter
        jitter = random.uniform(-jitter_amount, jitter_amount)  # nosec B311

        return max(0.1, backoff + jitter)

    def should_retry(self, attempt: int) -> bool:
        """Check if we should retry.

        Args:
            attempt: Current retry attempt number

        Returns:
            True if we should retry, False otherwise
        """
        return attempt < self.max_retries


class RateLimiter:
    """Main rate limiter class combining token bucket and exponential backoff."""

    def __init__(self, provider: str, config: Optional[RateLimitConfig] = None) -> None:
        """Initialize rate limiter for a provider.

        Args:
            provider: Provider name (anthropic, cohere, etc.)
            config: Custom rate limit configuration (uses provider default if None)
        """
        self.provider = provider

        if config is None:
            config = PROVIDER_RATE_LIMITS.get(provider, RateLimitConfig())

        self.config = config
        self.bucket = TokenBucket(config)
        self.backoff = ExponentialBackoff(config)

        logger.info(
            f"Initialized RateLimiter for {provider}: "
            f"{config.tokens_per_second} tokens/sec, "
            f"max retries: {config.max_retries}"
        )

    async def acquire_async(self, tokens: int = 1) -> None:
        """Acquire tokens with async waiting.

        Args:
            tokens: Number of tokens to acquire
        """
        attempt = 0

        while attempt <= self.config.max_retries:
            if self.bucket.acquire(tokens):
                if attempt > 0:
                    logger.info(
                        f"Successfully acquired {tokens} tokens after {attempt} attempts"
                    )
                return

            if not self.backoff.should_retry(attempt):
                raise RuntimeError(
                    f"Rate limit exceeded for {self.provider} after {attempt} attempts"
                )

            # Calculate wait time
            wait_time = max(
                self.bucket.wait_time(tokens), self.backoff.calculate_backoff(attempt)
            )

            logger.warning(
                f"Rate limited for {self.provider}. "
                f"Waiting {wait_time:.2f}s (attempt {attempt + 1})"
            )

            await asyncio.sleep(wait_time)
            attempt += 1

        raise RuntimeError(f"Rate limit exceeded for {self.provider}")

    def acquire(self, tokens: int = 1) -> None:
        """Acquire tokens with synchronous waiting.

        Args:
            tokens: Number of tokens to acquire
        """
        attempt = 0

        while attempt <= self.config.max_retries:
            if self.bucket.acquire(tokens):
                if attempt > 0:
                    logger.info(
                        f"Successfully acquired {tokens} tokens after {attempt} attempts"
                    )
                return

            if not self.backoff.should_retry(attempt):
                raise RuntimeError(
                    f"Rate limit exceeded for {self.provider} after {attempt} attempts"
                )

            # Calculate wait time
            wait_time = max(
                self.bucket.wait_time(tokens), self.backoff.calculate_backoff(attempt)
            )

            logger.warning(
                f"Rate limited for {self.provider}. "
                f"Waiting {wait_time:.2f}s (attempt {attempt + 1})"
            )

            time.sleep(wait_time)
            attempt += 1

        raise RuntimeError(f"Rate limit exceeded for {self.provider}")

    def try_acquire(self, tokens: int = 1) -> bool:
        """Try to acquire tokens without waiting.

        Args:
            tokens: Number of tokens to acquire

        Returns:
            True if tokens were acquired, False otherwise
        """
        return self.bucket.acquire(tokens)

    def get_wait_time(self, tokens: int = 1) -> float:
        """Get estimated wait time for tokens.

        Args:
            tokens: Number of tokens needed

        Returns:
            Estimated wait time in seconds
        """
        return self.bucket.wait_time(tokens)

    def get_status(self) -> Dict[str, any]:
        """Get rate limiter status.

        Returns:
            Dictionary with status information
        """
        bucket_status = self.bucket.get_status()
        return {
            "provider": self.provider,
            "bucket": bucket_status,
            "config": {
                "tokens_per_second": self.config.tokens_per_second,
                "max_tokens": self.config.max_tokens,
                "max_retries": self.config.max_retries,
                "base_backoff": self.config.base_backoff,
                "max_backoff": self.config.max_backoff,
            },
        }


class RateLimiterManager:
    """Manages rate limiters for multiple providers."""

    def __init__(self) -> None:
        """Initialize rate limiter manager."""
        self._limiters: Dict[str, RateLimiter] = {}
        self._lock = threading.Lock()

        logger.info("Initialized RateLimiterManager")

    def get_limiter(
        self, provider: str, config: Optional[RateLimitConfig] = None
    ) -> RateLimiter:
        """Get or create rate limiter for provider.

        Args:
            provider: Provider name
            config: Optional custom configuration

        Returns:
            Rate limiter instance
        """
        with self._lock:
            if provider not in self._limiters:
                self._limiters[provider] = RateLimiter(provider, config)
                logger.debug(f"Created new rate limiter for {provider}")

            return self._limiters[provider]

    def get_all_status(self) -> Dict[str, Dict[str, any]]:
        """Get status of all rate limiters.

        Returns:
            Dictionary with status for each provider
        """
        with self._lock:
            return {
                provider: limiter.get_status()
                for provider, limiter in self._limiters.items()
            }

    def clear_limiter(self, provider: str) -> None:
        """Remove rate limiter for provider.

        Args:
            provider: Provider name
        """
        with self._lock:
            if provider in self._limiters:
                del self._limiters[provider]
                logger.info(f"Cleared rate limiter for {provider}")

    def clear_all(self) -> None:
        """Clear all rate limiters."""
        with self._lock:
            self._limiters.clear()
            logger.info("Cleared all rate limiters")


# Global rate limiter manager instance
rate_limiter_manager = RateLimiterManager()

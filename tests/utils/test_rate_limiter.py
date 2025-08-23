"""Tests for rate limiting utilities."""

import asyncio
import threading
import time
from unittest.mock import Mock, patch

import pytest

from ml_agents.utils.rate_limiter import (
    PROVIDER_RATE_LIMITS,
    ExponentialBackoff,
    RateLimitConfig,
    RateLimiter,
    RateLimiterManager,
    TokenBucket,
    rate_limiter_manager,
)


class TestRateLimitConfig:
    """Test RateLimitConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = RateLimitConfig()

        assert config.tokens_per_second == 1.0
        assert config.max_tokens == 10
        assert config.max_retries == 5
        assert config.base_backoff == 1.0
        assert config.max_backoff == 60.0
        assert config.jitter == 0.1

    def test_custom_config(self):
        """Test custom configuration values."""
        config = RateLimitConfig(
            tokens_per_second=5.0,
            max_tokens=20,
            max_retries=3,
            base_backoff=2.0,
            max_backoff=30.0,
            jitter=0.2,
        )

        assert config.tokens_per_second == 5.0
        assert config.max_tokens == 20
        assert config.max_retries == 3
        assert config.base_backoff == 2.0
        assert config.max_backoff == 30.0
        assert config.jitter == 0.2


class TestTokenBucket:
    """Test TokenBucket implementation."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return RateLimitConfig(
            tokens_per_second=2.0,
            max_tokens=5,
        )

    @pytest.fixture
    def bucket(self, config):
        """Create token bucket instance."""
        return TokenBucket(config)

    def test_init(self, config):
        """Test bucket initialization."""
        bucket = TokenBucket(config)

        assert bucket.tokens_per_second == 2.0
        assert bucket.max_tokens == 5
        assert bucket._tokens == 5.0  # Starts full

    def test_acquire_success(self, bucket):
        """Test successful token acquisition."""
        # Should succeed - bucket starts full
        assert bucket.acquire(1) is True
        assert bucket.acquire(2) is True

        # Check remaining tokens (allow small floating point tolerance)
        status = bucket.get_status()
        assert abs(status["current_tokens"] - 2.0) < 0.01

    def test_acquire_insufficient_tokens(self, bucket):
        """Test token acquisition when insufficient tokens."""
        # Use all tokens
        assert bucket.acquire(5) is True

        # Should fail - no tokens left
        assert bucket.acquire(1) is False

    def test_token_replenishment(self, bucket):
        """Test token replenishment over time."""
        # Use all tokens
        bucket.acquire(5)

        # Simulate time passage (1 second = 2 tokens)
        with patch("ml_agents.utils.rate_limiter.time.time") as mock_time:
            # Set initial time
            mock_time.return_value = 0
            bucket._last_update = 0

            # Advance time by 1 second
            mock_time.return_value = 1

            # Should have 2 tokens after 1 second
            assert bucket.acquire(2) is True
            assert bucket.acquire(1) is False  # No more tokens

    def test_max_tokens_cap(self, bucket):
        """Test that tokens don't exceed maximum."""
        # Start with some tokens used
        bucket.acquire(2)

        # Simulate long time passage
        with patch("time.time") as mock_time:
            mock_time.side_effect = [0, 10]  # 10 seconds elapsed

            # Should cap at max_tokens (5)
            status = bucket.get_status()
            # Trigger token update
            bucket.acquire(0)
            status = bucket.get_status()
            assert status["current_tokens"] <= 5.0

    def test_wait_time_calculation(self, bucket):
        """Test wait time calculation."""
        # Use all tokens
        bucket.acquire(5)

        # Need 2 tokens, rate is 2 tokens/second
        wait_time = bucket.wait_time(2)
        assert wait_time == 1.0  # Should be 1 second

    def test_wait_time_with_existing_tokens(self, bucket):
        """Test wait time with partial tokens available."""
        # Use 3 tokens, leaving 2
        bucket.acquire(3)

        # Need 1 more token - should be immediate
        wait_time = bucket.wait_time(1)
        assert wait_time == 0.0

    def test_get_status(self, bucket):
        """Test status reporting."""
        status = bucket.get_status()

        assert "current_tokens" in status
        assert "max_tokens" in status
        assert "tokens_per_second" in status
        assert status["max_tokens"] == 5
        assert status["tokens_per_second"] == 2.0

    def test_thread_safety(self, bucket):
        """Test thread safety of token bucket."""
        results = []

        def acquire_tokens():
            for _ in range(10):
                results.append(bucket.acquire(1))

        # Start multiple threads
        threads = [threading.Thread(target=acquire_tokens) for _ in range(3)]
        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()

        # Should have some successes and failures
        assert True in results
        assert False in results


class TestExponentialBackoff:
    """Test ExponentialBackoff implementation."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return RateLimitConfig(
            base_backoff=1.0,
            max_backoff=10.0,
            max_retries=3,
            jitter=0.0,  # No jitter for predictable tests
        )

    @pytest.fixture
    def backoff(self, config):
        """Create backoff calculator."""
        return ExponentialBackoff(config)

    def test_calculate_backoff_progression(self, config):
        """Test exponential backoff progression."""
        # Set jitter to 0 for predictable results
        config.jitter = 0.0
        backoff = ExponentialBackoff(config)

        # Test exponential progression: 1, 2, 4
        assert backoff.calculate_backoff(0) == 1.0  # base * 2^0
        assert backoff.calculate_backoff(1) == 2.0  # base * 2^1
        assert backoff.calculate_backoff(2) == 4.0  # base * 2^2
        # Attempt 3 >= max_retries, so should return max_backoff
        assert backoff.calculate_backoff(3) == 10.0  # max_backoff

    def test_calculate_backoff_max_cap(self, config):
        """Test backoff capped at maximum."""
        config.jitter = 0.0
        backoff = ExponentialBackoff(config)

        # Should be capped at max_backoff (10.0)
        assert backoff.calculate_backoff(10) == 10.0

    def test_calculate_backoff_with_jitter(self, config):
        """Test backoff with jitter."""
        config.jitter = 0.1
        backoff = ExponentialBackoff(config)

        # Calculate multiple times to test jitter variation
        times = [backoff.calculate_backoff(1) for _ in range(10)]

        # Should have variation (not all the same)
        assert len(set(times)) > 1

        # All should be positive and reasonable
        assert all(t > 0 for t in times)
        assert all(
            t < 5.0 for t in times
        )  # Base is 2.0, jitter shouldn't make it too large

    def test_should_retry(self, backoff):
        """Test retry decision logic."""
        # Should retry within limits
        assert backoff.should_retry(0) is True
        assert backoff.should_retry(1) is True
        assert backoff.should_retry(2) is True

        # Should not retry at/beyond limit
        assert backoff.should_retry(3) is False
        assert backoff.should_retry(4) is False

    def test_should_retry_edge_cases(self):
        """Test retry logic with different max_retries."""
        config = RateLimitConfig(max_retries=0)
        backoff = ExponentialBackoff(config)

        # Should not retry if max_retries is 0
        assert backoff.should_retry(0) is False


class TestRateLimiter:
    """Test RateLimiter main class."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return RateLimitConfig(
            tokens_per_second=2.0,
            max_tokens=4,
            max_retries=2,
            base_backoff=0.1,  # Short backoff for testing
            max_backoff=0.5,
            jitter=0.0,
        )

    @pytest.fixture
    def limiter(self, config):
        """Create rate limiter."""
        return RateLimiter("test_provider", config)

    def test_init_with_config(self, config):
        """Test initialization with custom config."""
        limiter = RateLimiter("test_provider", config)

        assert limiter.provider == "test_provider"
        assert limiter.config == config

    def test_init_with_provider_defaults(self):
        """Test initialization with provider defaults."""
        limiter = RateLimiter("anthropic")

        assert limiter.provider == "anthropic"
        assert limiter.config == PROVIDER_RATE_LIMITS["anthropic"]

    def test_init_unknown_provider(self):
        """Test initialization with unknown provider."""
        limiter = RateLimiter("unknown_provider")

        assert limiter.provider == "unknown_provider"
        # Should use default config
        assert limiter.config.tokens_per_second == 1.0

    def test_try_acquire_success(self, limiter):
        """Test successful immediate acquisition."""
        assert limiter.try_acquire(1) is True
        assert limiter.try_acquire(2) is True

    def test_try_acquire_failure(self, limiter):
        """Test failed immediate acquisition."""
        # Use all tokens
        limiter.try_acquire(4)

        # Should fail
        assert limiter.try_acquire(1) is False

    def test_acquire_success_immediate(self, limiter):
        """Test successful synchronous acquisition."""
        # Should succeed immediately
        limiter.acquire(2)

    def test_acquire_with_waiting(self, limiter):
        """Test synchronous acquisition with waiting."""
        # Use most tokens
        limiter.try_acquire(3)

        start_time = time.time()

        # This should wait and then succeed
        limiter.acquire(2)

        elapsed = time.time() - start_time
        # Should have waited some time (but backoff is short)
        assert elapsed >= 0.1

    def test_acquire_failure_after_retries(self, limiter):
        """Test acquisition failure after max retries."""
        # Use all tokens
        limiter.try_acquire(4)

        # Mock bucket to always fail
        with patch.object(limiter.bucket, "acquire", return_value=False):
            with pytest.raises(RuntimeError, match="Rate limit exceeded"):
                limiter.acquire(1)

    @pytest.mark.asyncio
    async def test_acquire_async_success(self, limiter):
        """Test successful async acquisition."""
        await limiter.acquire_async(2)

    @pytest.mark.asyncio
    async def test_acquire_async_with_waiting(self, limiter):
        """Test async acquisition with waiting."""
        # Use most tokens
        limiter.try_acquire(3)

        start_time = time.time()

        # This should wait and then succeed
        await limiter.acquire_async(2)

        elapsed = time.time() - start_time
        assert elapsed >= 0.1

    @pytest.mark.asyncio
    async def test_acquire_async_failure(self, limiter):
        """Test async acquisition failure."""
        # Use all tokens
        limiter.try_acquire(4)

        # Mock bucket to always fail
        with patch.object(limiter.bucket, "acquire", return_value=False):
            with pytest.raises(RuntimeError, match="Rate limit exceeded"):
                await limiter.acquire_async(1)

    def test_get_wait_time(self, limiter):
        """Test wait time estimation."""
        # Use all tokens
        limiter.try_acquire(4)

        # Should need to wait for 1 token at 2 tokens/second
        wait_time = limiter.get_wait_time(1)
        assert wait_time == 0.5

    def test_get_status(self, limiter):
        """Test status reporting."""
        status = limiter.get_status()

        assert status["provider"] == "test_provider"
        assert "bucket" in status
        assert "config" in status
        assert status["config"]["tokens_per_second"] == 2.0


class TestRateLimiterManager:
    """Test RateLimiterManager."""

    @pytest.fixture
    def manager(self):
        """Create rate limiter manager."""
        return RateLimiterManager()

    def test_get_limiter_creates_new(self, manager):
        """Test getting limiter creates new instance."""
        limiter = manager.get_limiter("test_provider")

        assert isinstance(limiter, RateLimiter)
        assert limiter.provider == "test_provider"

    def test_get_limiter_reuses_existing(self, manager):
        """Test getting limiter reuses existing instance."""
        limiter1 = manager.get_limiter("test_provider")
        limiter2 = manager.get_limiter("test_provider")

        assert limiter1 is limiter2

    def test_get_limiter_with_custom_config(self, manager):
        """Test getting limiter with custom config."""
        config = RateLimitConfig(tokens_per_second=5.0)
        limiter = manager.get_limiter("test_provider", config)

        assert limiter.config.tokens_per_second == 5.0

    def test_get_all_status(self, manager):
        """Test getting status of all limiters."""
        manager.get_limiter("provider1")
        manager.get_limiter("provider2")

        status = manager.get_all_status()

        assert "provider1" in status
        assert "provider2" in status
        assert status["provider1"]["provider"] == "provider1"

    def test_clear_limiter(self, manager):
        """Test clearing specific limiter."""
        manager.get_limiter("test_provider")

        manager.clear_limiter("test_provider")

        # Should create new instance
        limiter1 = manager.get_limiter("test_provider")
        limiter2 = manager.get_limiter("test_provider")
        assert limiter1 is limiter2  # But reuse after recreation

    def test_clear_all(self, manager):
        """Test clearing all limiters."""
        manager.get_limiter("provider1")
        manager.get_limiter("provider2")

        manager.clear_all()

        # Should be empty
        status = manager.get_all_status()
        assert len(status) == 0

    def test_thread_safety(self, manager):
        """Test thread safety of manager."""
        limiters = []

        def get_limiter():
            limiter = manager.get_limiter("test_provider")
            limiters.append(limiter)

        # Start multiple threads
        threads = [threading.Thread(target=get_limiter) for _ in range(10)]
        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()

        # All should be the same instance
        assert all(limiter is limiters[0] for limiter in limiters)


class TestProviderRateLimits:
    """Test provider-specific rate limit configurations."""

    def test_all_providers_have_config(self):
        """Test that all expected providers have configurations."""
        expected_providers = ["anthropic", "cohere", "openrouter"]

        for provider in expected_providers:
            assert provider in PROVIDER_RATE_LIMITS
            config = PROVIDER_RATE_LIMITS[provider]
            assert isinstance(config, RateLimitConfig)

    def test_anthropic_config(self):
        """Test Anthropic-specific configuration."""
        config = PROVIDER_RATE_LIMITS["anthropic"]

        assert config.tokens_per_second == 5.0
        assert config.max_tokens == 20
        assert config.base_backoff == 2.0

    def test_cohere_config(self):
        """Test Cohere-specific configuration."""
        config = PROVIDER_RATE_LIMITS["cohere"]

        assert config.tokens_per_second == 10.0
        assert config.max_tokens == 50
        assert config.base_backoff == 1.0

    def test_openrouter_config(self):
        """Test OpenRouter-specific configuration."""
        config = PROVIDER_RATE_LIMITS["openrouter"]

        assert config.tokens_per_second == 2.0
        assert config.max_tokens == 10
        assert config.base_backoff == 1.5


class TestGlobalRateLimiterManager:
    """Test global rate limiter manager instance."""

    def test_global_instance_exists(self):
        """Test that global instance is available."""
        assert rate_limiter_manager is not None
        assert isinstance(rate_limiter_manager, RateLimiterManager)

    def test_global_instance_functionality(self):
        """Test that global instance works correctly."""
        limiter = rate_limiter_manager.get_limiter("test_provider")
        assert isinstance(limiter, RateLimiter)

        # Clean up
        rate_limiter_manager.clear_all()

"""Integration tests for reasoning approaches with real API calls.

These tests make actual API calls and should be run sparingly to minimize costs.
Run with: pytest tests/test_integration_reasoning.py -m integration
"""

import pytest

from src.config import ExperimentConfig
from src.core.reasoning_inference import ReasoningInference
from src.reasoning import create_reasoning_approach


class TestReasoningIntegration:
    """Integration tests for reasoning approaches."""

    @pytest.mark.integration
    def test_none_reasoning_integration(
        self, real_api_config, integration_test_prompts, skip_integration_if_no_keys
    ):
        """Test None reasoning with real API calls."""
        skip_integration_if_no_keys("openrouter")

        config = ExperimentConfig(**real_api_config)
        reasoning = create_reasoning_approach("None", config)

        # Test with minimal prompt to control costs
        prompt = integration_test_prompts[0]  # "What is 1+1?"
        result = reasoning.execute(prompt)

        # Verify real response structure
        assert len(result.text) > 0
        assert result.total_tokens > 0
        assert result.metadata["reasoning_approach"] == "None"
        assert result.metadata["reasoning_steps"] == 0

    @pytest.mark.integration
    def test_chain_of_thought_integration(
        self, real_api_config, integration_test_prompts, skip_integration_if_no_keys
    ):
        """Test Chain-of-Thought reasoning with real API calls."""
        skip_integration_if_no_keys("openrouter")

        config = ExperimentConfig(**real_api_config)
        reasoning = create_reasoning_approach("ChainOfThought", config)

        prompt = integration_test_prompts[0]
        result = reasoning.execute(prompt)

        # Verify CoT-enhanced response
        assert len(result.text) > 0
        assert result.total_tokens > 0
        assert result.metadata["reasoning_approach"] == "ChainOfThought"
        assert result.metadata["reasoning_steps"] > 0

    @pytest.mark.integration
    def test_reasoning_inference_integration(
        self, real_api_config, integration_test_prompts, skip_integration_if_no_keys
    ):
        """Test ReasoningInference engine with real API calls."""
        skip_integration_if_no_keys("openrouter")

        config = ExperimentConfig(**real_api_config)
        engine = ReasoningInference(config)

        # Test single inference
        prompt = integration_test_prompts[1]  # "Name one color."
        result = engine.run_inference(prompt, "None")

        # Verify result structure
        assert result.response.text != ""
        assert result.cost_estimate >= 0
        assert result.execution_time > 0
        assert result.approach_name == "None"

        # Test cost tracking
        cost_summary = engine.get_cost_summary()
        assert cost_summary["total_requests"] == 1
        assert cost_summary["total_cost"] >= 0

    @pytest.mark.integration
    def test_reasoning_comparison_integration(
        self, real_api_config, integration_test_prompts, skip_integration_if_no_keys
    ):
        """Test reasoning comparison with real API calls."""
        skip_integration_if_no_keys("openrouter")

        config = ExperimentConfig(**real_api_config)
        engine = ReasoningInference(config)

        # Compare None vs ChainOfThought
        prompt = integration_test_prompts[2]  # "Say hello."
        approaches = ["None", "ChainOfThought"]

        results = engine.run_comparison(prompt, approaches)

        # Verify comparison results
        assert len(results) == 2
        assert "None" in results
        assert "ChainOfThought" in results

        # Verify different approaches produce different responses
        none_result = results["None"]
        cot_result = results["ChainOfThought"]

        assert none_result.response.text != ""
        assert cot_result.response.text != ""
        assert none_result.metadata["reasoning_steps"] == 0
        assert cot_result.metadata["reasoning_steps"] > 0

    @pytest.mark.integration
    @pytest.mark.slow
    def test_batch_processing_integration(
        self, real_api_config, integration_test_prompts, skip_integration_if_no_keys
    ):
        """Test batch processing with real API calls (marked as slow)."""
        skip_integration_if_no_keys("openrouter")

        config = ExperimentConfig(**real_api_config)
        engine = ReasoningInference(config)

        # Use all integration test prompts for batch processing
        results = engine.run_batch_inference(integration_test_prompts, "None")

        # Verify batch results
        assert len(results) == len(integration_test_prompts)
        for result in results:
            assert result.response.text != ""
            assert result.cost_estimate >= 0
            assert result.approach_name == "None"

        # Verify cost accumulation
        cost_summary = engine.get_cost_summary()
        assert cost_summary["total_requests"] == len(integration_test_prompts)

    @pytest.mark.integration
    def test_error_handling_integration(self, skip_integration_if_no_keys):
        """Test error handling with invalid configuration."""
        skip_integration_if_no_keys("openrouter")

        # Create config with invalid model to test error handling
        invalid_config = ExperimentConfig(
            provider="openrouter",
            model="invalid/nonexistent-model",
            temperature=0.3,
            max_tokens=50,
        )

        engine = ReasoningInference(invalid_config)

        # This should handle errors gracefully
        with pytest.raises(Exception):  # Expect some kind of error
            engine.run_inference("Test prompt", "None")

    @pytest.mark.integration
    def test_cleanup_integration(self, real_api_config, skip_integration_if_no_keys):
        """Test resource cleanup after integration tests."""
        skip_integration_if_no_keys("openrouter")

        config = ExperimentConfig(**real_api_config)
        engine = ReasoningInference(config)

        # Run a simple inference
        engine.run_inference("Test cleanup", "None")

        # Test cleanup
        engine.cleanup()

        # Verify cache is cleared
        assert len(engine.approach_cache) == 0

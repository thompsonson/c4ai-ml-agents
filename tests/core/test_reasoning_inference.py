"""Tests for the ReasoningInference engine."""

from datetime import datetime
from unittest.mock import Mock, patch

import pytest

from ml_agents.config import ExperimentConfig
from ml_agents.core.reasoning_inference import ReasoningInference, ReasoningResult
from ml_agents.reasoning.base import BaseReasoning
from ml_agents.utils.api_clients import StandardResponse


class MockReasoningApproach(BaseReasoning):
    """Mock reasoning approach for testing."""

    def __init__(self, config, response_text="Mock response"):
        self.config = config
        self.approach_name = "Mock"
        self.response_text = response_text
        self.client = Mock()

    def execute(self, prompt: str) -> StandardResponse:
        response = StandardResponse(
            text=self.response_text,
            provider=self.config.provider,
            model=self.config.model,
            prompt_tokens=10,
            completion_tokens=20,
            total_tokens=30,
            generation_time=1.0,
            parameters={"temperature": 0.3},
            response_id="mock-123",
            metadata={"test": True},
        )
        return self._enhance_metadata(response, {"mock_steps": 2})


class TestReasoningInference:
    """Test cases for ReasoningInference engine."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return ExperimentConfig(
            provider="test_provider", model="test-model", temperature=0.3
        )

    @pytest.fixture
    def inference_engine(self, config):
        """Create ReasoningInference instance."""
        return ReasoningInference(config)

    def test_initialization(self, config):
        """Test ReasoningInference initialization."""
        engine = ReasoningInference(config)

        assert engine.config == config
        assert engine.total_cost == 0.0
        assert engine.total_requests == 0
        assert engine.approach_cache == {}
        assert engine.execution_history == []

    @patch("ml_agents.core.reasoning_inference.create_reasoning_approach")
    def test_run_inference_basic(self, mock_create_approach, inference_engine):
        """Test basic inference execution."""
        # Setup mock approach
        mock_approach = MockReasoningApproach(inference_engine.config)
        mock_create_approach.return_value = mock_approach

        test_prompt = "What is 2 + 2?"
        result = inference_engine.run_inference(test_prompt, "Mock")

        # Verify approach creation
        mock_create_approach.assert_called_once_with("Mock", inference_engine.config)

        # Verify result structure
        assert isinstance(result, ReasoningResult)
        assert result.approach_name == "Mock"
        assert result.response.text == "Mock response"
        assert result.execution_time > 0
        assert result.cost_estimate >= 0

        # Verify metadata
        assert result.metadata["prompt_length"] == len(test_prompt)
        assert result.metadata["response_length"] == len("Mock response")
        assert "timestamp" in result.metadata

    @patch("ml_agents.core.reasoning_inference.create_reasoning_approach")
    def test_run_inference_caching(self, mock_create_approach, inference_engine):
        """Test approach instance caching."""
        mock_approach = MockReasoningApproach(inference_engine.config)
        mock_create_approach.return_value = mock_approach

        # Run inference twice with same approach
        inference_engine.run_inference("First prompt", "Mock")
        inference_engine.run_inference("Second prompt", "Mock")

        # Should only create approach once (caching)
        mock_create_approach.assert_called_once()
        assert "Mock" in inference_engine.approach_cache

    def test_run_inference_empty_prompt(self, inference_engine):
        """Test inference with empty prompt."""
        with pytest.raises(ValueError, match="Prompt cannot be empty"):
            inference_engine.run_inference("", "Mock")

        with pytest.raises(ValueError, match="Prompt cannot be empty"):
            inference_engine.run_inference("   ", "Mock")

    def test_run_inference_empty_approach(self, inference_engine):
        """Test inference with empty approach name."""
        with pytest.raises(ValueError, match="Reasoning approach name cannot be empty"):
            inference_engine.run_inference("Test prompt", "")

    @patch("ml_agents.core.reasoning_inference.create_reasoning_approach")
    def test_run_inference_unknown_approach(
        self, mock_create_approach, inference_engine
    ):
        """Test inference with unknown approach."""
        mock_create_approach.side_effect = KeyError("Unknown approach")

        with pytest.raises(KeyError):
            inference_engine.run_inference("Test", "UnknownApproach")

    @patch("ml_agents.core.reasoning_inference.create_reasoning_approach")
    def test_run_inference_approach_error(self, mock_create_approach, inference_engine):
        """Test handling of approach execution errors."""
        # Create approach that raises error on execute
        mock_approach = Mock()
        mock_approach.execute.side_effect = RuntimeError("Approach failed")
        mock_create_approach.return_value = mock_approach

        with pytest.raises(RuntimeError, match="Approach failed"):
            inference_engine.run_inference("Test prompt", "FailingApproach")

        # Verify error result was tracked
        assert len(inference_engine.execution_history) == 1
        error_result = inference_engine.execution_history[0]
        assert (
            "Error occurred during FailingApproach reasoning"
            in error_result.response.text
        )
        assert error_result.cost_estimate == 0.0

    @patch("ml_agents.core.reasoning_inference.create_reasoning_approach")
    def test_run_comparison(self, mock_create_approach, inference_engine):
        """Test comparison across multiple approaches."""
        # Setup different mock approaches
        approaches = {
            "Mock1": MockReasoningApproach(inference_engine.config, "Response 1"),
            "Mock2": MockReasoningApproach(inference_engine.config, "Response 2"),
            "Mock3": MockReasoningApproach(inference_engine.config, "Response 3"),
        }

        def create_approach_side_effect(name, config):
            return approaches[name]

        mock_create_approach.side_effect = create_approach_side_effect

        test_prompt = "Compare these approaches"
        approach_names = ["Mock1", "Mock2", "Mock3"]

        results = inference_engine.run_comparison(test_prompt, approach_names)

        # Verify all approaches were executed
        assert len(results) == 3
        assert all(name in results for name in approach_names)

        # Verify each result
        for name in approach_names:
            assert isinstance(results[name], ReasoningResult)
            assert results[name].approach_name == name
            assert f"Response {name[-1]}" in results[name].response.text

    @patch("ml_agents.core.reasoning_inference.create_reasoning_approach")
    def test_run_comparison_with_failures(self, mock_create_approach, inference_engine):
        """Test comparison handling approach failures."""

        def create_approach_side_effect(name, config):
            if name == "FailingApproach":
                raise RuntimeError("This approach fails")
            return MockReasoningApproach(config, f"Response from {name}")

        mock_create_approach.side_effect = create_approach_side_effect

        approach_names = [
            "WorkingApproach",
            "FailingApproach",
            "AnotherWorkingApproach",
        ]
        results = inference_engine.run_comparison("Test", approach_names)

        # Should only have results for working approaches
        assert len(results) == 2
        assert "WorkingApproach" in results
        assert "AnotherWorkingApproach" in results
        assert "FailingApproach" not in results

    @patch("ml_agents.core.reasoning_inference.create_reasoning_approach")
    def test_run_batch_inference(self, mock_create_approach, inference_engine):
        """Test batch inference on multiple prompts."""
        mock_approach = MockReasoningApproach(inference_engine.config)
        mock_create_approach.return_value = mock_approach

        prompts = ["Question 1", "Question 2", "Question 3"]
        progress_calls = []

        def progress_callback(current, total, result):
            progress_calls.append((current, total, result.approach_name))

        results = inference_engine.run_batch_inference(
            prompts, "Mock", progress_callback
        )

        # Verify batch processing
        assert len(results) == 3
        assert all(isinstance(r, ReasoningResult) for r in results)
        assert all(r.approach_name == "Mock" for r in results)

        # Verify progress callbacks
        assert len(progress_calls) == 3
        assert progress_calls[0] == (1, 3, "Mock")
        assert progress_calls[1] == (2, 3, "Mock")
        assert progress_calls[2] == (3, 3, "Mock")

    @patch("ml_agents.core.reasoning_inference.create_reasoning_approach")
    def test_cost_tracking(self, mock_create_approach, inference_engine):
        """Test cost tracking functionality."""
        mock_approach = MockReasoningApproach(inference_engine.config)
        mock_create_approach.return_value = mock_approach

        # Run multiple inferences
        for i in range(3):
            inference_engine.run_inference(f"Prompt {i}", "Mock")

        # Check cost summary
        cost_summary = inference_engine.get_cost_summary()

        assert cost_summary["total_requests"] == 3
        assert cost_summary["total_cost"] > 0
        assert "Mock" in cost_summary["approach_costs"]
        assert cost_summary["approach_counts"]["Mock"] == 3
        assert cost_summary["execution_count"] == 3

    @patch("ml_agents.core.reasoning_inference.create_reasoning_approach")
    def test_performance_tracking(self, mock_create_approach, inference_engine):
        """Test performance tracking functionality."""
        mock_approach = MockReasoningApproach(inference_engine.config)
        mock_create_approach.return_value = mock_approach

        # Run multiple inferences
        for i in range(2):
            inference_engine.run_inference(f"Prompt {i}", "Mock")

        # Check performance summary
        perf_summary = inference_engine.get_performance_summary()

        assert "Mock" in perf_summary
        mock_stats = perf_summary["Mock"]
        assert "average_time" in mock_stats
        assert "average_tokens" in mock_stats
        assert mock_stats["total_executions"] == 2
        assert mock_stats["average_tokens"] == 30  # From mock response

    def test_performance_summary_no_history(self, inference_engine):
        """Test performance summary with no execution history."""
        perf_summary = inference_engine.get_performance_summary()
        assert "message" in perf_summary
        assert "No execution history available" in perf_summary["message"]

    def test_estimate_cost(self, inference_engine):
        """Test cost estimation logic."""
        response = StandardResponse(
            text="Test",
            provider="anthropic",
            model="test",
            prompt_tokens=1000,
            completion_tokens=500,
            total_tokens=1500,
            generation_time=1.0,
            parameters={},
            response_id="test",
            metadata={},
        )

        cost = inference_engine._estimate_cost(response)

        # Should be > 0 for anthropic (based on token count)
        assert cost > 0
        assert isinstance(cost, float)

    def test_estimate_cost_free_provider(self, inference_engine):
        """Test cost estimation for free providers."""
        response = StandardResponse(
            text="Test",
            provider="huggingface",
            model="test",
            prompt_tokens=1000,
            completion_tokens=500,
            total_tokens=1500,
            generation_time=1.0,
            parameters={},
            response_id="test",
            metadata={},
        )

        cost = inference_engine._estimate_cost(response)
        assert cost == 0.0  # HuggingFace should be free

    def test_cleanup(self, inference_engine):
        """Test cleanup functionality."""
        # Add some mock approaches to cache
        mock_approach1 = Mock()
        mock_approach2 = Mock()
        inference_engine.approach_cache["Approach1"] = mock_approach1
        inference_engine.approach_cache["Approach2"] = mock_approach2

        inference_engine.cleanup()

        # Verify cleanup was called on all approaches
        mock_approach1.cleanup.assert_called_once()
        mock_approach2.cleanup.assert_called_once()

        # Verify cache was cleared
        assert len(inference_engine.approach_cache) == 0

    def test_cleanup_with_error(self, inference_engine):
        """Test cleanup handling errors gracefully."""
        mock_approach = Mock()
        mock_approach.cleanup.side_effect = RuntimeError("Cleanup failed")
        inference_engine.approach_cache["FailingApproach"] = mock_approach

        # Should not raise error
        inference_engine.cleanup()

        # Cache should still be cleared
        assert len(inference_engine.approach_cache) == 0

    def test_reasoning_result_dataclass(self):
        """Test ReasoningResult dataclass."""
        response = StandardResponse("test", "provider", "model", 10, 20, 30, 1.0, {})

        result = ReasoningResult(
            response=response,
            approach_name="Test",
            execution_time=2.5,
            cost_estimate=0.05,
            metadata={"key": "value"},
        )

        assert result.response == response
        assert result.approach_name == "Test"
        assert result.execution_time == 2.5
        assert result.cost_estimate == 0.05
        assert result.metadata["key"] == "value"

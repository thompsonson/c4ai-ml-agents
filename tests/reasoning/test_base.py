"""Tests for the BaseReasoning class and reasoning infrastructure."""

from unittest.mock import Mock, patch

import pytest

from src.config import ExperimentConfig
from src.reasoning.base import BaseReasoning
from src.utils.api_clients import StandardResponse


class TestReasoningImplementation(BaseReasoning):
    """Test implementation of BaseReasoning for testing purposes."""

    def execute(self, prompt: str) -> StandardResponse:
        """Test implementation of execute method."""
        # Create a mock response
        response = StandardResponse(
            text="Test response",
            provider=self.config.provider,
            model=self.config.model,
            prompt_tokens=10,
            completion_tokens=20,
            total_tokens=30,
            generation_time=1.0,
            parameters={"temperature": 0.3},
            response_id="test-123",
            metadata={},
        )
        return self._enhance_metadata(response, {"test_data": True})


class TestBaseReasoning:
    """Test cases for BaseReasoning class."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return ExperimentConfig(
            provider="test_provider", model="test-model", temperature=0.3
        )

    @pytest.fixture
    def mock_client(self):
        """Create mock API client."""
        client = Mock()
        client.generate.return_value = StandardResponse(
            text="Mock response",
            provider="test_provider",
            model="test-model",
            prompt_tokens=10,
            completion_tokens=15,
            total_tokens=25,
            generation_time=0.5,
            parameters={"temperature": 0.3},
            response_id="mock-123",
            metadata={},
        )
        return client

    @patch("src.reasoning.base.create_api_client")
    def test_initialization(self, mock_create_client, config, mock_client):
        """Test BaseReasoning initialization."""
        mock_create_client.return_value = mock_client

        reasoning = TestReasoningImplementation(config)

        assert reasoning.config == config
        assert reasoning.client == mock_client
        assert reasoning.approach_name == "TestReasoningImplementation"
        mock_create_client.assert_called_once_with(config)

    @patch("src.reasoning.base.create_api_client")
    def test_execute_abstract_method(self, mock_create_client, config, mock_client):
        """Test that execute method is abstract and must be implemented."""
        mock_create_client.return_value = mock_client

        # BaseReasoning should raise NotImplementedError
        reasoning = BaseReasoning(config)

        with pytest.raises(NotImplementedError):
            reasoning.execute("test prompt")

    @patch("src.reasoning.base.create_api_client")
    def test_enhance_metadata(self, mock_create_client, config, mock_client):
        """Test metadata enhancement functionality."""
        mock_create_client.return_value = mock_client

        reasoning = TestReasoningImplementation(config)

        # Create a base response
        response = StandardResponse(
            text="Test",
            provider="test",
            model="test",
            prompt_tokens=5,
            completion_tokens=10,
            total_tokens=15,
            generation_time=1.0,
            parameters={},
            response_id="test",
            metadata={},
        )

        # Test enhancement with additional data
        reasoning_data = {"test_metric": 42, "test_flag": True}
        enhanced = reasoning._enhance_metadata(response, reasoning_data)

        assert enhanced.metadata["reasoning_approach"] == "TestReasoningImplementation"
        assert enhanced.metadata["test_metric"] == 42
        assert enhanced.metadata["test_flag"] is True

    @patch("src.reasoning.base.create_api_client")
    def test_enhance_metadata_no_existing_metadata(
        self, mock_create_client, config, mock_client
    ):
        """Test metadata enhancement when response has no existing metadata."""
        mock_create_client.return_value = mock_client

        reasoning = TestReasoningImplementation(config)

        # Create response with None metadata
        response = StandardResponse(
            text="Test",
            provider="test",
            model="test",
            prompt_tokens=5,
            completion_tokens=10,
            total_tokens=15,
            generation_time=1.0,
            parameters={},
            response_id="test",
            metadata=None,
        )

        enhanced = reasoning._enhance_metadata(response, {"new_data": "test"})

        assert enhanced.metadata is not None
        assert enhanced.metadata["reasoning_approach"] == "TestReasoningImplementation"
        assert enhanced.metadata["new_data"] == "test"

    @patch("src.reasoning.base.create_api_client")
    def test_count_reasoning_steps_basic(self, mock_create_client, config, mock_client):
        """Test basic reasoning step counting."""
        mock_create_client.return_value = mock_client

        reasoning = TestReasoningImplementation(config)

        # Test with common step indicators
        text_with_steps = """
        Step 1: First analysis
        Step 2: Second analysis
        Therefore, the answer is X.
        """

        steps = reasoning._count_reasoning_steps(text_with_steps)
        assert steps > 0

        # Test with no step indicators
        text_without_steps = "This is just a plain response with no step indicators."
        steps_none = reasoning._count_reasoning_steps(text_without_steps)
        assert steps_none == 0

    @patch("src.reasoning.base.create_api_client")
    def test_count_reasoning_steps_numbered_lists(
        self, mock_create_client, config, mock_client
    ):
        """Test step counting with numbered lists."""
        mock_create_client.return_value = mock_client

        reasoning = TestReasoningImplementation(config)

        text_with_numbers = """
        1. First point
        2. Second point
        3. Third point
        """

        steps = reasoning._count_reasoning_steps(text_with_numbers)
        assert steps >= 3

    @patch("src.reasoning.base.create_api_client")
    def test_count_reasoning_steps_sequence_words(
        self, mock_create_client, config, mock_client
    ):
        """Test step counting with sequence words."""
        mock_create_client.return_value = mock_client

        reasoning = TestReasoningImplementation(config)

        text_with_sequence = """
        First, we need to understand the problem.
        Second, we analyze the data.
        Finally, we reach a conclusion.
        """

        steps = reasoning._count_reasoning_steps(text_with_sequence)
        assert steps >= 3

    @patch("src.reasoning.base.create_api_client")
    def test_cleanup(self, mock_create_client, config, mock_client):
        """Test cleanup functionality."""
        mock_create_client.return_value = mock_client
        mock_client.cleanup = Mock()

        reasoning = TestReasoningImplementation(config)
        reasoning.cleanup()

        mock_client.cleanup.assert_called_once()

    @patch("src.reasoning.base.create_api_client")
    def test_cleanup_no_cleanup_method(self, mock_create_client, config, mock_client):
        """Test cleanup when client has no cleanup method."""
        mock_create_client.return_value = mock_client
        # Don't add cleanup method to mock_client

        reasoning = TestReasoningImplementation(config)
        # Should not raise an error
        reasoning.cleanup()

    @patch("src.reasoning.base.create_api_client")
    def test_approach_name_extraction(self, mock_create_client, config, mock_client):
        """Test approach name extraction from class name."""
        mock_create_client.return_value = mock_client

        reasoning = TestReasoningImplementation(config)
        assert reasoning.approach_name == "TestReasoningImplementation"

        # Test with typical reasoning class name
        class ChainOfThoughtReasoning(BaseReasoning):
            def execute(self, prompt: str) -> StandardResponse:
                return StandardResponse("test", "test", "test", 0, 0, 0, 0.0, {})

        cot_reasoning = ChainOfThoughtReasoning(config)
        assert cot_reasoning.approach_name == "ChainOfThoughtReasoning"

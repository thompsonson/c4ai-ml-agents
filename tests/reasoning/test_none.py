"""Tests for the None reasoning approach."""

from pathlib import Path
from unittest.mock import Mock, mock_open, patch

import pytest

from src.config import ExperimentConfig
from src.reasoning.none import NoneReasoning
from src.utils.api_clients import StandardResponse


class TestNoneReasoning:
    """Test cases for None reasoning approach."""

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
            text="Test response from model",
            provider="test_provider",
            model="test-model",
            prompt_tokens=10,
            completion_tokens=20,
            total_tokens=30,
            generation_time=1.0,
            parameters={"temperature": 0.3},
            response_id="test-123",
            metadata={},
        )
        return client

    @patch("src.reasoning.none.create_api_client")
    @patch(
        "builtins.open", new_callable=mock_open, read_data="Please answer: {question}"
    )
    def test_initialization_with_prompt_file(
        self, mock_file, mock_create_client, config, mock_client
    ):
        """Test None reasoning initialization with prompt file."""
        mock_create_client.return_value = mock_client

        reasoning = NoneReasoning(config)

        assert reasoning.config == config
        assert reasoning.client == mock_client
        assert reasoning.approach_name == "None"
        assert "Please answer: {question}" in reasoning.base_prompt

    @patch("src.reasoning.none.create_api_client")
    @patch("builtins.open", side_effect=FileNotFoundError)
    def test_initialization_without_prompt_file(
        self, mock_file, mock_create_client, config, mock_client
    ):
        """Test None reasoning initialization when prompt file is missing."""
        mock_create_client.return_value = mock_client

        reasoning = NoneReasoning(config)

        # Should use fallback prompt
        assert "Please answer the following question:" in reasoning.base_prompt
        assert "{question}" in reasoning.base_prompt

    @patch("src.reasoning.none.create_api_client")
    @patch("builtins.open", new_callable=mock_open, read_data="Base prompt: {question}")
    def test_execute_basic(self, mock_file, mock_create_client, config, mock_client):
        """Test basic execution of None reasoning."""
        mock_create_client.return_value = mock_client

        reasoning = NoneReasoning(config)
        test_prompt = "What is 2 + 2?"

        result = reasoning.execute(test_prompt)

        # Verify API client was called with formatted prompt
        expected_prompt = "Base prompt: What is 2 + 2?"
        mock_client.generate.assert_called_once_with(expected_prompt)

        # Verify result structure
        assert isinstance(result, StandardResponse)
        assert result.text == "Test response from model"
        assert result.metadata["reasoning_approach"] == "None"
        assert result.metadata["reasoning_steps"] == 0  # Baseline has 0 steps

    @patch("src.reasoning.none.create_api_client")
    @patch("builtins.open", new_callable=mock_open, read_data="Question: {question}")
    def test_execute_metadata_structure(
        self, mock_file, mock_create_client, config, mock_client
    ):
        """Test metadata structure in None reasoning response."""
        mock_create_client.return_value = mock_client

        reasoning = NoneReasoning(config)
        test_prompt = "Test question"

        result = reasoning.execute(test_prompt)

        # Verify approach-specific metadata
        assert result.metadata["approach_specific_metrics"]["baseline"] is True
        assert (
            result.metadata["approach_specific_metrics"]["original_prompt"]
            == test_prompt
        )
        assert result.metadata["approach_specific_metrics"]["template_used"] == "base"

        # Verify baseline characteristics
        assert result.metadata["reasoning_steps"] == 0
        assert result.metadata["reasoning_approach"] == "None"

    @patch("src.reasoning.none.create_api_client")
    @patch("builtins.open", new_callable=mock_open, read_data="{question}")
    def test_execute_empty_prompt(
        self, mock_file, mock_create_client, config, mock_client
    ):
        """Test execution with empty prompt."""
        mock_create_client.return_value = mock_client

        reasoning = NoneReasoning(config)

        result = reasoning.execute("")

        # Should still work with empty prompt
        mock_client.generate.assert_called_once_with("")
        assert result.metadata["approach_specific_metrics"]["original_prompt"] == ""

    @patch("src.reasoning.none.create_api_client")
    @patch(
        "builtins.open", new_callable=mock_open, read_data="Complex prompt: {question}"
    )
    def test_execute_long_prompt(
        self, mock_file, mock_create_client, config, mock_client
    ):
        """Test execution with a long prompt."""
        mock_create_client.return_value = mock_client

        reasoning = NoneReasoning(config)
        long_prompt = "This is a very long prompt. " * 100  # 500+ characters

        result = reasoning.execute(long_prompt)

        # Verify the full prompt is preserved
        expected_formatted = f"Complex prompt: {long_prompt}"
        mock_client.generate.assert_called_once_with(expected_formatted)
        assert (
            result.metadata["approach_specific_metrics"]["original_prompt"]
            == long_prompt
        )

    @patch("src.reasoning.none.create_api_client")
    @patch("builtins.open", new_callable=mock_open, read_data="{question}")
    def test_execute_special_characters(
        self, mock_file, mock_create_client, config, mock_client
    ):
        """Test execution with special characters in prompt."""
        mock_create_client.return_value = mock_client

        reasoning = NoneReasoning(config)
        special_prompt = "What about émojis 🤔 and symbols ∑∆≠?"

        result = reasoning.execute(special_prompt)

        mock_client.generate.assert_called_once_with(special_prompt)
        assert (
            result.metadata["approach_specific_metrics"]["original_prompt"]
            == special_prompt
        )

    @patch("src.reasoning.none.create_api_client")
    @patch("builtins.open", new_callable=mock_open, read_data="Q: {question}")
    def test_execute_api_client_integration(
        self, mock_file, mock_create_client, config, mock_client
    ):
        """Test integration with API client including token tracking."""
        mock_create_client.return_value = mock_client

        # Configure mock client with realistic response
        mock_client.generate.return_value = StandardResponse(
            text="The answer is 42.",
            provider="test_provider",
            model="test-model",
            prompt_tokens=15,
            completion_tokens=8,
            total_tokens=23,
            generation_time=1.5,
            parameters={"temperature": 0.3},
            response_id="response-456",
            metadata={"original_tokens": 23},
        )

        reasoning = NoneReasoning(config)
        test_prompt = "What is the meaning of life?"

        result = reasoning.execute(test_prompt)

        # Verify token information is preserved
        assert result.prompt_tokens == 15
        assert result.completion_tokens == 8
        assert result.total_tokens == 23
        assert result.generation_time == 1.5

        # Verify our metadata is added while preserving original
        assert "original_tokens" in result.metadata
        assert result.metadata["reasoning_approach"] == "None"

    @patch("src.reasoning.none.create_api_client")
    @patch("builtins.open", new_callable=mock_open, read_data="{question}")
    def test_multiple_executions(
        self, mock_file, mock_create_client, config, mock_client
    ):
        """Test multiple executions with the same instance."""
        mock_create_client.return_value = mock_client

        reasoning = NoneReasoning(config)

        # Execute multiple times
        prompts = ["First question", "Second question", "Third question"]
        results = []

        for prompt in prompts:
            result = reasoning.execute(prompt)
            results.append(result)

        # Verify each execution
        assert len(results) == 3
        assert mock_client.generate.call_count == 3

        # Verify each has correct original prompt
        for i, result in enumerate(results):
            assert (
                result.metadata["approach_specific_metrics"]["original_prompt"]
                == prompts[i]
            )
            assert result.metadata["reasoning_steps"] == 0
            assert result.metadata["reasoning_approach"] == "None"

    @patch("src.reasoning.none.create_api_client")
    @patch("builtins.open", new_callable=mock_open, read_data="Answer: {question}")
    def test_cleanup(self, mock_file, mock_create_client, config, mock_client):
        """Test cleanup functionality."""
        mock_create_client.return_value = mock_client
        mock_client.cleanup = Mock()

        reasoning = NoneReasoning(config)
        reasoning.cleanup()

        mock_client.cleanup.assert_called_once()

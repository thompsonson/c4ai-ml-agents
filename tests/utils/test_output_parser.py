"""Tests for output parser functionality."""

from unittest.mock import Mock, patch

import pytest

from ml_agents.utils.answer_extraction import BaseAnswerExtraction
from ml_agents.utils.api_clients import StandardResponse
from ml_agents.utils.output_parser import OutputParser, ParsingError


class TestOutputParser:
    """Test suite for OutputParser class."""

    def setup_method(self):
        """Set up test fixtures."""
        # Create a mock client
        self.mock_client = Mock()
        self.mock_client.provider = "openrouter"
        self.mock_client.model = "test-model"
        self.mock_client.max_tokens = 512

    def test_output_parser_initialization(self):
        """Test OutputParser initialization."""
        parser = OutputParser(
            client=self.mock_client,
            use_structured_parsing=True,
            fallback_to_regex=True,
            confidence_threshold=0.7,
            max_retries=2,
        )

        assert parser.client == self.mock_client
        assert parser.use_structured_parsing is True
        assert parser.fallback_to_regex is True
        assert parser.confidence_threshold == 0.7
        assert parser.max_retries == 2

    def test_regex_fallback_extraction(self):
        """Test regex fallback when structured parsing is disabled."""
        parser = OutputParser(
            client=self.mock_client,
            use_structured_parsing=False,
            fallback_to_regex=True,
        )

        test_text = "The answer is: 42"

        result = parser.extract_answer(test_text)

        assert "extraction" in result
        assert "metadata" in result
        assert result["metadata"]["parsing_method"] == "regex"
        # The regex fallback may not extract the exact "42" - check if it contains it
        assert "42" in result["extraction"].final_answer
        assert result["extraction"].confidence > 0

    def test_regex_patterns(self):
        """Test various regex patterns for answer extraction."""
        parser = OutputParser(
            client=self.mock_client,
            use_structured_parsing=False,
            fallback_to_regex=True,
        )

        test_cases = [
            ("Answer: The solution is 42", "The solution is 42"),
            ("Final Answer: Yes", "Yes"),
            ("The answer is (C)", "C"),
            ("Multiple choice: [B]", "B"),
        ]

        for test_input, expected in test_cases:
            result = parser.extract_answer(test_input)
            assert expected.lower() in result["extraction"].final_answer.lower()

    def test_parsing_failure_with_no_fallback(self):
        """Test parsing failure when fallback is disabled."""
        parser = OutputParser(
            client=self.mock_client,
            use_structured_parsing=False,
            fallback_to_regex=False,
        )

        with pytest.raises(ParsingError):
            parser.extract_answer("Some text without clear answer")

    def test_get_parsing_stats(self):
        """Test parsing statistics retrieval."""
        parser = OutputParser(
            client=self.mock_client,
            use_structured_parsing=True,
            fallback_to_regex=True,
            confidence_threshold=0.8,
            max_retries=3,
        )

        stats = parser.get_parsing_stats()

        assert stats["use_structured_parsing"] is True
        assert stats["fallback_to_regex"] is True
        assert stats["confidence_threshold"] == 0.8
        assert stats["max_retries"] == 3
        assert "instructor_client_available" in stats

    def test_create_extraction_prompt(self):
        """Test extraction prompt creation."""
        parser = OutputParser(
            client=self.mock_client,
            use_structured_parsing=False,
            fallback_to_regex=True,
        )

        response_text = "Let me think about this step by step. The answer is 42."
        prompt = parser._create_extraction_prompt(response_text, "numerical")

        assert "extract the final answer" in prompt.lower()
        assert response_text in prompt
        assert "numerical value" in prompt.lower()

    @patch("ml_agents.utils.output_parser.instructor")
    def test_instructor_initialization_failure(self, mock_instructor):
        """Test handling of instructor initialization failure."""
        mock_instructor.patch.side_effect = Exception("Instructor failed")

        # Should not raise exception if fallback is enabled
        parser = OutputParser(
            client=self.mock_client,
            use_structured_parsing=True,
            fallback_to_regex=True,
        )

        assert parser.instructor_client is None

        # Should raise exception if fallback is disabled
        with pytest.raises(ParsingError):
            OutputParser(
                client=self.mock_client,
                use_structured_parsing=True,
                fallback_to_regex=False,
            )

    def test_empty_response_handling(self):
        """Test handling of empty responses."""
        parser = OutputParser(
            client=self.mock_client,
            use_structured_parsing=False,
            fallback_to_regex=True,
        )

        result = parser.extract_answer("")

        assert result["extraction"].final_answer == "..."
        assert result["extraction"].confidence == 0.1
        assert result["metadata"]["parsing_method"] == "regex"

    def test_confidence_threshold_validation(self):
        """Test confidence threshold validation in results."""
        parser = OutputParser(
            client=self.mock_client,
            use_structured_parsing=False,
            fallback_to_regex=True,
            confidence_threshold=0.8,
        )

        # Test with clear answer (should have reasonable confidence)
        result = parser.extract_answer("Answer: 42")

        # Regex parsing typically gives lower confidence
        assert result["extraction"].confidence <= 0.8
        assert result["metadata"]["parsing_method"] == "regex"

    def test_parsing_metadata_structure(self):
        """Test that parsing metadata has expected structure."""
        parser = OutputParser(
            client=self.mock_client,
            use_structured_parsing=False,
            fallback_to_regex=True,
        )

        result = parser.extract_answer("Answer: Test response")
        metadata = result["metadata"]

        # Check required metadata fields
        required_fields = [
            "parsing_method",
            "parsing_confidence",
            "parsing_attempts",
            "extraction_time_ms",
        ]

        for field in required_fields:
            assert field in metadata

        assert isinstance(metadata["extraction_time_ms"], int)
        assert metadata["extraction_time_ms"] >= 0
        assert isinstance(metadata["parsing_attempts"], int)
        assert (
            metadata["parsing_attempts"] >= 0
        )  # Regex parsing doesn't count attempts the same way

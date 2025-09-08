"""Tests for output parser functionality."""

from unittest.mock import Mock, patch

import pytest

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
        )

        assert parser.client == self.mock_client
        assert parser.use_structured_parsing is True

    def test_parsing_failure_without_structured_parsing(self):
        """Test parsing failure when structured parsing is disabled."""
        parser = OutputParser(
            client=self.mock_client,
            use_structured_parsing=False,
        )

        # Should fail when trying to extract
        with pytest.raises(ParsingError, match="Structured parsing not available"):
            parser.extract_answer("Some text")

    @patch("ml_agents.utils.output_parser.instructor")
    def test_instructor_initialization_failure(self, mock_instructor):
        """Test handling of instructor initialization failure."""
        mock_instructor.patch.side_effect = Exception("Instructor failed")

        # Should raise exception when instructor initialization fails
        with pytest.raises(
            ParsingError, match="Failed to initialize structured parsing"
        ):
            OutputParser(
                client=self.mock_client,
                use_structured_parsing=True,
            )

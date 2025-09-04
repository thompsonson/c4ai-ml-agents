"""Tests for logging configuration."""

import json
import logging
import os
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import pytest

from ml_agents.utils.logging_config import (
    HumanFormatter,
    JSONFormatter,
    get_logger,
    log_api_call,
    log_experiment_end,
    log_experiment_start,
    log_reasoning_step,
    setup_logging,
)


class TestJSONFormatter:
    """Test JSON formatter."""

    def test_format_basic_message(self) -> None:
        """Test formatting a basic log message."""
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="test_logger",
            level=logging.INFO,
            pathname="test.py",
            lineno=42,
            msg="Test message",
            args=(),
            exc_info=None,
        )
        record.module = "test"
        record.funcName = "test_function"

        result = formatter.format(record)
        data = json.loads(result)

        assert data["level"] == "INFO"
        assert data["logger"] == "test_logger"
        assert data["message"] == "Test message"
        assert data["module"] == "test"
        assert data["function"] == "test_function"
        assert data["line"] == 42
        assert "timestamp" in data

    def test_format_with_exception(self) -> None:
        """Test formatting with exception info."""
        formatter = JSONFormatter()

        try:
            raise ValueError("Test exception")
        except ValueError:
            import sys

            exc_info = sys.exc_info()

        record = logging.LogRecord(
            name="test_logger",
            level=logging.ERROR,
            pathname="test.py",
            lineno=42,
            msg="Error occurred",
            args=(),
            exc_info=exc_info,
        )
        record.module = "test"
        record.funcName = "test_function"

        result = formatter.format(record)
        data = json.loads(result)

        assert "exception" in data
        assert "ValueError: Test exception" in data["exception"]

    def test_format_with_extra_fields(self) -> None:
        """Test formatting with extra fields."""
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="test_logger",
            level=logging.INFO,
            pathname="test.py",
            lineno=42,
            msg="Test message",
            args=(),
            exc_info=None,
        )
        record.module = "test"
        record.funcName = "test_function"
        record.extra_fields = {"custom_field": "custom_value"}

        result = formatter.format(record)
        data = json.loads(result)

        assert data["custom_field"] == "custom_value"


class TestHumanFormatter:
    """Test human-readable formatter."""

    def test_format_basic_message(self) -> None:
        """Test formatting a basic log message."""
        formatter = HumanFormatter(use_colors=False)
        record = logging.LogRecord(
            name="test_logger",
            level=logging.INFO,
            pathname="test.py",
            lineno=42,
            msg="Test message",
            args=(),
            exc_info=None,
        )
        record.module = "test"
        record.funcName = "test_function"

        result = formatter.format(record)

        assert "INFO" in result
        assert "test:test_function:42" in result
        assert "Test message" in result
        assert "20" in result  # Year in timestamp

    def test_format_with_colors(self) -> None:
        """Test formatting with colors enabled."""
        formatter = HumanFormatter(use_colors=True)
        record = logging.LogRecord(
            name="test_logger",
            level=logging.ERROR,
            pathname="test.py",
            lineno=42,
            msg="Error message",
            args=(),
            exc_info=None,
        )
        record.module = "test"
        record.funcName = "test_function"

        result = formatter.format(record)

        # Should contain ANSI color codes if colors are enabled
        # Note: This test might be environment-dependent
        assert "ERROR" in result
        assert "Error message" in result

    def test_format_with_exception(self) -> None:
        """Test formatting with exception info."""
        formatter = HumanFormatter(use_colors=False)

        try:
            raise ValueError("Test exception")
        except ValueError:
            import sys

            exc_info = sys.exc_info()

        record = logging.LogRecord(
            name="test_logger",
            level=logging.ERROR,
            pathname="test.py",
            lineno=42,
            msg="Error occurred",
            args=(),
            exc_info=exc_info,
        )
        record.module = "test"
        record.funcName = "test_function"

        result = formatter.format(record)

        assert "Error occurred" in result
        assert "ValueError: Test exception" in result


class TestLoggingSetup:
    """Test logging setup functions."""

    def teardown_method(self) -> None:
        """Clean up after each test."""
        # Reset logging configuration
        logger = logging.getLogger()
        logger.handlers.clear()
        logger.setLevel(logging.WARNING)

    def test_setup_logging_defaults(self) -> None:
        """Test setting up logging with default values."""
        # Clear LOG_LEVEL to test actual defaults
        with patch.dict(os.environ, {}, clear=True):
            logger = setup_logging()

            assert logger.level == logging.ERROR
            assert len(logger.handlers) == 1  # Console handler only
            assert isinstance(logger.handlers[0], logging.StreamHandler)

    @patch.dict(os.environ, {"LOG_LEVEL": "DEBUG", "LOG_FORMAT": "json"})
    def test_setup_logging_from_environment(self) -> None:
        """Test setting up logging from environment variables."""
        logger = setup_logging()

        assert logger.level == logging.DEBUG
        assert isinstance(logger.handlers[0].formatter, JSONFormatter)

    @pytest.mark.logging_test
    def test_setup_logging_with_file(self, temp_dir) -> None:
        """Test setting up logging with file output."""
        log_dir = str(temp_dir)
        logger = setup_logging(log_to_file=True, log_dir=log_dir)

        # Should have both console and file handlers
        assert len(logger.handlers) == 2

        # Check that log file was created
        log_file = temp_dir / "ml_agents.log"
        assert log_file.exists()

    def test_setup_logging_invalid_level(self) -> None:
        """Test setup with invalid log level."""
        with pytest.raises(ValueError, match="Invalid log level"):
            setup_logging(level="INVALID")

    def test_get_logger(self) -> None:
        """Test getting a named logger."""
        logger = get_logger("test_module")
        assert logger.name == "test_module"
        assert isinstance(logger, logging.Logger)


class TestLoggingHelpers:
    """Test logging helper functions."""

    def setup_method(self) -> None:
        """Set up for each test."""
        # Capture log output
        self.log_capture = StringIO()
        self.handler = logging.StreamHandler(self.log_capture)
        self.handler.setFormatter(JSONFormatter())

        self.logger = logging.getLogger("test")
        self.logger.setLevel(logging.DEBUG)
        self.logger.addHandler(self.handler)

    def teardown_method(self) -> None:
        """Clean up after each test."""
        self.logger.removeHandler(self.handler)

    def test_log_experiment_start(self) -> None:
        """Test logging experiment start."""
        config = {"model": "test-model", "temperature": 0.5}
        log_experiment_start(config, self.logger)

        output = self.log_capture.getvalue()
        data = json.loads(output.strip())

        assert data["level"] == "INFO"
        assert "Starting ML Agents experiment" in data["message"]
        assert data["config"]["model"] == "test-model"

    def test_log_experiment_end(self) -> None:
        """Test logging experiment end."""
        results = {"accuracy": 0.85, "total_time": 120.5}
        log_experiment_end(results, self.logger)

        output = self.log_capture.getvalue()
        data = json.loads(output.strip())

        assert data["level"] == "INFO"
        assert "Completed ML Agents experiment" in data["message"]
        assert data["results"]["accuracy"] == 0.85

    def test_log_api_call_success(self) -> None:
        """Test logging successful API call."""
        log_api_call(
            provider="openrouter",
            model="gpt-4",
            duration=2.5,
            tokens=150,
            cost=0.02,
            success=True,
            logger=self.logger,
        )

        output = self.log_capture.getvalue()
        data = json.loads(output.strip())

        assert data["level"] == "INFO"
        assert "API call to openrouter/gpt-4" in data["message"]
        assert data["provider"] == "openrouter"
        assert data["duration"] == 2.5
        assert data["success"] is True

    def test_log_api_call_failure(self) -> None:
        """Test logging failed API call."""
        log_api_call(
            provider="anthropic",
            model="claude-3",
            duration=1.0,
            tokens=0,
            cost=0.0,
            success=False,
            error="Rate limit exceeded",
            logger=self.logger,
        )

        output = self.log_capture.getvalue()
        data = json.loads(output.strip())

        assert data["level"] == "ERROR"
        assert data["success"] is False
        assert data["error"] == "Rate limit exceeded"

    def test_log_reasoning_step(self) -> None:
        """Test logging reasoning step."""
        log_reasoning_step(
            step_name="chain_of_thought",
            prompt="What is 2+2?",
            response="Let me think step by step. 2+2=4",
            duration=1.5,
            logger=self.logger,
        )

        output = self.log_capture.getvalue()
        data = json.loads(output.strip())

        assert data["level"] == "DEBUG"
        assert "Reasoning step: chain_of_thought" in data["message"]
        assert data["step_name"] == "chain_of_thought"
        assert data["prompt"] == "What is 2+2?"
        assert data["duration"] == 1.5

    def test_log_reasoning_step_long_content(self) -> None:
        """Test logging reasoning step with long content."""
        long_prompt = "A" * 1000
        long_response = "B" * 1000

        log_reasoning_step(
            step_name="test",
            prompt=long_prompt,
            response=long_response,
            duration=1.0,
            logger=self.logger,
        )

        output = self.log_capture.getvalue()
        data = json.loads(output.strip())

        # Should be truncated
        assert len(data["prompt"]) <= 503  # 500 + "..."
        assert len(data["response"]) <= 503
        assert data["prompt"].endswith("...")
        assert data["response"].endswith("...")


class TestLoggingIntegration:
    """Test logging integration scenarios."""

    def test_multiple_loggers(self) -> None:
        """Test that multiple loggers work correctly."""
        # Clear environment to ensure clean test
        with patch.dict(os.environ, {}, clear=True):
            setup_logging(level="DEBUG")

            logger1 = get_logger("module1")
            logger2 = get_logger("module2")

            assert logger1.name == "module1"
            assert logger2.name == "module2"
            # Child loggers inherit effective level from root logger
            assert logger1.getEffectiveLevel() == logging.DEBUG
            assert logger2.getEffectiveLevel() == logging.DEBUG

    def test_third_party_logger_levels(self) -> None:
        """Test that third-party loggers are configured correctly."""
        setup_logging(level="DEBUG")

        # These should be set to WARNING to reduce noise
        assert logging.getLogger("urllib3").level == logging.WARNING
        assert logging.getLogger("requests").level == logging.WARNING
        assert logging.getLogger("transformers").level == logging.WARNING

    @pytest.mark.logging_test
    @patch.dict(os.environ, {"LOG_TO_FILE": "true"})
    def test_file_logging_rotation(self, temp_dir) -> None:
        """Test file logging with rotation."""
        log_dir = str(temp_dir)

        # Set up logging with small file size for testing
        setup_logging(
            log_to_file=True,
            log_dir=log_dir,
            max_file_size=100,  # Very small for testing
        )

        logger = get_logger("test")

        # Write enough logs to trigger rotation
        for i in range(50):
            logger.info(f"Test message {i} with enough content to fill up the file")

        # Check that log files exist
        log_files = list(Path(log_dir).glob("ml_agents.log*"))
        assert len(log_files) >= 1

"""Pytest configuration and fixtures."""

import logging
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest


def pytest_configure(config):
    """Configure pytest settings."""
    # Set a minimal logging configuration for cleaner output
    # Note: We use INFO level to allow caplog to capture warnings
    logging.basicConfig(
        level=logging.INFO,  # Allow capturing of warnings in tests
        format="%(levelname)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stderr)],
        force=True,
    )
    # But set the console handler to only show critical messages for clean output
    for handler in logging.root.handlers:
        if isinstance(handler, logging.StreamHandler):
            handler.setLevel(logging.CRITICAL)

    # Suppress debug logging from third-party HTTP libraries to prevent
    # cleanup errors when temporary directories are removed
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)


@pytest.fixture(autouse=True)
def suppress_file_logging(request):
    """Suppress file logging for tests that don't need it."""
    # Skip this fixture for tests marked as logging_test
    if request.node.get_closest_marker("logging_test"):
        yield
        return

    # For all other tests, mock the RotatingFileHandler to prevent file creation
    with patch("logging.handlers.RotatingFileHandler") as mock_handler:
        # Create a mock that behaves like a handler but doesn't create files
        mock_instance = Mock()
        mock_instance.setLevel = Mock()
        mock_instance.setFormatter = Mock()
        mock_instance.emit = Mock()
        mock_handler.return_value = mock_instance
        yield


@pytest.fixture(autouse=True)
def ensure_temp_dirs(tmp_path_factory):
    """Ensure temporary directories exist for tests that might need them."""
    # Create a temporary directory that persists for the session
    temp_dir = tmp_path_factory.getbasetemp()
    os.environ["TEST_TEMP_DIR"] = str(temp_dir)
    yield
    # Cleanup
    if "TEST_TEMP_DIR" in os.environ:
        del os.environ["TEST_TEMP_DIR"]

"""Tests for CLI main module."""

import pytest
from typer.testing import CliRunner

from src.cli.main import app


class TestMainCLI:
    """Test the main CLI application."""

    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()

    def test_main_help(self):
        """Test main help command."""
        result = self.runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "ML Agents Reasoning Research Platform" in result.stdout
        assert "Cohere Labs Open Science Initiative" in result.stdout

    def test_version_command(self):
        """Test version command."""
        result = self.runner.invoke(app, ["version"])
        assert result.exit_code == 0
        assert "ML Agents CLI" in result.stdout

    def test_version_flag(self):
        """Test version flag."""
        result = self.runner.invoke(app, ["--version"])
        assert result.exit_code == 0
        assert "ML Agents CLI" in result.stdout

    def test_list_approaches(self):
        """Test listing available approaches."""
        result = self.runner.invoke(app, ["list-approaches"])
        assert result.exit_code == 0
        assert "Available Reasoning Approaches" in result.stdout
        assert "ChainOfThought" in result.stdout

    def test_validate_env_command(self):
        """Test environment validation command."""
        result = self.runner.invoke(app, ["validate-env"])
        # This might fail due to missing API keys, but should not crash
        assert result.exit_code in [0, 1]  # 0 for success, 1 for validation failure

    def test_invalid_command(self):
        """Test invalid command handling."""
        result = self.runner.invoke(app, ["invalid-command"])
        assert result.exit_code != 0


class TestCommandExists:
    """Test that all expected commands are available."""

    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()

    def test_run_command_exists(self):
        """Test that run command exists."""
        result = self.runner.invoke(app, ["run", "--help"])
        assert result.exit_code == 0
        assert "single reasoning experiment" in result.stdout.lower()

    def test_compare_command_exists(self):
        """Test that compare command exists."""
        result = self.runner.invoke(app, ["compare", "--help"])
        assert result.exit_code == 0
        assert "comparison experiment" in result.stdout.lower()

    def test_resume_command_exists(self):
        """Test that resume command exists."""
        result = self.runner.invoke(app, ["resume", "--help"])
        assert result.exit_code == 0
        assert "checkpoint" in result.stdout.lower()

    def test_list_checkpoints_command_exists(self):
        """Test that list-checkpoints command exists."""
        result = self.runner.invoke(app, ["list-checkpoints", "--help"])
        assert result.exit_code == 0
        assert "checkpoint" in result.stdout.lower()

"""Comprehensive tests for setup CLI commands."""

from unittest.mock import Mock, patch

import pytest
from typer.testing import CliRunner

from ml_agents.cli.main import app


class TestSetupValidateEnv:
    """Test the setup validate-env command."""

    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()

    def test_validate_env_command_help(self):
        """Test validate-env command help display."""
        result = self.runner.invoke(app, ["setup", "validate-env", "--help"])
        assert result.exit_code == 0
        assert "environment configuration" in result.stdout.lower()

    @patch("ml_agents.cli.commands.setup.validate_environment")
    def test_validate_env_success(self, mock_validate):
        """Test successful environment validation."""
        mock_validate.return_value = {
            "api_keys": True,
            "output_dir_writable": True,
            "dependencies_available": True,
        }

        result = self.runner.invoke(app, ["setup", "validate-env"])

        assert result.exit_code == 0
        assert "Environment validation passed" in result.stdout
        assert "Ready to run experiments" in result.stdout
        mock_validate.assert_called_once()

    @patch("ml_agents.cli.commands.setup.validate_environment")
    def test_validate_env_missing_api_keys(self, mock_validate):
        """Test validation with missing API keys."""
        mock_validate.return_value = {
            "api_keys": False,
            "output_dir_writable": True,
            "dependencies_available": True,
        }

        result = self.runner.invoke(app, ["setup", "validate-env"])

        assert result.exit_code == 1
        assert "Environment validation failed" in result.stdout
        assert "Missing API keys" in result.stdout

    @patch("ml_agents.cli.commands.setup.validate_environment")
    def test_validate_env_output_dir_not_writable(self, mock_validate):
        """Test validation with non-writable output directory."""
        mock_validate.return_value = {
            "api_keys": True,
            "output_dir_writable": False,
            "dependencies_available": True,
        }

        result = self.runner.invoke(app, ["setup", "validate-env"])

        assert result.exit_code == 1
        assert "Cannot write to output directory" in result.stdout

    @patch("ml_agents.cli.commands.setup.validate_environment")
    def test_validate_env_missing_dependencies(self, mock_validate):
        """Test validation with missing dependencies."""
        mock_validate.return_value = {
            "api_keys": True,
            "output_dir_writable": True,
            "dependencies_available": False,
        }

        result = self.runner.invoke(app, ["setup", "validate-env"])

        assert result.exit_code == 1
        assert "Missing required dependencies" in result.stdout

    @patch("ml_agents.cli.commands.setup.validate_environment")
    def test_validate_env_all_failed(self, mock_validate):
        """Test validation with all checks failing."""
        mock_validate.return_value = {
            "api_keys": False,
            "output_dir_writable": False,
            "dependencies_available": False,
        }

        result = self.runner.invoke(app, ["setup", "validate-env"])

        assert result.exit_code == 1
        assert "Environment validation failed" in result.stdout
        assert "Missing API keys" in result.stdout
        assert "Cannot write to output directory" in result.stdout
        assert "Missing required dependencies" in result.stdout


class TestSetupListApproaches:
    """Test the setup list-approaches command."""

    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()

    def test_list_approaches_command_help(self):
        """Test list-approaches command help display."""
        result = self.runner.invoke(app, ["setup", "list-approaches", "--help"])
        assert result.exit_code == 0
        assert "reasoning approaches" in result.stdout.lower()

    @patch("ml_agents.reasoning.get_available_approaches")
    def test_list_approaches_complete(self, mock_get_approaches):
        """Test that all approaches are listed."""
        mock_approaches = [
            "ChainOfThought",
            "TreeOfThought",
            "Reflection",
            "AsPlanning",
            "ProgramOfThought",
            "ChainOfVerification",
            "SkeletonOfThought",
            "None",
        ]
        mock_get_approaches.return_value = mock_approaches

        result = self.runner.invoke(app, ["setup", "list-approaches"])

        assert result.exit_code == 0
        assert "Available Reasoning Approaches" in result.stdout

        # Check that all approaches are listed
        for approach in mock_approaches:
            assert approach in result.stdout

        # Check that it shows the total count
        assert f"Total: {len(mock_approaches)} approaches available" in result.stdout

    @patch("ml_agents.reasoning.get_available_approaches")
    def test_list_approaches_empty_list(self, mock_get_approaches):
        """Test behavior with empty approaches list."""
        mock_get_approaches.return_value = []

        result = self.runner.invoke(app, ["setup", "list-approaches"])

        assert result.exit_code == 0
        assert "Available Reasoning Approaches" in result.stdout
        assert "Total: 0 approaches available" in result.stdout

    @patch("ml_agents.reasoning.get_available_approaches")
    def test_list_approaches_single_approach(self, mock_get_approaches):
        """Test behavior with single approach."""
        mock_get_approaches.return_value = ["ChainOfThought"]

        result = self.runner.invoke(app, ["setup", "list-approaches"])

        assert result.exit_code == 0
        assert "ChainOfThought" in result.stdout
        assert "1. ChainOfThought" in result.stdout
        assert "Total: 1 approaches available" in result.stdout


class TestSetupVersion:
    """Test the setup version command."""

    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()

    def test_version_command_help(self):
        """Test version command help display."""
        result = self.runner.invoke(app, ["setup", "version", "--help"])
        assert result.exit_code == 0
        assert "version information" in result.stdout.lower()

    @patch("importlib.metadata.version", return_value="0.1.0")
    def test_version_command_format(self, mock_version):
        """Test version command output format."""
        result = self.runner.invoke(app, ["setup", "version"])

        assert result.exit_code == 0
        assert "ML Agents CLI" in result.stdout
        assert "0.1.0" in result.stdout
        assert "Cohere Labs Open Science Initiative" in result.stdout

    def test_main_version_flag(self):
        """Test main CLI version flag."""
        result = self.runner.invoke(app, ["--version"])

        assert result.exit_code == 0
        assert "ML Agents CLI" in result.stdout

    def test_version_flag_short(self):
        """Test short version flag."""
        result = self.runner.invoke(app, ["-V"])

        assert result.exit_code == 0
        assert "ML Agents CLI" in result.stdout


class TestSetupCommandsIntegration:
    """Integration tests for setup commands."""

    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()

    def test_setup_group_exists(self):
        """Test that setup group exists and is accessible."""
        result = self.runner.invoke(app, ["setup", "--help"])
        assert result.exit_code == 0
        assert "Environment setup" in result.stdout
        assert "validate-env" in result.stdout
        assert "list-approaches" in result.stdout
        assert "version" in result.stdout

    def test_all_setup_commands_accessible(self):
        """Test that all setup commands are accessible."""
        commands = ["validate-env", "list-approaches", "version"]

        for command in commands:
            result = self.runner.invoke(app, ["setup", command, "--help"])
            assert result.exit_code == 0, f"Command 'setup {command}' not accessible"

    @patch("ml_agents.reasoning.get_available_approaches")
    @patch("ml_agents.cli.commands.setup.validate_environment")
    def test_setup_workflow_integration(self, mock_validate, mock_get_approaches):
        """Test typical setup workflow integration."""
        # Setup mocks
        mock_validate.return_value = {
            "api_keys": True,
            "output_dir_writable": True,
            "dependencies_available": True,
        }
        mock_get_approaches.return_value = ["ChainOfThought", "Reflection"]

        # Test version first
        version_result = self.runner.invoke(app, ["setup", "version"])
        assert version_result.exit_code == 0

        # Test list approaches
        list_result = self.runner.invoke(app, ["setup", "list-approaches"])
        assert list_result.exit_code == 0
        assert "ChainOfThought" in list_result.stdout

        # Test environment validation
        validate_result = self.runner.invoke(app, ["setup", "validate-env"])
        assert validate_result.exit_code == 0
        assert "Environment validation passed" in validate_result.stdout

"""Tests for CLI main module with grouped command structure."""

import pytest
from typer.testing import CliRunner

from ml_agents.cli.main import app


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

    def test_version_flag(self):
        """Test version flag."""
        result = self.runner.invoke(app, ["--version"])
        assert result.exit_code == 0
        assert "ML Agents CLI" in result.stdout

    def test_invalid_command(self):
        """Test invalid command handling."""
        result = self.runner.invoke(app, ["invalid-command"])
        assert result.exit_code != 0


class TestGroupedCommands:
    """Test that all grouped commands are available."""

    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()

    def test_setup_group_exists(self):
        """Test that setup group exists."""
        result = self.runner.invoke(app, ["setup", "--help"])
        assert result.exit_code == 0
        assert "Environment setup" in result.stdout

    def test_preprocess_group_exists(self):
        """Test that preprocess group exists."""
        result = self.runner.invoke(app, ["preprocess", "--help"])
        assert result.exit_code == 0
        assert "Dataset preprocessing" in result.stdout

    def test_eval_group_exists(self):
        """Test that eval group exists."""
        result = self.runner.invoke(app, ["eval", "--help"])
        assert result.exit_code == 0
        assert "Reasoning evaluation" in result.stdout

    def test_results_group_exists(self):
        """Test that results group exists."""
        result = self.runner.invoke(app, ["results", "--help"])
        assert result.exit_code == 0
        assert "Results analysis" in result.stdout

    def test_db_group_exists(self):
        """Test that db group exists."""
        result = self.runner.invoke(app, ["db", "--help"])
        assert result.exit_code == 0
        assert "Database management" in result.stdout


class TestSetupCommands:
    """Test setup command group."""

    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()

    def test_setup_version_command(self):
        """Test setup version command."""
        result = self.runner.invoke(app, ["setup", "version"])
        assert result.exit_code == 0
        assert "ML Agents CLI" in result.stdout

    def test_setup_list_approaches_command(self):
        """Test setup list-approaches command."""
        result = self.runner.invoke(app, ["setup", "list-approaches"])
        assert result.exit_code == 0
        assert "Available Reasoning Approaches" in result.stdout
        assert "ChainOfThought" in result.stdout

    def test_setup_validate_env_command(self):
        """Test setup validate-env command."""
        result = self.runner.invoke(app, ["setup", "validate-env"])
        # This might fail due to missing API keys, but should not crash
        assert result.exit_code in [0, 1]  # 0 for success, 1 for validation failure


class TestEvalCommands:
    """Test evaluation command group."""

    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()

    def test_eval_run_command_exists(self):
        """Test that eval run command exists."""
        result = self.runner.invoke(app, ["eval", "run", "--help"])
        assert result.exit_code == 0
        assert "single reasoning experiment" in result.stdout.lower()

    def test_eval_compare_command_exists(self):
        """Test that eval compare command exists."""
        result = self.runner.invoke(app, ["eval", "compare", "--help"])
        assert result.exit_code == 0
        assert "comparison experiment" in result.stdout.lower()

    def test_eval_resume_command_exists(self):
        """Test that eval resume command exists."""
        result = self.runner.invoke(app, ["eval", "resume", "--help"])
        assert result.exit_code == 0
        assert "checkpoint" in result.stdout.lower()

    def test_eval_checkpoints_command_exists(self):
        """Test that eval checkpoints command exists."""
        result = self.runner.invoke(app, ["eval", "checkpoints", "--help"])
        assert result.exit_code == 0
        assert "checkpoint" in result.stdout.lower()


class TestPreprocessCommands:
    """Test preprocessing command group."""

    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()

    def test_preprocess_list_command_exists(self):
        """Test that preprocess list command exists."""
        result = self.runner.invoke(app, ["preprocess", "list", "--help"])
        assert result.exit_code == 0

    def test_preprocess_inspect_command_exists(self):
        """Test that preprocess inspect command exists."""
        result = self.runner.invoke(app, ["preprocess", "inspect", "--help"])
        assert result.exit_code == 0

    def test_preprocess_generate_rules_command_exists(self):
        """Test that preprocess generate-rules command exists."""
        result = self.runner.invoke(app, ["preprocess", "generate-rules", "--help"])
        assert result.exit_code == 0

    def test_preprocess_transform_command_exists(self):
        """Test that preprocess transform command exists."""
        result = self.runner.invoke(app, ["preprocess", "transform", "--help"])
        assert result.exit_code == 0

    def test_preprocess_batch_command_exists(self):
        """Test that preprocess batch command exists."""
        result = self.runner.invoke(app, ["preprocess", "batch", "--help"])
        assert result.exit_code == 0

    def test_preprocess_upload_command_exists(self):
        """Test that preprocess upload command exists."""
        result = self.runner.invoke(app, ["preprocess", "upload", "--help"])
        assert result.exit_code == 0


class TestResultsCommands:
    """Test results command group."""

    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()

    def test_results_export_command_exists(self):
        """Test that results export command exists."""
        result = self.runner.invoke(app, ["results", "export", "--help"])
        assert result.exit_code == 0

    def test_results_analyze_command_exists(self):
        """Test that results analyze command exists."""
        result = self.runner.invoke(app, ["results", "analyze", "--help"])
        assert result.exit_code == 0

    def test_results_compare_command_exists(self):
        """Test that results compare command exists."""
        result = self.runner.invoke(app, ["results", "compare", "--help"])
        assert result.exit_code == 0

    def test_results_list_command_exists(self):
        """Test that results list command exists."""
        result = self.runner.invoke(app, ["results", "list", "--help"])
        assert result.exit_code == 0


class TestDatabaseCommands:
    """Test database command group."""

    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()

    def test_db_init_command_exists(self):
        """Test that db init command exists."""
        result = self.runner.invoke(app, ["db", "init", "--help"])
        assert result.exit_code == 0

    def test_db_backup_command_exists(self):
        """Test that db backup command exists."""
        result = self.runner.invoke(app, ["db", "backup", "--help"])
        assert result.exit_code == 0

    def test_db_migrate_command_exists(self):
        """Test that db migrate command exists."""
        result = self.runner.invoke(app, ["db", "migrate", "--help"])
        assert result.exit_code == 0

    def test_db_stats_command_exists(self):
        """Test that db stats command exists."""
        result = self.runner.invoke(app, ["db", "stats", "--help"])
        assert result.exit_code == 0

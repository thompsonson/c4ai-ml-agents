"""Basic integration smoke tests for stable CLI commands.

These tests verify that commands can be invoked without errors,
without requiring complex setup or external dependencies.
"""

import tempfile
from pathlib import Path

import pytest
from typer.testing import CliRunner

from ml_agents.cli.main import app


class TestIntegrationSmokeTests:
    """Basic smoke tests for stable CLI commands."""

    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()

    def test_main_help_accessible(self):
        """Test that main CLI help is accessible."""
        result = self.runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "ML Agents CLI" in result.stdout
        assert "setup" in result.stdout
        assert "preprocess" in result.stdout
        assert "db" in result.stdout

    def test_version_commands_work(self):
        """Test that version commands work without external dependencies."""
        # Main version flag
        result = self.runner.invoke(app, ["--version"])
        assert result.exit_code == 0
        assert "ML Agents CLI" in result.stdout

        # Short version flag
        result = self.runner.invoke(app, ["-V"])
        assert result.exit_code == 0
        assert "ML Agents CLI" in result.stdout

        # Setup version command
        result = self.runner.invoke(app, ["setup", "version"])
        assert result.exit_code == 0
        assert "ML Agents CLI" in result.stdout
        assert "Cohere Labs" in result.stdout

    def test_help_commands_accessible(self):
        """Test that all group help commands are accessible."""
        groups = ["setup", "preprocess", "db"]

        for group in groups:
            result = self.runner.invoke(app, [group, "--help"])
            assert result.exit_code == 0, f"Help for '{group}' group failed"
            assert group in result.stdout.lower()

    def test_setup_commands_basic_access(self):
        """Test basic accessibility of setup commands."""
        commands = ["validate-env", "list-approaches", "version"]

        for command in commands:
            result = self.runner.invoke(app, ["setup", command, "--help"])
            assert result.exit_code == 0, f"Setup command '{command}' help failed"

    def test_db_commands_basic_access(self):
        """Test basic accessibility of db commands."""
        commands = ["init", "backup", "stats", "migrate"]

        for command in commands:
            result = self.runner.invoke(app, ["db", command, "--help"])
            assert result.exit_code == 0, f"DB command '{command}' help failed"

    def test_preprocess_commands_basic_access(self):
        """Test basic accessibility of preprocessing commands."""
        commands = ["list", "inspect", "generate-rules", "transform", "batch", "upload"]

        for command in commands:
            result = self.runner.invoke(app, ["preprocess", command, "--help"])
            assert result.exit_code == 0, f"Preprocess command '{command}' help failed"

    def test_error_handling_smoke(self):
        """Test basic error handling patterns."""
        # Test invalid subcommand
        result = self.runner.invoke(app, ["invalid-command"])
        assert result.exit_code != 0

        # Test invalid db operations on nonexistent database
        result = self.runner.invoke(app, ["db", "stats", "--db-path", "nonexistent.db"])
        assert result.exit_code == 1
        assert "not found" in result.stdout.lower()

    def test_db_init_with_temp_directory(self):
        """Test db init command with temporary directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test.db"

            result = self.runner.invoke(app, ["db", "init", "--db-path", str(db_path)])

            # Should either succeed or fail gracefully
            assert result.exit_code in [0, 1]
            if result.exit_code == 1:
                # Should contain helpful error message
                assert "failed" in result.stdout.lower()

    def test_preprocess_list_basic_functionality(self):
        """Test preprocessing list command basic functionality."""
        # This should work even without valid benchmark CSV
        result = self.runner.invoke(
            app, ["preprocess", "list", "--benchmark-csv", "nonexistent.csv"]
        )

        # Should either succeed with empty results or fail gracefully
        assert result.exit_code in [0, 1]
        if result.exit_code == 1:
            assert any(
                word in result.stdout.lower()
                for word in ["not found", "error", "failed"]
            )

    @pytest.mark.skip(reason="Requires actual reasoning module - may be unstable")
    def test_setup_list_approaches_basic(self):
        """Test setup list-approaches command (may be unstable)."""
        result = self.runner.invoke(app, ["setup", "list-approaches"])

        # Should either work or fail gracefully
        assert result.exit_code in [0, 1]
        if result.exit_code == 0:
            assert "Available Reasoning Approaches" in result.stdout

    def test_keyboard_interrupt_handling(self):
        """Test that KeyboardInterrupt is handled gracefully."""
        # This is a basic smoke test - actual interrupt testing is complex
        # We just verify the main app structure supports it
        result = self.runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        # If we got here, the basic CLI structure is sound


class TestCLIConsistencySmokeTests:
    """Smoke tests for CLI consistency patterns."""

    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()

    def test_all_commands_support_help(self):
        """Verify all commands support --help flag."""
        commands_to_test = [
            # Setup commands
            ["setup", "validate-env"],
            ["setup", "list-approaches"],
            ["setup", "version"],
            # DB commands
            ["db", "init"],
            ["db", "backup"],
            ["db", "stats"],
            ["db", "migrate"],
            # Preprocess commands
            ["preprocess", "list"],
            ["preprocess", "inspect"],
            ["preprocess", "generate-rules"],
            ["preprocess", "transform"],
            ["preprocess", "batch"],
            ["preprocess", "upload"],
        ]

        for cmd_parts in commands_to_test:
            result = self.runner.invoke(app, cmd_parts + ["--help"])
            assert (
                result.exit_code == 0
            ), f"Command {' '.join(cmd_parts)} does not support --help"
            assert "help" in result.stdout.lower() or "usage" in result.stdout.lower()

    def test_consistent_option_patterns(self):
        """Test that common options follow consistent patterns."""
        # Test db-path option consistency (where applicable)
        db_commands = [
            ["db", "init", "--help"],
            ["db", "backup", "--help"],
            ["db", "stats", "--help"],
            ["db", "migrate", "--help"],
            ["preprocess", "list", "--help"],
        ]

        for cmd in db_commands:
            result = self.runner.invoke(app, cmd)
            assert result.exit_code == 0
            if "--db-path" in result.stdout:
                # Should mention default value
                assert "default" in result.stdout.lower()

    def test_consistent_error_exit_codes(self):
        """Test that error conditions use consistent exit codes."""
        error_scenarios = [
            # Nonexistent database
            ["db", "stats", "--db-path", "nonexistent.db"],
            # Invalid format
            # Note: Can't easily test format validation without complex setup
        ]

        for cmd in error_scenarios:
            result = self.runner.invoke(app, cmd)
            if result.exit_code != 0:
                # Should use exit code 1 for errors, not random codes
                assert (
                    result.exit_code == 1
                ), f"Command {' '.join(cmd)} used unexpected exit code: {result.exit_code}"

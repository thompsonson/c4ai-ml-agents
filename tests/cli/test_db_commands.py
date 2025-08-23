"""Comprehensive tests for database CLI commands."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest
from typer.testing import CliRunner

from ml_agents.cli.main import app


class TestDbInitCommand:
    """Test the db init command."""

    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()

    def test_db_init_command_help(self):
        """Test db init command help display."""
        result = self.runner.invoke(app, ["db", "init", "--help"])
        assert result.exit_code == 0
        assert "Initialize" in result.stdout
        assert "database" in result.stdout.lower()

    @patch("ml_agents.core.database_manager.DatabaseManager")
    @patch("ml_agents.core.database_manager.DatabaseConfig")
    def test_db_init_new_database(self, mock_config, mock_manager):
        """Test initializing a new database."""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test.db"

            # Mock the database manager
            mock_manager_instance = Mock()
            mock_manager_instance.get_database_stats.return_value = {
                "schema_version": "1.2.0",
                "experiments_count": 0,
                "runs_count": 0,
                "database_size_bytes": 1024,
            }
            mock_manager.return_value = mock_manager_instance

            result = self.runner.invoke(app, ["db", "init", "--db-path", str(db_path)])

            assert result.exit_code == 0
            assert "Database initialized successfully" in result.stdout
            assert "Schema version: 1.2.0" in result.stdout
            assert "Total experiments: 0" in result.stdout
            assert "Database size: 1024 bytes" in result.stdout

    @patch("ml_agents.core.database_manager.DatabaseManager")
    @patch("ml_agents.core.database_manager.DatabaseConfig")
    @patch("pathlib.Path")
    def test_db_init_existing_database_without_force(
        self, mock_path, mock_config, mock_manager
    ):
        """Test initializing when database exists without force flag."""
        # Mock path to simulate existing database
        mock_path_instance = Mock()
        mock_path_instance.exists.return_value = True
        mock_path.return_value = mock_path_instance

        # Mock user declining to continue
        with patch("typer.confirm", return_value=False):
            result = self.runner.invoke(app, ["db", "init", "--db-path", "test.db"])

            assert result.exit_code == 0
            assert "Database already exists" in result.stdout
            assert "Database initialization cancelled" in result.stdout

    @patch("ml_agents.core.database_manager.DatabaseManager")
    @patch("ml_agents.core.database_manager.DatabaseConfig")
    @patch("pathlib.Path")
    def test_db_init_existing_database_with_force(
        self, mock_path, mock_config, mock_manager
    ):
        """Test initializing when database exists with force flag."""
        mock_path_instance = Mock()
        mock_path_instance.exists.return_value = True
        mock_path.return_value = mock_path_instance

        mock_manager_instance = Mock()
        mock_manager_instance.get_database_stats.return_value = {
            "schema_version": "1.2.0",
            "experiments_count": 5,
            "runs_count": 25,
            "database_size_bytes": 2048,
        }
        mock_manager.return_value = mock_manager_instance

        result = self.runner.invoke(
            app, ["db", "init", "--db-path", "test.db", "--force"]
        )

        assert result.exit_code == 0
        assert "Database initialized successfully" in result.stdout

    @patch("ml_agents.core.database_manager.DatabaseManager")
    @patch("ml_agents.core.database_manager.DatabaseConfig")
    def test_db_init_default_path(self, mock_config, mock_manager):
        """Test db init with default database path."""
        mock_manager_instance = Mock()
        mock_manager_instance.get_database_stats.return_value = {
            "schema_version": "1.2.0",
            "experiments_count": 0,
            "runs_count": 0,
            "database_size_bytes": 1024,
        }
        mock_manager.return_value = mock_manager_instance

        result = self.runner.invoke(app, ["db", "init"])

        assert result.exit_code == 0
        # Should use default path
        mock_config.assert_called_with(db_path="./ml_agents_results.db")

    @patch("ml_agents.core.database_manager.DatabaseManager")
    @patch("ml_agents.core.database_manager.DatabaseConfig")
    def test_db_init_error_handling(self, mock_config, mock_manager):
        """Test db init error handling."""
        mock_manager.side_effect = Exception("Database connection failed")

        result = self.runner.invoke(app, ["db", "init", "--db-path", "test.db"])

        assert result.exit_code == 1
        assert "Failed to initialize database" in result.stdout


class TestDbBackupCommand:
    """Test the db backup command."""

    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()

    def test_db_backup_command_help(self):
        """Test db backup command help display."""
        result = self.runner.invoke(app, ["db", "backup", "--help"])
        assert result.exit_code == 0
        assert "backup" in result.stdout.lower()

    @patch("ml_agents.core.database_manager.DatabaseManager")
    @patch("ml_agents.core.database_manager.DatabaseConfig")
    @patch("pathlib.Path")
    def test_db_backup_success(self, mock_path, mock_config, mock_manager):
        """Test successful database backup."""
        # Mock source database exists
        mock_source_path = Mock()
        mock_source_path.exists.return_value = True
        mock_path.return_value = mock_source_path

        # Mock backup file stats
        mock_backup_path = Mock()
        mock_backup_path.stat.return_value.st_size = 2048

        with patch(
            "ml_agents.cli.commands.db.Path",
            side_effect=lambda x: (
                mock_source_path if "source" in str(x) else mock_backup_path
            ),
        ):
            mock_manager_instance = Mock()
            mock_manager.return_value = mock_manager_instance

            result = self.runner.invoke(app, ["db", "backup", "--source", "source.db"])

            assert result.exit_code == 0
            assert "Database backup created successfully" in result.stdout
            assert "Source: source.db" in result.stdout
            assert "Size: 2048 bytes" in result.stdout

    @patch("pathlib.Path")
    def test_db_backup_source_not_exists(self, mock_path):
        """Test backup when source database doesn't exist."""
        mock_path_instance = Mock()
        mock_path_instance.exists.return_value = False
        mock_path.return_value = mock_path_instance

        result = self.runner.invoke(app, ["db", "backup", "--source", "nonexistent.db"])

        assert result.exit_code == 1
        assert "Database not found" in result.stdout

    @patch("ml_agents.core.database_manager.DatabaseManager")
    @patch("ml_agents.core.database_manager.DatabaseConfig")
    @patch("pathlib.Path")
    def test_db_backup_custom_path(self, mock_path, mock_config, mock_manager):
        """Test backup with custom backup path."""
        mock_source_path = Mock()
        mock_source_path.exists.return_value = True
        mock_backup_path = Mock()
        mock_backup_path.stat.return_value.st_size = 1500

        def path_side_effect(path):
            if "custom_backup" in str(path):
                return mock_backup_path
            return mock_source_path

        with patch("ml_agents.cli.commands.db.Path", side_effect=path_side_effect):
            mock_manager_instance = Mock()
            mock_manager.return_value = mock_manager_instance

            result = self.runner.invoke(
                app,
                [
                    "db",
                    "backup",
                    "--source",
                    "source.db",
                    "--backup-path",
                    "custom_backup.db",
                ],
            )

            assert result.exit_code == 0
            assert "Backup: custom_backup.db" in result.stdout

    @patch("ml_agents.core.database_manager.DatabaseManager")
    @patch("ml_agents.core.database_manager.DatabaseConfig")
    @patch("pathlib.Path")
    def test_db_backup_default_source(self, mock_path, mock_config, mock_manager):
        """Test backup with default source path."""
        mock_path_instance = Mock()
        mock_path_instance.exists.return_value = True
        mock_path_instance.stat.return_value.st_size = 1024
        mock_path.return_value = mock_path_instance

        mock_manager_instance = Mock()
        mock_manager.return_value = mock_manager_instance

        result = self.runner.invoke(app, ["db", "backup"])

        assert result.exit_code == 0
        # Should use default source path
        mock_config.assert_called_with(db_path="./ml_agents_results.db")


class TestDbStatsCommand:
    """Test the db stats command."""

    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()

    def test_db_stats_command_help(self):
        """Test db stats command help display."""
        result = self.runner.invoke(app, ["db", "stats", "--help"])
        assert result.exit_code == 0
        assert "statistics" in result.stdout.lower()

    @patch("ml_agents.core.database_manager.DatabaseManager")
    @patch("ml_agents.core.database_manager.DatabaseConfig")
    @patch("pathlib.Path")
    def test_db_stats_display(self, mock_path, mock_config, mock_manager):
        """Test database statistics display."""
        mock_path_instance = Mock()
        mock_path_instance.exists.return_value = True
        mock_path.return_value = mock_path_instance

        mock_manager_instance = Mock()
        mock_manager_instance.get_database_stats.return_value = {
            "schema_version": "1.2.0",
            "experiments_count": 10,
            "runs_count": 150,
            "parsing_metrics_count": 75,
            "database_size_bytes": 5120,
        }
        mock_manager_instance.validate_integrity.return_value = True
        mock_manager.return_value = mock_manager_instance

        result = self.runner.invoke(app, ["db", "stats", "--db-path", "test.db"])

        assert result.exit_code == 0
        assert "Database Statistics" in result.stdout
        assert "Schema Version" in result.stdout
        assert "1.2.0" in result.stdout
        assert "Experiments" in result.stdout
        assert "10" in result.stdout
        assert "Runs" in result.stdout
        assert "150" in result.stdout
        assert "5,120 bytes" in result.stdout
        assert "Database integrity check passed" in result.stdout

    @patch("pathlib.Path")
    def test_db_stats_database_not_found(self, mock_path):
        """Test stats when database doesn't exist."""
        mock_path_instance = Mock()
        mock_path_instance.exists.return_value = False
        mock_path.return_value = mock_path_instance

        result = self.runner.invoke(app, ["db", "stats", "--db-path", "nonexistent.db"])

        assert result.exit_code == 1
        assert "Database not found" in result.stdout

    @patch("ml_agents.core.database_manager.DatabaseManager")
    @patch("ml_agents.core.database_manager.DatabaseConfig")
    @patch("pathlib.Path")
    def test_db_stats_integrity_failed(self, mock_path, mock_config, mock_manager):
        """Test stats with failed integrity check."""
        mock_path_instance = Mock()
        mock_path_instance.exists.return_value = True
        mock_path.return_value = mock_path_instance

        mock_manager_instance = Mock()
        mock_manager_instance.get_database_stats.return_value = {
            "schema_version": "1.2.0",
            "experiments_count": 5,
            "runs_count": 50,
            "parsing_metrics_count": 25,
            "database_size_bytes": 2048,
        }
        mock_manager_instance.validate_integrity.return_value = False
        mock_manager.return_value = mock_manager_instance

        result = self.runner.invoke(app, ["db", "stats"])

        assert result.exit_code == 0
        assert "Database integrity check failed" in result.stdout
        assert "consider running repair" in result.stdout


class TestDbMigrateCommand:
    """Test the db migrate command."""

    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()

    def test_db_migrate_command_help(self):
        """Test db migrate command help display."""
        result = self.runner.invoke(app, ["db", "migrate", "--help"])
        assert result.exit_code == 0
        assert "migrate" in result.stdout.lower()
        assert "schema" in result.stdout.lower()

    @patch("ml_agents.core.database_manager.DatabaseManager")
    @patch("ml_agents.core.database_manager.DatabaseConfig")
    @patch("pathlib.Path")
    def test_db_migrate_schema_update(self, mock_path, mock_config, mock_manager):
        """Test schema migration process."""
        mock_path_instance = Mock()
        mock_path_instance.exists.return_value = True
        mock_path.return_value = mock_path_instance

        # Mock database manager
        mock_manager_instance = Mock()
        mock_manager_instance.CURRENT_SCHEMA_VERSION = "1.3.0"
        mock_manager_instance.validate_integrity.return_value = True

        # Mock connection and cursor for version check
        mock_cursor = Mock()
        mock_cursor.fetchone.return_value = ("1.2.0",)
        mock_conn = Mock()
        mock_conn.cursor.return_value = mock_cursor
        mock_conn.__enter__ = Mock(return_value=mock_conn)
        mock_conn.__exit__ = Mock(return_value=None)
        mock_manager_instance.get_connection.return_value = mock_conn

        mock_manager.return_value = mock_manager_instance

        with patch("typer.confirm", return_value=True):
            result = self.runner.invoke(app, ["db", "migrate", "--db-path", "test.db"])

            assert result.exit_code == 0
            assert "Current schema version: 1.2.0" in result.stdout
            assert "Target schema version: 1.3.0" in result.stdout
            assert "Database migration completed successfully" in result.stdout

    @patch("ml_agents.core.database_manager.DatabaseManager")
    @patch("ml_agents.core.database_manager.DatabaseConfig")
    @patch("pathlib.Path")
    def test_db_migrate_already_up_to_date(self, mock_path, mock_config, mock_manager):
        """Test migration when schema is already up to date."""
        mock_path_instance = Mock()
        mock_path_instance.exists.return_value = True
        mock_path.return_value = mock_path_instance

        mock_manager_instance = Mock()
        mock_manager_instance.CURRENT_SCHEMA_VERSION = "1.2.0"

        mock_cursor = Mock()
        mock_cursor.fetchone.return_value = ("1.2.0",)
        mock_conn = Mock()
        mock_conn.cursor.return_value = mock_cursor
        mock_conn.__enter__ = Mock(return_value=mock_conn)
        mock_conn.__exit__ = Mock(return_value=None)
        mock_manager_instance.get_connection.return_value = mock_conn

        mock_manager.return_value = mock_manager_instance

        result = self.runner.invoke(app, ["db", "migrate", "--db-path", "test.db"])

        assert result.exit_code == 0
        assert "Database schema is already up to date" in result.stdout

    @patch("pathlib.Path")
    def test_db_migrate_database_not_found(self, mock_path):
        """Test migration when database doesn't exist."""
        mock_path_instance = Mock()
        mock_path_instance.exists.return_value = False
        mock_path.return_value = mock_path_instance

        result = self.runner.invoke(
            app, ["db", "migrate", "--db-path", "nonexistent.db"]
        )

        assert result.exit_code == 1
        assert "Database not found" in result.stdout

    @patch("ml_agents.core.database_manager.DatabaseManager")
    @patch("ml_agents.core.database_manager.DatabaseConfig")
    @patch("pathlib.Path")
    def test_db_migrate_user_cancellation(self, mock_path, mock_config, mock_manager):
        """Test migration cancellation by user."""
        mock_path_instance = Mock()
        mock_path_instance.exists.return_value = True
        mock_path.return_value = mock_path_instance

        mock_manager_instance = Mock()
        mock_manager_instance.CURRENT_SCHEMA_VERSION = "1.3.0"

        mock_cursor = Mock()
        mock_cursor.fetchone.return_value = ("1.2.0",)
        mock_conn = Mock()
        mock_conn.cursor.return_value = mock_cursor
        mock_conn.__enter__ = Mock(return_value=mock_conn)
        mock_conn.__exit__ = Mock(return_value=None)
        mock_manager_instance.get_connection.return_value = mock_conn

        mock_manager.return_value = mock_manager_instance

        with patch("typer.confirm", return_value=False):
            result = self.runner.invoke(app, ["db", "migrate", "--db-path", "test.db"])

            assert result.exit_code == 0
            assert "Migration cancelled" in result.stdout


class TestDbCommandsIntegration:
    """Integration tests for database commands."""

    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()

    def test_db_group_exists(self):
        """Test that database group exists and is accessible."""
        result = self.runner.invoke(app, ["db", "--help"])
        assert result.exit_code == 0
        assert "Database management" in result.stdout
        assert "init" in result.stdout
        assert "backup" in result.stdout
        assert "stats" in result.stdout
        assert "migrate" in result.stdout

    def test_all_db_commands_accessible(self):
        """Test that all database commands are accessible."""
        commands = ["init", "backup", "stats", "migrate"]

        for command in commands:
            result = self.runner.invoke(app, ["db", command, "--help"])
            assert result.exit_code == 0, f"Command 'db {command}' not accessible"

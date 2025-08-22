"""Tests for database_manager module."""

import json
import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from ml_agents.core.database_manager import DatabaseConfig, DatabaseManager


class TestDatabaseConfig:
    """Test DatabaseConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = DatabaseConfig()

        assert config.db_path == "./ml_agents_results.db"
        assert config.schema_version == "1.0.0"
        assert config.backup_frequency == 100
        assert config.auto_vacuum is True
        assert config.connection_timeout == 30
        assert config.enable_wal is True

    def test_custom_config(self):
        """Test custom configuration values."""
        config = DatabaseConfig(
            db_path="/custom/path.db",
            backup_frequency=50,
            auto_vacuum=False,
            connection_timeout=60,
        )

        assert config.db_path == "/custom/path.db"
        assert config.backup_frequency == 50
        assert config.auto_vacuum is False
        assert config.connection_timeout == 60


class TestDatabaseManager:
    """Test DatabaseManager class."""

    @pytest.fixture
    def temp_db(self):
        """Create a temporary database for testing."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
            db_path = tmp.name

        config = DatabaseConfig(db_path=db_path)
        manager = DatabaseManager(config)

        yield manager, db_path

        # Cleanup
        Path(db_path).unlink(missing_ok=True)
        # Also cleanup WAL files
        Path(f"{db_path}-wal").unlink(missing_ok=True)
        Path(f"{db_path}-shm").unlink(missing_ok=True)

    def test_database_initialization(self, temp_db):
        """Test database initialization creates required tables."""
        manager, db_path = temp_db

        # Check that database file exists
        assert Path(db_path).exists()

        # Check that required tables exist
        with manager.get_connection() as conn:
            cursor = conn.cursor()

            # Check for main tables
            cursor.execute(
                """
                SELECT name FROM sqlite_master
                WHERE type='table' AND name IN ('experiments', 'runs', 'parsing_metrics', 'schema_version')
            """
            )
            tables = {row[0] for row in cursor.fetchall()}

            assert "experiments" in tables
            assert "runs" in tables
            assert "parsing_metrics" in tables
            assert "schema_version" in tables

            # Check for indexes
            cursor.execute(
                """
                SELECT name FROM sqlite_master
                WHERE type='index' AND name LIKE 'idx_%'
            """
            )
            indexes = {row[0] for row in cursor.fetchall()}

            expected_indexes = {
                "idx_runs_experiment_approach",
                "idx_runs_created_at",
                "idx_runs_accuracy",
                "idx_parsing_confidence",
                "idx_experiments_status",
            }

            assert expected_indexes.issubset(indexes)

    def test_schema_version_tracking(self, temp_db):
        """Test schema version is properly tracked."""
        manager, db_path = temp_db

        with manager.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT version FROM schema_version ORDER BY applied_at DESC LIMIT 1"
            )
            version = cursor.fetchone()[0]

            assert version == DatabaseManager.CURRENT_SCHEMA_VERSION

    def test_foreign_keys_enabled(self, temp_db):
        """Test that foreign key constraints are enabled."""
        manager, db_path = temp_db

        with manager.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("PRAGMA foreign_keys")
            fk_enabled = cursor.fetchone()[0]

            assert fk_enabled == 1

    def test_get_connection_context_manager(self, temp_db):
        """Test database connection context manager."""
        manager, db_path = temp_db

        # Test successful connection
        with manager.get_connection() as conn:
            assert isinstance(conn, sqlite3.Connection)
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            result = cursor.fetchone()[0]
            assert result == 1

        # Connection should be closed after context
        with pytest.raises(sqlite3.ProgrammingError):
            cursor.execute("SELECT 1")

    def test_backup_database(self, temp_db):
        """Test database backup functionality."""
        manager, db_path = temp_db

        # Insert some test data
        with manager.get_connection() as conn:
            conn.execute(
                """
                INSERT INTO experiments (id, name, config_json)
                VALUES ('test_exp', 'Test Experiment', '{}')
            """
            )
            conn.commit()

        # Create backup
        backup_path = f"{db_path}_backup"
        manager.backup_database(backup_path)

        # Verify backup exists
        assert Path(backup_path).exists()

        # Verify backup contains data
        backup_config = DatabaseConfig(db_path=backup_path)
        backup_manager = DatabaseManager(backup_config)

        with backup_manager.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM experiments")
            count = cursor.fetchone()[0]
            assert count == 1

        # Cleanup
        Path(backup_path).unlink(missing_ok=True)

    def test_backup_nonexistent_database(self, tmp_path):
        """Test backup of non-existent database."""
        nonexistent_db = tmp_path / "nonexistent.db"
        backup_path = tmp_path / "backup.db"

        config = DatabaseConfig(db_path=str(nonexistent_db))
        manager = DatabaseManager(config)

        # This should not raise an error, just log a warning
        manager.backup_database(str(backup_path))

    def test_validate_integrity_success(self, temp_db):
        """Test database integrity validation with valid database."""
        manager, db_path = temp_db

        # Insert valid test data
        with manager.get_connection() as conn:
            conn.execute(
                """
                INSERT INTO experiments (id, name, config_json)
                VALUES ('test_exp', 'Test Experiment', '{}')
            """
            )
            conn.execute(
                """
                INSERT INTO runs (id, experiment_id, approach_name, provider, model,
                                sample_index, input_text, is_correct)
                VALUES ('run_1', 'test_exp', 'CoT', 'openai', 'gpt-4', 0, 'test input', 1)
            """
            )
            conn.commit()

        assert manager.validate_integrity() is True

    def test_validate_integrity_foreign_key_violation(self, temp_db):
        """Test database integrity validation with foreign key violations."""
        manager, db_path = temp_db

        # Insert data that violates foreign key constraints
        with manager.get_connection() as conn:
            # Disable foreign keys temporarily to insert invalid data
            conn.execute("PRAGMA foreign_keys = OFF")
            conn.execute(
                """
                INSERT INTO runs (id, experiment_id, approach_name, provider, model,
                                sample_index, input_text, is_correct)
                VALUES ('run_1', 'nonexistent_exp', 'CoT', 'openai', 'gpt-4', 0, 'test input', 1)
            """
            )
            conn.commit()
            conn.execute("PRAGMA foreign_keys = ON")

        assert manager.validate_integrity() is False

    def test_get_database_stats(self, temp_db):
        """Test database statistics collection."""
        manager, db_path = temp_db

        # Insert test data
        with manager.get_connection() as conn:
            conn.execute(
                """
                INSERT INTO experiments (id, name, config_json)
                VALUES ('test_exp', 'Test Experiment', '{}')
            """
            )
            conn.execute(
                """
                INSERT INTO runs (id, experiment_id, approach_name, provider, model,
                                sample_index, input_text, is_correct)
                VALUES ('run_1', 'test_exp', 'CoT', 'openai', 'gpt-4', 0, 'test input', 1)
            """
            )
            conn.execute(
                """
                INSERT INTO parsing_metrics (run_id, parsing_attempts, confidence_score)
                VALUES ('run_1', 1, 0.9)
            """
            )
            conn.commit()

        stats = manager.get_database_stats()

        assert "experiments_count" in stats
        assert "runs_count" in stats
        assert "parsing_metrics_count" in stats
        assert "database_size_bytes" in stats
        assert "schema_version" in stats

        assert stats["experiments_count"] == 1
        assert stats["runs_count"] == 1
        assert stats["parsing_metrics_count"] == 1
        assert stats["database_size_bytes"] > 0
        assert stats["schema_version"] == DatabaseManager.CURRENT_SCHEMA_VERSION

    def test_vacuum(self, temp_db):
        """Test database vacuum operation."""
        manager, db_path = temp_db

        # Get initial size
        initial_stats = manager.get_database_stats()
        initial_size = initial_stats["database_size_bytes"]

        # Insert and delete data to create fragmentation
        with manager.get_connection() as conn:
            for i in range(100):
                conn.execute(
                    """
                    INSERT INTO experiments (id, name, config_json)
                    VALUES (?, ?, '{}')
                """,
                    (f"exp_{i}", f"Experiment {i}"),
                )
            conn.commit()

            # Delete half the data
            conn.execute("DELETE FROM experiments WHERE id LIKE 'exp_5%'")
            conn.commit()

        # Vacuum should not raise an error
        manager.vacuum()

        # Verify database is still functional
        stats = manager.get_database_stats()
        assert stats["experiments_count"] > 0

    def test_check_auto_backup(self, temp_db):
        """Test automatic backup functionality."""
        manager, db_path = temp_db

        # Set backup frequency to 2 for testing
        manager.config.backup_frequency = 2
        manager._run_count = 0

        # First call should not trigger backup
        with patch.object(manager, "backup_database") as mock_backup:
            manager.check_auto_backup()
            assert manager._run_count == 1
            mock_backup.assert_not_called()

        # Second call should trigger backup
        with patch.object(manager, "backup_database") as mock_backup:
            manager.check_auto_backup()
            assert manager._run_count == 0  # Reset after backup
            mock_backup.assert_called_once()

    def test_export_schema(self, temp_db):
        """Test schema export functionality."""
        manager, db_path = temp_db

        schema_sql = manager.export_schema()

        assert "CREATE TABLE" in schema_sql
        assert "experiments" in schema_sql
        assert "runs" in schema_sql
        assert "parsing_metrics" in schema_sql
        assert "CREATE INDEX" in schema_sql

    def test_import_csv_data(self, temp_db):
        """Test CSV data import functionality."""
        manager, db_path = temp_db

        # Create test CSV content
        csv_content = """approach,provider,model,input,expected,output,is_correct,execution_time_ms,cost
CoT,openai,gpt-4,"2+2=?",4,"The answer is 4",true,1500,0.001
PoT,openai,gpt-4,"3*3=?",9,"The answer is 9",true,1200,0.001"""

        # Create temporary CSV file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as tmp:
            tmp.write(csv_content)
            csv_path = tmp.name

        try:
            # First create an experiment
            experiment_id = "test_import"
            with manager.get_connection() as conn:
                conn.execute(
                    """
                    INSERT INTO experiments (id, name, config_json)
                    VALUES (?, 'Import Test', '{}')
                """,
                    (experiment_id,),
                )
                conn.commit()

            # Import CSV data
            rows_imported = manager.import_csv_data(csv_path, experiment_id)

            assert rows_imported == 2

            # Verify data was imported
            with manager.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT COUNT(*) FROM runs WHERE experiment_id = ?",
                    (experiment_id,),
                )
                count = cursor.fetchone()[0]
                assert count == 2

                # Check specific data
                cursor.execute(
                    """
                    SELECT approach_name, provider, model, is_correct
                    FROM runs WHERE experiment_id = ? ORDER BY sample_index
                """,
                    (experiment_id,),
                )

                rows = cursor.fetchall()
                assert len(rows) == 2
                assert tuple(rows[0]) == ("CoT", "openai", "gpt-4", True)
                assert tuple(rows[1]) == ("PoT", "openai", "gpt-4", True)

        finally:
            # Cleanup
            Path(csv_path).unlink(missing_ok=True)

    def test_import_csv_invalid_file(self, temp_db):
        """Test CSV import with invalid file."""
        manager, db_path = temp_db

        with pytest.raises(Exception):
            manager.import_csv_data("/nonexistent/file.csv", "test_exp")

    def test_migration_support(self, temp_db):
        """Test schema migration functionality."""
        manager, db_path = temp_db

        # Simulate different schema version
        with manager.get_connection() as conn:
            conn.execute("DELETE FROM schema_version")
            conn.execute("INSERT INTO schema_version (version) VALUES ('0.9.0')")
            conn.commit()

        # Test migration (should handle gracefully)
        with patch.object(manager, "_migrate_schema") as mock_migrate:
            with manager.get_connection() as conn:
                manager._check_schema_version(conn)
                mock_migrate.assert_called_once_with(
                    conn, "0.9.0", DatabaseManager.CURRENT_SCHEMA_VERSION
                )

    def test_concurrent_access(self, temp_db):
        """Test concurrent database access."""
        manager, db_path = temp_db

        # Test multiple connections work properly
        for i in range(5):
            with manager.get_connection() as conn:
                # Each connection should be able to execute queries
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM experiments")
                result = cursor.fetchone()[0]
                assert isinstance(result, int)

    def test_error_handling(self):
        """Test error handling for various scenarios."""
        # Test with invalid database path
        config = DatabaseConfig(db_path="/invalid/path/db.sqlite")

        with pytest.raises(Exception):
            DatabaseManager(config)

    def test_database_constraints(self, temp_db):
        """Test database constraints are properly enforced."""
        manager, db_path = temp_db

        with manager.get_connection() as conn:
            # Test status constraint on experiments table
            with pytest.raises(sqlite3.IntegrityError):
                conn.execute(
                    """
                    INSERT INTO experiments (id, name, config_json, status)
                    VALUES ('test', 'Test', '{}', 'invalid_status')
                """
                )

            # Test parsing method constraint on runs table
            with pytest.raises(sqlite3.IntegrityError):
                conn.execute(
                    """
                    INSERT INTO experiments (id, name, config_json)
                    VALUES ('test_exp', 'Test', '{}')
                """
                )
                conn.execute(
                    """
                    INSERT INTO runs (id, experiment_id, approach_name, provider, model,
                                    sample_index, input_text, parsing_method)
                    VALUES ('run_1', 'test_exp', 'CoT', 'openai', 'gpt-4', 0, 'test', 'invalid_method')
                """
                )

            # Test confidence score constraint
            with pytest.raises(sqlite3.IntegrityError):
                conn.execute(
                    """
                    INSERT INTO parsing_metrics (run_id, confidence_score)
                    VALUES ('run_1', 1.5)  -- > 1.0 should fail
                """
                )


if __name__ == "__main__":
    pytest.main([__file__])

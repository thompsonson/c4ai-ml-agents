"""Database management for ML Agents experiment results.

This module provides SQLite database functionality for persisting experiment
results, with support for schema initialization, migration, backup, and
data integrity validation.
"""

import json
import logging
import shutil
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class DatabaseConfig:
    """Database configuration settings."""

    db_path: str = "./ml_agents_results.db"
    schema_version: str = "1.0.0"
    backup_frequency: int = 100  # Backup every N runs
    auto_vacuum: bool = True
    connection_timeout: int = 30
    enable_wal: bool = True  # Write-Ahead Logging for better concurrency


class DatabaseManager:
    """Manages SQLite database for ML Agents experiment results."""

    # Schema version for migration tracking
    CURRENT_SCHEMA_VERSION = "1.2.0"

    # SQL for creating tables
    SCHEMA_SQL = """
    -- Experiments: High-level experiment metadata
    CREATE TABLE IF NOT EXISTS experiments (
        id TEXT PRIMARY KEY,
        name TEXT,
        description TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        config_json TEXT NOT NULL,
        status TEXT DEFAULT 'running' CHECK(status IN ('running', 'completed', 'failed')),
        preprocessing_id TEXT,
        preprocessing_rules_path TEXT,
        dataset_name TEXT,
        eval_output_path TEXT
    );

    -- Runs: Individual reasoning approach executions
    CREATE TABLE IF NOT EXISTS runs (
        id TEXT PRIMARY KEY,
        experiment_id TEXT NOT NULL,
        approach_name TEXT NOT NULL,
        provider TEXT NOT NULL,
        model TEXT NOT NULL,
        sample_index INTEGER NOT NULL,
        input_text TEXT NOT NULL,
        expected_answer TEXT,
        raw_output TEXT,
        parsed_answer TEXT,
        parsing_method TEXT CHECK(parsing_method IN ('instructor', 'instructor_multiple', 'regex', 'manual', 'none')),
        parsing_confidence REAL CHECK(parsing_confidence >= 0 AND parsing_confidence <= 1),
        is_correct BOOLEAN,
        execution_time_ms INTEGER,
        cost_estimate REAL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        metadata_json TEXT,
        FOREIGN KEY (experiment_id) REFERENCES experiments(id) ON DELETE CASCADE
    );

    -- Parsing_metrics: Track parsing performance
    CREATE TABLE IF NOT EXISTS parsing_metrics (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        run_id TEXT NOT NULL,
        parsing_attempts INTEGER DEFAULT 0,
        fallback_used BOOLEAN DEFAULT 0,
        confidence_score REAL CHECK(confidence_score >= 0 AND confidence_score <= 1),
        extraction_time_ms INTEGER,
        error_details TEXT,
        FOREIGN KEY (run_id) REFERENCES runs(id) ON DELETE CASCADE
    );

    -- Dataset_preprocessing: Track dataset preprocessing metadata
    CREATE TABLE IF NOT EXISTS dataset_preprocessing (
        id TEXT PRIMARY KEY,
        dataset_name TEXT NOT NULL,
        dataset_config TEXT,  -- HuggingFace config name if applicable
        dataset_url TEXT,
        status TEXT DEFAULT 'pending' CHECK(status IN ('pending', 'processed', 'failed')),
        schema_analysis TEXT,  -- JSON with detected patterns and field mapping
        transformation_rules TEXT,  -- JSON with transformation configuration
        confidence_score REAL CHECK(confidence_score >= 0 AND confidence_score <= 1),
        original_samples INTEGER,
        processed_samples INTEGER,
        validation_results TEXT,  -- JSON with validation metrics
        output_path TEXT,
        rules_path TEXT,
        analysis_path TEXT,
        processed_at TIMESTAMP,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(dataset_name, dataset_config)
    );

    -- Preprocessing-Evaluation linkage table
    CREATE TABLE IF NOT EXISTS preprocessing_eval_link (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        experiment_id TEXT NOT NULL,
        preprocessing_id TEXT NOT NULL,
        dataset_name TEXT NOT NULL,
        dataset_config TEXT,
        preprocessing_path TEXT,
        rules_path TEXT,
        processed_data_path TEXT,
        eval_output_path TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY(experiment_id) REFERENCES experiments(id) ON DELETE CASCADE,
        FOREIGN KEY(preprocessing_id) REFERENCES dataset_preprocessing(id)
    );

    -- Schema version tracking
    CREATE TABLE IF NOT EXISTS schema_version (
        version TEXT PRIMARY KEY,
        applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );

    -- Performance indexes
    CREATE INDEX IF NOT EXISTS idx_runs_experiment_approach ON runs(experiment_id, approach_name);
    CREATE INDEX IF NOT EXISTS idx_runs_created_at ON runs(created_at);
    CREATE INDEX IF NOT EXISTS idx_runs_accuracy ON runs(is_correct);
    CREATE INDEX IF NOT EXISTS idx_parsing_confidence ON parsing_metrics(confidence_score);
    CREATE INDEX IF NOT EXISTS idx_experiments_status ON experiments(status);
    CREATE INDEX IF NOT EXISTS idx_preprocessing_status ON dataset_preprocessing(status);
    CREATE INDEX IF NOT EXISTS idx_preprocessing_confidence ON dataset_preprocessing(confidence_score);
    CREATE INDEX IF NOT EXISTS idx_preprocessing_dataset ON dataset_preprocessing(dataset_name, dataset_config);
    CREATE INDEX IF NOT EXISTS idx_prep_eval_link_experiment ON preprocessing_eval_link(experiment_id);
    CREATE INDEX IF NOT EXISTS idx_prep_eval_link_preprocessing ON preprocessing_eval_link(preprocessing_id);
    CREATE INDEX IF NOT EXISTS idx_prep_eval_link_dataset ON preprocessing_eval_link(dataset_name);
    """

    def __init__(self, config: DatabaseConfig):
        """Initialize database manager with configuration.

        Args:
            config: Database configuration settings
        """
        self.config = config
        self.db_path = Path(config.db_path)
        self._ensure_database_directory()
        self._init_database()
        self._run_count = 0  # Track runs for auto-backup

    def _ensure_database_directory(self) -> None:
        """Ensure the database directory exists."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    def _init_database(self) -> None:
        """Initialize database with schema."""
        with self.get_connection() as conn:
            # Enable foreign keys
            conn.execute("PRAGMA foreign_keys = ON")

            # Enable WAL mode for better concurrency
            if self.config.enable_wal:
                conn.execute("PRAGMA journal_mode = WAL")

            # Set auto-vacuum if enabled
            if self.config.auto_vacuum:
                conn.execute("PRAGMA auto_vacuum = FULL")

            # Create schema
            conn.executescript(self.SCHEMA_SQL)

            # Check and update schema version
            self._check_schema_version(conn)

            # Run any necessary migrations
            self._run_migrations(conn)

            conn.commit()

    def _check_schema_version(self, conn: sqlite3.Connection) -> None:
        """Check and update schema version.

        Args:
            conn: Database connection
        """
        cursor = conn.cursor()
        cursor.execute(
            "SELECT version FROM schema_version ORDER BY applied_at DESC LIMIT 1"
        )
        result = cursor.fetchone()

        current_version = result[0] if result else None

        if current_version != self.CURRENT_SCHEMA_VERSION:
            # Apply migration if needed
            if current_version:
                self._migrate_schema(conn, current_version, self.CURRENT_SCHEMA_VERSION)

            # Update version
            conn.execute(
                "INSERT INTO schema_version (version) VALUES (?)",
                (self.CURRENT_SCHEMA_VERSION,),
            )

    def _migrate_schema(
        self, conn: sqlite3.Connection, from_version: str, to_version: str
    ) -> None:
        """Migrate database schema between versions.

        Args:
            conn: Database connection
            from_version: Current schema version
            to_version: Target schema version
        """
        logger.info(f"Migrating database schema from {from_version} to {to_version}")

        # Create backup before migration
        backup_path = self.db_path.with_suffix(f".backup_{from_version}")
        self.backup_database(str(backup_path))

        # Version-specific migrations
        if from_version == "1.0.0" and to_version == "1.1.0":
            self._migrate_1_0_0_to_1_1_0(conn)
        # Add more migrations here as needed

        logger.info("Schema migration completed")

    def _run_migrations(self, conn: sqlite3.Connection) -> None:
        """Run any necessary database migrations.

        Args:
            conn: Database connection
        """
        # Check if we need to add new columns to existing tables
        self._add_missing_columns(conn)

    def _add_missing_columns(self, conn: sqlite3.Connection) -> None:
        """Add any missing columns to existing tables.

        Args:
            conn: Database connection
        """
        try:
            # Check experiments table for new columns
            cursor = conn.execute("PRAGMA table_info(experiments)")
            columns = {row[1] for row in cursor.fetchall()}

            # Add missing columns to experiments table
            if "preprocessing_id" not in columns:
                conn.execute("ALTER TABLE experiments ADD COLUMN preprocessing_id TEXT")
                logger.info("Added preprocessing_id column to experiments table")

            if "preprocessing_rules_path" not in columns:
                conn.execute(
                    "ALTER TABLE experiments ADD COLUMN preprocessing_rules_path TEXT"
                )
                logger.info(
                    "Added preprocessing_rules_path column to experiments table"
                )

            if "dataset_name" not in columns:
                conn.execute("ALTER TABLE experiments ADD COLUMN dataset_name TEXT")
                logger.info("Added dataset_name column to experiments table")

            if "eval_output_path" not in columns:
                conn.execute("ALTER TABLE experiments ADD COLUMN eval_output_path TEXT")
                logger.info("Added eval_output_path column to experiments table")

            # Check dataset_preprocessing table
            cursor = conn.execute("PRAGMA table_info(dataset_preprocessing)")
            columns = {row[1] for row in cursor.fetchall()}

            if "dataset_config" not in columns:
                conn.execute(
                    "ALTER TABLE dataset_preprocessing ADD COLUMN dataset_config TEXT"
                )
                logger.info(
                    "Added dataset_config column to dataset_preprocessing table"
                )

            if "rules_path" not in columns:
                conn.execute(
                    "ALTER TABLE dataset_preprocessing ADD COLUMN rules_path TEXT"
                )
                logger.info("Added rules_path column to dataset_preprocessing table")

            if "analysis_path" not in columns:
                conn.execute(
                    "ALTER TABLE dataset_preprocessing ADD COLUMN analysis_path TEXT"
                )
                logger.info("Added analysis_path column to dataset_preprocessing table")

            # Create preprocessing_eval_link table if it doesn't exist
            cursor = conn.execute(
                """
                SELECT name FROM sqlite_master
                WHERE type='table' AND name='preprocessing_eval_link'
            """
            )
            if not cursor.fetchone():
                conn.execute(
                    """
                    CREATE TABLE preprocessing_eval_link (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        experiment_id TEXT NOT NULL,
                        preprocessing_id TEXT NOT NULL,
                        dataset_name TEXT NOT NULL,
                        dataset_config TEXT,
                        preprocessing_path TEXT,
                        rules_path TEXT,
                        processed_data_path TEXT,
                        eval_output_path TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY(experiment_id) REFERENCES experiments(id) ON DELETE CASCADE,
                        FOREIGN KEY(preprocessing_id) REFERENCES dataset_preprocessing(id)
                    )
                """
                )
                logger.info("Created preprocessing_eval_link table")

                # Add indexes
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_prep_eval_link_experiment ON preprocessing_eval_link(experiment_id)"
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_prep_eval_link_preprocessing ON preprocessing_eval_link(preprocessing_id)"
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_prep_eval_link_dataset ON preprocessing_eval_link(dataset_name)"
                )

        except sqlite3.Error as e:
            logger.warning(f"Error adding missing columns: {e}")
            # Continue without failing - schema might already be up to date

    def _migrate_1_0_0_to_1_1_0(self, conn: sqlite3.Connection) -> None:
        """Migrate from schema version 1.0.0 to 1.1.0.

        Changes:
        - Add 'instructor_multiple' to parsing_method CHECK constraint
        """
        logger.info(
            "Applying migration 1.0.0 -> 1.1.0: Adding instructor_multiple parsing method"
        )

        try:
            # Check if the runs table already has the instructor_multiple constraint
            cursor = conn.execute(
                "SELECT sql FROM sqlite_master WHERE type='table' AND name='runs'"
            )
            table_schema = cursor.fetchone()

            if table_schema and "instructor_multiple" in table_schema[0]:
                logger.info(
                    "Migration 1.0.0 -> 1.1.0: instructor_multiple already exists in schema, skipping migration"
                )
                return

            # SQLite doesn't support modifying CHECK constraints directly
            # We need to recreate the table with the new constraint

            # Step 1: Create new table with updated constraint
            conn.execute(
                """
                CREATE TABLE runs_new (
                    id TEXT PRIMARY KEY,
                    experiment_id TEXT NOT NULL,
                    approach_name TEXT NOT NULL,
                    provider TEXT NOT NULL,
                    model TEXT NOT NULL,
                    sample_index INTEGER NOT NULL,
                    input_text TEXT NOT NULL,
                    expected_answer TEXT,
                    raw_output TEXT,
                    parsed_answer TEXT,
                    parsing_method TEXT CHECK(parsing_method IN ('instructor', 'instructor_multiple', 'regex', 'manual', 'none')),
                    parsing_confidence REAL CHECK(parsing_confidence >= 0 AND parsing_confidence <= 1),
                    is_correct BOOLEAN,
                    execution_time_ms INTEGER,
                    cost_estimate REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    metadata_json TEXT,
                    FOREIGN KEY (experiment_id) REFERENCES experiments(id) ON DELETE CASCADE
                )
            """
            )

            # Step 2: Copy data from old table to new table
            conn.execute(
                """
                INSERT INTO runs_new
                SELECT * FROM runs
            """
            )

            # Step 3: Drop old table
            conn.execute("DROP TABLE runs")

            # Step 4: Rename new table
            conn.execute("ALTER TABLE runs_new RENAME TO runs")

            # Step 5: Recreate indexes if any existed
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_runs_experiment_approach
                ON runs(experiment_id, approach_name)
            """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_runs_created_at
                ON runs(created_at)
            """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_runs_accuracy
                ON runs(is_correct)
            """
            )

            conn.commit()
            logger.info(
                "Successfully migrated runs table to support instructor_multiple parsing method"
            )

        except Exception as e:
            logger.error(f"Migration 1.0.0 -> 1.1.0 failed: {e}")
            conn.rollback()
            raise

    @contextmanager
    def get_connection(self) -> sqlite3.Connection:
        """Get a database connection with proper settings.

        Yields:
            Database connection
        """
        conn = sqlite3.connect(
            str(self.db_path),
            timeout=self.config.connection_timeout,
            isolation_level="DEFERRED",  # Better for concurrent access
        )
        conn.row_factory = sqlite3.Row  # Enable column access by name
        conn.execute(
            "PRAGMA foreign_keys = ON"
        )  # Enable foreign keys for this connection
        try:
            yield conn
        finally:
            conn.close()

    def backup_database(self, backup_path: str) -> None:
        """Create a backup of the database.

        Args:
            backup_path: Path for the backup file
        """
        if not self.db_path.exists():
            logger.warning("No database to backup")
            return

        logger.info(f"Creating database backup at {backup_path}")
        shutil.copy2(self.db_path, backup_path)

        # Also backup WAL file if it exists
        wal_path = self.db_path.with_suffix(".db-wal")
        if wal_path.exists():
            shutil.copy2(wal_path, Path(backup_path).with_suffix(".db-wal"))

    def validate_integrity(self) -> bool:
        """Check database integrity.

        Returns:
            True if database is valid, False otherwise
        """
        try:
            with self.get_connection() as conn:
                # Run integrity check
                cursor = conn.cursor()
                cursor.execute("PRAGMA integrity_check")
                result = cursor.fetchone()

                if result[0] != "ok":
                    logger.error(f"Database integrity check failed: {result[0]}")
                    return False

                # Check foreign key constraints
                cursor.execute("PRAGMA foreign_key_check")
                fk_errors = cursor.fetchall()

                if fk_errors:
                    logger.error(f"Foreign key constraint violations: {fk_errors}")
                    return False

                return True

        except Exception as e:
            logger.error(f"Error validating database integrity: {e}")
            return False

    def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics.

        Returns:
            Dictionary with database statistics
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()

            # Get table counts
            stats = {}
            for table in ["experiments", "runs", "parsing_metrics"]:
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                stats[f"{table}_count"] = cursor.fetchone()[0]

            # Get database size
            cursor.execute(
                "SELECT page_count * page_size as size FROM pragma_page_count(), pragma_page_size()"
            )
            stats["database_size_bytes"] = cursor.fetchone()[0]

            # Get schema version
            cursor.execute(
                "SELECT version FROM schema_version ORDER BY applied_at DESC LIMIT 1"
            )
            result = cursor.fetchone()
            stats["schema_version"] = result[0] if result else "unknown"

            return stats

    def vacuum(self) -> None:
        """Vacuum the database to reclaim space and optimize."""
        logger.info("Vacuuming database...")
        with self.get_connection() as conn:
            conn.execute("VACUUM")
        logger.info("Database vacuum completed")

    def check_auto_backup(self) -> None:
        """Check if auto-backup is needed based on run count."""
        self._run_count += 1
        if self._run_count >= self.config.backup_frequency:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = self.db_path.with_name(
                f"{self.db_path.stem}_auto_{timestamp}.db"
            )
            self.backup_database(str(backup_path))
            self._run_count = 0

    def export_schema(self) -> str:
        """Export the database schema as SQL.

        Returns:
            SQL schema definition
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT sql FROM sqlite_master WHERE type='table' OR type='index'"
            )
            schema_parts = [row[0] for row in cursor.fetchall() if row[0]]
            return ";\n\n".join(schema_parts) + ";"

    def import_csv_data(self, csv_path: str, experiment_id: str) -> int:
        """Import existing CSV results into the database.

        Args:
            csv_path: Path to CSV file with results
            experiment_id: ID of the experiment to associate with

        Returns:
            Number of rows imported
        """
        import pandas as pd

        logger.info(f"Importing CSV data from {csv_path}")

        try:
            df = pd.read_csv(csv_path)
            rows_imported = 0

            with self.get_connection() as conn:
                for _, row in df.iterrows():
                    # Map CSV columns to database fields
                    run_data = {
                        "id": f"{experiment_id}_{rows_imported}",
                        "experiment_id": experiment_id,
                        "approach_name": row.get("approach", "unknown"),
                        "provider": row.get("provider", "unknown"),
                        "model": row.get("model", "unknown"),
                        "sample_index": rows_imported,
                        "input_text": row.get("input", ""),
                        "expected_answer": row.get("expected", ""),
                        "raw_output": row.get("output", ""),
                        "parsed_answer": row.get("parsed_answer", ""),
                        "is_correct": row.get("is_correct", False),
                        "execution_time_ms": row.get("execution_time_ms", 0),
                        "cost_estimate": row.get("cost", 0.0),
                        "metadata_json": json.dumps(
                            {
                                "imported_from": csv_path,
                                "import_timestamp": datetime.now().isoformat(),
                            }
                        ),
                    }

                    # Insert run
                    conn.execute(
                        """
                        INSERT OR IGNORE INTO runs (
                            id, experiment_id, approach_name, provider, model,
                            sample_index, input_text, expected_answer, raw_output,
                            parsed_answer, is_correct, execution_time_ms, cost_estimate,
                            metadata_json
                        ) VALUES (
                            :id, :experiment_id, :approach_name, :provider, :model,
                            :sample_index, :input_text, :expected_answer, :raw_output,
                            :parsed_answer, :is_correct, :execution_time_ms, :cost_estimate,
                            :metadata_json
                        )
                    """,
                        run_data,
                    )

                    rows_imported += 1

                conn.commit()

            logger.info(f"Successfully imported {rows_imported} rows")
            return rows_imported

        except Exception as e:
            logger.error(f"Error importing CSV data: {e}")
            raise

    def link_preprocessing_to_experiment(
        self,
        experiment_id: str,
        preprocessing_id: str,
        dataset_name: str,
        dataset_config: Optional[str] = None,
        preprocessing_path: Optional[str] = None,
        rules_path: Optional[str] = None,
        processed_data_path: Optional[str] = None,
        eval_output_path: Optional[str] = None,
    ) -> None:
        """Link a preprocessing run to an evaluation experiment.

        Args:
            experiment_id: ID of the evaluation experiment
            preprocessing_id: ID of the preprocessing run
            dataset_name: Name of the dataset
            dataset_config: Optional HuggingFace config name
            preprocessing_path: Path to preprocessing output directory
            rules_path: Path to the rules file used
            processed_data_path: Path to the processed dataset
            eval_output_path: Path to evaluation output directory
        """
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute(
                """
                INSERT INTO preprocessing_eval_link (
                    experiment_id, preprocessing_id, dataset_name, dataset_config,
                    preprocessing_path, rules_path, processed_data_path, eval_output_path
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    experiment_id,
                    preprocessing_id,
                    dataset_name,
                    dataset_config,
                    preprocessing_path,
                    rules_path,
                    processed_data_path,
                    eval_output_path,
                ),
            )

            # Also update the experiment record with preprocessing info
            conn.execute(
                """
                UPDATE experiments
                SET preprocessing_id = ?, preprocessing_rules_path = ?,
                    dataset_name = ?, eval_output_path = ?
                WHERE id = ?
                """,
                (
                    preprocessing_id,
                    rules_path,
                    dataset_name,
                    eval_output_path,
                    experiment_id,
                ),
            )

            conn.commit()
            logger.debug(
                f"Linked preprocessing {preprocessing_id} to experiment {experiment_id}"
            )

    def get_experiment_preprocessing_info(
        self, experiment_id: str
    ) -> Optional[Dict[str, Any]]:
        """Get preprocessing information for an experiment.

        Args:
            experiment_id: ID of the experiment

        Returns:
            Dictionary with preprocessing details or None if not found
        """
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                """
                SELECT pel.*, dp.*
                FROM preprocessing_eval_link pel
                JOIN dataset_preprocessing dp ON pel.preprocessing_id = dp.id
                WHERE pel.experiment_id = ?
                """,
                (experiment_id,),
            )
            row = cursor.fetchone()

            if row:
                return dict(row)
            return None

    def get_preprocessing_usage_stats(self, preprocessing_id: str) -> Dict[str, Any]:
        """Get usage statistics for a preprocessing run.

        Args:
            preprocessing_id: ID of the preprocessing run

        Returns:
            Dictionary with usage statistics
        """
        with sqlite3.connect(str(self.db_path)) as conn:
            # Count experiments using this preprocessing
            cursor = conn.execute(
                """
                SELECT COUNT(*) as experiment_count
                FROM preprocessing_eval_link
                WHERE preprocessing_id = ?
                """,
                (preprocessing_id,),
            )
            experiment_count = cursor.fetchone()[0]

            # Get list of experiments
            cursor = conn.execute(
                """
                SELECT experiment_id, created_at
                FROM preprocessing_eval_link
                WHERE preprocessing_id = ?
                ORDER BY created_at DESC
                """,
                (preprocessing_id,),
            )
            experiments = [
                {"id": row[0], "created_at": row[1]} for row in cursor.fetchall()
            ]

            return {
                "preprocessing_id": preprocessing_id,
                "experiment_count": experiment_count,
                "experiments": experiments,
            }

    def get_latest_preprocessing_for_dataset(
        self, dataset_name: str, dataset_config: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Get the most recent preprocessing for a dataset.

        Args:
            dataset_name: Name of the dataset
            dataset_config: Optional HuggingFace config name

        Returns:
            Dictionary with preprocessing details or None if not found
        """
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.row_factory = sqlite3.Row

            if dataset_config:
                query = """
                    SELECT * FROM dataset_preprocessing
                    WHERE dataset_name = ? AND dataset_config = ?
                        AND status = 'processed'
                    ORDER BY processed_at DESC
                    LIMIT 1
                """
                params = (dataset_name, dataset_config)
            else:
                query = """
                    SELECT * FROM dataset_preprocessing
                    WHERE dataset_name = ? AND status = 'processed'
                    ORDER BY processed_at DESC
                    LIMIT 1
                """
                params = (dataset_name,)

            cursor = conn.execute(query, params)
            row = cursor.fetchone()

            if row:
                return dict(row)
            return None

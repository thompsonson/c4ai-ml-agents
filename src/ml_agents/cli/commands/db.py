"""Database management and maintenance commands."""

from datetime import datetime
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from ml_agents.cli.display import (
    display_error,
    display_info,
    display_success,
    display_warning,
)

console = Console()


def db_init(
    db_path: Optional[str] = typer.Option(
        None,
        "--db-path",
        help="Path to database file (default: ./ml_agents_results.db)",
    ),
    force: bool = typer.Option(
        False, "--force", help="Force initialization even if database exists"
    ),
) -> None:
    """Initialize the database for storing experiment results."""
    from ml_agents.core.database_manager import DatabaseConfig, DatabaseManager

    db_path = db_path or "./ml_agents_results.db"

    try:
        if Path(db_path).exists() and not force:
            display_warning(f"Database already exists at {db_path}")
            if not typer.confirm(
                "Do you want to continue? This will not modify existing data."
            ):
                display_info("Database initialization cancelled")
                return

        config = DatabaseConfig(db_path=db_path)
        db_manager = DatabaseManager(config)

        stats = db_manager.get_database_stats()

        display_success(f"Database initialized successfully at {db_path}")
        console.print(f"📊 Schema version: {stats['schema_version']}")
        console.print(f"📈 Total experiments: {stats['experiments_count']}")
        console.print(f"📈 Total runs: {stats['runs_count']}")
        console.print(f"💾 Database size: {stats['database_size_bytes']} bytes")

    except Exception as e:
        display_error(f"Failed to initialize database: {e}")
        raise typer.Exit(1)


def db_backup(
    source_db: Optional[str] = typer.Option(
        None, "--source", help="Source database path (default: ./ml_agents_results.db)"
    ),
    backup_path: Optional[str] = typer.Option(
        None, "--backup-path", help="Backup file path (default: auto-generated)"
    ),
) -> None:
    """Create a backup of the experiment database."""
    from ml_agents.core.database_manager import DatabaseConfig, DatabaseManager

    source_db = source_db or "./ml_agents_results.db"

    if not Path(source_db).exists():
        display_error(f"Database not found: {source_db}")
        raise typer.Exit(1)

    if not backup_path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = f"{source_db}_backup_{timestamp}.db"

    try:
        config = DatabaseConfig(db_path=source_db)
        db_manager = DatabaseManager(config)

        db_manager.backup_database(backup_path)

        display_success(f"Database backup created successfully")
        console.print(f"📁 Source: {source_db}")
        console.print(f"💾 Backup: {backup_path}")
        console.print(f"📊 Size: {Path(backup_path).stat().st_size} bytes")

    except Exception as e:
        display_error(f"Failed to create backup: {e}")
        raise typer.Exit(1)


def db_stats(
    db_path: Optional[str] = typer.Option(
        None, "--db-path", help="Database path (default: ./ml_agents_results.db)"
    )
) -> None:
    """Show database statistics and information."""
    from ml_agents.core.database_manager import DatabaseConfig, DatabaseManager

    db_path = db_path or "./ml_agents_results.db"

    if not Path(db_path).exists():
        display_error(f"Database not found: {db_path}")
        raise typer.Exit(1)

    try:
        config = DatabaseConfig(db_path=db_path)
        db_manager = DatabaseManager(config)

        stats = db_manager.get_database_stats()

        # Create statistics table
        table = Table(title=f"Database Statistics: {db_path}")
        table.add_column("Metric", style="cyan", no_wrap=True)
        table.add_column("Value", style="magenta")

        table.add_row("Schema Version", stats["schema_version"])
        table.add_row("Experiments", str(stats["experiments_count"]))
        table.add_row("Runs", str(stats["runs_count"]))
        table.add_row("Parsing Metrics", str(stats["parsing_metrics_count"]))
        table.add_row("Database Size", f"{stats['database_size_bytes']:,} bytes")

        console.print(table)

        # Check database integrity
        display_info("Checking database integrity...")
        if db_manager.validate_integrity():
            display_success("Database integrity check passed")
        else:
            display_warning("Database integrity check failed - consider running repair")

    except Exception as e:
        display_error(f"Failed to get database statistics: {e}")
        raise typer.Exit(1)


def db_migrate(
    db_path: Optional[str] = typer.Option(
        None, "--db-path", help="Database path (default: ./ml_agents_results.db)"
    ),
    backup_before: bool = typer.Option(
        True, "--backup/--no-backup", help="Create backup before migration"
    ),
) -> None:
    """Migrate database schema to the latest version."""
    from ml_agents.core.database_manager import DatabaseConfig, DatabaseManager

    db_path = db_path or "./ml_agents_results.db"

    if not Path(db_path).exists():
        display_error(f"Database not found: {db_path}")
        raise typer.Exit(1)

    try:
        config = DatabaseConfig(db_path=db_path)
        db_manager = DatabaseManager(config)

        # Check current schema version
        with db_manager.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT version FROM schema_version ORDER BY applied_at DESC LIMIT 1"
            )
            result = cursor.fetchone()
            current_version = result[0] if result else "unknown"

        display_info(f"Current schema version: {current_version}")
        display_info(f"Target schema version: {db_manager.CURRENT_SCHEMA_VERSION}")

        if current_version == db_manager.CURRENT_SCHEMA_VERSION:
            display_success("Database schema is already up to date!")
            return

        # Create backup if requested
        if backup_before:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = Path(db_path).with_name(
                f"{Path(db_path).stem}_migration_backup_{timestamp}.db"
            )
            display_info(f"Creating backup: {backup_path}")
            db_manager.backup_database(str(backup_path))

        # Confirm migration
        if not typer.confirm(
            f"Migrate database from {current_version} to {db_manager.CURRENT_SCHEMA_VERSION}?"
        ):
            display_info("Migration cancelled")
            return

        # Perform migration by triggering schema check
        display_info("Starting database migration...")

        # The migration happens automatically when we create a new DatabaseManager
        # because it checks and updates the schema version in __init__
        new_manager = DatabaseManager(config)

        display_success("Database migration completed successfully!")
        display_info(f"Schema updated to version: {new_manager.CURRENT_SCHEMA_VERSION}")

        # Validate integrity after migration
        if db_manager.validate_integrity():
            display_success("Database integrity check passed")
        else:
            display_warning("Database integrity check failed - please review migration")

    except Exception as e:
        display_error(f"Failed to migrate database: {e}")
        raise typer.Exit(1)

"""CLI commands for ML Agents experiments."""

import json
import sys
from pathlib import Path
from typing import List, Optional

import typer
from rich.console import Console
from rich.progress import track
from rich.table import Table

from src.cli.config_loader import load_and_validate_config
from src.cli.display import (
    create_accuracy_breakdown_table,
    create_experiment_table,
    display_error,
    display_experiment_complete,
    display_experiment_start,
    display_info,
    display_success,
    display_warning,
)
from src.cli.validators import (
    check_environment_ready,
    validate_checkpoint_file,
    validate_config_file,
    validate_max_tokens,
    validate_max_workers,
    validate_output_directory,
    validate_reasoning_approaches,
    validate_sample_count,
    validate_temperature,
    validate_top_p,
)
from src.core.experiment_runner import ExperimentRunner

console = Console()


def run_single_experiment(
    approach: str = typer.Option(
        "ChainOfThought", "--approach", "-a", help="Reasoning approach to use"
    ),
    samples: int = typer.Option(
        50,
        "--samples",
        "-n",
        help="Number of samples to process",
        callback=lambda x: validate_sample_count(x) if x else 50,
    ),
    config: Optional[str] = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to configuration file",
        callback=lambda x: validate_config_file(x) if x else None,
    ),
    # Model settings
    provider: Optional[str] = typer.Option(None, "--provider", help="Model provider"),
    model: Optional[str] = typer.Option(None, "--model", help="Model name"),
    temperature: Optional[float] = typer.Option(
        None,
        "--temperature",
        help="Sampling temperature (0.0-2.0)",
        callback=lambda x: validate_temperature(x) if x is not None else None,
    ),
    max_tokens: Optional[int] = typer.Option(
        None,
        "--max-tokens",
        help="Maximum tokens to generate",
        callback=lambda x: validate_max_tokens(x) if x else None,
    ),
    top_p: Optional[float] = typer.Option(
        None,
        "--top-p",
        help="Top-p sampling parameter (0.0-1.0)",
        callback=lambda x: validate_top_p(x) if x is not None else None,
    ),
    # Execution settings
    output_dir: Optional[str] = typer.Option(
        None,
        "--output",
        "-o",
        help="Output directory",
        callback=lambda x: validate_output_directory(x) if x else None,
    ),
    save_checkpoints: bool = typer.Option(
        True, "--checkpoints/--no-checkpoints", help="Save experiment checkpoints"
    ),
    # Advanced reasoning settings
    multi_step_reflection: bool = typer.Option(
        False, "--multi-step-reflection", help="Enable multi-step reflection"
    ),
    multi_step_verification: bool = typer.Option(
        False, "--multi-step-verification", help="Enable multi-step verification"
    ),
    max_reasoning_calls: int = typer.Option(
        5, "--max-reasoning-calls", help="Maximum reasoning API calls"
    ),
    # Output parsing settings
    use_structured_parsing: Optional[bool] = typer.Option(
        None,
        "--structured-parsing/--no-structured-parsing",
        help="Enable/disable structured output parsing with Instructor",
    ),
    fallback_to_regex: Optional[bool] = typer.Option(
        None,
        "--fallback-regex/--no-fallback-regex",
        help="Enable/disable fallback to regex parsing",
    ),
    confidence_threshold: Optional[float] = typer.Option(
        None,
        "--confidence-threshold",
        help="Minimum confidence threshold for parsing (0.0-1.0)",
    ),
    max_parsing_retries: Optional[int] = typer.Option(
        None, "--max-parsing-retries", help="Maximum number of parsing retry attempts"
    ),
    # Verbosity
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Enable verbose output"
    ),
) -> None:
    """Run a single reasoning experiment with the specified approach."""

    try:
        # Validate reasoning approach
        from src.cli.validators import validate_reasoning_approach

        approach = validate_reasoning_approach(approach)

        # Load and validate configuration
        experiment_config = load_and_validate_config(
            config_file=config,
            reasoning_approaches=[approach],
            sample_count=samples,
            provider=provider,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            output_dir=output_dir,
            multi_step_reflection=multi_step_reflection,
            multi_step_verification=multi_step_verification,
            max_reasoning_calls=max_reasoning_calls,
            use_structured_parsing=use_structured_parsing,
            fallback_to_regex=fallback_to_regex,
            confidence_threshold=confidence_threshold,
            max_parsing_retries=max_parsing_retries,
        )

        # Check environment is ready
        check_environment_ready(experiment_config.provider)

        # Display experiment start info
        display_experiment_start(
            config=experiment_config.to_dict(),
            approaches=[approach],
            total_samples=samples,
        )

        # Create and run experiment
        runner = ExperimentRunner(experiment_config)

        console.print("🚀 [blue]Starting single experiment...[/blue]")

        result = runner.run_single_experiment(
            approach=approach,
            progress_callback=lambda msg: (
                console.print(f"   {msg}") if verbose else None
            ),
        )

        # Display results
        if result:
            display_success("Experiment completed successfully!")

            # Create results table - extract data from results_summary
            approach_data = result.results_summary.get(approach, {})
            results_table = create_experiment_table(
                [
                    {
                        "approach": approach,
                        "total_samples": result.total_samples,
                        "accuracy": approach_data.get("accuracy", "N/A"),
                        "avg_time": f"{approach_data.get('avg_execution_time', 0):.2f}s",
                        "total_cost": f"${approach_data.get('total_cost', 0):.4f}",
                        "completed": approach_data.get("error_count", 0) == 0,
                        "error_count": approach_data.get("error_count", 0),
                    }
                ]
            )

            console.print("\n📊 [bold blue]Experiment Results:[/bold blue]")
            console.print(results_table)

            # Display accuracy breakdown by reading the saved CSV
            try:
                import csv

                # Find the most recent results CSV file
                output_dir = Path(experiment_config.output_dir)
                csv_files = list(output_dir.glob("*_results.csv"))
                if csv_files:
                    latest_csv = max(csv_files, key=lambda f: f.stat().st_mtime)

                    # Read the CSV and extract sample data
                    sample_results = []
                    with open(latest_csv, "r") as f:
                        reader = csv.DictReader(f)
                        for row in reader:
                            sample_results.append(
                                {
                                    "sample_id": row["sample_id"],
                                    "expected_output": row["expected_output"],
                                    "response_text": row["response_text"],
                                    "extracted_answer": row.get("extracted_answer", ""),
                                }
                            )

                    if sample_results:
                        accuracy_table = create_accuracy_breakdown_table(sample_results)
                        console.print(accuracy_table)
            except Exception as e:
                console.print(f"[dim]Could not display accuracy breakdown: {e}[/dim]")

            # Display completion info - calculate total cost from cost_summary
            total_cost = sum(result.cost_summary.values())
            display_experiment_complete(
                experiment_id=result.experiment_id,
                duration=result.duration,
                total_cost=total_cost,
                output_dir=experiment_config.output_dir,
            )
        else:
            display_error("Experiment failed to complete")
            raise typer.Exit(1)

    except KeyboardInterrupt:
        display_warning("Experiment interrupted by user")
        console.print(
            "💾 [dim]Partial results may be saved in the output directory[/dim]"
        )
        raise typer.Exit(130)
    except Exception as e:
        display_error(f"Experiment failed: {e}")
        if verbose:
            console.print_exception()
        raise typer.Exit(1)


def run_comparison_experiment(
    approaches: str = typer.Option(
        "ChainOfThought,AsPlanning",
        "--approaches",
        "-a",
        help="Comma-separated list of reasoning approaches",
        callback=lambda x: validate_reasoning_approaches(x) if x else None,
    ),
    samples: int = typer.Option(
        50,
        "--samples",
        "-n",
        help="Number of samples to process",
        callback=lambda x: validate_sample_count(x) if x else 50,
    ),
    config: Optional[str] = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to configuration file",
        callback=lambda x: validate_config_file(x) if x else None,
    ),
    # Model settings
    provider: Optional[str] = typer.Option(None, "--provider", help="Model provider"),
    model: Optional[str] = typer.Option(None, "--model", help="Model name"),
    temperature: Optional[float] = typer.Option(
        None,
        "--temperature",
        help="Sampling temperature (0.0-2.0)",
        callback=lambda x: validate_temperature(x) if x is not None else None,
    ),
    max_tokens: Optional[int] = typer.Option(
        None,
        "--max-tokens",
        help="Maximum tokens to generate",
        callback=lambda x: validate_max_tokens(x) if x else None,
    ),
    top_p: Optional[float] = typer.Option(
        None,
        "--top-p",
        help="Top-p sampling parameter (0.0-1.0)",
        callback=lambda x: validate_top_p(x) if x is not None else None,
    ),
    # Execution settings
    parallel: bool = typer.Option(
        False, "--parallel", help="Run approaches in parallel"
    ),
    max_workers: int = typer.Option(
        4,
        "--max-workers",
        help="Maximum parallel workers",
        callback=lambda x: validate_max_workers(x) if x else 4,
    ),
    output_dir: Optional[str] = typer.Option(
        None,
        "--output",
        "-o",
        help="Output directory",
        callback=lambda x: validate_output_directory(x) if x else None,
    ),
    save_checkpoints: bool = typer.Option(
        True, "--checkpoints/--no-checkpoints", help="Save experiment checkpoints"
    ),
    # Advanced reasoning settings
    multi_step_reflection: bool = typer.Option(
        False, "--multi-step-reflection", help="Enable multi-step reflection"
    ),
    multi_step_verification: bool = typer.Option(
        False, "--multi-step-verification", help="Enable multi-step verification"
    ),
    max_reasoning_calls: int = typer.Option(
        5, "--max-reasoning-calls", help="Maximum reasoning API calls"
    ),
    # Output parsing settings
    use_structured_parsing: Optional[bool] = typer.Option(
        None,
        "--structured-parsing/--no-structured-parsing",
        help="Enable/disable structured output parsing with Instructor",
    ),
    fallback_to_regex: Optional[bool] = typer.Option(
        None,
        "--fallback-regex/--no-fallback-regex",
        help="Enable/disable fallback to regex parsing",
    ),
    confidence_threshold: Optional[float] = typer.Option(
        None,
        "--confidence-threshold",
        help="Minimum confidence threshold for parsing (0.0-1.0)",
    ),
    max_parsing_retries: Optional[int] = typer.Option(
        None, "--max-parsing-retries", help="Maximum number of parsing retry attempts"
    ),
    # Verbosity
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Enable verbose output"
    ),
) -> None:
    """Run a comparison experiment across multiple reasoning approaches."""

    try:
        # Load and validate configuration
        experiment_config = load_and_validate_config(
            config_file=config,
            reasoning_approaches=approaches,
            sample_count=samples,
            provider=provider,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            output_dir=output_dir,
            multi_step_reflection=multi_step_reflection,
            multi_step_verification=multi_step_verification,
            max_reasoning_calls=max_reasoning_calls,
            use_structured_parsing=use_structured_parsing,
            fallback_to_regex=fallback_to_regex,
            confidence_threshold=confidence_threshold,
            max_parsing_retries=max_parsing_retries,
        )

        # Check environment is ready
        check_environment_ready(experiment_config.provider)

        # Display experiment start info
        display_experiment_start(
            config=experiment_config.to_dict(),
            approaches=approaches,
            total_samples=samples,
        )

        # Create and run experiment
        runner = ExperimentRunner(experiment_config)

        console.print(
            f"🚀 [blue]Starting comparison experiment with {len(approaches)} approaches...[/blue]"
        )

        if parallel:
            console.print(
                f"⚡ [yellow]Running in parallel with {max_workers} workers[/yellow]"
            )

        result = runner.run_comparison(
            approaches=approaches,
            parallel=parallel,
            max_workers=max_workers if parallel else 1,
            progress_callback=lambda msg: (
                console.print(f"   {msg}") if verbose else None
            ),
        )

        # Display results
        if result:
            display_success("Comparison experiment completed successfully!")

            # Extract results for each approach
            approach_results = []
            for approach in approaches:
                approach_data = result.results_summary.get(approach, {})
                approach_results.append(
                    {
                        "approach": approach,
                        "total_samples": approach_data.get("total_samples", 0),
                        "accuracy": approach_data.get("accuracy"),
                        "avg_time": approach_data.get("avg_time"),
                        "total_cost": approach_data.get("total_cost", 0.0),
                        "completed": approach_data.get("completed", False),
                        "error_count": approach_data.get("error_count", 0),
                    }
                )

            # Create results table
            results_table = create_experiment_table(approach_results)

            console.print("\n📊 [bold blue]Comparison Results:[/bold blue]")
            console.print(results_table)

            # Display accuracy breakdown by reading the saved CSV
            try:
                import csv

                # Find the most recent results CSV file
                output_dir = Path(experiment_config.output_dir)
                csv_files = list(output_dir.glob("*_results.csv"))
                if csv_files:
                    latest_csv = max(csv_files, key=lambda f: f.stat().st_mtime)

                    # Read the CSV and extract sample data
                    sample_results = []
                    with open(latest_csv, "r") as f:
                        reader = csv.DictReader(f)
                        for row in reader:
                            sample_results.append(
                                {
                                    "sample_id": row["sample_id"],
                                    "expected_output": row["expected_output"],
                                    "response_text": row["response_text"],
                                    "extracted_answer": row.get("extracted_answer", ""),
                                    "approach": row["approach"],
                                }
                            )

                    if sample_results:
                        # Group by approach for comparison experiments
                        for approach in approaches:
                            approach_samples = [
                                r for r in sample_results if r["approach"] == approach
                            ]
                            if approach_samples:
                                console.print(
                                    f"\n📋 [bold yellow]{approach} - Accuracy Breakdown:[/bold yellow]"
                                )
                                accuracy_table = create_accuracy_breakdown_table(
                                    approach_samples
                                )
                                console.print(accuracy_table)
            except Exception as e:
                console.print(f"[dim]Could not display accuracy breakdown: {e}[/dim]")

            # Display completion info
            display_experiment_complete(
                experiment_id=result.experiment_id,
                duration=result.duration,
                total_cost=sum(r["total_cost"] for r in approach_results),
                output_dir=experiment_config.output_dir,
            )
        else:
            display_error("Comparison experiment failed to complete")
            raise typer.Exit(1)

    except KeyboardInterrupt:
        display_warning("Experiment interrupted by user")
        console.print(
            "💾 [dim]Partial results may be saved in the output directory[/dim]"
        )
        raise typer.Exit(130)
    except Exception as e:
        display_error(f"Experiment failed: {e}")
        if verbose:
            console.print_exception()
        raise typer.Exit(1)


def resume_experiment(
    checkpoint: str = typer.Argument(
        ..., help="Path to checkpoint file", callback=validate_checkpoint_file
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Enable verbose output"
    ),
) -> None:
    """Resume an interrupted experiment from a checkpoint."""

    try:
        console.print(f"📂 [blue]Loading checkpoint:[/blue] {checkpoint}")

        # Load checkpoint data
        with open(checkpoint, "r") as f:
            checkpoint_data = json.load(f)

        experiment_id = checkpoint_data["experiment_id"]
        config_dict = checkpoint_data["config"]
        progress_data = checkpoint_data["progress"]

        display_info(f"Resuming experiment: {experiment_id}")

        # Recreate experiment config
        from src.config import ExperimentConfig

        experiment_config = ExperimentConfig.from_dict(config_dict)

        # Check environment is ready
        check_environment_ready(experiment_config.provider)

        # Create runner and resume
        runner = ExperimentRunner(experiment_config)

        console.print("🔄 [blue]Resuming experiment...[/blue]")

        # Resume the experiment
        result = runner.resume_from_checkpoint(checkpoint)

        # Display results
        if result:
            display_success("Experiment resumed and completed successfully!")

            # Extract the approach from checkpoint data
            approach = checkpoint_data.get("approach", "Unknown")

            # Create results table
            results_table = create_experiment_table(
                [
                    {
                        "approach": approach,
                        "total_samples": result.total_samples,
                        "accuracy": None,  # Would need to be calculated from results
                        "avg_time": None,  # Would need to be calculated from results
                        "total_cost": 0.0,  # Would need to be calculated from results
                        "completed": True,
                        "error_count": result.error_summary.get(approach, 0),
                    }
                ]
            )

            console.print("\n📊 [bold blue]Resumed Experiment Results:[/bold blue]")
            console.print(results_table)

            # Display completion info
            display_experiment_complete(
                experiment_id=result.experiment_id,
                duration=result.duration,
                total_cost=0.0,  # Would need to be calculated
                output_dir=experiment_config.output_dir,
            )
        else:
            display_error("Failed to resume experiment")
            raise typer.Exit(1)

    except Exception as e:
        display_error(f"Failed to resume experiment: {e}")
        if verbose:
            console.print_exception()
        raise typer.Exit(1)


def list_checkpoints(
    output_dir: str = typer.Option(
        "./outputs", "--output-dir", "-o", help="Directory to search for checkpoints"
    ),
) -> None:
    """List available experiment checkpoints."""

    try:
        output_path = Path(output_dir)

        if not output_path.exists():
            display_warning(f"Output directory not found: {output_dir}")
            return

        # Find checkpoint files
        checkpoints = list(output_path.glob("**/checkpoint_*.json"))

        if not checkpoints:
            display_info("No checkpoints found")
            return

        console.print(
            f"\n📂 [bold blue]Found {len(checkpoints)} checkpoints:[/bold blue]\n"
        )

        for checkpoint in sorted(
            checkpoints, key=lambda x: x.stat().st_mtime, reverse=True
        ):
            try:
                with open(checkpoint, "r") as f:
                    data = json.load(f)

                experiment_id = data.get("experiment_id", "Unknown")
                timestamp = data.get("timestamp", "Unknown")
                approaches = data.get("config", {}).get("reasoning_approaches", [])

                console.print(f"📁 [cyan]{checkpoint.name}[/cyan]")
                console.print(f"   ID: {experiment_id}")
                console.print(f"   Time: {timestamp}")
                console.print(f"   Approaches: {', '.join(approaches)}")
                console.print()

            except Exception as e:
                console.print(
                    f"❌ [red]Invalid checkpoint:[/red] {checkpoint.name} ({e})"
                )

    except Exception as e:
        display_error(f"Failed to list checkpoints: {e}")
        raise typer.Exit(1)


# ==================== Database Management Commands ====================


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
    from src.core.database_manager import DatabaseConfig, DatabaseManager

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
    from datetime import datetime

    from src.core.database_manager import DatabaseConfig, DatabaseManager

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
    from src.core.database_manager import DatabaseConfig, DatabaseManager

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
    from datetime import datetime

    from src.core.database_manager import DatabaseConfig, DatabaseManager

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


# ==================== Export and Analysis Commands ====================


def export_experiment(
    experiment_id: str = typer.Argument(..., help="Experiment ID to export"),
    format: str = typer.Option(
        "csv", "--format", "-f", help="Export format: csv, json, excel"
    ),
    output: Optional[str] = typer.Option(
        None,
        "--output",
        "-o",
        help="Output file path (auto-generated if not specified)",
    ),
    db_path: Optional[str] = typer.Option(
        None, "--db-path", help="Database path (default: ./ml_agents_results.db)"
    ),
    include_raw: bool = typer.Option(
        False, "--include-raw", help="Include raw model outputs in JSON export"
    ),
) -> None:
    """Export experiment results to various formats."""
    from datetime import datetime

    from src.core.database_manager import DatabaseConfig
    from src.core.results_processor import ResultsProcessor

    db_path = db_path or "./ml_agents_results.db"

    if not Path(db_path).exists():
        display_error(f"Database not found: {db_path}")
        raise typer.Exit(1)

    # Validate format
    if format not in ["csv", "json", "excel"]:
        display_error("Format must be one of: csv, json, excel")
        raise typer.Exit(1)

    try:
        config = DatabaseConfig(db_path=db_path)
        processor = ResultsProcessor(config)

        # Check if experiment exists
        summary = processor.get_experiment_summary(experiment_id)
        if not summary:
            display_error(f"Experiment not found: {experiment_id}")
            raise typer.Exit(1)

        # Generate output filename if not provided
        if not output:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output = f"{experiment_id}_{format}_{timestamp}.{format}"

        display_info(
            f"Exporting experiment {experiment_id} to {format.upper()} format..."
        )

        # Export based on format
        if format == "csv":
            processor.export_to_csv(experiment_id, output)
        elif format == "json":
            processor.export_to_json(
                experiment_id, output, include_raw_output=include_raw
            )
        elif format == "excel":
            processor.export_to_excel([experiment_id], output)

        display_success(f"Export completed successfully")
        console.print(f"📁 Output file: {output}")
        console.print(
            f"📊 Experiment: {summary.total_runs} runs, {summary.accuracy:.2%} accuracy"
        )

    except Exception as e:
        display_error(f"Failed to export experiment: {e}")
        raise typer.Exit(1)


def compare_experiments(
    experiment_ids: str = typer.Argument(
        ..., help="Comma-separated experiment IDs to compare"
    ),
    output: Optional[str] = typer.Option(
        None, "--output", "-o", help="Output file path for Excel comparison"
    ),
    db_path: Optional[str] = typer.Option(
        None, "--db-path", help="Database path (default: ./ml_agents_results.db)"
    ),
) -> None:
    """Compare results across multiple experiments."""
    from datetime import datetime

    from src.core.database_manager import DatabaseConfig
    from src.core.results_processor import ResultsProcessor

    db_path = db_path or "./ml_agents_results.db"
    exp_ids = [exp_id.strip() for exp_id in experiment_ids.split(",")]

    if not Path(db_path).exists():
        display_error(f"Database not found: {db_path}")
        raise typer.Exit(1)

    try:
        config = DatabaseConfig(db_path=db_path)
        processor = ResultsProcessor(config)

        # Validate all experiments exist
        missing_experiments = []
        valid_experiments = []

        for exp_id in exp_ids:
            summary = processor.get_experiment_summary(exp_id)
            if summary:
                valid_experiments.append(summary)
            else:
                missing_experiments.append(exp_id)

        if missing_experiments:
            display_warning(f"Missing experiments: {', '.join(missing_experiments)}")

        if not valid_experiments:
            display_error("No valid experiments found")
            raise typer.Exit(1)

        # Create comparison table
        table = Table(title="Experiment Comparison")
        table.add_column("Experiment ID", style="cyan")
        table.add_column("Total Runs", justify="right")
        table.add_column("Accuracy", justify="right")
        table.add_column("Avg Time (ms)", justify="right")
        table.add_column("Total Cost", justify="right")
        table.add_column("Approaches", style="dim")

        for summary in valid_experiments:
            table.add_row(
                summary.experiment_id[:12] + "...",
                str(summary.total_runs),
                f"{summary.accuracy:.2%}",
                f"{summary.avg_execution_time_ms:.0f}",
                f"${summary.total_cost:.4f}",
                ", ".join(summary.approaches_tested),
            )

        console.print(table)

        # Export to Excel if requested
        if output:
            processor.export_to_excel(
                [exp.experiment_id for exp in valid_experiments], output
            )
            display_success(f"Comparison exported to {output}")

    except Exception as e:
        display_error(f"Failed to compare experiments: {e}")
        raise typer.Exit(1)


def analyze_experiment(
    experiment_id: str = typer.Argument(..., help="Experiment ID to analyze"),
    report_type: str = typer.Option(
        "summary", "--type", "-t", help="Report type: summary, accuracy, failures"
    ),
    db_path: Optional[str] = typer.Option(
        None, "--db-path", help="Database path (default: ./ml_agents_results.db)"
    ),
) -> None:
    """Generate detailed analysis reports for an experiment."""
    from src.core.database_manager import DatabaseConfig
    from src.core.results_processor import ResultsProcessor

    db_path = db_path or "./ml_agents_results.db"

    if not Path(db_path).exists():
        display_error(f"Database not found: {db_path}")
        raise typer.Exit(1)

    if report_type not in ["summary", "accuracy", "failures"]:
        display_error("Report type must be one of: summary, accuracy, failures")
        raise typer.Exit(1)

    try:
        config = DatabaseConfig(db_path=db_path)
        processor = ResultsProcessor(config)

        # Check if experiment exists
        summary = processor.get_experiment_summary(experiment_id)
        if not summary:
            display_error(f"Experiment not found: {experiment_id}")
            raise typer.Exit(1)

        if report_type == "summary":
            # Display experiment summary
            console.print(f"\n📊 [bold]Experiment Summary: {experiment_id}[/bold]\n")
            console.print(f"Total Runs: {summary.total_runs}")
            console.print(f"Completed: {summary.completed_runs}")
            console.print(f"Failed: {summary.failed_runs}")
            console.print(f"Accuracy: {summary.accuracy:.2%}")
            console.print(f"Avg Execution Time: {summary.avg_execution_time_ms:.2f}ms")
            console.print(f"Total Cost: ${summary.total_cost:.4f}")
            console.print(f"Approaches: {', '.join(summary.approaches_tested)}")
            console.print(f"Models: {', '.join(summary.models_used)}")
            console.print(f"Parsing Success Rate: {summary.parsing_success_rate:.2%}")

        elif report_type == "accuracy":
            # Display accuracy analysis
            accuracy_report = processor.generate_accuracy_report(experiment_id)

            console.print(f"\n📈 [bold]Accuracy Analysis: {experiment_id}[/bold]\n")

            # Accuracy by approach
            table = Table(title="Accuracy by Approach")
            table.add_column("Approach", style="cyan")
            table.add_column("Total", justify="right")
            table.add_column("Correct", justify="right")
            table.add_column("Accuracy", justify="right")

            for approach, data in accuracy_report["accuracy_by_approach"].items():
                table.add_row(
                    approach,
                    str(data["total"]),
                    str(data["correct"]),
                    f"{data['accuracy']:.2%}",
                )

            console.print(table)

        elif report_type == "failures":
            # Display failure analysis
            failures = processor.identify_failure_patterns(experiment_id)

            console.print(f"\n❌ [bold]Failure Analysis: {experiment_id}[/bold]\n")

            for i, pattern in enumerate(failures[:10], 1):  # Show top 10
                console.print(
                    f"{i}. [red]{pattern['failure_type']}[/red] - {pattern['approach']} ({pattern['model']})"
                )
                console.print(f"   Count: {pattern['count']}")
                if pattern["examples"]:
                    console.print(
                        f"   Example: {pattern['examples'][0]['input'][:100]}..."
                    )
                console.print()

    except Exception as e:
        display_error(f"Failed to analyze experiment: {e}")
        raise typer.Exit(1)


def list_experiments(
    status: Optional[str] = typer.Option(
        None, "--status", help="Filter by status: running, completed, failed"
    ),
    limit: int = typer.Option(
        10, "--limit", help="Maximum number of experiments to show"
    ),
    db_path: Optional[str] = typer.Option(
        None, "--db-path", help="Database path (default: ./ml_agents_results.db)"
    ),
) -> None:
    """List experiments stored in the database."""
    from src.core.database_manager import DatabaseConfig
    from src.core.results_processor import ResultsProcessor

    db_path = db_path or "./ml_agents_results.db"

    if not Path(db_path).exists():
        display_error(f"Database not found: {db_path}")
        raise typer.Exit(1)

    try:
        config = DatabaseConfig(db_path=db_path)
        processor = ResultsProcessor(config)

        experiments = processor.get_experiments_list(status=status)

        if not experiments:
            display_info("No experiments found")
            return

        # Create experiments table
        table = Table(title="Experiments")
        table.add_column("ID", style="cyan")
        table.add_column("Name", style="white")
        table.add_column("Status", style="yellow")
        table.add_column("Created", style="dim")
        table.add_column("Approaches", style="green")

        for exp in experiments[:limit]:
            table.add_row(
                exp["id"][:12] + "...",
                exp["name"][:30] + "..." if len(exp["name"]) > 30 else exp["name"],
                exp["status"],
                exp["created_at"][:10],  # Just the date
                ", ".join(exp["config"].get("reasoning_approaches", []))[:30] + "...",
            )

        console.print(table)

        if len(experiments) > limit:
            console.print(
                f"\n[dim]Showing {limit} of {len(experiments)} experiments. Use --limit to see more.[/dim]"
            )

    except Exception as e:
        display_error(f"Failed to list experiments: {e}")
        raise typer.Exit(1)

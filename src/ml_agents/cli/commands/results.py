"""Results analysis and export commands."""

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


def display_pre_alpha_warning():
    """Display pre-alpha warning for results commands."""
    console.print("\n⚠️  [bold yellow]PRE-ALPHA WARNING[/bold yellow]")
    console.print(
        "[yellow]The 'results' command group is in pre-alpha development.[/yellow]"
    )
    console.print(
        "[yellow]Features may be incomplete, unstable, or subject to breaking changes.[/yellow]"
    )
    console.print(
        "[yellow]For production use, consider using the stable preprocessing and database commands.[/yellow]"
    )
    console.print("[dim]Use --skip-warnings to suppress this message.[/dim]\n")


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
    skip_warnings: bool = typer.Option(
        False, "--skip-warnings", help="Skip pre-alpha warnings"
    ),
) -> None:
    """⚠️ PRE-ALPHA: Export experiment results to various formats.

    This command is in pre-alpha development and may be unstable."""
    # Display pre-alpha warning
    if not skip_warnings:
        display_pre_alpha_warning()

    from ml_agents.core.database_manager import DatabaseConfig
    from ml_agents.core.results_processor import ResultsProcessor

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
    skip_warnings: bool = typer.Option(
        False, "--skip-warnings", help="Skip pre-alpha warnings"
    ),
) -> None:
    """⚠️ PRE-ALPHA: Compare results across multiple experiments.

    This command is in pre-alpha development and may be unstable."""
    # Display pre-alpha warning
    if not skip_warnings:
        display_pre_alpha_warning()

    from ml_agents.core.database_manager import DatabaseConfig
    from ml_agents.core.results_processor import ResultsProcessor

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
    skip_warnings: bool = typer.Option(
        False, "--skip-warnings", help="Skip pre-alpha warnings"
    ),
) -> None:
    """⚠️ PRE-ALPHA: Generate detailed analysis reports for an experiment.

    This command is in pre-alpha development and may be unstable."""
    # Display pre-alpha warning
    if not skip_warnings:
        display_pre_alpha_warning()

    from ml_agents.core.database_manager import DatabaseConfig
    from ml_agents.core.results_processor import ResultsProcessor

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
    skip_warnings: bool = typer.Option(
        False, "--skip-warnings", help="Skip pre-alpha warnings"
    ),
) -> None:
    """⚠️ PRE-ALPHA: List experiments stored in the database.

    This command is in pre-alpha development and may be unstable."""
    # Display pre-alpha warning
    if not skip_warnings:
        display_pre_alpha_warning()

    from ml_agents.core.database_manager import DatabaseConfig
    from ml_agents.core.results_processor import ResultsProcessor

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

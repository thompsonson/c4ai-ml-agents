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
    full_ids: bool = typer.Option(
        False, "--full-ids", help="Show full experiment IDs instead of truncated"
    ),
    show_accuracy: bool = typer.Option(
        True, "--show-accuracy/--no-accuracy", help="Show accuracy column"
    ),
    dataset_filter: Optional[str] = typer.Option(
        None, "--dataset", help="Filter by dataset name (partial match)"
    ),
    model_filter: Optional[str] = typer.Option(
        None, "--model", help="Filter by model name (partial match)"
    ),
    approach_filter: Optional[str] = typer.Option(
        None, "--approach", help="Filter by reasoning approach"
    ),
    accuracy_min: Optional[float] = typer.Option(
        None, "--accuracy-min", help="Minimum accuracy threshold (0.0-1.0)"
    ),
    sort_by: str = typer.Option(
        "date", "--sort-by", help="Sort by: date, accuracy, samples, cost"
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

        # Get enhanced experiment data with accuracy
        experiments = processor.get_experiments_list_enhanced(
            status=status,
            dataset_filter=dataset_filter,
            model_filter=model_filter,
            approach_filter=approach_filter,
            accuracy_min=accuracy_min,
            sort_by=sort_by,
        )

        if not experiments:
            display_info("No experiments found")
            return

        # Create experiments table with dynamic columns
        table = Table(title=f"ML Agents Experiments (Total: {len(experiments)})")

        # ID column - full or truncated based on flag
        if full_ids:
            table.add_column("Experiment ID", style="cyan", min_width=40)
        else:
            table.add_column("ID", style="cyan", min_width=15)

        # Core columns
        table.add_column("Dataset", style="white", min_width=20)
        table.add_column("Approach", style="green", min_width=15)
        table.add_column("Status", style="yellow", min_width=10)

        # Metrics columns
        if show_accuracy:
            table.add_column("Accuracy", style="magenta", justify="right", min_width=10)
        table.add_column("Samples", justify="right", min_width=8)
        table.add_column("Time (s)", justify="right", min_width=8)
        table.add_column("Cost ($)", justify="right", min_width=10)
        table.add_column("Created", style="dim", min_width=16)

        # Add rows with proper formatting
        for exp in experiments[:limit]:
            # Extract dataset name from experiment data
            dataset = exp.get("dataset_name", "")
            if not dataset and exp.get("name"):
                # Try to extract from experiment name
                name = exp["name"]
                if "BENCHMARK-" in name:
                    # Extract benchmark dataset name from anywhere in the string
                    import re

                    match = re.search(r"BENCHMARK-\d+-[^_\]]+", name)
                    if match:
                        dataset = match.group(0).replace(".csv", "")
                    else:
                        # Fallback: look for parts
                        parts = name.split("_")
                        for part in parts:
                            if "BENCHMARK-" in part:
                                dataset = (
                                    part.replace(".csv", "")
                                    .replace("]", "")
                                    .replace("[", "")
                                )
                                break
                elif "LOCAL_TEST" in name:
                    dataset = "LOCAL_TEST"
                else:
                    # Extract from name pattern: provider_model_approach_dataset
                    parts = name.split("_")
                    if len(parts) >= 3:
                        # Check if third part looks like approach list
                        if parts[2].startswith("[") and parts[2].endswith("]"):
                            dataset = parts[3] if len(parts) > 3 else "Unknown"
                        else:
                            dataset = parts[2]
                    else:
                        dataset = "Unknown"

            # Format ID
            exp_id = exp["id"] if full_ids else exp["id"][:12] + "..."

            # Format accuracy with color coding
            accuracy_str = ""
            if show_accuracy and exp.get("accuracy") is not None:
                accuracy = exp["accuracy"]
                if accuracy >= 0.8:
                    accuracy_str = f"[green]{accuracy:.1%}[/green]"
                elif accuracy >= 0.5:
                    accuracy_str = f"[yellow]{accuracy:.1%}[/yellow]"
                else:
                    accuracy_str = f"[red]{accuracy:.1%}[/red]"

            # Format approach - extract from config or metadata
            approach = "Unknown"
            if exp.get("config") and exp["config"].get("reasoning_approaches"):
                approach = exp["config"]["reasoning_approaches"][0]
            elif exp.get("approach"):
                approach = exp["approach"]

            # Format time and cost
            avg_time = f"{exp.get('avg_time', 0):.1f}" if exp.get("avg_time") else "0.0"
            total_cost = (
                f"{exp.get('total_cost', 0):.4f}" if exp.get("total_cost") else "0.0000"
            )

            # Build row
            row = [
                exp_id,
                dataset[:30] + "..." if len(dataset) > 30 else dataset,
                approach[:20] + "..." if len(approach) > 20 else approach,
                exp["status"],
            ]

            if show_accuracy:
                row.append(accuracy_str if accuracy_str else "N/A")

            row.extend(
                [
                    str(exp.get("total_samples", 0)),
                    avg_time,
                    total_cost,
                    exp["created_at"][:16] if exp.get("created_at") else "Unknown",
                ]
            )

            table.add_row(*row)

        console.print(table)

        if len(experiments) > limit:
            console.print(
                f"\n[dim]Showing {limit} of {len(experiments)} experiments. Use --limit to see more.[/dim]"
            )
            console.print(
                "[dim]Tip: Use --full-ids to see complete experiment IDs for copying.[/dim]"
            )

    except Exception as e:
        display_error(f"Failed to list experiments: {e}")
        raise typer.Exit(1)

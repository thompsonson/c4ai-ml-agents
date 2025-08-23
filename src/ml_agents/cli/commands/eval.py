"""Evaluation and experiment execution commands."""

import json
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

from ml_agents.cli.config_loader import load_and_validate_config
from ml_agents.cli.display import (
    create_accuracy_breakdown_table,
    create_experiment_table,
    display_error,
    display_experiment_complete,
    display_experiment_start,
    display_info,
    display_success,
    display_warning,
)
from ml_agents.cli.validators import (
    check_environment_ready,
    validate_checkpoint_file,
    validate_config_file,
    validate_max_tokens,
    validate_max_workers,
    validate_output_directory,
    validate_reasoning_approach,
    validate_reasoning_approaches,
    validate_sample_count,
    validate_temperature,
    validate_top_p,
)
from ml_agents.core.experiment_runner import ExperimentRunner

console = Console()


def display_pre_alpha_warning() -> None:
    """Display pre-alpha warning for evaluation commands."""
    console.print("\n⚠️  [bold yellow]PRE-ALPHA WARNING[/bold yellow]")
    console.print(
        "[yellow]The 'eval' command group is in pre-alpha development.[/yellow]"
    )
    console.print(
        "[yellow]Features may be incomplete, unstable, or subject to breaking changes.[/yellow]"
    )
    console.print("[yellow]Preprocessing and database commands are stable.[/yellow]")
    console.print("[dim]Use --skip-warnings to suppress this message.[/dim]\n")


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
    skip_warnings: bool = typer.Option(
        False, "--skip-warnings", help="Skip pre-alpha warnings"
    ),
) -> None:
    """⚠️ PRE-ALPHA: Run a single reasoning experiment with the specified approach.

    This command is in pre-alpha development and may be unstable."""

    try:
        # Display pre-alpha warning
        if not skip_warnings:
            display_pre_alpha_warning()

        # Validate reasoning approach
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
        callback=lambda x: validate_max_workers(x),
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
    skip_warnings: bool = typer.Option(
        False, "--skip-warnings", help="Skip pre-alpha warnings"
    ),
) -> None:
    """⚠️ PRE-ALPHA: Run a comparison experiment across multiple reasoning approaches.

    This command is in pre-alpha development and may be unstable."""

    try:
        # Display pre-alpha warning
        if not skip_warnings:
            display_pre_alpha_warning()

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
    skip_warnings: bool = typer.Option(
        False, "--skip-warnings", help="Skip pre-alpha warnings"
    ),
) -> None:
    """⚠️ PRE-ALPHA: Resume an interrupted experiment from a checkpoint.

    This command is in pre-alpha development and may be unstable."""

    try:
        # Display pre-alpha warning
        if not skip_warnings:
            display_pre_alpha_warning()

        console.print(f"📂 [blue]Loading checkpoint:[/blue] {checkpoint}")

        # Load checkpoint data
        with open(checkpoint, "r") as f:
            checkpoint_data = json.load(f)

        experiment_id = checkpoint_data["experiment_id"]
        config_dict = checkpoint_data["config"]
        progress_data = checkpoint_data["progress"]

        display_info(f"Resuming experiment: {experiment_id}")

        # Recreate experiment config
        from ml_agents.config import ExperimentConfig

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
    skip_warnings: bool = typer.Option(
        False, "--skip-warnings", help="Skip pre-alpha warnings"
    ),
) -> None:
    """⚠️ PRE-ALPHA: List available experiment checkpoints.

    This command is in pre-alpha development and may be unstable."""

    try:
        # Display pre-alpha warning
        if not skip_warnings:
            display_pre_alpha_warning()

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

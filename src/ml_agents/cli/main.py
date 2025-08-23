"""Main CLI entry point for ML Agents reasoning research platform."""

from typing import Optional

import typer
from rich.console import Console

from ml_agents.cli.commands import (
    analyze_experiment,
    compare_experiments,
    db_backup,
    db_init,
    db_migrate,
    db_stats,
    export_experiment,
    list_checkpoints,
    list_experiments,
    preprocess_batch,
    preprocess_generate_rules,
    preprocess_inspect,
    preprocess_list_unprocessed,
    preprocess_transform,
    preprocess_upload,
    resume_experiment,
    run_comparison_experiment,
    run_single_experiment,
)
from ml_agents.cli.display import display_banner, display_error
from ml_agents.config import validate_environment

app = typer.Typer(
    name="ml-agents",
    help="🧠 ML Agents Reasoning Research Platform - Cohere Labs Open Science Initiative",
    add_completion=False,
    rich_markup_mode="rich",
)
console = Console()

# Add experiment commands
app.command("run")(run_single_experiment)
app.command("compare")(run_comparison_experiment)
app.command("resume")(resume_experiment)
app.command("list-checkpoints")(list_checkpoints)

# Add database management commands
app.command("db-init")(db_init)
app.command("db-backup")(db_backup)
app.command("db-migrate")(db_migrate)
app.command("db-stats")(db_stats)

# Add export and analysis commands
app.command("export")(export_experiment)
app.command("compare-experiments")(compare_experiments)
app.command("analyze")(analyze_experiment)
app.command("list-experiments")(list_experiments)

# Add preprocessing commands
app.command("preprocess-list")(preprocess_list_unprocessed)
app.command("preprocess-inspect")(preprocess_inspect)
app.command("preprocess-generate-rules")(preprocess_generate_rules)
app.command("preprocess-transform")(preprocess_transform)
app.command("preprocess-batch")(preprocess_batch)
app.command("preprocess-upload")(preprocess_upload)


@app.command()
def validate_env() -> None:
    """Validate environment configuration and API keys."""
    console.print("\n🔍 [bold blue]Validating Environment...[/bold blue]")

    validation_results = validate_environment()

    if all(validation_results.values()):
        console.print("✅ [bold green]Environment validation passed![/bold green]")
        console.print("🚀 [green]Ready to run experiments![/green]")
    else:
        console.print("❌ [bold red]Environment validation failed![/bold red]")

        if not validation_results["api_keys"]:
            console.print("   • Missing API keys. Check your .env file.")
        if not validation_results["output_dir_writable"]:
            console.print("   • Cannot write to output directory.")
        if not validation_results["dependencies_available"]:
            console.print("   • Missing required dependencies.")

        console.print("\n📖 See documentation for setup instructions.")
        raise typer.Exit(1)


@app.command()
def list_approaches() -> None:
    """List all available reasoning approaches."""
    from ml_agents.reasoning import get_available_approaches

    approaches = get_available_approaches()

    console.print("\n🧠 [bold blue]Available Reasoning Approaches:[/bold blue]\n")

    for i, approach in enumerate(approaches, 1):
        console.print(f"  {i:2d}. [cyan]{approach}[/cyan]")

    console.print(f"\n📊 Total: [bold]{len(approaches)}[/bold] approaches available")


@app.command()
def version() -> None:
    """Show version information."""
    from ml_agents import __version__

    console.print(f"\n🧠 [bold blue]ML Agents CLI[/bold blue] v{__version__}")
    console.print("🔬 [dim]Cohere Labs Open Science Initiative[/dim]")


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    version_flag: Optional[bool] = typer.Option(
        None, "--version", "-V", help="Show version and exit"
    ),
    verbose: Optional[bool] = typer.Option(
        False, "--verbose", "-v", help="Enable verbose output"
    ),
) -> None:
    """🧠 ML Agents Reasoning Research Platform - Cohere Labs Open Science Initiative

    A comprehensive platform for conducting reasoning research across different AI models and approaches.
    Supports 8 reasoning methods including Chain-of-Thought, Tree-of-Thought, and more.
    """
    if version_flag:
        version()
        raise typer.Exit()

    if ctx.invoked_subcommand is None:
        display_banner()
        console.print("💡 [dim]Use --help to see available commands[/dim]\n")


if __name__ == "__main__":
    try:
        app()
    except KeyboardInterrupt:
        console.print("\n\n⚠️  [yellow]Experiment interrupted by user[/yellow]")
        console.print(
            "💾 [dim]Partial results may be saved in the output directory[/dim]"
        )
        raise typer.Exit(130)
    except Exception as e:
        display_error(f"Unexpected error: {e}")
        raise typer.Exit(1)

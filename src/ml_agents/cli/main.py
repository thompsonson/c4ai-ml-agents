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
    list_approaches,
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
    validate_env,
    version,
)
from ml_agents.cli.display import display_banner, display_error

app = typer.Typer(
    name="ml-agents",
    help="🧠 ML Agents Reasoning Research Platform - Cohere Labs Open Science Initiative",
    add_completion=False,
    rich_markup_mode="rich",
)
console = Console()

# Create sub-apps for grouped commands
setup_app = typer.Typer(
    name="setup",
    help="🔧 Environment setup and system validation commands",
    rich_markup_mode="rich",
)

preprocess_app = typer.Typer(
    name="preprocess",
    help="🔄 Dataset preprocessing and transformation commands",
    rich_markup_mode="rich",
)

eval_app = typer.Typer(
    name="eval",
    help="🧪 Reasoning evaluation and experiment execution commands",
    rich_markup_mode="rich",
)

results_app = typer.Typer(
    name="results",
    help="📊 Results analysis and export commands",
    rich_markup_mode="rich",
)

db_app = typer.Typer(
    name="db",
    help="🗄️ Database management and maintenance commands",
    rich_markup_mode="rich",
)

# Add sub-apps to main app
app.add_typer(setup_app, name="setup")
app.add_typer(preprocess_app, name="preprocess")
app.add_typer(eval_app, name="eval")
app.add_typer(results_app, name="results")
app.add_typer(db_app, name="db")

# Add commands to respective sub-apps

# Setup commands
setup_app.command("validate-env")(validate_env)
setup_app.command("list-approaches")(list_approaches)
setup_app.command("version")(version)

# Preprocess commands
preprocess_app.command("list")(preprocess_list_unprocessed)
preprocess_app.command("inspect")(preprocess_inspect)
preprocess_app.command("generate-rules")(preprocess_generate_rules)
preprocess_app.command("transform")(preprocess_transform)
preprocess_app.command("batch")(preprocess_batch)
preprocess_app.command("upload")(preprocess_upload)

# Eval commands
eval_app.command("run")(run_single_experiment)
eval_app.command("compare")(run_comparison_experiment)
eval_app.command("resume")(resume_experiment)
eval_app.command("checkpoints")(list_checkpoints)

# Results commands
results_app.command("export")(export_experiment)
results_app.command("analyze")(analyze_experiment)
results_app.command("compare")(compare_experiments)
results_app.command("list")(list_experiments)

# Database commands
db_app.command("init")(db_init)
db_app.command("backup")(db_backup)
db_app.command("migrate")(db_migrate)
db_app.command("stats")(db_stats)


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

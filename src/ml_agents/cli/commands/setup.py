"""Setup and environment validation commands."""

import typer
from rich.console import Console

from ml_agents.config import validate_environment

console = Console()


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


def list_approaches() -> None:
    """List all available reasoning approaches."""
    from ml_agents.reasoning import get_available_approaches

    approaches = get_available_approaches()

    console.print("\n🧠 [bold blue]Available Reasoning Approaches:[/bold blue]\n")

    for i, approach in enumerate(approaches, 1):
        console.print(f"  {i:2d}. [cyan]{approach}[/cyan]")

    console.print(f"\n📊 Total: [bold]{len(approaches)}[/bold] approaches available")


def version() -> None:
    """Show version information."""
    try:
        from importlib.metadata import version as get_version

        version_str = get_version("ml-agents-reasoning")
    except Exception:
        # Fallback for development installations
        version_str = "dev"

    console.print(f"\n🧠 [bold blue]ML Agents CLI[/bold blue] v{version_str}")
    console.print("🔬 [dim]Cohere Labs Open Science Initiative[/dim]")

"""Rich display utilities for CLI output formatting."""

from datetime import datetime
from typing import Any, Dict, List, Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table
from rich.text import Text

console = Console()


def display_banner() -> None:
    """Display the ML Agents CLI banner."""
    banner_text = """
🧠 [bold blue]ML Agents[/bold blue] [dim]Reasoning Research Platform[/dim]

[dim]Cohere Labs Open Science Initiative[/dim]
[dim]Investigating the efficacy of reasoning in AI models[/dim]
"""

    console.print(Panel(banner_text.strip(), border_style="blue", padding=(1, 2)))


def display_error(message: str) -> None:
    """Display an error message."""
    console.print(f"\n❌ [bold red]Error:[/bold red] {message}")


def display_warning(message: str) -> None:
    """Display a warning message."""
    console.print(f"⚠️  [yellow]Warning:[/yellow] {message}")


def display_success(message: str) -> None:
    """Display a success message."""
    console.print(f"✅ [green]{message}[/green]")


def display_info(message: str) -> None:
    """Display an info message."""
    console.print(f"ℹ️  [blue]{message}[/blue]")


def create_experiment_table(results: List[Dict[str, Any]]) -> Table:
    """Create a Rich table for experiment results."""
    table = Table(show_header=True, header_style="bold cyan")

    table.add_column("Approach", style="cyan", no_wrap=True)
    table.add_column("Samples", justify="right")
    table.add_column("Accuracy", justify="right")
    table.add_column("Avg Time", justify="right")
    table.add_column("Total Cost", justify="right", style="green")
    table.add_column("Status", justify="center")

    for result in results:
        # Extract key metrics
        approach = result.get("approach", "Unknown")
        samples = str(result.get("total_samples", 0))

        # Format numeric values as strings
        accuracy_val = result.get("accuracy")
        if accuracy_val is not None:
            accuracy = (
                f"{accuracy_val:.2%}"
                if isinstance(accuracy_val, (int, float))
                else str(accuracy_val)
            )
        else:
            accuracy = "N/A"

        avg_time_val = result.get("avg_time")
        if avg_time_val is not None:
            avg_time = (
                f"{avg_time_val:.2f}s"
                if isinstance(avg_time_val, (int, float))
                else str(avg_time_val)
            )
        else:
            avg_time = "N/A"

        total_cost_val = result.get("total_cost")
        if total_cost_val is not None:
            total_cost = (
                f"${total_cost_val:.2f}"
                if isinstance(total_cost_val, (int, float))
                else str(total_cost_val)
            )
        else:
            total_cost = "$0.00"

        # Status with color coding
        if result.get("error_count", 0) > 0:
            status = "⚠️  Partial"
            status_style = "yellow"
        elif result.get("completed", False):
            status = "✅ Complete"
            status_style = "green"
        else:
            status = "⏳ Running"
            status_style = "blue"

        table.add_row(
            approach,
            samples,
            accuracy,
            avg_time,
            total_cost,
            f"[{status_style}]{status}[/{status_style}]",
        )

    return table


def create_accuracy_breakdown_table(results_data: List[Dict[str, Any]]) -> Table:
    """Create a Rich table showing expected vs actual answers for accuracy analysis."""
    table = Table(
        show_header=True, header_style="bold yellow", title="📋 Accuracy Breakdown"
    )

    table.add_column("Sample ID", justify="right", style="dim")
    table.add_column("Expected", style="cyan", max_width=30)
    table.add_column("Actual", style="magenta", max_width=30)
    table.add_column("Correct", justify="center")

    for result in results_data:
        sample_id = str(result.get("sample_id", "?"))
        expected = str(result.get("expected_output", "N/A"))

        # Use extracted_answer if available, otherwise fall back to response_text
        extracted_answer = result.get("extracted_answer")
        if extracted_answer and extracted_answer.strip():
            actual = str(extracted_answer)
        else:
            actual = str(result.get("response_text", "N/A"))

        # Truncate long responses for display
        expected_display = expected[:27] + "..." if len(expected) > 30 else expected
        actual_display = actual[:27] + "..." if len(actual) > 30 else actual

        # Determine if correct (simple string comparison)
        is_correct = expected.strip().lower() == actual.strip().lower()
        correct_symbol = "✅" if is_correct else "❌"

        table.add_row(sample_id, expected_display, actual_display, correct_symbol)

    return table


def create_cost_summary_table(cost_breakdown: Dict[str, Any]) -> Table:
    """Create a Rich table for cost breakdown."""
    table = Table(show_header=True, header_style="bold green", title="💰 Cost Summary")

    table.add_column("Provider", style="cyan")
    table.add_column("Model", style="dim")
    table.add_column("Requests", justify="right")
    table.add_column("Tokens", justify="right")
    table.add_column("Cost", justify="right", style="green")

    total_cost = 0.0

    for provider, models in cost_breakdown.items():
        if isinstance(models, dict):
            for model, data in models.items():
                requests = data.get("requests", 0)
                tokens = data.get("tokens", 0)
                cost = data.get("cost", 0.0)
                total_cost += cost

                table.add_row(
                    provider, model, str(requests), f"{tokens:,}", f"${cost:.2f}"
                )

    table.add_row(
        "", "[bold]TOTAL[/bold]", "", "", f"[bold green]${total_cost:.2f}[/bold green]"
    )

    return table


def create_progress_display(
    total_samples: int, approaches: List[str], parallel: bool = False
) -> Progress:
    """Create a Rich progress display for experiments."""
    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console,
        expand=True,
    )

    return progress


def display_experiment_start(
    config: Dict[str, Any], approaches: List[str], total_samples: int
) -> None:
    """Display experiment start information."""
    console.print("\n🚀 [bold blue]Starting Experiment[/bold blue]")

    # Configuration summary
    config_table = Table(show_header=False, box=None, padding=(0, 1))
    config_table.add_column("Setting", style="dim")
    config_table.add_column("Value", style="cyan")

    config_table.add_row("Dataset:", config.get("dataset_name", "Unknown"))
    config_table.add_row(
        "Model:",
        f"{config.get('provider', 'Unknown')}/{config.get('model', 'Unknown')}",
    )
    config_table.add_row("Samples:", str(total_samples))
    config_table.add_row("Approaches:", ", ".join(approaches))
    config_table.add_row("Parallel:", "Yes" if config.get("parallel", False) else "No")

    console.print(config_table)
    console.print()


def display_experiment_complete(
    experiment_id: str, duration: float, total_cost: float, output_dir: str
) -> None:
    """Display experiment completion summary."""
    console.print("\n🎉 [bold green]Experiment Complete![/bold green]")

    summary_table = Table(show_header=False, box=None, padding=(0, 1))
    summary_table.add_column("Metric", style="dim")
    summary_table.add_column("Value", style="green")

    summary_table.add_row("Experiment ID:", experiment_id)
    summary_table.add_row("Duration:", f"{duration:.1f} seconds")
    summary_table.add_row("Total Cost:", f"${total_cost:.2f}")
    summary_table.add_row("Results Saved:", output_dir)

    console.print(summary_table)
    console.print()


def format_approach_name(approach: str) -> str:
    """Format reasoning approach name for display."""
    # Convert internal names to display names
    name_mapping = {
        "None": "Baseline (No Reasoning)",
        "ChainOfThought": "Chain of Thought",
        "ProgramOfThought": "Program of Thought",
        "AsPlanning": "Reasoning as Planning",
        "Reflection": "Reflection",
        "ChainOfVerification": "Chain of Verification",
        "SkeletonOfThought": "Skeleton of Thought",
        "TreeOfThought": "Tree of Thought",
    }

    return name_mapping.get(approach, approach)


def display_checkpoint_info(checkpoint_file: str, resume_from: str) -> None:
    """Display checkpoint resumption information."""
    console.print(f"📂 [blue]Resuming from checkpoint:[/blue] {checkpoint_file}")
    console.print(f"⏰ [dim]Last saved:[/dim] {resume_from}")
    console.print()


def display_validation_errors(errors: List[str]) -> None:
    """Display configuration validation errors."""
    console.print("\n❌ [bold red]Configuration Validation Failed:[/bold red]")

    for i, error in enumerate(errors, 1):
        console.print(f"  {i}. {error}")

    console.print("\n💡 [dim]Fix these issues and try again[/dim]")


def display_cost_warning(estimated_cost: float, threshold: float = 10.0) -> bool:
    """Display cost warning and get user confirmation."""
    if estimated_cost > threshold:
        console.print(f"\n⚠️  [yellow]High Cost Warning![/yellow]")
        console.print(f"   Estimated cost: [bold red]${estimated_cost:.2f}[/bold red]")

        confirm = typer.confirm("Do you want to continue?")
        if not confirm:
            console.print("❌ [red]Experiment cancelled by user[/red]")
            return False

    return True

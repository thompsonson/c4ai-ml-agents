"""CLI-specific validation utilities."""

from pathlib import Path
from typing import List, Optional

import typer
from rich.console import Console

from ml_agents.config import API_KEYS, SUPPORTED_MODELS
from ml_agents.reasoning import get_available_approaches

console = Console()


def validate_reasoning_approach(approach: str) -> str:
    """Validate that a reasoning approach is available."""
    available_approaches = get_available_approaches()

    if approach not in available_approaches:
        console.print(f"❌ [red]Invalid reasoning approach:[/red] {approach}")
        console.print(
            f"📋 [blue]Available approaches:[/blue] {', '.join(available_approaches)}"
        )
        raise typer.BadParameter(f"Invalid reasoning approach: {approach}")

    return approach


def validate_reasoning_approaches(approaches: str) -> List[str]:
    """Validate a comma-separated list of reasoning approaches."""
    approach_list = [a.strip() for a in approaches.split(",")]
    validated_approaches = []

    for approach in approach_list:
        validated_approaches.append(validate_reasoning_approach(approach))

    return validated_approaches


def validate_provider_model(provider: str, model: str) -> tuple[str, str]:
    """Validate provider and model combination."""
    if provider not in SUPPORTED_MODELS:
        console.print(f"❌ [red]Invalid provider:[/red] {provider}")
        console.print(
            f"📋 [blue]Supported providers:[/blue] {', '.join(SUPPORTED_MODELS.keys())}"
        )
        raise typer.BadParameter(f"Invalid provider: {provider}")

    if model not in SUPPORTED_MODELS[provider]:
        console.print(f"❌ [red]Invalid model for provider {provider}:[/red] {model}")
        console.print(
            f"📋 [blue]Supported models for {provider}:[/blue] {', '.join(SUPPORTED_MODELS[provider])}"
        )
        raise typer.BadParameter(f"Invalid model for provider {provider}: {model}")

    return provider, model


def validate_output_directory(output_dir: str) -> str:
    """Validate that output directory can be created/written to."""
    output_path = Path(output_dir)

    try:
        output_path.mkdir(parents=True, exist_ok=True)
        # Test write permissions
        test_file = output_path / ".test_write"
        test_file.write_text("test")
        test_file.unlink()
    except Exception as e:
        console.print(f"❌ [red]Cannot write to output directory:[/red] {output_dir}")
        console.print(f"   Error: {e}")
        raise typer.BadParameter(f"Cannot write to output directory: {output_dir}")

    return str(output_path.absolute())


def validate_api_key_available(provider: str) -> bool:
    """Check if API key is available for provider."""
    api_key = API_KEYS.get(provider)
    return bool(api_key and api_key != "your_key_here")


def validate_config_file(config_path: str) -> str:
    """Validate that config file exists and is readable."""
    config_file = Path(config_path)

    if not config_file.exists():
        console.print(f"❌ [red]Config file not found:[/red] {config_path}")
        raise typer.BadParameter(f"Config file not found: {config_path}")

    if config_file.suffix.lower() not in [".yaml", ".yml", ".json"]:
        console.print(
            f"❌ [red]Unsupported config file format:[/red] {config_file.suffix}"
        )
        console.print("📋 [blue]Supported formats:[/blue] .yaml, .yml, .json")
        raise typer.BadParameter(
            f"Unsupported config file format: {config_file.suffix}"
        )

    try:
        with open(config_file, "r") as f:
            f.read()
    except Exception as e:
        console.print(f"❌ [red]Cannot read config file:[/red] {e}")
        raise typer.BadParameter(f"Cannot read config file: {e}")

    return str(config_file.absolute())


def validate_checkpoint_file(checkpoint_path: str) -> str:
    """Validate that checkpoint file exists and is valid."""
    checkpoint_file = Path(checkpoint_path)

    if not checkpoint_file.exists():
        console.print(f"❌ [red]Checkpoint file not found:[/red] {checkpoint_path}")
        raise typer.BadParameter(f"Checkpoint file not found: {checkpoint_path}")

    if checkpoint_file.suffix.lower() != ".json":
        console.print(
            f"❌ [red]Invalid checkpoint file format:[/red] {checkpoint_file.suffix}"
        )
        console.print("📋 [blue]Expected format:[/blue] .json")
        raise typer.BadParameter(
            f"Invalid checkpoint file format: {checkpoint_file.suffix}"
        )

    try:
        import json

        with open(checkpoint_file, "r") as f:
            data = json.load(f)

        # Basic validation of checkpoint structure
        required_fields = ["experiment_id", "config", "progress"]
        for field in required_fields:
            if field not in data:
                raise ValueError(f"Missing required field: {field}")

    except Exception as e:
        console.print(f"❌ [red]Invalid checkpoint file:[/red] {e}")
        raise typer.BadParameter(f"Invalid checkpoint file: {e}")

    return str(checkpoint_file.absolute())


def validate_sample_count(sample_count: int) -> int:
    """Validate sample count is reasonable."""
    if sample_count < 1:
        raise typer.BadParameter("Sample count must be at least 1")

    if sample_count > 10000:
        console.print(f"⚠️  [yellow]Large sample count:[/yellow] {sample_count}")
        console.print("   This may take a long time and incur significant costs.")

        if not typer.confirm("Do you want to continue?"):
            raise typer.Abort()

    return sample_count


def validate_temperature(temperature: float) -> float:
    """Validate temperature parameter."""
    if not 0.0 <= temperature <= 2.0:
        raise typer.BadParameter("Temperature must be between 0.0 and 2.0")
    return temperature


def validate_top_p(top_p: float) -> float:
    """Validate top_p parameter."""
    if not 0.0 <= top_p <= 1.0:
        raise typer.BadParameter("top_p must be between 0.0 and 1.0")
    return top_p


def validate_max_tokens(max_tokens: int) -> int:
    """Validate max_tokens parameter.

    Range allows for modern models:
    - Claude Sonnet 4: up to 128K output tokens
    - GPT-4o Long Output: up to 64K output tokens
    - GPT-4o mini: up to 16K output tokens
    """
    if not 1 <= max_tokens <= 128000:
        raise typer.BadParameter("max_tokens must be between 1 and 128,000")
    return max_tokens


def validate_max_workers(max_workers: int) -> int:
    """Validate max_workers for parallel execution."""
    if max_workers < 1:
        raise typer.BadParameter("max_workers must be at least 1")

    if max_workers > 16:
        console.print(f"⚠️  [yellow]High worker count:[/yellow] {max_workers}")
        console.print("   This may overwhelm API rate limits.")

        if not typer.confirm("Do you want to continue?"):
            raise typer.Abort()

    return max_workers


def check_environment_ready(provider: str) -> None:
    """Check if environment is ready for the specified provider."""
    if not validate_api_key_available(provider):
        console.print(f"❌ [red]API key not found for provider:[/red] {provider}")
        console.print("💡 [blue]Set your API key in the .env file:[/blue]")

        if provider == "openrouter":
            console.print("   OPENROUTER_API_KEY=your_key_here")
        elif provider == "anthropic":
            console.print("   ANTHROPIC_API_KEY=your_key_here")
        elif provider == "cohere":
            console.print("   COHERE_API_KEY=your_key_here")

        raise typer.Exit(1)

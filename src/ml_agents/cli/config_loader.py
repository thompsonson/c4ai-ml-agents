"""Configuration loading and validation for CLI interface."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml
from pydantic import BaseModel, Field, ValidationError
from rich.console import Console

from ml_agents.config import ExperimentConfig

console = Console()


class CLIExperimentConfig(BaseModel):
    """Pydantic model for CLI configuration validation."""

    # Experiment settings
    experiment_name: Optional[str] = Field(None, description="Name for the experiment")
    sample_count: int = Field(50, ge=1, description="Number of samples to process")
    output_dir: str = Field("./outputs", description="Output directory for results")

    # Dataset settings
    dataset_name: str = Field(
        "MrLight/bbeh-eval", description="HuggingFace dataset name"
    )

    # Model settings
    provider: str = Field("openrouter", description="Model provider")
    model: str = Field("openai/gpt-oss-120b", description="Model name")
    temperature: float = Field(0.3, ge=0.0, le=2.0, description="Sampling temperature")
    max_tokens: int = Field(
        16384, ge=1, le=131000, description="Maximum tokens to generate"
    )
    top_p: float = Field(0.9, ge=0.0, le=1.0, description="Top-p sampling parameter")

    # Reasoning settings
    reasoning_approaches: List[str] = Field(
        default_factory=lambda: ["ChainOfThought"],
        description="List of reasoning approaches to test",
    )
    multi_step_reflection: bool = Field(
        False, description="Enable multi-step reflection"
    )
    multi_step_verification: bool = Field(
        False, description="Enable multi-step verification"
    )
    max_reasoning_calls: int = Field(5, ge=1, description="Maximum reasoning API calls")
    max_reflection_iterations: int = Field(
        2, ge=1, description="Maximum reflection iterations"
    )
    reflection_threshold: float = Field(
        0.7, ge=0.0, le=1.0, description="Reflection confidence threshold"
    )

    # Execution settings
    parallel: bool = Field(False, description="Enable parallel execution")
    max_workers: int = Field(4, ge=1, description="Maximum parallel workers")
    parallel_requests: int = Field(
        1, ge=1, description="Concurrent requests per approach"
    )
    retry_attempts: int = Field(3, ge=0, description="Number of retry attempts")
    request_timeout: int = Field(30, ge=1, description="Request timeout in seconds")

    # Output settings
    save_checkpoints: bool = Field(True, description="Save experiment checkpoints")
    output_format: List[str] = Field(
        default_factory=lambda: ["csv", "json"],
        description="Output formats (csv, json)",
    )

    # Parsing settings
    use_structured_parsing: bool = Field(
        True, description="Enable structured output parsing with Instructor"
    )
    fallback_to_regex: bool = Field(
        True, description="Enable fallback to regex parsing"
    )
    confidence_threshold: float = Field(
        0.7, ge=0.0, le=1.0, description="Minimum confidence threshold for parsing"
    )
    max_parsing_retries: int = Field(
        2, ge=0, description="Maximum number of parsing retry attempts"
    )

    def to_experiment_config(self) -> ExperimentConfig:
        """Convert to standard ExperimentConfig."""
        from ml_agents.config import ParsingConfig

        # Create parsing config
        parsing_config = ParsingConfig(
            use_structured_parsing=self.use_structured_parsing,
            fallback_to_regex=self.fallback_to_regex,
            confidence_threshold=self.confidence_threshold,
            max_parsing_retries=self.max_parsing_retries,
        )

        return ExperimentConfig(
            dataset_name=self.dataset_name,
            sample_count=self.sample_count,
            provider=self.provider,
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            top_p=self.top_p,
            reasoning_approaches=self.reasoning_approaches,
            output_dir=self.output_dir,
            parallel_requests=self.parallel_requests,
            retry_attempts=self.retry_attempts,
            request_timeout=self.request_timeout,
            multi_step_reflection=self.multi_step_reflection,
            multi_step_verification=self.multi_step_verification,
            max_reasoning_calls=self.max_reasoning_calls,
            max_reflection_iterations=self.max_reflection_iterations,
            reflection_threshold=self.reflection_threshold,
            parsing=parsing_config,
        )


def load_config_file(config_path: Union[str, Path]) -> Dict[str, Any]:
    """Load configuration from YAML or JSON file."""
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r") as f:
        if config_path.suffix.lower() in [".yaml", ".yml"]:
            return yaml.safe_load(f)
        elif config_path.suffix.lower() == ".json":
            return json.load(f)
        else:
            raise ValueError(f"Unsupported config file format: {config_path.suffix}")


def flatten_nested_config(nested_config: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten nested configuration to match CLIExperimentConfig structure."""
    flattened = {}

    # Handle experiment section
    if "experiment" in nested_config:
        exp = nested_config["experiment"]
        if "name" in exp:
            flattened["experiment_name"] = exp["name"]
        if "sample_count" in exp:
            flattened["sample_count"] = exp["sample_count"]
        if "output_dir" in exp:
            flattened["output_dir"] = exp["output_dir"]

    # Handle dataset section
    if "dataset" in nested_config:
        dataset = nested_config["dataset"]
        if "name" in dataset:
            flattened["dataset_name"] = dataset["name"]

    # Handle model section
    if "model" in nested_config:
        model = nested_config["model"]
        for key in ["provider", "name", "temperature", "max_tokens", "top_p"]:
            if key in model:
                if key == "name":
                    flattened["model"] = model[key]
                else:
                    flattened[key] = model[key]

    # Handle reasoning section
    if "reasoning" in nested_config:
        reasoning = nested_config["reasoning"]
        if "approaches" in reasoning:
            flattened["reasoning_approaches"] = reasoning["approaches"]
        for key in [
            "multi_step_reflection",
            "multi_step_verification",
            "max_reasoning_calls",
            "max_reflection_iterations",
            "reflection_threshold",
        ]:
            if key in reasoning:
                flattened[key] = reasoning[key]

    # Handle execution section
    if "execution" in nested_config:
        execution = nested_config["execution"]
        for key in [
            "parallel",
            "max_workers",
            "parallel_requests",
            "retry_attempts",
            "request_timeout",
            "save_checkpoints",
        ]:
            if key in execution:
                flattened[key] = execution[key]

    # Handle output section
    if "output" in nested_config:
        output = nested_config["output"]
        if "formats" in output:
            flattened["output_format"] = output["formats"]

    # Handle any top-level keys (for backward compatibility)
    for key, value in nested_config.items():
        if key not in [
            "experiment",
            "dataset",
            "model",
            "reasoning",
            "execution",
            "output",
        ]:
            flattened[key] = value

    return flattened


def merge_config_sources(
    cli_args: Dict[str, Any],
    config_file: Optional[Dict[str, Any]] = None,
    base_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Merge configuration from multiple sources with priority: CLI > config file > base config."""

    # Start with base config (lowest priority)
    merged_config = base_config.copy() if base_config else {}

    # Override with config file (medium priority)
    if config_file:
        # Check if config file is nested structure (values must be dictionaries)
        if any(
            key in config_file and isinstance(config_file[key], dict)
            for key in ["experiment", "model", "reasoning", "execution"]
        ):
            # Flatten nested structure
            flat_config = flatten_nested_config(config_file)
            merged_config.update(
                {k: v for k, v in flat_config.items() if v is not None}
            )
        else:
            # Use flat structure as-is
            merged_config.update(
                {k: v for k, v in config_file.items() if v is not None}
            )

    # Override with CLI args (highest priority)
    merged_config.update({k: v for k, v in cli_args.items() if v is not None})

    return merged_config


def validate_cli_config(config_dict: Dict[str, Any]) -> CLIExperimentConfig:
    """Validate CLI configuration using Pydantic."""
    try:
        return CLIExperimentConfig(**config_dict)
    except ValidationError as e:
        errors = []
        for error in e.errors():
            field = error["loc"][-1] if error["loc"] else "unknown"
            message = error["msg"]
            errors.append(f"{field}: {message}")

        console.print("\n❌ [bold red]Configuration validation failed:[/bold red]")
        for error in errors:
            console.print(f"  • {error}")

        raise ValueError("Configuration validation failed")


def load_and_validate_config(
    config_file: Optional[str] = None, **cli_overrides: Any
) -> ExperimentConfig:
    """Load and validate configuration from all sources."""

    # Load config file if provided
    file_config = None
    if config_file:
        try:
            file_config = load_config_file(config_file)
            console.print(f"📝 [green]Loaded config from:[/green] {config_file}")
        except Exception as e:
            console.print(f"❌ [red]Failed to load config file:[/red] {e}")
            raise

    # Get base configuration (from defaults)
    base_config = CLIExperimentConfig().model_dump()

    # Filter out None values from CLI overrides
    cli_args = {k: v for k, v in cli_overrides.items() if v is not None}

    # Merge all configuration sources
    merged_config = merge_config_sources(cli_args, file_config, base_config)

    # Validate the merged configuration
    cli_config = validate_cli_config(merged_config)

    # Convert to standard ExperimentConfig
    experiment_config = cli_config.to_experiment_config()

    return experiment_config


def create_example_config() -> Dict[str, Any]:
    """Create an example configuration dictionary."""
    return {
        "experiment": {
            "name": "reasoning_comparison_study",
            "sample_count": 100,
            "output_dir": "./results",
        },
        "dataset": {"name": "MrLight/bbeh-eval"},
        "model": {
            "provider": "openrouter",
            "name": "openai/gpt-oss-120b",
            "temperature": 0.3,
            "max_tokens": 512,
            "top_p": 0.9,
        },
        "reasoning": {
            "approaches": ["ChainOfThought", "AsPlanning", "TreeOfThought"],
            "multi_step_verification": True,
            "max_reasoning_calls": 5,
        },
        "execution": {"parallel": True, "max_workers": 4, "save_checkpoints": True},
        "output": {"formats": ["csv", "json"]},
    }


def save_config_template(output_path: Union[str, Path], format: str = "yaml") -> None:
    """Save an example configuration file."""
    config = create_example_config()
    output_path = Path(output_path)

    with open(output_path, "w") as f:
        if format.lower() == "yaml":
            yaml.dump(config, f, default_flow_style=False, indent=2)
        elif format.lower() == "json":
            json.dump(config, f, indent=2)
        else:
            raise ValueError(f"Unsupported format: {format}")

    console.print(f"📝 [green]Config template saved to:[/green] {output_path}")

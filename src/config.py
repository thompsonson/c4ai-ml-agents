"""Configuration module for ML Agents experiments."""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# API Keys dictionary - maintain backward compatibility
API_KEYS = {
    "anthropic": os.getenv("ANTHROPIC_API_KEY"),
    "cohere": os.getenv("COHERE_API_KEY"),
    "openrouter": os.getenv("OPENROUTER_API_KEY"),
    "huggingface": os.getenv("HUGGINGFACE_API_KEY"),
}

# Supported providers and their models
SUPPORTED_MODELS = {
    "huggingface": [
        "google/gemma-2-2b-it",
        "mistralai/Mistral-7B-Instruct-v0.2",
        "meta-llama/Meta-Llama-3-8B-Instruct",
    ],
    "anthropic": [
        "claude-opus-4-1-20250805",
        "claude-sonnet-4-20250514",
        "claude-3-5-haiku-20241022",
    ],
    "cohere": ["command-r-plus", "command-r", "command-light"],
    "openrouter": [
        "openai/gpt-5-chat",
        "openai/gpt-5-mini",
        "openai/gpt-oss-120b",
        "google/gemini-2.5-flash-lite",
    ],
}

# Supported reasoning approaches
SUPPORTED_REASONING = [
    "None",
    "Chain-of-Thought (CoT)",
    "Program-of-Thought (PoT)",
    "Reasoning-as-Planning (RAP)",
    "Reflection",
    "Chain-of-Verification (CoVe)",
    "Skeleton-of-Thought (SoT)",
    "Tree-of-Thought (ToT)",
    "Graph-of-Thought (GoT)",
    "ReWOO",
    "Buffer-of-Thoughts (BoT)",
]


def get_api_key(provider: str) -> Optional[str]:
    """Get API key for a specific provider.

    Args:
        provider: The provider name (anthropic, cohere, openrouter, huggingface)

    Returns:
        The API key string or None if not found
    """
    return API_KEYS.get(provider)


def validate_api_keys() -> Dict[str, bool]:
    """Validate that all API keys are set.

    Returns:
        Dictionary with provider names as keys and boolean values
        indicating if key is set
    """
    return {
        provider: bool(key) and key != "your_key_here"
        for provider, key in API_KEYS.items()
    }


@dataclass
class ExperimentConfig:
    """Configuration for ML Agents experiments with validation."""

    # Dataset configuration
    dataset_name: str = "MrLight/bbeh-eval"
    sample_count: int = 50

    # Model configuration
    provider: str = "openrouter"
    model: str = "openai/gpt-oss-120b"
    temperature: float = 0.3
    max_tokens: int = 512
    top_p: float = 0.9

    # Experiment configuration
    reasoning_approaches: List[str] = field(default_factory=lambda: ["None"])
    output_dir: str = "./outputs"

    # Advanced configuration
    parallel_requests: int = 1
    retry_attempts: int = 3
    request_timeout: int = 30

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        self.validate()

    def validate(self) -> None:
        """Validate all configuration parameters."""
        errors = []

        # Validate dataset
        if not self.dataset_name or not self.dataset_name.strip():
            errors.append("dataset_name cannot be empty")

        if self.sample_count < 1:
            errors.append("sample_count must be >= 1")

        # Validate provider and model
        if self.provider not in SUPPORTED_MODELS:
            errors.append(f"provider must be one of {list(SUPPORTED_MODELS.keys())}")
        elif self.model not in SUPPORTED_MODELS[self.provider]:
            errors.append(
                f"model '{self.model}' not supported for provider '{self.provider}'"
            )

        # Validate model parameters
        if not (0.0 <= self.temperature <= 2.0):
            errors.append("temperature must be between 0.0 and 2.0")

        if not (1 <= self.max_tokens <= 4096):
            errors.append("max_tokens must be between 1 and 4096")

        if not (0.0 <= self.top_p <= 1.0):
            errors.append("top_p must be between 0.0 and 1.0")

        # Validate reasoning approaches
        if not self.reasoning_approaches:
            errors.append("reasoning_approaches cannot be empty")

        invalid_approaches = [
            approach
            for approach in self.reasoning_approaches
            if approach not in SUPPORTED_REASONING
        ]
        if invalid_approaches:
            errors.append(f"Invalid reasoning approaches: {invalid_approaches}")

        # Validate output directory
        if not self.output_dir or not self.output_dir.strip():
            errors.append("output_dir cannot be empty")

        # Validate advanced parameters
        if self.parallel_requests < 1:
            errors.append("parallel_requests must be >= 1")

        if self.retry_attempts < 0:
            errors.append("retry_attempts must be >= 0")

        if self.request_timeout < 1:
            errors.append("request_timeout must be >= 1")

        if errors:
            raise ValueError(f"Configuration validation failed: {'; '.join(errors)}")

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "ExperimentConfig":
        """Create config from dictionary."""
        return cls(**config_dict)

    @classmethod
    def from_yaml(cls, yaml_path: Union[str, Path]) -> "ExperimentConfig":
        """Load configuration from YAML file."""
        with open(yaml_path, "r") as f:
            config_dict = yaml.safe_load(f)
        return cls.from_dict(config_dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "dataset_name": self.dataset_name,
            "sample_count": self.sample_count,
            "provider": self.provider,
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "reasoning_approaches": self.reasoning_approaches,
            "output_dir": self.output_dir,
            "parallel_requests": self.parallel_requests,
            "retry_attempts": self.retry_attempts,
            "request_timeout": self.request_timeout,
        }

    def to_yaml(self, yaml_path: Union[str, Path]) -> None:
        """Save configuration to YAML file."""
        with open(yaml_path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)

    def update_from_args(self, args: Any) -> None:
        """Update configuration from command line arguments."""
        if hasattr(args, "dataset_name") and args.dataset_name:
            self.dataset_name = args.dataset_name
        if hasattr(args, "sample_count") and args.sample_count:
            self.sample_count = args.sample_count
        if hasattr(args, "provider") and args.provider:
            self.provider = args.provider
        if hasattr(args, "model") and args.model:
            self.model = args.model
        if hasattr(args, "temperature") and args.temperature is not None:
            self.temperature = args.temperature
        if hasattr(args, "max_tokens") and args.max_tokens:
            self.max_tokens = args.max_tokens
        if hasattr(args, "top_p") and args.top_p is not None:
            self.top_p = args.top_p
        if hasattr(args, "reasoning_approaches") and args.reasoning_approaches:
            self.reasoning_approaches = args.reasoning_approaches
        if hasattr(args, "output_dir") and args.output_dir:
            self.output_dir = args.output_dir
        if hasattr(args, "parallel_requests") and args.parallel_requests:
            self.parallel_requests = args.parallel_requests
        if hasattr(args, "retry_attempts") and args.retry_attempts is not None:
            self.retry_attempts = args.retry_attempts
        if hasattr(args, "request_timeout") and args.request_timeout:
            self.request_timeout = args.request_timeout

        # Re-validate after updates
        self.validate()


def get_default_config() -> ExperimentConfig:
    """Get default experiment configuration."""
    return ExperimentConfig()


def validate_environment() -> Dict[str, bool]:
    """Validate that the environment is properly configured."""
    validation_results = {
        "api_keys": all(validate_api_keys().values()),
        "output_dir_writable": True,
        "dependencies_available": True,
    }

    # Check if we can create output directory
    try:
        output_dir = Path("./outputs")
        output_dir.mkdir(exist_ok=True)
        test_file = output_dir / "test_write.tmp"
        test_file.write_text("test")
        test_file.unlink()
    except Exception:
        validation_results["output_dir_writable"] = False

    # Check critical dependencies
    try:
        import datasets  # noqa: F401
        import pandas  # noqa: F401
        import tqdm  # noqa: F401
    except ImportError:
        validation_results["dependencies_available"] = False

    return validation_results

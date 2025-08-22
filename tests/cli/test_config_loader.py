"""Tests for CLI configuration loading and validation."""

import json
import tempfile
from pathlib import Path
from typing import Any, Dict

import pytest
import yaml

from ml_agents.cli.config_loader import (
    CLIExperimentConfig,
    create_example_config,
    flatten_nested_config,
    load_and_validate_config,
    load_config_file,
    merge_config_sources,
    save_config_template,
    validate_cli_config,
)
from ml_agents.config import ExperimentConfig


class TestCLIExperimentConfig:
    """Test CLIExperimentConfig Pydantic model."""

    def test_default_config_creation(self):
        """Test creating config with defaults."""
        config = CLIExperimentConfig()

        assert config.sample_count == 50
        assert config.provider == "openrouter"
        assert config.temperature == 0.3
        assert config.reasoning_approaches == ["ChainOfThought"]
        assert config.multi_step_reflection is False
        assert config.parallel is False

    def test_config_validation_valid_values(self):
        """Test validation with valid values."""
        config = CLIExperimentConfig(
            sample_count=100,
            temperature=1.0,
            max_tokens=1000,
            top_p=0.8,
            reasoning_approaches=["ChainOfThought", "AsPlanning"],
            multi_step_verification=True,
            max_workers=8,
        )

        assert config.sample_count == 100
        assert config.temperature == 1.0
        assert config.max_tokens == 1000
        assert config.top_p == 0.8
        assert len(config.reasoning_approaches) == 2
        assert config.multi_step_verification is True
        assert config.max_workers == 8

    def test_config_validation_invalid_temperature(self):
        """Test validation fails for invalid temperature."""
        with pytest.raises(
            ValueError, match="ensure this value is less than or equal to 2"
        ):
            CLIExperimentConfig(temperature=3.0)

    def test_config_validation_invalid_top_p(self):
        """Test validation fails for invalid top_p."""
        with pytest.raises(
            ValueError, match="ensure this value is less than or equal to 1"
        ):
            CLIExperimentConfig(top_p=1.5)

    def test_config_validation_invalid_sample_count(self):
        """Test validation fails for invalid sample count."""
        with pytest.raises(
            ValueError, match="ensure this value is greater than or equal to 1"
        ):
            CLIExperimentConfig(sample_count=0)

    def test_to_experiment_config_conversion(self):
        """Test conversion to ExperimentConfig."""
        cli_config = CLIExperimentConfig(
            sample_count=25,
            provider="anthropic",
            model="claude-sonnet-4-20250514",
            temperature=0.5,
            reasoning_approaches=["Reflection", "ChainOfThought"],
        )

        exp_config = cli_config.to_experiment_config()

        assert isinstance(exp_config, ExperimentConfig)
        assert exp_config.sample_count == 25
        assert exp_config.provider == "anthropic"
        assert exp_config.model == "claude-sonnet-4-20250514"
        assert exp_config.temperature == 0.5
        assert exp_config.reasoning_approaches == ["Reflection", "ChainOfThought"]


class TestConfigFileLoading:
    """Test configuration file loading functions."""

    def test_load_yaml_config(self):
        """Test loading YAML configuration file."""
        config_data = {
            "sample_count": 75,
            "provider": "cohere",
            "temperature": 0.7,
            "reasoning_approaches": ["TreeOfThought", "AsPlanning"],
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            config_path = f.name

        try:
            loaded_config = load_config_file(config_path)

            assert loaded_config["sample_count"] == 75
            assert loaded_config["provider"] == "cohere"
            assert loaded_config["temperature"] == 0.7
            assert loaded_config["reasoning_approaches"] == [
                "TreeOfThought",
                "AsPlanning",
            ]
        finally:
            Path(config_path).unlink()

    def test_load_json_config(self):
        """Test loading JSON configuration file."""
        config_data = {
            "sample_count": 30,
            "provider": "anthropic",
            "max_tokens": 2048,
            "parallel": True,
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config_data, f)
            config_path = f.name

        try:
            loaded_config = load_config_file(config_path)

            assert loaded_config["sample_count"] == 30
            assert loaded_config["provider"] == "anthropic"
            assert loaded_config["max_tokens"] == 2048
            assert loaded_config["parallel"] is True
        finally:
            Path(config_path).unlink()

    def test_load_config_file_not_found(self):
        """Test loading non-existent config file."""
        with pytest.raises(FileNotFoundError, match="Configuration file not found"):
            load_config_file("/nonexistent/config.yaml")

    def test_load_config_unsupported_format(self):
        """Test loading config file with unsupported format."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("some content")
            config_path = f.name

        try:
            with pytest.raises(ValueError, match="Unsupported config file format"):
                load_config_file(config_path)
        finally:
            Path(config_path).unlink()


class TestNestedConfigFlattening:
    """Test nested configuration flattening functionality."""

    def test_flatten_nested_experiment_config(self):
        """Test flattening nested experiment configuration."""
        nested_config = {
            "experiment": {
                "name": "test_experiment",
                "sample_count": 100,
                "output_dir": "./test_results",
            },
            "dataset": {"name": "custom/dataset"},
        }

        flattened = flatten_nested_config(nested_config)

        assert flattened["experiment_name"] == "test_experiment"
        assert flattened["sample_count"] == 100
        assert flattened["output_dir"] == "./test_results"
        assert flattened["dataset_name"] == "custom/dataset"

    def test_flatten_nested_model_config(self):
        """Test flattening nested model configuration."""
        nested_config = {
            "model": {
                "provider": "openrouter",
                "name": "gpt-4",
                "temperature": 0.8,
                "max_tokens": 1500,
                "top_p": 0.95,
            }
        }

        flattened = flatten_nested_config(nested_config)

        assert flattened["provider"] == "openrouter"
        assert flattened["model"] == "gpt-4"  # "name" becomes "model"
        assert flattened["temperature"] == 0.8
        assert flattened["max_tokens"] == 1500
        assert flattened["top_p"] == 0.95

    def test_flatten_nested_reasoning_config(self):
        """Test flattening nested reasoning configuration."""
        nested_config = {
            "reasoning": {
                "approaches": ["ChainOfThought", "Reflection"],
                "multi_step_verification": True,
                "max_reasoning_calls": 5,
                "reflection_threshold": 0.8,
            }
        }

        flattened = flatten_nested_config(nested_config)

        assert flattened["reasoning_approaches"] == ["ChainOfThought", "Reflection"]
        assert flattened["multi_step_verification"] is True
        assert flattened["max_reasoning_calls"] == 5
        assert flattened["reflection_threshold"] == 0.8

    def test_flatten_nested_execution_config(self):
        """Test flattening nested execution configuration."""
        nested_config = {
            "execution": {
                "parallel": True,
                "max_workers": 6,
                "retry_attempts": 5,
                "save_checkpoints": False,
            }
        }

        flattened = flatten_nested_config(nested_config)

        assert flattened["parallel"] is True
        assert flattened["max_workers"] == 6
        assert flattened["retry_attempts"] == 5
        assert flattened["save_checkpoints"] is False

    def test_flatten_mixed_nested_and_flat_config(self):
        """Test flattening config with both nested and flat keys."""
        nested_config = {
            "sample_count": 50,  # Top-level (flat)
            "model": {"provider": "anthropic", "name": "claude-3"},  # Nested
            "parallel": True,  # Top-level (flat)
        }

        flattened = flatten_nested_config(nested_config)

        assert flattened["sample_count"] == 50
        assert flattened["provider"] == "anthropic"
        assert flattened["model"] == "claude-3"
        assert flattened["parallel"] is True


class TestConfigMerging:
    """Test configuration merging from multiple sources."""

    def test_merge_cli_over_config_file(self):
        """Test CLI arguments override config file values."""
        cli_args = {"sample_count": 100, "temperature": 0.5}
        config_file = {"sample_count": 50, "provider": "cohere", "temperature": 0.3}
        base_config = {"sample_count": 25, "provider": "openrouter", "max_tokens": 512}

        merged = merge_config_sources(cli_args, config_file, base_config)

        assert merged["sample_count"] == 100  # CLI wins
        assert merged["temperature"] == 0.5  # CLI wins
        assert merged["provider"] == "cohere"  # Config file wins
        assert merged["max_tokens"] == 512  # Base config used

    def test_merge_config_file_over_base(self):
        """Test config file overrides base config."""
        cli_args = {}
        config_file = {"provider": "anthropic", "max_tokens": 1000}
        base_config = {"provider": "openrouter", "max_tokens": 512, "temperature": 0.3}

        merged = merge_config_sources(cli_args, config_file, base_config)

        assert merged["provider"] == "anthropic"  # Config file wins
        assert merged["max_tokens"] == 1000  # Config file wins
        assert merged["temperature"] == 0.3  # Base config used

    def test_merge_with_none_values_filtered(self):
        """Test that None values are filtered out during merge."""
        cli_args = {"sample_count": None, "temperature": 0.5, "provider": None}
        config_file = {"sample_count": 75, "provider": "cohere"}

        merged = merge_config_sources(cli_args, config_file)

        assert merged["sample_count"] == 75  # None filtered, config file used
        assert merged["temperature"] == 0.5  # CLI used
        assert merged["provider"] == "cohere"  # None filtered, config file used

    def test_merge_with_nested_config_file(self):
        """Test merging with nested config file structure."""
        cli_args = {"temperature": 0.8}
        config_file = {
            "model": {"provider": "anthropic", "temperature": 0.3, "max_tokens": 2000},
            "reasoning": {"approaches": ["TreeOfThought"]},
        }

        merged = merge_config_sources(cli_args, config_file)

        assert merged["temperature"] == 0.8  # CLI wins
        assert merged["provider"] == "anthropic"  # From nested config
        assert merged["max_tokens"] == 2000  # From nested config
        assert merged["reasoning_approaches"] == ["TreeOfThought"]  # Flattened


class TestConfigValidation:
    """Test CLI configuration validation."""

    def test_validate_valid_config(self):
        """Test validation of valid configuration."""
        config_dict = {
            "sample_count": 50,
            "provider": "openrouter",
            "temperature": 0.3,
            "reasoning_approaches": ["ChainOfThought"],
        }

        validated = validate_cli_config(config_dict)

        assert isinstance(validated, CLIExperimentConfig)
        assert validated.sample_count == 50
        assert validated.provider == "openrouter"

    def test_validate_config_with_errors(self):
        """Test validation fails with meaningful error messages."""
        config_dict = {
            "sample_count": -5,  # Invalid: must be >= 1
            "temperature": 3.0,  # Invalid: must be <= 2.0
            "max_tokens": 5000,  # Invalid: must be <= 4096
            "top_p": 1.5,  # Invalid: must be <= 1.0
        }

        with pytest.raises(ValueError, match="Configuration validation failed"):
            validate_cli_config(config_dict)


class TestFullConfigLoading:
    """Test the complete configuration loading and validation process."""

    def test_load_and_validate_config_file_only(self):
        """Test loading config from file only."""
        config_data = {
            "model": {
                "provider": "cohere",
                "name": "command-r-plus",
                "temperature": 0.6,
            },
            "reasoning": {"approaches": ["AsPlanning", "TreeOfThought"]},
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            config_path = f.name

        try:
            result = load_and_validate_config(config_file=config_path)

            assert isinstance(result, ExperimentConfig)
            assert result.provider == "cohere"
            assert result.model == "command-r-plus"
            assert result.temperature == 0.6
            assert result.reasoning_approaches == ["AsPlanning", "TreeOfThought"]
        finally:
            Path(config_path).unlink()

    def test_load_and_validate_cli_overrides(self):
        """Test CLI arguments override config file."""
        config_data = {"sample_count": 25, "provider": "anthropic", "temperature": 0.2}

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config_data, f)
            config_path = f.name

        try:
            result = load_and_validate_config(
                config_file=config_path,
                sample_count=100,  # CLI override
                temperature=0.8,  # CLI override
                reasoning_approaches=["Reflection"],  # CLI override
            )

            assert result.sample_count == 100  # CLI wins
            assert result.temperature == 0.8  # CLI wins
            assert result.provider == "anthropic"  # Config file used
            assert result.reasoning_approaches == ["Reflection"]  # CLI wins
        finally:
            Path(config_path).unlink()

    def test_load_and_validate_defaults_only(self):
        """Test loading with defaults when no config file provided."""
        result = load_and_validate_config()

        assert isinstance(result, ExperimentConfig)
        assert result.sample_count == 50  # Default
        assert result.provider == "openrouter"  # Default
        assert result.reasoning_approaches == ["ChainOfThought"]  # Default


class TestConfigTemplates:
    """Test configuration template creation and saving."""

    def test_create_example_config(self):
        """Test creating example configuration."""
        config = create_example_config()

        assert "experiment" in config
        assert "model" in config
        assert "reasoning" in config
        assert "execution" in config

        assert config["experiment"]["sample_count"] == 100
        assert config["model"]["provider"] == "openrouter"
        assert "ChainOfThought" in config["reasoning"]["approaches"]
        assert config["execution"]["parallel"] is True

    def test_save_yaml_template(self):
        """Test saving YAML configuration template."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            output_path = f.name

        try:
            save_config_template(output_path, format="yaml")

            # Verify file was created and contains expected structure
            with open(output_path, "r") as f:
                saved_config = yaml.safe_load(f)

            assert "experiment" in saved_config
            assert "model" in saved_config
            assert saved_config["model"]["provider"] == "openrouter"
        finally:
            Path(output_path).unlink()

    def test_save_json_template(self):
        """Test saving JSON configuration template."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            output_path = f.name

        try:
            save_config_template(output_path, format="json")

            # Verify file was created and contains expected structure
            with open(output_path, "r") as f:
                saved_config = json.load(f)

            assert "experiment" in saved_config
            assert "reasoning" in saved_config
            assert "ChainOfThought" in saved_config["reasoning"]["approaches"]
        finally:
            Path(output_path).unlink()

    def test_save_template_unsupported_format(self):
        """Test saving template with unsupported format."""
        with pytest.raises(ValueError, match="Unsupported format"):
            save_config_template("config.txt", format="txt")

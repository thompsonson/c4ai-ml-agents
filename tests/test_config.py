"""Tests for configuration module."""

import os
from unittest.mock import MagicMock, patch

import pytest
import yaml

from ml_agents.config import (
    SUPPORTED_MODELS,
    SUPPORTED_REASONING,
    ExperimentConfig,
    get_api_key,
    get_default_config,
    validate_api_keys,
    validate_environment,
)


class TestAPIKeyFunctions:
    """Test API key related functions."""

    def test_get_api_key_existing(self, mock_env_vars):
        """Test getting an existing API key."""
        key = get_api_key("anthropic")
        assert key == "test-anthropic-key"

    def test_get_api_key_nonexistent(self):
        """Test getting a non-existent API key."""
        key = get_api_key("nonexistent")
        assert key is None

    def test_validate_api_keys_all_valid(self, mock_env_vars):
        """Test validation when all API keys are valid."""
        result = validate_api_keys()
        assert all(result.values())

    def test_validate_api_keys_some_missing(self):
        """Test validation when some API keys are missing."""
        # Mock API_KEYS with some missing
        mock_keys = {
            "anthropic": "",  # Empty string should be invalid
            "cohere": "valid-key",
            "openrouter": "valid-key",
            "huggingface": "valid-key",
        }
        with patch("ml_agents.config.API_KEYS", mock_keys):
            result = validate_api_keys()
            assert not result["anthropic"]
            assert result["cohere"]

    def test_validate_api_keys_default_value(self):
        """Test validation rejects default placeholder values."""
        # Mock API_KEYS with default placeholder
        mock_keys = {
            "anthropic": "your_key_here",  # Default placeholder should be invalid
            "cohere": "valid-key",
            "openrouter": "valid-key",
            "huggingface": "valid-key",
        }
        with patch("ml_agents.config.API_KEYS", mock_keys):
            result = validate_api_keys()
            assert not result["anthropic"]  # Should be False for placeholder value


class TestExperimentConfig:
    """Test ExperimentConfig class."""

    def test_default_config(self):
        """Test creating config with default values."""
        config = ExperimentConfig()
        assert config.dataset_name == "MrLight/bbeh-eval"
        assert config.sample_count == 50
        assert config.provider == "openrouter"
        assert config.temperature == 0.3
        assert config.reasoning_approaches == ["None"]

    def test_config_with_custom_values(self, experiment_config_data):
        """Test creating config with custom values."""
        config = ExperimentConfig(**experiment_config_data)
        assert config.dataset_name == experiment_config_data["dataset_name"]
        assert config.sample_count == experiment_config_data["sample_count"]
        assert config.provider == experiment_config_data["provider"]

    def test_config_validation_success(self, experiment_config_data):
        """Test successful configuration validation."""
        # Should not raise any exception
        config = ExperimentConfig(**experiment_config_data)
        assert config is not None

    def test_config_validation_invalid_temperature(self):
        """Test validation fails for invalid temperature."""
        with pytest.raises(ValueError, match="temperature must be between"):
            ExperimentConfig(temperature=3.0)

    def test_config_validation_invalid_top_p(self):
        """Test validation fails for invalid top_p."""
        with pytest.raises(ValueError, match="top_p must be between"):
            ExperimentConfig(top_p=1.5)

    def test_config_validation_invalid_max_tokens(self):
        """Test validation fails for invalid max_tokens."""
        with pytest.raises(ValueError, match="max_tokens must be between"):
            ExperimentConfig(max_tokens=0)

    def test_config_validation_invalid_provider(self):
        """Test validation fails for invalid provider."""
        with pytest.raises(ValueError, match="provider must be one of"):
            ExperimentConfig(provider="invalid_provider")

    def test_config_validation_invalid_model_for_provider(self):
        """Test validation fails for invalid model/provider combination."""
        with pytest.raises(ValueError, match="model .* not supported"):
            ExperimentConfig(provider="anthropic", model="openai/gpt-4")

    def test_config_validation_invalid_reasoning_approach(self):
        """Test validation fails for invalid reasoning approach."""
        with pytest.raises(ValueError, match="Invalid reasoning approaches"):
            ExperimentConfig(reasoning_approaches=["Invalid Approach"])

    def test_config_validation_empty_dataset_name(self):
        """Test validation fails for empty dataset name."""
        with pytest.raises(ValueError, match="dataset_name cannot be empty"):
            ExperimentConfig(dataset_name="")

    def test_config_validation_negative_sample_count(self):
        """Test validation fails for negative sample count."""
        with pytest.raises(ValueError, match="sample_count must be >= 1"):
            ExperimentConfig(sample_count=-1)

    def test_config_validation_empty_reasoning_approaches(self):
        """Test validation fails for empty reasoning approaches."""
        with pytest.raises(ValueError, match="reasoning_approaches cannot be empty"):
            ExperimentConfig(reasoning_approaches=[])

    def test_config_to_dict(self, experiment_config_data):
        """Test converting config to dictionary."""
        config = ExperimentConfig(**experiment_config_data)
        config_dict = config.to_dict()

        for key, value in experiment_config_data.items():
            assert config_dict[key] == value

    def test_config_from_dict(self, experiment_config_data):
        """Test creating config from dictionary."""
        config = ExperimentConfig.from_dict(experiment_config_data)
        assert config.dataset_name == experiment_config_data["dataset_name"]
        assert config.provider == experiment_config_data["provider"]

    def test_config_yaml_round_trip(self, experiment_config_data, temp_dir):
        """Test saving and loading config from YAML."""
        config = ExperimentConfig(**experiment_config_data)
        yaml_path = temp_dir / "config.yaml"

        # Save to YAML
        config.to_yaml(yaml_path)
        assert yaml_path.exists()

        # Load from YAML
        loaded_config = ExperimentConfig.from_yaml(yaml_path)
        assert loaded_config.to_dict() == config.to_dict()

    def test_config_update_from_args(self):
        """Test updating config from command line arguments."""
        config = ExperimentConfig()

        # Create a simple object with attributes instead of MagicMock
        class Args:
            dataset_name = "new/dataset"
            temperature = 0.8
            provider = "anthropic"
            model = "claude-sonnet-4-20250514"
            sample_count = None  # Not set
            max_tokens = None
            top_p = None
            reasoning_approaches = None
            output_dir = None
            parallel_requests = None
            retry_attempts = None
            request_timeout = None

        args = Args()
        config.update_from_args(args)

        assert config.dataset_name == "new/dataset"
        assert config.temperature == 0.8
        assert config.provider == "anthropic"
        assert config.model == "claude-sonnet-4-20250514"

    def test_config_update_from_args_invalid(self):
        """Test updating config with invalid args triggers validation."""
        config = ExperimentConfig()

        # Create a simple object with invalid temperature
        class Args:
            temperature = 5.0  # Invalid - exceeds 2.0 max
            dataset_name = None
            sample_count = None
            provider = None
            model = None
            max_tokens = None
            top_p = None
            reasoning_approaches = None
            output_dir = None
            parallel_requests = None
            retry_attempts = None
            request_timeout = None

        args = Args()
        with pytest.raises(ValueError):
            config.update_from_args(args)


class TestConfigurationValidation:
    """Test configuration validation functions."""

    def test_get_default_config(self):
        """Test getting default configuration."""
        config = get_default_config()
        assert isinstance(config, ExperimentConfig)
        assert config.dataset_name == "MrLight/bbeh-eval"

    @patch("ml_agents.config.Path")
    @patch("builtins.open")
    def test_validate_environment_success(self, mock_open, mock_path, mock_env_vars):
        """Test environment validation success."""
        # Mock file operations
        mock_path.return_value.mkdir.return_value = None
        mock_path.return_value.__truediv__.return_value.write_text.return_value = None
        mock_path.return_value.__truediv__.return_value.unlink.return_value = None

        result = validate_environment()

        assert result["api_keys"] is True
        assert result["output_dir_writable"] is True
        assert result["dependencies_available"] is True

    def test_supported_models_structure(self):
        """Test supported models dictionary structure."""
        assert isinstance(SUPPORTED_MODELS, dict)
        assert "anthropic" in SUPPORTED_MODELS
        assert "cohere" in SUPPORTED_MODELS
        assert "openrouter" in SUPPORTED_MODELS
        assert "huggingface" in SUPPORTED_MODELS

        for provider, models in SUPPORTED_MODELS.items():
            assert isinstance(models, list)
            assert len(models) > 0

    def test_supported_reasoning_structure(self):
        """Test supported reasoning approaches structure."""
        assert isinstance(SUPPORTED_REASONING, list)
        assert "None" in SUPPORTED_REASONING
        assert "Chain-of-Thought (CoT)" in SUPPORTED_REASONING
        assert len(SUPPORTED_REASONING) == 11  # Should have 11 approaches


class TestConfigurationEdgeCases:
    """Test edge cases and error conditions."""

    def test_config_with_minimal_valid_values(self):
        """Test config with minimal valid values."""
        config = ExperimentConfig(
            dataset_name="a",
            sample_count=1,
            temperature=0.0,
            max_tokens=1,
            top_p=0.0,
        )
        assert config.dataset_name == "a"
        assert config.sample_count == 1

    def test_config_with_maximum_valid_values(self):
        """Test config with maximum valid values."""
        config = ExperimentConfig(
            temperature=2.0,
            max_tokens=4096,
            top_p=1.0,
        )
        assert config.temperature == 2.0
        assert config.max_tokens == 4096
        assert config.top_p == 1.0

    def test_config_yaml_file_not_found(self, temp_dir):
        """Test loading config from non-existent YAML file."""
        yaml_path = temp_dir / "nonexistent.yaml"

        with pytest.raises(FileNotFoundError):
            ExperimentConfig.from_yaml(yaml_path)

    def test_config_yaml_invalid_format(self, temp_dir):
        """Test loading config from invalid YAML file."""
        yaml_path = temp_dir / "invalid.yaml"
        yaml_path.write_text("invalid: yaml: content: [")

        with pytest.raises(yaml.YAMLError):
            ExperimentConfig.from_yaml(yaml_path)

    def test_config_multiple_validation_errors(self):
        """Test that multiple validation errors are reported."""
        with pytest.raises(ValueError) as exc_info:
            ExperimentConfig(
                dataset_name="",
                sample_count=-1,
                temperature=3.0,
                top_p=1.5,
            )

        error_msg = str(exc_info.value)
        assert "dataset_name cannot be empty" in error_msg
        assert "sample_count must be >= 1" in error_msg
        assert "temperature must be between" in error_msg
        assert "top_p must be between" in error_msg

"""Tests for CLI validators."""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import typer

from ml_agents.cli.validators import (
    check_environment_ready,
    validate_api_key_available,
    validate_checkpoint_file,
    validate_config_file,
    validate_max_tokens,
    validate_max_workers,
    validate_output_directory,
    validate_provider_model,
    validate_reasoning_approach,
    validate_reasoning_approaches,
    validate_sample_count,
    validate_temperature,
    validate_top_p,
)


class TestReasoningApproachValidation:
    """Test reasoning approach validation functions."""

    @patch("ml_agents.cli.validators.get_available_approaches")
    def test_validate_valid_reasoning_approach(self, mock_get_approaches):
        """Test validation of valid reasoning approach."""
        mock_get_approaches.return_value = [
            "ChainOfThought",
            "AsPlanning",
            "TreeOfThought",
        ]

        result = validate_reasoning_approach("ChainOfThought")
        assert result == "ChainOfThought"

    @patch("ml_agents.cli.validators.get_available_approaches")
    def test_validate_invalid_reasoning_approach(self, mock_get_approaches):
        """Test validation fails for invalid reasoning approach."""
        mock_get_approaches.return_value = ["ChainOfThought", "AsPlanning"]

        with pytest.raises(
            typer.BadParameter, match="Invalid reasoning approach: InvalidApproach"
        ):
            validate_reasoning_approach("InvalidApproach")

    @patch("ml_agents.cli.validators.get_available_approaches")
    def test_validate_reasoning_approaches_list_valid(self, mock_get_approaches):
        """Test validation of valid reasoning approaches list."""
        mock_get_approaches.return_value = [
            "ChainOfThought",
            "AsPlanning",
            "TreeOfThought",
            "Reflection",
        ]

        result = validate_reasoning_approaches(
            "ChainOfThought,AsPlanning,TreeOfThought"
        )
        assert result == ["ChainOfThought", "AsPlanning", "TreeOfThought"]

    @patch("ml_agents.cli.validators.get_available_approaches")
    def test_validate_reasoning_approaches_with_spaces(self, mock_get_approaches):
        """Test validation handles spaces in approaches list."""
        mock_get_approaches.return_value = [
            "ChainOfThought",
            "AsPlanning",
            "Reflection",
        ]

        result = validate_reasoning_approaches(
            "ChainOfThought, AsPlanning , Reflection"
        )
        assert result == ["ChainOfThought", "AsPlanning", "Reflection"]

    @patch("ml_agents.cli.validators.get_available_approaches")
    def test_validate_reasoning_approaches_with_invalid(self, mock_get_approaches):
        """Test validation fails when one approach is invalid."""
        mock_get_approaches.return_value = ["ChainOfThought", "AsPlanning"]

        with pytest.raises(
            typer.BadParameter, match="Invalid reasoning approach: InvalidApproach"
        ):
            validate_reasoning_approaches("ChainOfThought,InvalidApproach")


class TestProviderModelValidation:
    """Test provider and model validation functions."""

    @patch("ml_agents.cli.validators.SUPPORTED_MODELS")
    def test_validate_valid_provider_model(self, mock_supported_models):
        """Test validation of valid provider/model combination."""
        mock_supported_models.return_value = {
            "openrouter": ["gpt-3.5-turbo", "gpt-4"],
            "anthropic": ["claude-sonnet-4-20250514"],
        }

        provider, model = validate_provider_model("openrouter", "gpt-3.5-turbo")
        assert provider == "openrouter"
        assert model == "gpt-3.5-turbo"

    @patch("ml_agents.cli.validators.SUPPORTED_MODELS")
    def test_validate_invalid_provider(self, mock_supported_models):
        """Test validation fails for invalid provider."""
        mock_supported_models.return_value = {
            "openrouter": ["gpt-3.5-turbo"],
            "anthropic": ["claude-sonnet-4-20250514"],
        }

        with pytest.raises(
            typer.BadParameter, match="Invalid provider: invalid_provider"
        ):
            validate_provider_model("invalid_provider", "gpt-3.5-turbo")

    @patch("ml_agents.cli.validators.SUPPORTED_MODELS")
    def test_validate_invalid_model_for_provider(self, mock_supported_models):
        """Test validation fails for invalid model for provider."""
        mock_supported_models.return_value = {
            "openrouter": ["gpt-3.5-turbo"],
            "anthropic": ["claude-sonnet-4-20250514"],
        }

        with pytest.raises(
            typer.BadParameter,
            match="Invalid model for provider openrouter: claude-sonnet-4-20250514",
        ):
            validate_provider_model("openrouter", "claude-sonnet-4-20250514")


class TestOutputDirectoryValidation:
    """Test output directory validation."""

    def test_validate_existing_writable_directory(self):
        """Test validation of existing writable directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            result = validate_output_directory(temp_dir)
            assert Path(result).exists()
            assert Path(result).is_dir()

    def test_validate_non_existing_directory_creates_it(self):
        """Test validation creates non-existing directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            new_dir = Path(temp_dir) / "new_output_dir"

            result = validate_output_directory(str(new_dir))
            assert Path(result).exists()
            assert Path(result).is_dir()

    def test_validate_nested_directory_creation(self):
        """Test validation creates nested directory structure."""
        with tempfile.TemporaryDirectory() as temp_dir:
            nested_dir = Path(temp_dir) / "level1" / "level2" / "output"

            result = validate_output_directory(str(nested_dir))
            assert Path(result).exists()
            assert Path(result).is_dir()

    @patch("pathlib.Path.mkdir")
    def test_validate_directory_creation_fails(self, mock_mkdir):
        """Test validation fails when directory cannot be created."""
        mock_mkdir.side_effect = PermissionError("Permission denied")

        with pytest.raises(
            typer.BadParameter, match="Cannot write to output directory"
        ):
            validate_output_directory("/invalid/path")

    @patch("pathlib.Path.write_text")
    def test_validate_directory_not_writable(self, mock_write_text):
        """Test validation fails when directory is not writable."""
        mock_write_text.side_effect = PermissionError("Permission denied")

        with tempfile.TemporaryDirectory() as temp_dir:
            with pytest.raises(
                typer.BadParameter, match="Cannot write to output directory"
            ):
                validate_output_directory(temp_dir)


class TestAPIKeyValidation:
    """Test API key availability validation."""

    @patch("ml_agents.cli.validators.API_KEYS")
    def test_validate_api_key_available_valid(self, mock_api_keys):
        """Test API key is available and valid."""
        mock_api_keys.get.return_value = "sk-valid-api-key-12345"

        result = validate_api_key_available("openrouter")
        assert result is True

    @patch("ml_agents.cli.validators.API_KEYS")
    def test_validate_api_key_missing(self, mock_api_keys):
        """Test API key is missing."""
        mock_api_keys.get.return_value = None

        result = validate_api_key_available("openrouter")
        assert result is False

    @patch("ml_agents.cli.validators.API_KEYS")
    def test_validate_api_key_placeholder(self, mock_api_keys):
        """Test API key is placeholder value."""
        mock_api_keys.get.return_value = "your_key_here"

        result = validate_api_key_available("anthropic")
        assert result is False

    @patch("ml_agents.cli.validators.API_KEYS")
    def test_validate_api_key_empty_string(self, mock_api_keys):
        """Test API key is empty string."""
        mock_api_keys.get.return_value = ""

        result = validate_api_key_available("cohere")
        assert result is False


class TestConfigFileValidation:
    """Test configuration file validation."""

    def test_validate_yaml_config_file(self):
        """Test validation of valid YAML config file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("sample_count: 50\nprovider: openrouter")
            config_path = f.name

        try:
            result = validate_config_file(config_path)
            assert Path(result).exists()
            assert result.endswith(".yaml")
        finally:
            Path(config_path).unlink()

    def test_validate_yml_config_file(self):
        """Test validation of valid YML config file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as f:
            f.write("sample_count: 100")
            config_path = f.name

        try:
            result = validate_config_file(config_path)
            assert Path(result).exists()
            assert result.endswith(".yml")
        finally:
            Path(config_path).unlink()

    def test_validate_json_config_file(self):
        """Test validation of valid JSON config file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write('{"sample_count": 75}')
            config_path = f.name

        try:
            result = validate_config_file(config_path)
            assert Path(result).exists()
            assert result.endswith(".json")
        finally:
            Path(config_path).unlink()

    def test_validate_config_file_not_found(self):
        """Test validation fails for non-existent config file."""
        with pytest.raises(typer.BadParameter, match="Config file not found"):
            validate_config_file("/nonexistent/config.yaml")

    def test_validate_config_file_unsupported_format(self):
        """Test validation fails for unsupported file format."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("some content")
            config_path = f.name

        try:
            with pytest.raises(
                typer.BadParameter, match="Unsupported config file format"
            ):
                validate_config_file(config_path)
        finally:
            Path(config_path).unlink()

    @patch("builtins.open")
    def test_validate_config_file_read_error(self, mock_open):
        """Test validation fails when config file cannot be read."""
        mock_open.side_effect = PermissionError("Permission denied")

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            config_path = f.name

        try:
            with pytest.raises(typer.BadParameter, match="Cannot read config file"):
                validate_config_file(config_path)
        finally:
            Path(config_path).unlink()


class TestCheckpointFileValidation:
    """Test checkpoint file validation."""

    def test_validate_valid_checkpoint_file(self):
        """Test validation of valid checkpoint file."""
        checkpoint_data = {
            "experiment_id": "exp_12345",
            "config": {"sample_count": 50},
            "progress": {"completed": 25},
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            import json

            json.dump(checkpoint_data, f)
            checkpoint_path = f.name

        try:
            result = validate_checkpoint_file(checkpoint_path)
            assert Path(result).exists()
            assert result.endswith(".json")
        finally:
            Path(checkpoint_path).unlink()

    def test_validate_checkpoint_file_not_found(self):
        """Test validation fails for non-existent checkpoint file."""
        with pytest.raises(typer.BadParameter, match="Checkpoint file not found"):
            validate_checkpoint_file("/nonexistent/checkpoint.json")

    def test_validate_checkpoint_file_wrong_format(self):
        """Test validation fails for non-JSON checkpoint file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("experiment_id: exp_12345")
            checkpoint_path = f.name

        try:
            with pytest.raises(
                typer.BadParameter, match="Invalid checkpoint file format"
            ):
                validate_checkpoint_file(checkpoint_path)
        finally:
            Path(checkpoint_path).unlink()

    def test_validate_checkpoint_file_invalid_json(self):
        """Test validation fails for invalid JSON content."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write('{"invalid": json content}')  # Invalid JSON
            checkpoint_path = f.name

        try:
            with pytest.raises(typer.BadParameter, match="Invalid checkpoint file"):
                validate_checkpoint_file(checkpoint_path)
        finally:
            Path(checkpoint_path).unlink()

    def test_validate_checkpoint_file_missing_required_fields(self):
        """Test validation fails for checkpoint missing required fields."""
        incomplete_data = {
            "experiment_id": "exp_12345",
            # Missing "config" and "progress" fields
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            import json

            json.dump(incomplete_data, f)
            checkpoint_path = f.name

        try:
            with pytest.raises(typer.BadParameter, match="Invalid checkpoint file"):
                validate_checkpoint_file(checkpoint_path)
        finally:
            Path(checkpoint_path).unlink()


class TestParameterValidation:
    """Test numeric parameter validation functions."""

    def test_validate_sample_count_valid(self):
        """Test valid sample count validation."""
        assert validate_sample_count(1) == 1
        assert validate_sample_count(50) == 50
        assert validate_sample_count(1000) == 1000

    def test_validate_sample_count_zero(self):
        """Test sample count validation fails for zero."""
        with pytest.raises(typer.BadParameter, match="Sample count must be at least 1"):
            validate_sample_count(0)

    def test_validate_sample_count_negative(self):
        """Test sample count validation fails for negative values."""
        with pytest.raises(typer.BadParameter, match="Sample count must be at least 1"):
            validate_sample_count(-5)

    @patch("typer.confirm")
    def test_validate_sample_count_large_with_confirmation(self, mock_confirm):
        """Test large sample count with user confirmation."""
        mock_confirm.return_value = True

        result = validate_sample_count(15000)
        assert result == 15000
        mock_confirm.assert_called_once()

    @patch("typer.confirm")
    def test_validate_sample_count_large_without_confirmation(self, mock_confirm):
        """Test large sample count without user confirmation."""
        mock_confirm.return_value = False

        with pytest.raises(typer.Abort):
            validate_sample_count(15000)

    def test_validate_temperature_valid(self):
        """Test valid temperature validation."""
        assert validate_temperature(0.0) == 0.0
        assert validate_temperature(0.5) == 0.5
        assert validate_temperature(1.0) == 1.0
        assert validate_temperature(2.0) == 2.0

    def test_validate_temperature_invalid_low(self):
        """Test temperature validation fails for values below 0.0."""
        with pytest.raises(
            typer.BadParameter, match="Temperature must be between 0.0 and 2.0"
        ):
            validate_temperature(-0.1)

    def test_validate_temperature_invalid_high(self):
        """Test temperature validation fails for values above 2.0."""
        with pytest.raises(
            typer.BadParameter, match="Temperature must be between 0.0 and 2.0"
        ):
            validate_temperature(2.1)

    def test_validate_top_p_valid(self):
        """Test valid top_p validation."""
        assert validate_top_p(0.0) == 0.0
        assert validate_top_p(0.5) == 0.5
        assert validate_top_p(0.9) == 0.9
        assert validate_top_p(1.0) == 1.0

    def test_validate_top_p_invalid_low(self):
        """Test top_p validation fails for values below 0.0."""
        with pytest.raises(
            typer.BadParameter, match="top_p must be between 0.0 and 1.0"
        ):
            validate_top_p(-0.1)

    def test_validate_top_p_invalid_high(self):
        """Test top_p validation fails for values above 1.0."""
        with pytest.raises(
            typer.BadParameter, match="top_p must be between 0.0 and 1.0"
        ):
            validate_top_p(1.1)

    def test_validate_max_tokens_valid(self):
        """Test valid max_tokens validation."""
        assert validate_max_tokens(1) == 1
        assert validate_max_tokens(512) == 512
        assert validate_max_tokens(4096) == 4096

    def test_validate_max_tokens_invalid_low(self):
        """Test max_tokens validation fails for values below 1."""
        with pytest.raises(
            typer.BadParameter, match="max_tokens must be between 1 and 4096"
        ):
            validate_max_tokens(0)

    def test_validate_max_tokens_invalid_high(self):
        """Test max_tokens validation fails for values above 4096."""
        with pytest.raises(
            typer.BadParameter, match="max_tokens must be between 1 and 4096"
        ):
            validate_max_tokens(5000)

    def test_validate_max_workers_valid(self):
        """Test valid max_workers validation."""
        assert validate_max_workers(1) == 1
        assert validate_max_workers(4) == 4
        assert validate_max_workers(8) == 8

    def test_validate_max_workers_invalid_low(self):
        """Test max_workers validation fails for values below 1."""
        with pytest.raises(typer.BadParameter, match="max_workers must be at least 1"):
            validate_max_workers(0)

    @patch("typer.confirm")
    def test_validate_max_workers_high_with_confirmation(self, mock_confirm):
        """Test high max_workers with user confirmation."""
        mock_confirm.return_value = True

        result = validate_max_workers(20)
        assert result == 20
        mock_confirm.assert_called_once()

    @patch("typer.confirm")
    def test_validate_max_workers_high_without_confirmation(self, mock_confirm):
        """Test high max_workers without user confirmation."""
        mock_confirm.return_value = False

        with pytest.raises(typer.Abort):
            validate_max_workers(20)


class TestEnvironmentValidation:
    """Test environment readiness validation."""

    @patch("ml_agents.cli.validators.validate_api_key_available")
    def test_check_environment_ready_valid_key(self, mock_validate_api_key):
        """Test environment ready with valid API key."""
        mock_validate_api_key.return_value = True

        # Should not raise any exception
        check_environment_ready("openrouter")
        mock_validate_api_key.assert_called_once_with("openrouter")

    @patch("ml_agents.cli.validators.validate_api_key_available")
    def test_check_environment_ready_invalid_key(self, mock_validate_api_key):
        """Test environment not ready with invalid API key."""
        mock_validate_api_key.return_value = False

        with pytest.raises(typer.Exit):
            check_environment_ready("openrouter")

    @patch("ml_agents.cli.validators.validate_api_key_available")
    def test_check_environment_ready_anthropic_instructions(
        self, mock_validate_api_key
    ):
        """Test environment check shows Anthropic API key instructions."""
        mock_validate_api_key.return_value = False

        with pytest.raises(typer.Exit):
            check_environment_ready("anthropic")

        # Would need to capture console output to verify instruction message

    @patch("ml_agents.cli.validators.validate_api_key_available")
    def test_check_environment_ready_cohere_instructions(self, mock_validate_api_key):
        """Test environment check shows Cohere API key instructions."""
        mock_validate_api_key.return_value = False

        with pytest.raises(typer.Exit):
            check_environment_ready("cohere")

    @patch("ml_agents.cli.validators.validate_api_key_available")
    def test_check_environment_ready_huggingface_instructions(
        self, mock_validate_api_key
    ):
        """Test environment check shows HuggingFace API key instructions."""
        mock_validate_api_key.return_value = False

        with pytest.raises(typer.Exit):
            check_environment_ready("huggingface")

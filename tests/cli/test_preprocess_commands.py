"""Comprehensive tests for preprocessing CLI commands."""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest
from typer.testing import CliRunner

from ml_agents.cli.main import app


class TestPreprocessListCommand:
    """Test the preprocess list command."""

    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()

    def test_preprocess_list_command_help(self):
        """Test preprocess list command help display."""
        result = self.runner.invoke(app, ["preprocess", "list", "--help"])
        assert result.exit_code == 0
        assert "haven't been preprocessed yet" in result.stdout.lower()

    @patch("ml_agents.core.dataset_preprocessor.DatasetPreprocessor")
    def test_preprocess_list_unprocessed(self, mock_preprocessor):
        """Test listing unprocessed datasets."""
        mock_unprocessed = [
            {
                "name": "dataset1",
                "task_type": "reasoning",
                "status": "unprocessed",
                "description": "A test dataset for reasoning tasks",
            },
            {
                "name": "dataset2",
                "task_type": "classification",
                "status": "unprocessed",
                "description": "Another test dataset",
            },
        ]

        mock_preprocessor_instance = Mock()
        mock_preprocessor_instance.get_unprocessed_datasets.return_value = (
            mock_unprocessed
        )
        mock_preprocessor.return_value = mock_preprocessor_instance

        result = self.runner.invoke(app, ["preprocess", "list"])

        assert result.exit_code == 0
        assert "Unprocessed Datasets (2 found)" in result.stdout
        assert "dataset1" in result.stdout
        assert "dataset2" in result.stdout
        assert "reasoning" in result.stdout
        assert "Total unprocessed datasets: 2" in result.stdout

    @patch("ml_agents.core.dataset_preprocessor.DatasetPreprocessor")
    def test_preprocess_list_no_unprocessed(self, mock_preprocessor):
        """Test listing when no unprocessed datasets exist."""
        mock_preprocessor_instance = Mock()
        mock_preprocessor_instance.get_unprocessed_datasets.return_value = []
        mock_preprocessor.return_value = mock_preprocessor_instance

        result = self.runner.invoke(app, ["preprocess", "list"])

        assert result.exit_code == 0
        assert "No unprocessed datasets found" in result.stdout

    @patch("ml_agents.core.dataset_preprocessor.DatasetPreprocessor")
    def test_preprocess_list_json_format(self, mock_preprocessor):
        """Test listing with JSON output format."""
        mock_unprocessed = [
            {
                "name": "dataset1",
                "task_type": "reasoning",
                "status": "unprocessed",
            }
        ]

        mock_preprocessor_instance = Mock()
        mock_preprocessor_instance.get_unprocessed_datasets.return_value = (
            mock_unprocessed
        )
        mock_preprocessor.return_value = mock_preprocessor_instance

        result = self.runner.invoke(app, ["preprocess", "list", "--format", "json"])

        assert result.exit_code == 0
        # Should contain JSON formatted output
        assert '"name": "dataset1"' in result.stdout
        assert '"task_type": "reasoning"' in result.stdout

    @patch("ml_agents.core.dataset_preprocessor.DatasetPreprocessor")
    def test_preprocess_list_custom_benchmark_csv(self, mock_preprocessor):
        """Test listing with custom benchmark CSV path."""
        mock_preprocessor_instance = Mock()
        mock_preprocessor_instance.get_unprocessed_datasets.return_value = []
        mock_preprocessor.return_value = mock_preprocessor_instance

        result = self.runner.invoke(
            app, ["preprocess", "list", "--benchmark-csv", "custom.csv"]
        )

        assert result.exit_code == 0
        # Check that preprocessor was initialized with custom path
        mock_preprocessor.assert_called_with("custom.csv", "./ml_agents_results.db")


class TestPreprocessInspectCommand:
    """Test the preprocess inspect command."""

    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()

    def test_preprocess_inspect_command_help(self):
        """Test preprocess inspect command help display."""
        result = self.runner.invoke(app, ["preprocess", "inspect", "--help"])
        assert result.exit_code == 0
        assert "inspect dataset schema" in result.stdout.lower()

    @patch("ml_agents.core.dataset_preprocessor.DatasetPreprocessor")
    def test_preprocess_inspect_dataset(self, mock_preprocessor):
        """Test dataset schema inspection."""
        mock_schema_info = {
            "dataset_name": "test-dataset",
            "total_samples": 1000,
            "sample_size_analyzed": 100,
            "columns": ["input", "output", "reasoning"],
            "column_types": {
                "input": "string",
                "output": "string",
                "reasoning": "string",
            },
            "detected_patterns": {
                "recommended_pattern": {
                    "type": "input_output",
                    "confidence": 0.95,
                    "input_fields": ["input"],
                    "output_field": "output",
                    "reasoning": "Clear input/output structure detected",
                }
            },
        }

        mock_preprocessor_instance = Mock()
        mock_preprocessor_instance.inspect_dataset_schema.return_value = (
            mock_schema_info
        )
        mock_preprocessor.return_value = mock_preprocessor_instance

        result = self.runner.invoke(app, ["preprocess", "inspect", "test-dataset"])

        assert result.exit_code == 0
        assert "Dataset Schema Analysis" in result.stdout
        assert "test-dataset" in result.stdout
        assert "Total samples: 1,000" in result.stdout
        assert "Analyzed samples: 100" in result.stdout
        assert "Columns: 3" in result.stdout
        assert "Confidence: 0.95" in result.stdout
        assert "Dataset inspection completed" in result.stdout

    @patch("ml_agents.core.dataset_preprocessor.DatasetPreprocessor")
    @patch("ml_agents.cli.commands.preprocess.ensure_preprocessing_output_dir")
    def test_preprocess_inspect_saves_output(self, mock_ensure_dir, mock_preprocessor):
        """Test that inspection results are saved to file."""
        mock_output_dir = Path("/tmp/test_output")
        mock_ensure_dir.return_value = mock_output_dir

        mock_schema_info = {
            "dataset_name": "test-dataset",
            "total_samples": 500,
            "sample_size_analyzed": 50,
            "columns": ["question", "answer"],
            "column_types": {"question": "string", "answer": "string"},
            "detected_patterns": {"recommended_pattern": {"type": "qa"}},
        }

        mock_preprocessor_instance = Mock()
        mock_preprocessor_instance.inspect_dataset_schema.return_value = (
            mock_schema_info
        )
        mock_preprocessor.return_value = mock_preprocessor_instance

        with patch("builtins.open", create=True) as mock_open:
            with patch("json.dump") as mock_json_dump:
                result = self.runner.invoke(
                    app, ["preprocess", "inspect", "test-dataset"]
                )

                assert result.exit_code == 0
                mock_json_dump.assert_called_once_with(
                    mock_schema_info,
                    mock_open.return_value.__enter__.return_value,
                    indent=2,
                )

    @patch("ml_agents.core.dataset_preprocessor.DatasetPreprocessor")
    def test_preprocess_inspect_with_config(self, mock_preprocessor):
        """Test dataset inspection with configuration."""
        mock_preprocessor_instance = Mock()
        mock_preprocessor_instance.inspect_dataset_schema.return_value = {
            "dataset_name": "test-dataset",
            "total_samples": 200,
            "sample_size_analyzed": 50,
            "columns": ["text"],
            "column_types": {"text": "string"},
            "detected_patterns": {"recommended_pattern": {"type": "text"}},
        }
        mock_preprocessor.return_value = mock_preprocessor_instance

        result = self.runner.invoke(
            app, ["preprocess", "inspect", "test-dataset", "--config", "subset_a"]
        )

        assert result.exit_code == 0
        # Check that config was passed to inspect method
        mock_preprocessor_instance.inspect_dataset_schema.assert_called_with(
            "test-dataset", 100, "subset_a"
        )

    @patch("ml_agents.core.dataset_preprocessor.DatasetPreprocessor")
    def test_preprocess_inspect_custom_samples(self, mock_preprocessor):
        """Test dataset inspection with custom sample size."""
        mock_preprocessor_instance = Mock()
        mock_preprocessor_instance.inspect_dataset_schema.return_value = {
            "dataset_name": "test-dataset",
            "total_samples": 5000,
            "sample_size_analyzed": 500,
            "columns": ["data"],
            "column_types": {"data": "string"},
            "detected_patterns": {"recommended_pattern": {"type": "unknown"}},
        }
        mock_preprocessor.return_value = mock_preprocessor_instance

        result = self.runner.invoke(
            app, ["preprocess", "inspect", "test-dataset", "--samples", "500"]
        )

        assert result.exit_code == 0
        assert "Analyzed samples: 500" in result.stdout
        # Check that custom sample size was passed
        mock_preprocessor_instance.inspect_dataset_schema.assert_called_with(
            "test-dataset", 500, None
        )


class TestPreprocessGenerateRulesCommand:
    """Test the preprocess generate-rules command."""

    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()

    def test_preprocess_generate_rules_command_help(self):
        """Test preprocess generate-rules command help display."""
        result = self.runner.invoke(app, ["preprocess", "generate-rules", "--help"])
        assert result.exit_code == 0
        assert "transformation rules" in result.stdout.lower()

    @patch("ml_agents.core.dataset_preprocessor.DatasetPreprocessor")
    @patch("ml_agents.cli.commands.preprocess.ensure_preprocessing_output_dir")
    def test_preprocess_generate_rules(self, mock_ensure_dir, mock_preprocessor):
        """Test transformation rule generation."""
        mock_output_dir = Path("/tmp/test_output")
        mock_ensure_dir.return_value = mock_output_dir

        mock_schema_info = {
            "dataset_name": "test-dataset",
            "columns": ["input", "output"],
        }

        mock_rules = {
            "dataset_name": "test-dataset",
            "transformation_type": "input_output_mapping",
            "confidence": 0.9,
            "input_format": "question_answer",
            "rules": {"input_field": "input", "output_field": "output"},
        }

        mock_preprocessor_instance = Mock()
        mock_preprocessor_instance.inspect_dataset_schema.return_value = (
            mock_schema_info
        )
        mock_preprocessor_instance.generate_transformation_rules.return_value = (
            mock_rules
        )
        mock_preprocessor.return_value = mock_preprocessor_instance

        with patch("builtins.open", create=True) as mock_open:
            with patch("json.dump") as mock_json_dump:
                result = self.runner.invoke(
                    app, ["preprocess", "generate-rules", "test-dataset"]
                )

                assert result.exit_code == 0
                assert "Transformation Rules Generated" in result.stdout
                assert "test-dataset" in result.stdout
                assert "input_output_mapping" in result.stdout
                assert "Confidence: 0.90" in result.stdout
                assert "Transformation rules generated successfully" in result.stdout
                mock_json_dump.assert_called_once()


class TestPreprocessTransformCommand:
    """Test the preprocess transform command."""

    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()

    def test_preprocess_transform_command_help(self):
        """Test preprocess transform command help display."""
        result = self.runner.invoke(app, ["preprocess", "transform", "--help"])
        assert result.exit_code == 0
        assert "transformation rules" in result.stdout.lower()
        assert "INPUT, OUTPUT" in result.stdout

    @patch("ml_agents.core.dataset_preprocessor.DatasetPreprocessor")
    def test_preprocess_transform_apply(self, mock_preprocessor):
        """Test applying transformations."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            mock_rules = {
                "dataset_name": "test-dataset",
                "transformation_type": "input_output_mapping",
                "rules": {"input_field": "question", "output_field": "answer"},
            }
            json.dump(mock_rules, f)
            rules_path = f.name

        try:
            mock_transformed = [
                {"INPUT": "What is 2+2?", "OUTPUT": "4"},
                {"INPUT": "What is 3+3?", "OUTPUT": "6"},
            ]

            mock_preprocessor_instance = Mock()
            mock_preprocessor_instance.apply_transformation.return_value = (
                mock_transformed
            )
            # Mock validation results that the function expects
            mock_preprocessor_instance.validate_transformation.return_value = {
                "validation_passed": True,
                "issues": [],
                "original_samples": 100,
                "transformed_samples": 100,
                "empty_inputs": 0,
                "empty_outputs": 0,
            }
            # Mock the export method
            mock_preprocessor_instance.export_standardized.return_value = None
            mock_preprocessor.return_value = mock_preprocessor_instance

            with patch("datasets.load_dataset") as mock_load:
                with patch("datasets.get_dataset_config_info") as mock_config_info:
                    mock_config_info.return_value.splits = {"train": Mock()}
                    mock_load.return_value = Mock()

                    result = self.runner.invoke(
                        app, ["preprocess", "transform", "test-dataset", rules_path]
                    )

                    assert result.exit_code == 0
                    assert "Transforming dataset: test-dataset" in result.stdout
                    assert "Dataset transformation completed" in result.stdout

        finally:
            Path(rules_path).unlink()

    @patch("ml_agents.core.dataset_preprocessor.DatasetPreprocessor")
    def test_preprocess_transform_with_validation(self, mock_preprocessor):
        """Test transformation with validation enabled."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            mock_rules = {
                "dataset_name": "test-dataset",
                "transformation_type": "input_output_mapping",
            }
            json.dump(mock_rules, f)
            rules_path = f.name

        try:
            mock_transformed = [{"INPUT": "test", "OUTPUT": "result"}]
            mock_validation_results = {
                "validation_passed": True,
                "original_samples": 100,
                "transformed_samples": 100,
                "empty_inputs": 0,
                "empty_outputs": 0,
                "issues": [],
            }

            mock_preprocessor_instance = Mock()
            mock_preprocessor_instance.apply_transformation.return_value = (
                mock_transformed
            )
            mock_preprocessor_instance.validate_transformation.return_value = (
                mock_validation_results
            )
            mock_preprocessor.return_value = mock_preprocessor_instance

            with patch("datasets.load_dataset") as mock_load:
                with patch("datasets.get_dataset_config_info") as mock_config_info:
                    mock_config_info.return_value.splits = {"train": Mock()}
                    mock_load.return_value = Mock()

                    result = self.runner.invoke(
                        app,
                        [
                            "preprocess",
                            "transform",
                            "test-dataset",
                            rules_path,
                            "--validate",
                        ],
                    )

                    assert result.exit_code == 0
                    assert "Transformation validation passed" in result.stdout
                    assert "Original samples: 100" in result.stdout

        finally:
            Path(rules_path).unlink()


class TestPreprocessBatchCommand:
    """Test the preprocess batch command."""

    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()

    def test_preprocess_batch_command_help(self):
        """Test preprocess batch command help display."""
        result = self.runner.invoke(app, ["preprocess", "batch", "--help"])
        assert result.exit_code == 0
        assert "batch process" in result.stdout.lower()

    @patch("ml_agents.core.dataset_preprocessor.DatasetPreprocessor")
    def test_preprocess_batch_processing(self, mock_preprocessor):
        """Test batch processing workflow."""
        mock_unprocessed = [
            {"name": "dataset1", "url": "org/dataset1"},
            {"name": "dataset2", "url": "org/dataset2"},
        ]

        mock_schema_info = {"dataset_name": "test", "columns": ["input", "output"]}

        mock_rules = {
            "dataset_name": "test",
            "confidence": 0.8,
            "transformation_type": "input_output",
        }

        mock_transformed = [{"INPUT": "test", "OUTPUT": "result"}]
        mock_validation = {"validation_passed": True}

        mock_preprocessor_instance = Mock()
        mock_preprocessor_instance.get_unprocessed_datasets.return_value = (
            mock_unprocessed
        )
        mock_preprocessor_instance.inspect_dataset_schema.return_value = (
            mock_schema_info
        )
        mock_preprocessor_instance.generate_transformation_rules.return_value = (
            mock_rules
        )
        mock_preprocessor_instance.apply_transformation.return_value = mock_transformed
        mock_preprocessor_instance.validate_transformation.return_value = (
            mock_validation
        )
        mock_preprocessor.return_value = mock_preprocessor_instance

        with patch("datasets.load_dataset") as mock_load:
            with patch("datasets.get_dataset_config_info") as mock_config_info:
                mock_config_info.return_value.splits = {"train": Mock()}
                mock_load.return_value = Mock()

                result = self.runner.invoke(app, ["preprocess", "batch", "--max", "2"])

                assert result.exit_code == 0
                assert "Batch Processing Summary" in result.stdout
                assert "Successful: 2" in result.stdout

    @patch("ml_agents.core.dataset_preprocessor.DatasetPreprocessor")
    def test_preprocess_batch_no_datasets(self, mock_preprocessor):
        """Test batch processing when no unprocessed datasets exist."""
        mock_preprocessor_instance = Mock()
        mock_preprocessor_instance.get_unprocessed_datasets.return_value = []
        mock_preprocessor.return_value = mock_preprocessor_instance

        result = self.runner.invoke(app, ["preprocess", "batch"])

        assert result.exit_code == 0
        assert "No unprocessed datasets found" in result.stdout


class TestPreprocessUploadCommand:
    """Test the preprocess upload command."""

    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()

    def test_preprocess_upload_command_help(self):
        """Test preprocess upload command help display."""
        result = self.runner.invoke(app, ["preprocess", "upload", "--help"])
        assert result.exit_code == 0
        assert "huggingface" in result.stdout.lower()

    @patch("ml_agents.core.dataset_uploader.DatasetUploader")
    def test_preprocess_upload_dataset(self, mock_uploader):
        """Test dataset upload to HuggingFace Hub."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            test_data = [{"INPUT": "test", "OUTPUT": "result"}]
            json.dump(test_data, f)
            dataset_path = f.name

        try:
            mock_validation_results = {
                "validation_passed": True,
                "format": "json",
                "sample_count": 1,
                "file_size_mb": 0.001,
                "has_input_output_schema": True,
                "issues": [],
            }

            mock_uploader_instance = Mock()
            mock_uploader_instance.validate_processed_file.return_value = (
                mock_validation_results
            )
            mock_uploader_instance.upload_dataset.return_value = (
                "c4ai-ml-agents/test-dataset"
            )
            mock_uploader.return_value = mock_uploader_instance

            with patch("typer.confirm", return_value=True):
                result = self.runner.invoke(
                    app,
                    [
                        "preprocess",
                        "upload",
                        dataset_path,
                        "--source-dataset",
                        "org/original-dataset",
                        "--target-name",
                        "test-dataset",
                    ],
                )

                assert result.exit_code == 0
                assert "Dataset validation passed" in result.stdout
                assert "Dataset successfully uploaded" in result.stdout
                assert "c4ai-ml-agents/test-dataset" in result.stdout

        finally:
            Path(dataset_path).unlink()

    @patch("ml_agents.core.dataset_uploader.DatasetUploader")
    def test_preprocess_upload_dry_run(self, mock_uploader):
        """Test dataset upload dry run."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            test_data = [{"INPUT": "test", "OUTPUT": "result"}]
            json.dump(test_data, f)
            dataset_path = f.name

        try:
            mock_validation_results = {
                "validation_passed": True,
                "format": "json",
                "sample_count": 1,
                "file_size_mb": 0.001,
                "has_input_output_schema": True,
                "issues": [],
            }

            mock_uploader_instance = Mock()
            mock_uploader_instance.validate_processed_file.return_value = (
                mock_validation_results
            )
            mock_uploader.return_value = mock_uploader_instance

            result = self.runner.invoke(
                app,
                [
                    "preprocess",
                    "upload",
                    dataset_path,
                    "--source-dataset",
                    "org/original",
                    "--target-name",
                    "test",
                    "--dry-run",
                ],
            )

            assert result.exit_code == 0
            assert "Dry run successful" in result.stdout
            assert "ready for upload" in result.stdout
            # Should not call upload_dataset in dry run
            mock_uploader_instance.upload_dataset.assert_not_called()

        finally:
            Path(dataset_path).unlink()

    @patch("ml_agents.core.dataset_uploader.DatasetUploader")
    def test_preprocess_upload_validation_failed(self, mock_uploader):
        """Test dataset upload with validation failure."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            test_data = [{"INVALID": "data"}]
            json.dump(test_data, f)
            dataset_path = f.name

        try:
            mock_validation_results = {
                "validation_passed": False,
                "issues": ["Missing INPUT field", "Missing OUTPUT field"],
            }

            mock_uploader_instance = Mock()
            mock_uploader_instance.validate_processed_file.return_value = (
                mock_validation_results
            )
            mock_uploader.return_value = mock_uploader_instance

            result = self.runner.invoke(
                app,
                [
                    "preprocess",
                    "upload",
                    dataset_path,
                    "--source-dataset",
                    "org/original",
                    "--target-name",
                    "test",
                ],
            )

            assert result.exit_code == 1
            assert "Dataset validation failed" in result.stdout
            assert "Missing INPUT field" in result.stdout

        finally:
            Path(dataset_path).unlink()


class TestPreprocessCommandsIntegration:
    """Integration tests for preprocessing commands."""

    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()

    def test_preprocess_group_exists(self):
        """Test that preprocess group exists and is accessible."""
        result = self.runner.invoke(app, ["preprocess", "--help"])
        assert result.exit_code == 0
        assert "Dataset preprocessing" in result.stdout
        assert "list" in result.stdout
        assert "inspect" in result.stdout
        assert "generate-rules" in result.stdout
        assert "transform" in result.stdout
        assert "batch" in result.stdout
        assert "upload" in result.stdout

    def test_all_preprocess_commands_accessible(self):
        """Test that all preprocessing commands are accessible."""
        commands = ["list", "inspect", "generate-rules", "transform", "batch", "upload"]

        for command in commands:
            if command in ["inspect", "generate-rules", "transform", "upload"]:
                # These commands require arguments, so just check they show help correctly
                result = self.runner.invoke(app, ["preprocess", command, "--help"])
            else:
                # These commands work without arguments
                result = self.runner.invoke(app, ["preprocess", command, "--help"])
            assert (
                result.exit_code == 0
            ), f"Command 'preprocess {command}' not accessible"

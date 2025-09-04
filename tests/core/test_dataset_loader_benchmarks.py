"""Tests for dataset loader benchmark integration."""

from unittest.mock import Mock, patch

import pytest
from datasets import Dataset

from ml_agents.config import ExperimentConfig
from ml_agents.core.benchmark_registry import (
    BenchmarkFormatError,
    BenchmarkNotFoundError,
)
from ml_agents.core.dataset_loader import BBEHDatasetLoader


class TestDatasetLoaderBenchmarks:
    """Test cases for dataset loader with benchmark registry."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config = ExperimentConfig(benchmark_id="GPQA", sample_count=10)
        self.loader = BBEHDatasetLoader(self.config)

    @patch("ml_agents.core.dataset_loader.BenchmarkRegistry")
    def test_load_dataset_with_benchmark_id(self, mock_registry_class):
        """Test loading dataset with benchmark ID."""
        # Mock the registry and its methods
        mock_registry = Mock()
        mock_registry_class.return_value = mock_registry

        # Mock dataset
        mock_dataset = Mock()
        mock_dataset.column_names = ["INPUT", "OUTPUT"]
        mock_dataset.__len__ = Mock(return_value=100)
        mock_dataset.__getitem__ = Mock(
            return_value={"INPUT": "test", "OUTPUT": "result"}
        )
        mock_registry.load_benchmark.return_value = mock_dataset

        # Create loader inside test (after patch is applied)
        loader = BBEHDatasetLoader(self.config)

        # Load dataset
        result = loader.load_dataset("GPQA")

        # Verify registry was called correctly
        mock_registry.load_benchmark.assert_called_once_with("GPQA")
        assert result == mock_dataset

    @patch("ml_agents.core.benchmark_registry.load_dataset")
    def test_load_dataset_benchmark_not_found(self, mock_load_dataset):
        """Test error handling when benchmark not found."""
        mock_load_dataset.side_effect = FileNotFoundError("File not found")

        with pytest.raises(BenchmarkNotFoundError):
            self.loader.load_dataset("NonExistent")

    @patch("ml_agents.core.benchmark_registry.load_dataset")
    def test_load_dataset_invalid_format(self, mock_load_dataset):
        """Test error handling for invalid benchmark format."""
        # Mock dataset with invalid format (missing INPUT/OUTPUT columns)
        mock_dataset = Mock()
        mock_dataset.column_names = ["question", "answer"]  # Wrong column names
        mock_load_dataset.return_value = mock_dataset

        with pytest.raises(BenchmarkFormatError):  # BenchmarkFormatError from registry
            self.loader.load_dataset("InvalidFormat")

    def test_validate_format_valid_benchmark(self):
        """Test validation with valid benchmark format."""
        mock_dataset = Mock()
        mock_dataset.column_names = ["INPUT", "OUTPUT", "extra"]
        mock_dataset.__len__ = Mock(return_value=10)
        mock_dataset.__getitem__ = Mock(
            return_value={"INPUT": "test", "OUTPUT": "result"}
        )

        # Should not raise exception
        self.loader.validate_format(mock_dataset)

    def test_validate_format_missing_input(self):
        """Test validation fails when INPUT column missing."""
        mock_dataset = Mock()
        mock_dataset.column_names = ["OUTPUT", "extra"]
        mock_dataset.__len__ = Mock(return_value=10)

        with pytest.raises(ValueError) as exc_info:
            self.loader.validate_format(mock_dataset)

        assert "INPUT column" in str(exc_info.value)

    def test_validate_format_missing_output(self):
        """Test validation fails when OUTPUT column missing."""
        mock_dataset = Mock()
        mock_dataset.column_names = ["INPUT", "extra"]
        mock_dataset.__len__ = Mock(return_value=10)

        with pytest.raises(ValueError) as exc_info:
            self.loader.validate_format(mock_dataset)

        assert "OUTPUT column" in str(exc_info.value)

    def test_validate_format_empty_dataset(self):
        """Test validation fails for empty dataset."""
        mock_dataset = Mock()
        mock_dataset.__len__ = Mock(return_value=0)

        with pytest.raises(ValueError) as exc_info:
            self.loader.validate_format(mock_dataset)

        assert "empty" in str(exc_info.value)

    def test_validate_format_empty_values(self):
        """Test validation fails for empty INPUT/OUTPUT values."""
        mock_dataset = Mock()
        mock_dataset.column_names = ["INPUT", "OUTPUT"]
        mock_dataset.__len__ = Mock(return_value=1)
        mock_dataset.__getitem__ = Mock(return_value={"INPUT": "", "OUTPUT": "result"})

        with pytest.raises(ValueError) as exc_info:
            self.loader.validate_format(mock_dataset)

        assert "INPUT column contains empty values" in str(exc_info.value)

    @patch("ml_agents.core.dataset_loader.BenchmarkRegistry")
    def test_sample_data_functionality(self, mock_registry_class):
        """Test data sampling functionality."""
        # Mock the registry
        mock_registry = Mock()
        mock_registry_class.return_value = mock_registry

        # Create mock dataset
        mock_dataset = Mock()
        mock_dataset.column_names = ["INPUT", "OUTPUT"]
        mock_dataset.__len__ = Mock(return_value=100)
        mock_dataset.__getitem__ = Mock(
            return_value={"INPUT": "test", "OUTPUT": "result"}
        )

        # Mock shuffle and select methods
        mock_shuffled = Mock()
        mock_sampled = Mock()
        mock_sampled.__len__ = Mock(return_value=5)
        mock_shuffled.select.return_value = mock_sampled
        mock_dataset.shuffle.return_value = mock_shuffled

        # Load and sample
        self.loader._dataset = mock_dataset
        result = self.loader.sample_data(sample_size=5)

        # Verify sampling
        mock_dataset.shuffle.assert_called_once_with(seed=42)
        mock_shuffled.select.assert_called_once_with(range(5))
        assert result == mock_sampled

    def test_sample_data_larger_than_dataset(self):
        """Test sampling when requested size >= dataset size."""
        mock_dataset = Mock()
        mock_dataset.__len__ = Mock(return_value=5)

        result = self.loader.sample_data(mock_dataset, sample_size=10)

        # Should return original dataset
        assert result == mock_dataset

    def test_sample_data_no_dataset_loaded(self):
        """Test error when sampling without loaded dataset."""
        self.loader._dataset = None

        with pytest.raises(ValueError) as exc_info:
            self.loader.sample_data(sample_size=5)

        assert "No dataset loaded" in str(exc_info.value)

    def test_get_input_column_name(self):
        """Test getting input column name."""
        assert self.loader.get_input_column_name() == "INPUT"

    def test_get_output_column_name(self):
        """Test getting output column name."""
        assert self.loader.get_output_column_name() == "OUTPUT"

    @patch("ml_agents.core.dataset_loader.BenchmarkRegistry")
    def test_list_available_benchmarks_integration(self, mock_registry_class):
        """Test listing available benchmarks through loader."""
        mock_registry = Mock()
        mock_registry_class.return_value = mock_registry
        mock_registry.list_available_benchmarks.return_value = ["GPQA", "MMLU", "ARC"]

        # Create loader inside test (after patch is applied)
        loader = BBEHDatasetLoader(self.config)
        result = loader.list_available_benchmarks()

        assert result == ["GPQA", "MMLU", "ARC"]
        mock_registry.list_available_benchmarks.assert_called_once()

    @patch("ml_agents.core.dataset_loader.BenchmarkRegistry")
    def test_get_benchmark_info_integration(self, mock_registry_class):
        """Test getting benchmark info through loader."""
        mock_registry = Mock()
        mock_registry_class.return_value = mock_registry
        expected_info = {
            "benchmark_id": "GPQA",
            "num_samples": 1000,
            "columns": ["INPUT", "OUTPUT"],
        }
        mock_registry.get_benchmark_info.return_value = expected_info

        # Create loader inside test (after patch is applied)
        loader = BBEHDatasetLoader(self.config)
        result = loader.get_benchmark_info("GPQA")

        assert result == expected_info
        mock_registry.get_benchmark_info.assert_called_once_with("GPQA")

    @patch("ml_agents.core.dataset_loader.BenchmarkRegistry")
    def test_get_dataset_info_with_loaded_dataset(self, mock_registry_class):
        """Test getting dataset info when dataset is loaded."""
        mock_dataset = Mock()
        mock_dataset.__len__ = Mock(return_value=50)
        mock_dataset.column_names = ["INPUT", "OUTPUT"]
        mock_dataset.features = {"INPUT": "string", "OUTPUT": "string"}
        mock_dataset.__getitem__ = Mock(
            side_effect=lambda i: {"INPUT": f"question_{i}", "OUTPUT": f"answer_{i}"}
        )

        self.loader._dataset = mock_dataset
        result = self.loader.get_dataset_info()

        assert result["size"] == 50
        assert result["columns"] == ["INPUT", "OUTPUT"]
        assert "sample_examples" in result
        assert len(result["sample_examples"]) == 3  # Default sample size

    def test_get_dataset_info_no_dataset(self):
        """Test getting dataset info when no dataset loaded."""
        self.loader._dataset = None
        result = self.loader.get_dataset_info()

        assert "error" in result
        assert result["error"] == "No dataset loaded"

    def test_initialization_with_config(self):
        """Test loader initialization with experiment config."""
        config = ExperimentConfig(benchmark_id="TEST", sample_count=25)
        loader = BBEHDatasetLoader(config)

        assert loader.config == config
        assert loader.sample_count == 25
        assert loader._dataset is None

    def test_sample_data_with_custom_seed(self):
        """Test data sampling with custom random seed."""
        mock_dataset = Mock()
        mock_dataset.__len__ = Mock(return_value=100)

        mock_shuffled = Mock()
        mock_sampled = Mock()
        mock_sampled.__len__ = Mock(return_value=10)  # Add missing __len__ method
        mock_shuffled.select.return_value = mock_sampled
        mock_dataset.shuffle.return_value = mock_shuffled

        self.loader.sample_data(mock_dataset, sample_size=10, random_seed=123)

        mock_dataset.shuffle.assert_called_once_with(seed=123)

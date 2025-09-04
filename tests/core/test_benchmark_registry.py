"""Tests for benchmark registry functionality."""

from unittest.mock import Mock, patch

import pytest
from datasets import Dataset

from ml_agents.core.benchmark_registry import (
    BenchmarkFormatError,
    BenchmarkNotFoundError,
    BenchmarkRegistry,
)


class TestBenchmarkRegistry:
    """Test cases for BenchmarkRegistry class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.registry = BenchmarkRegistry()

    @patch("ml_agents.core.benchmark_registry.load_dataset")
    def test_load_benchmark_success(self, mock_load_dataset):
        """Test successful benchmark loading."""
        # Mock dataset with correct format
        mock_dataset = Mock()
        mock_dataset.column_names = ["INPUT", "OUTPUT", "other_column"]
        mock_dataset.__len__ = Mock(return_value=100)
        mock_load_dataset.return_value = mock_dataset

        # Load benchmark
        result = self.registry.load_benchmark("GPQA")

        # Verify calls and result
        mock_load_dataset.assert_called_once_with(
            "c4ai-ml-agents/benchmarks-base", data_files="GPQA.csv", split="train"
        )
        assert result == mock_dataset

    @patch("ml_agents.core.benchmark_registry.load_dataset")
    def test_load_benchmark_not_found(self, mock_load_dataset):
        """Test error handling for missing benchmark."""
        mock_load_dataset.side_effect = FileNotFoundError("File not found")

        with pytest.raises(BenchmarkNotFoundError) as exc_info:
            self.registry.load_benchmark("NonExistentBenchmark")

        assert "NonExistentBenchmark" in str(exc_info.value)

    @patch("ml_agents.core.benchmark_registry.load_dataset")
    def test_load_benchmark_invalid_format(self, mock_load_dataset):
        """Test error handling for invalid benchmark format."""
        # Mock dataset without required columns
        mock_dataset = Mock()
        mock_dataset.column_names = ["question", "answer"]  # Missing INPUT/OUTPUT
        mock_load_dataset.return_value = mock_dataset

        with pytest.raises(BenchmarkFormatError) as exc_info:
            self.registry.load_benchmark("InvalidBenchmark")

        assert "INPUT and OUTPUT columns" in str(exc_info.value)

    @patch("ml_agents.core.benchmark_registry.load_dataset")
    def test_validate_benchmark_format_valid(self, mock_load_dataset):
        """Test validation with valid benchmark format."""
        mock_dataset = Mock()
        mock_dataset.column_names = ["INPUT", "OUTPUT"]

        assert self.registry._validate_benchmark_format(mock_dataset) is True

    @patch("ml_agents.core.benchmark_registry.load_dataset")
    def test_validate_benchmark_format_invalid(self, mock_load_dataset):
        """Test validation with invalid benchmark format."""
        mock_dataset = Mock()
        mock_dataset.column_names = ["question", "answer"]

        assert self.registry._validate_benchmark_format(mock_dataset) is False

    @patch("ml_agents.core.benchmark_registry.load_dataset")
    def test_get_benchmark_info_success(self, mock_load_dataset):
        """Test successful benchmark info retrieval."""
        # Mock sample dataset for info
        mock_sample_dataset = Mock()
        mock_sample_dataset.column_names = ["INPUT", "OUTPUT"]
        mock_sample_dataset.__len__ = Mock(return_value=10)
        mock_sample_dataset.__getitem__ = Mock(
            return_value={"INPUT": "What is 2+2?", "OUTPUT": "4"}
        )

        # Mock full dataset for count
        mock_full_dataset = Mock()
        mock_full_dataset.__len__ = Mock(return_value=1000)

        # Configure mock to return different datasets for different calls
        mock_load_dataset.side_effect = [mock_sample_dataset, mock_full_dataset]

        result = self.registry.get_benchmark_info("GPQA")

        assert result["benchmark_id"] == "GPQA"
        assert result["columns"] == ["INPUT", "OUTPUT"]
        assert result["num_samples"] == 1000
        assert result["has_input_output"] is True
        assert "sample" in result
        assert result["sample"]["INPUT"] == "What is 2+2?"
        assert result["sample"]["OUTPUT"] == "4"

    @patch("ml_agents.core.benchmark_registry.load_dataset")
    def test_get_benchmark_info_not_found(self, mock_load_dataset):
        """Test error handling when benchmark info not found."""
        mock_load_dataset.side_effect = FileNotFoundError("Not found")

        with pytest.raises(BenchmarkNotFoundError):
            self.registry.get_benchmark_info("NonExistent")

    def test_list_available_benchmarks_fallback(self):
        """Test listing available benchmarks using fallback detection."""
        # Mock the _load_benchmark_list method
        expected_benchmarks = ["GPQA", "MMLU", "HellaSwag"]
        self.registry._available_benchmarks = expected_benchmarks
        self.registry._benchmarks_loaded = True

        result = self.registry.list_available_benchmarks()

        assert result == sorted(expected_benchmarks)

    def test_list_available_benchmarks_empty(self):
        """Test listing when no benchmarks are available."""
        self.registry._available_benchmarks = []
        self.registry._benchmarks_loaded = True

        result = self.registry.list_available_benchmarks()

        assert result == []

    @patch("ml_agents.core.benchmark_registry.load_dataset")
    def test_load_benchmark_list_fallback_detection(self, mock_load_dataset):
        """Test fallback benchmark detection logic."""

        # Configure mock to succeed for some benchmarks and fail for others
        def load_dataset_side_effect(*args, **kwargs):
            # Handle the repository info call (first call with split=None)
            if kwargs.get("split") is None and not kwargs.get("data_files"):
                # This is the repo_info call, just return a mock
                return Mock()

            data_files = kwargs.get("data_files", "")
            if "GPQA" in data_files:
                mock_dataset = Mock()
                mock_dataset.__len__ = Mock(return_value=1)
                return mock_dataset
            elif "MMLU" in data_files:
                mock_dataset = Mock()
                mock_dataset.__len__ = Mock(return_value=1)
                return mock_dataset
            else:
                raise FileNotFoundError("Not found")

        mock_load_dataset.side_effect = load_dataset_side_effect

        # Trigger benchmark list loading
        self.registry._load_benchmark_list()

        # Should find GPQA and MMLU
        assert "GPQA" in self.registry._available_benchmarks
        assert "MMLU" in self.registry._available_benchmarks
        assert self.registry._benchmarks_loaded is True

    @patch("ml_agents.core.benchmark_registry.load_dataset")
    def test_load_benchmark_format_error_handling(self, mock_load_dataset):
        """Test generic error handling in benchmark loading."""
        mock_load_dataset.side_effect = ValueError("Invalid data format")

        with pytest.raises(BenchmarkFormatError) as exc_info:
            self.registry.load_benchmark("BadFormat")

        assert "Invalid benchmark format" in str(exc_info.value)
        assert "BadFormat" in str(exc_info.value)

    def test_repository_constant(self):
        """Test that repository constant is set correctly."""
        assert self.registry.REPOSITORY == "c4ai-ml-agents/benchmarks-base"

    @patch("ml_agents.core.benchmark_registry.load_dataset")
    def test_benchmark_id_to_filename_resolution(self, mock_load_dataset):
        """Test that benchmark IDs are correctly resolved to filenames."""
        mock_dataset = Mock()
        mock_dataset.column_names = ["INPUT", "OUTPUT"]
        mock_dataset.__len__ = Mock(return_value=100)
        mock_load_dataset.return_value = mock_dataset

        self.registry.load_benchmark("Custom-Benchmark-Name")

        mock_load_dataset.assert_called_once_with(
            "c4ai-ml-agents/benchmarks-base",
            data_files="Custom-Benchmark-Name.csv",
            split="train",
        )

    def test_initialization(self):
        """Test benchmark registry initialization."""
        registry = BenchmarkRegistry()

        assert registry.REPOSITORY == "c4ai-ml-agents/benchmarks-base"
        assert registry._available_benchmarks == []
        assert registry._benchmarks_loaded is False

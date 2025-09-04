"""Tests for BBEHDatasetLoader."""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pandas as pd
import pytest
from datasets import Dataset

from ml_agents.config import ExperimentConfig
from ml_agents.core.benchmark_registry import (
    BenchmarkFormatError,
    BenchmarkNotFoundError,
)
from ml_agents.core.dataset_loader import BBEHDatasetLoader


class TestBBEHDatasetLoader:
    """Test suite for BBEHDatasetLoader."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return ExperimentConfig(
            benchmark_id="TEST_BENCHMARK",  # Phase 12: Use benchmark_id instead of dataset_name
            sample_count=10,
            provider="openrouter",
            model="openai/gpt-oss-120b",
        )

    @pytest.fixture
    def loader(self, config):
        """Create dataset loader instance."""
        return BBEHDatasetLoader(config)

    @pytest.fixture
    def sample_dataset(self):
        """Create sample dataset for testing (Phase 12 INPUT/OUTPUT format)."""
        data = {
            "INPUT": [
                "What is 2+2?",
                "What is the capital of France?",
                "Explain photosynthesis",
            ],
            "OUTPUT": ["4", "Paris", "Process by which plants make food"],
            "id": [1, 2, 3],
        }
        return Dataset.from_dict(data)

    @pytest.fixture
    def invalid_dataset(self):
        """Create dataset with invalid format (missing INPUT/OUTPUT)."""
        data = {
            "question": ["What is 2+2?", "What is the capital of France?"],
            "answer": ["4", "Paris"],
            "metadata": [{"difficulty": "easy"}, {"difficulty": "medium"}],
        }
        return Dataset.from_dict(data)

    @patch("ml_agents.core.dataset_loader.BenchmarkRegistry")
    def test_init(self, mock_registry_class, config):
        """Test loader initialization (Phase 12)."""
        loader = BBEHDatasetLoader(config)

        assert loader.config == config
        assert loader.sample_count == 10
        assert loader._dataset is None
        # Phase 12: No more cache_dir, has benchmark_registry instead
        assert hasattr(loader, "benchmark_registry")
        mock_registry_class.assert_called_once()

    @patch("ml_agents.core.dataset_loader.BenchmarkRegistry")
    def test_load_dataset_success(self, mock_registry_class, config, sample_dataset):
        """Test successful dataset loading (Phase 12)."""
        # Mock the registry
        mock_registry = Mock()
        mock_registry_class.return_value = mock_registry
        mock_registry.load_benchmark.return_value = sample_dataset

        # Create loader inside test (after patch is applied)
        loader = BBEHDatasetLoader(config)

        result = loader.load_dataset("TEST_BENCHMARK")

        assert result == sample_dataset
        assert loader._dataset == sample_dataset
        mock_registry.load_benchmark.assert_called_once_with("TEST_BENCHMARK")

    @patch("ml_agents.core.dataset_loader.BenchmarkRegistry")
    def test_load_dataset_not_found(self, mock_registry_class, config):
        """Test dataset loading when benchmark not found (Phase 12)."""
        # Mock the registry to raise BenchmarkNotFoundError
        mock_registry = Mock()
        mock_registry_class.return_value = mock_registry
        mock_registry.load_benchmark.side_effect = BenchmarkNotFoundError(
            "Benchmark not found"
        )

        # Create loader inside test (after patch is applied)
        loader = BBEHDatasetLoader(config)

        with pytest.raises(BenchmarkNotFoundError):
            loader.load_dataset("NONEXISTENT")

    @patch("ml_agents.core.dataset_loader.BenchmarkRegistry")
    def test_load_dataset_format_error(self, mock_registry_class, config):
        """Test dataset loading with format error (Phase 12)."""
        # Mock the registry to raise BenchmarkFormatError
        mock_registry = Mock()
        mock_registry_class.return_value = mock_registry
        mock_registry.load_benchmark.side_effect = BenchmarkFormatError(
            "Invalid format"
        )

        # Create loader inside test (after patch is applied)
        loader = BBEHDatasetLoader(config)

        with pytest.raises(BenchmarkFormatError):
            loader.load_dataset("INVALID_FORMAT")

    @patch("ml_agents.core.dataset_loader.BenchmarkRegistry")
    def test_validate_format_valid_dataset(
        self, mock_registry_class, config, sample_dataset
    ):
        """Test validation of valid dataset (Phase 12 INPUT/OUTPUT format)."""
        loader = BBEHDatasetLoader(config)
        # Should not raise any exception
        loader.validate_format(sample_dataset)

    @patch("ml_agents.core.dataset_loader.BenchmarkRegistry")
    def test_validate_format_missing_input(
        self, mock_registry_class, config, invalid_dataset
    ):
        """Test validation with missing INPUT column (Phase 12)."""
        loader = BBEHDatasetLoader(config)
        # Should raise ValueError for missing INPUT column
        with pytest.raises(ValueError, match="INPUT column"):
            loader.validate_format(invalid_dataset)

    @patch("ml_agents.core.dataset_loader.BenchmarkRegistry")
    def test_validate_format_empty_dataset(self, mock_registry_class, config):
        """Test validation of empty dataset (Phase 12)."""
        loader = BBEHDatasetLoader(config)
        empty_dataset = Dataset.from_dict({"INPUT": [], "OUTPUT": []})

        with pytest.raises(ValueError, match="Dataset is empty"):
            loader.validate_format(empty_dataset)

    @patch("ml_agents.core.dataset_loader.BenchmarkRegistry")
    def test_validate_format_missing_output(self, mock_registry_class, config):
        """Test validation with missing OUTPUT column (Phase 12)."""
        loader = BBEHDatasetLoader(config)
        data = {"INPUT": ["question1", "question2"], "other": ["data1", "data2"]}
        bad_dataset = Dataset.from_dict(data)

        with pytest.raises(ValueError, match="OUTPUT column"):
            loader.validate_format(bad_dataset)

    @patch("ml_agents.core.dataset_loader.BenchmarkRegistry")
    def test_sample_data_basic(self, mock_registry_class, config, sample_dataset):
        """Test basic data sampling (Phase 12)."""
        loader = BBEHDatasetLoader(config)
        loader._dataset = sample_dataset

        sampled = loader.sample_data(sample_size=2)

        assert len(sampled) == 2
        assert isinstance(sampled, Dataset)

    @patch("ml_agents.core.dataset_loader.BenchmarkRegistry")
    def test_sample_data_larger_than_dataset(
        self, mock_registry_class, config, sample_dataset
    ):
        """Test sampling more data than available (Phase 12)."""
        loader = BBEHDatasetLoader(config)
        loader._dataset = sample_dataset

        sampled = loader.sample_data(sample_size=10)

        # Should return full dataset
        assert len(sampled) == len(sample_dataset)

    @patch("ml_agents.core.dataset_loader.BenchmarkRegistry")
    def test_sample_data_no_dataset_loaded(self, mock_registry_class, config):
        """Test sampling without loaded dataset (Phase 12)."""
        loader = BBEHDatasetLoader(config)
        with pytest.raises(ValueError, match="No dataset loaded"):
            loader.sample_data()

    @patch("ml_agents.core.dataset_loader.BenchmarkRegistry")
    def test_sample_data_invalid_size(
        self, mock_registry_class, config, sample_dataset
    ):
        """Test sampling with invalid size (Phase 12)."""
        loader = BBEHDatasetLoader(config)
        with pytest.raises(ValueError, match="Sample size must be positive"):
            loader.sample_data(dataset=sample_dataset, sample_size=0)

    @patch("ml_agents.core.dataset_loader.BenchmarkRegistry")
    def test_sample_data_reproducible(
        self, mock_registry_class, config, sample_dataset
    ):
        """Test that sampling is reproducible with same seed (Phase 12)."""
        loader = BBEHDatasetLoader(config)
        sample1 = loader.sample_data(
            dataset=sample_dataset, sample_size=2, random_seed=42
        )
        sample2 = loader.sample_data(
            dataset=sample_dataset, sample_size=2, random_seed=42
        )

        # Should get same samples with same seed
        assert sample1[0] == sample2[0]
        assert sample1[1] == sample2[1]

    @patch("ml_agents.core.dataset_loader.BenchmarkRegistry")
    def test_get_dataset_info_with_data(
        self, mock_registry_class, config, sample_dataset
    ):
        """Test getting dataset info with loaded data (Phase 12)."""
        loader = BBEHDatasetLoader(config)
        loader._dataset = sample_dataset

        info = loader.get_dataset_info()

        assert info["size"] == 3
        assert "INPUT" in info["columns"]
        assert "OUTPUT" in info["columns"]
        assert len(info["sample_examples"]) == 3

    @patch("ml_agents.core.dataset_loader.BenchmarkRegistry")
    def test_get_dataset_info_no_data(self, mock_registry_class, config):
        """Test getting dataset info without loaded data (Phase 12)."""
        loader = BBEHDatasetLoader(config)
        info = loader.get_dataset_info()

        assert "error" in info
        assert info["error"] == "No dataset loaded"

    @patch("ml_agents.core.dataset_loader.BenchmarkRegistry")
    def test_get_input_column_name(self, mock_registry_class, config):
        """Test getting input column name (Phase 12)."""
        loader = BBEHDatasetLoader(config)
        assert loader.get_input_column_name() == "INPUT"

    @patch("ml_agents.core.dataset_loader.BenchmarkRegistry")
    def test_get_output_column_name(self, mock_registry_class, config):
        """Test getting output column name (Phase 12)."""
        loader = BBEHDatasetLoader(config)
        assert loader.get_output_column_name() == "OUTPUT"

    @patch("ml_agents.core.dataset_loader.BenchmarkRegistry")
    def test_list_available_benchmarks(self, mock_registry_class, config):
        """Test listing available benchmarks (Phase 12)."""
        mock_registry = Mock()
        mock_registry_class.return_value = mock_registry
        mock_registry.list_available_benchmarks.return_value = ["GPQA", "MMLU", "ARC"]

        loader = BBEHDatasetLoader(config)
        result = loader.list_available_benchmarks()

        assert result == ["GPQA", "MMLU", "ARC"]
        mock_registry.list_available_benchmarks.assert_called_once()

    @patch("ml_agents.core.dataset_loader.BenchmarkRegistry")
    def test_get_benchmark_info(self, mock_registry_class, config):
        """Test getting benchmark info (Phase 12)."""
        mock_registry = Mock()
        mock_registry_class.return_value = mock_registry
        expected_info = {
            "benchmark_id": "TEST_BENCHMARK",
            "num_samples": 1000,
            "columns": ["INPUT", "OUTPUT"],
        }
        mock_registry.get_benchmark_info.return_value = expected_info

        loader = BBEHDatasetLoader(config)
        result = loader.get_benchmark_info("TEST_BENCHMARK")

        assert result == expected_info
        mock_registry.get_benchmark_info.assert_called_once_with("TEST_BENCHMARK")

    @patch("ml_agents.core.dataset_loader.BenchmarkRegistry")
    def test_config_integration(self, mock_registry_class, config):
        """Test that loader uses config parameters correctly (Phase 12)."""
        loader = BBEHDatasetLoader(config)

        assert loader.config == config
        assert loader.sample_count == config.sample_count

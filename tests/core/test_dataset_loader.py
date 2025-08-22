"""Tests for BBEHDatasetLoader."""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pandas as pd
import pytest
from datasets import Dataset

from ml_agents.config import ExperimentConfig
from ml_agents.core.dataset_loader import BBEHDatasetLoader


class TestBBEHDatasetLoader:
    """Test suite for BBEHDatasetLoader."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return ExperimentConfig(
            dataset_name="test/dataset",
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
        """Create sample dataset for testing."""
        data = {
            "input": [
                "What is 2+2?",
                "What is the capital of France?",
                "Explain photosynthesis",
            ],
            "answer": ["4", "Paris", "Process by which plants make food"],
            "id": [1, 2, 3],
        }
        return Dataset.from_dict(data)

    @pytest.fixture
    def alternative_dataset(self):
        """Create dataset with alternative column names."""
        data = {
            "question": ["What is 2+2?", "What is the capital of France?"],
            "target": ["4", "Paris"],
            "metadata": [{"difficulty": "easy"}, {"difficulty": "medium"}],
        }
        return Dataset.from_dict(data)

    def test_init(self, config):
        """Test loader initialization."""
        loader = BBEHDatasetLoader(config)

        assert loader.config == config
        assert loader.dataset_name == "test/dataset"
        assert loader.sample_count == 10
        assert loader._dataset is None
        assert loader._cache_dir.exists()

    @patch("ml_agents.core.dataset_loader.load_dataset")
    def test_load_dataset_success(self, mock_load_dataset, loader, sample_dataset):
        """Test successful dataset loading."""
        mock_load_dataset.return_value = sample_dataset

        with (
            patch.object(loader, "_load_from_cache", return_value=None),
            patch.object(loader, "_save_to_cache"),
        ):
            result = loader.load_dataset()

            assert result == sample_dataset
            assert loader._dataset == sample_dataset
            mock_load_dataset.assert_called_once_with(
                "test/dataset", split="test", trust_remote_code=True
            )

    @patch("ml_agents.core.dataset_loader.load_dataset")
    def test_load_dataset_from_cache(self, mock_load_dataset, loader, sample_dataset):
        """Test loading dataset from cache."""
        with patch.object(loader, "_load_from_cache", return_value=sample_dataset):
            result = loader.load_dataset()

            assert result == sample_dataset
            assert loader._dataset == sample_dataset
            mock_load_dataset.assert_not_called()

    @patch("ml_agents.core.dataset_loader.load_dataset")
    def test_load_dataset_failure(self, mock_load_dataset, loader):
        """Test dataset loading failure."""
        mock_load_dataset.side_effect = Exception("API Error")

        with patch.object(loader, "_load_from_cache", return_value=None):
            with pytest.raises(RuntimeError, match="Failed to load dataset"):
                loader.load_dataset()

    def test_validate_format_valid_dataset(self, loader, sample_dataset):
        """Test validation of valid dataset."""
        # Should not raise any exception
        loader.validate_format(sample_dataset)

    def test_validate_format_alternative_columns(self, loader, alternative_dataset):
        """Test validation with alternative column names."""
        # Should not raise exception, should work with 'question' column
        loader.validate_format(alternative_dataset)

    def test_validate_format_empty_dataset(self, loader):
        """Test validation of empty dataset."""
        empty_dataset = Dataset.from_dict({})

        with pytest.raises(ValueError, match="Dataset is empty"):
            loader.validate_format(empty_dataset)

    def test_validate_format_no_text_columns(self, loader):
        """Test validation with no valid text columns."""
        bad_data = {
            "numeric_col": [1, 2, 3],
            "empty_text": ["", "", ""],
            "none_col": [None, None, None],
        }
        bad_dataset = Dataset.from_dict(bad_data)

        with pytest.raises(
            ValueError, match="must contain at least one text input column"
        ):
            loader.validate_format(bad_dataset)

    def test_sample_data_basic(self, loader, sample_dataset):
        """Test basic data sampling."""
        loader._dataset = sample_dataset

        sampled = loader.sample_data(sample_size=2)

        assert len(sampled) == 2
        assert isinstance(sampled, Dataset)

    def test_sample_data_larger_than_dataset(self, loader, sample_dataset):
        """Test sampling more data than available."""
        loader._dataset = sample_dataset

        sampled = loader.sample_data(sample_size=10)

        # Should return full dataset
        assert len(sampled) == len(sample_dataset)

    def test_sample_data_no_dataset_loaded(self, loader):
        """Test sampling without loaded dataset."""
        with pytest.raises(ValueError, match="No dataset loaded"):
            loader.sample_data()

    def test_sample_data_invalid_size(self, loader, sample_dataset):
        """Test sampling with invalid size."""
        with pytest.raises(ValueError, match="Sample size must be positive"):
            loader.sample_data(dataset=sample_dataset, sample_size=0)

    def test_sample_data_reproducible(self, loader, sample_dataset):
        """Test that sampling is reproducible with same seed."""
        sample1 = loader.sample_data(
            dataset=sample_dataset, sample_size=2, random_seed=42
        )
        sample2 = loader.sample_data(
            dataset=sample_dataset, sample_size=2, random_seed=42
        )

        # Should get same samples with same seed
        assert sample1[0] == sample2[0]
        assert sample1[1] == sample2[1]

    def test_get_dataset_info_with_data(self, loader, sample_dataset):
        """Test getting dataset info with loaded data."""
        loader._dataset = sample_dataset

        info = loader.get_dataset_info()

        assert info["name"] == "test/dataset"
        assert info["size"] == 3
        assert "input" in info["columns"]
        assert "answer" in info["columns"]
        assert len(info["sample_examples"]) == 3

    def test_get_dataset_info_no_data(self, loader):
        """Test getting dataset info without loaded data."""
        info = loader.get_dataset_info()

        assert "error" in info
        assert info["error"] == "No dataset loaded"

    def test_cache_path_generation(self, loader):
        """Test cache path generation."""
        path1 = loader._get_cache_path("test")
        path2 = loader._get_cache_path("train")

        assert path1 != path2
        assert path1.suffix == ".json"
        assert path1.parent == loader._cache_dir

    def test_cache_save_and_load(self, loader, sample_dataset):
        """Test caching functionality."""
        with tempfile.TemporaryDirectory() as temp_dir:
            loader._cache_dir = Path(temp_dir)

            # Save to cache
            loader._save_to_cache(sample_dataset, "test")

            # Load from cache
            cached_dataset = loader._load_from_cache("test")

            assert cached_dataset is not None
            assert len(cached_dataset) == len(sample_dataset)
            assert cached_dataset.column_names == sample_dataset.column_names

    def test_cache_load_nonexistent(self, loader):
        """Test loading from nonexistent cache."""
        with tempfile.TemporaryDirectory() as temp_dir:
            loader._cache_dir = Path(temp_dir)

            result = loader._load_from_cache("nonexistent")

            assert result is None

    def test_cache_load_corrupted(self, loader):
        """Test loading corrupted cache file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            loader._cache_dir = Path(temp_dir)
            cache_path = loader._get_cache_path("test")

            # Create corrupted cache file
            cache_path.write_text("invalid json")

            result = loader._load_from_cache("test")

            assert result is None
            assert not cache_path.exists()  # Should be cleaned up

    def test_clear_cache(self, loader):
        """Test cache clearing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            loader._cache_dir = Path(temp_dir)

            # Create some cache files
            (loader._cache_dir / "cache1.json").write_text("{}")
            (loader._cache_dir / "cache2.json").write_text("{}")

            loader.clear_cache()

            assert len(list(loader._cache_dir.glob("*.json"))) == 0

    @patch("ml_agents.core.dataset_loader.load_dataset")
    def test_load_dataset_different_split(
        self, mock_load_dataset, loader, sample_dataset
    ):
        """Test loading different dataset split."""
        mock_load_dataset.return_value = sample_dataset

        with (
            patch.object(loader, "_load_from_cache", return_value=None),
            patch.object(loader, "_save_to_cache"),
        ):
            loader.load_dataset(split="train")

            mock_load_dataset.assert_called_once_with(
                "test/dataset", split="train", trust_remote_code=True
            )

    def test_config_integration(self, config):
        """Test that loader uses config parameters correctly."""
        loader = BBEHDatasetLoader(config)

        assert loader.dataset_name == config.dataset_name
        assert loader.sample_count == config.sample_count

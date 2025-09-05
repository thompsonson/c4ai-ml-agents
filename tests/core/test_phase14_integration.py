"""Integration tests for Phase 14 implementation."""

import pytest
from datasets import Dataset

from ml_agents.config import ExperimentConfig
from ml_agents.core.dataset_loader import BBEHDatasetLoader
from ml_agents.core.local_test_dataset import LocalTestDataset
from ml_agents.core.phase14_test_data import get_test_dataset


class TestPhase14Integration:
    """Test Phase 14 implementation components."""

    def test_local_test_dataset_creation(self):
        """Test LOCAL_TEST dataset creation."""
        # Test dataset creation
        data = get_test_dataset("all")
        assert len(data) == 20  # 5 basic + 5 reasoning + 10 extended

        # Verify format
        for item in data:
            assert "INPUT" in item
            assert "OUTPUT" in item
            assert isinstance(item["INPUT"], str)
            assert isinstance(item["OUTPUT"], str)
            assert len(item["INPUT"]) > 0
            assert len(item["OUTPUT"]) > 0

    def test_local_test_dataset_loader(self):
        """Test LocalTestDataset loader functionality."""
        # Test dataset identification
        assert LocalTestDataset.is_local_test_dataset("LOCAL_TEST")
        assert LocalTestDataset.is_local_test_dataset("LOCAL_TEST_LARGE")
        assert not LocalTestDataset.is_local_test_dataset("OTHER_DATASET")

        # Test dataset loading
        dataset = LocalTestDataset.load_dataset("LOCAL_TEST")
        assert isinstance(dataset, Dataset)
        assert len(dataset) == 20
        assert "INPUT" in dataset.column_names
        assert "OUTPUT" in dataset.column_names

        # Test large dataset
        large_dataset = LocalTestDataset.load_dataset(
            "LOCAL_TEST_LARGE", sample_size=100
        )
        assert len(large_dataset) == 100

    def test_dataset_loader_integration(self):
        """Test BBEHDatasetLoader with LOCAL_TEST datasets."""
        config = ExperimentConfig(
            benchmark_id="LOCAL_TEST",
            sample_count=20,
            provider="anthropic",
            model="claude-sonnet-4-20250514",
        )

        loader = BBEHDatasetLoader(config)

        # Test loading
        dataset = loader.load_dataset("LOCAL_TEST")
        assert isinstance(dataset, Dataset)
        assert len(dataset) == 20

        # Test validation
        loader.validate_format(dataset)  # Should not raise

        # Test sampling
        sampled = loader.sample_data(dataset, sample_size=5)
        assert len(sampled) == 5

    def test_dataset_info(self):
        """Test dataset information retrieval."""
        info = LocalTestDataset.get_dataset_info("LOCAL_TEST")

        assert info["id"] == "LOCAL_TEST"
        assert info["size"] == 20
        assert info["format"] == "INPUT/OUTPUT"
        assert "Phase 14" in info["description"]

    def test_question_categories(self):
        """Test different question categories."""
        basic = get_test_dataset("basic")
        reasoning = get_test_dataset("reasoning")
        extended = get_test_dataset("extended")

        assert len(basic) == 5
        assert len(reasoning) == 5
        assert len(extended) == 10

        # Test sample limits
        limited = get_test_dataset("all", sample_size=3)
        assert len(limited) == 3

    def test_dataset_content_quality(self):
        """Test that dataset content is appropriate for reasoning evaluation."""
        data = get_test_dataset("all")

        # Check for variety in question types
        inputs = [item["INPUT"] for item in data]
        outputs = [item["OUTPUT"] for item in data]

        # Should have questions and answers of reasonable length
        assert all(len(inp) > 5 for inp in inputs)
        assert all(len(out) > 0 for out in outputs)

        # Should have some mathematical questions
        math_questions = [
            inp for inp in inputs if any(op in inp for op in ["+", "-", "*", "/", "="])
        ]
        assert len(math_questions) > 0

        # Should have some reasoning questions
        reasoning_words = ["if", "then", "because", "therefore", "conclude"]
        reasoning_questions = [
            inp
            for inp in inputs
            if any(word in inp.lower() for word in reasoning_words)
        ]
        assert len(reasoning_questions) > 0

    def test_benchmark_list_integration(self):
        """Test that LOCAL_TEST datasets appear in benchmark lists."""
        config = ExperimentConfig(
            benchmark_id="LOCAL_TEST",
            sample_count=20,
            provider="anthropic",
            model="claude-sonnet-4-20250514",
        )

        loader = BBEHDatasetLoader(config)
        benchmarks = loader.list_available_benchmarks()

        assert "LOCAL_TEST" in benchmarks
        assert "LOCAL_TEST_LARGE" in benchmarks

    def test_error_handling(self):
        """Test error handling for invalid inputs."""
        # Invalid dataset type
        with pytest.raises(ValueError):
            get_test_dataset("invalid_type")

        # Invalid dataset ID
        with pytest.raises(ValueError):
            LocalTestDataset.load_dataset("INVALID_DATASET")

        # Invalid info request
        with pytest.raises(ValueError):
            LocalTestDataset.get_dataset_info("INVALID_DATASET")

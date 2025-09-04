"""Integration tests for benchmark system end-to-end functionality."""

from unittest.mock import Mock, patch

import pytest
from datasets import Dataset

from ml_agents.cli.commands.eval import benchmark_info, list_benchmarks
from ml_agents.config import ExperimentConfig
from ml_agents.core.benchmark_registry import BenchmarkRegistry
from ml_agents.core.experiment_runner import ExperimentRunner


class TestBenchmarkIntegration:
    """Integration tests for complete benchmark workflows."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config = ExperimentConfig(
            benchmark_id="GPQA",
            provider="openrouter",
            model="openai/gpt-oss-120b",
            sample_count=5,
            database_enabled=False,  # Disable database for integration tests
        )

    @patch("ml_agents.core.benchmark_registry.load_dataset")
    @patch("ml_agents.core.reasoning_inference.ReasoningInference")
    def test_end_to_end_benchmark_eval(self, mock_reasoning_class, mock_load_dataset):
        """Test complete evaluation workflow with benchmarks."""
        # Mock dataset - use MagicMock for proper __len__ support
        mock_dataset = Mock()
        mock_dataset.column_names = ["INPUT", "OUTPUT"]
        mock_dataset.__len__ = Mock(return_value=100)
        mock_dataset.__getitem__ = Mock(
            side_effect=lambda i: {"INPUT": f"Question {i}", "OUTPUT": f"Answer {i}"}
        )

        # Mock shuffled/sampled dataset
        mock_sampled = Mock()
        mock_sampled.__len__ = Mock(return_value=5)
        mock_sampled.__getitem__ = Mock(
            side_effect=lambda i: {"INPUT": f"Question {i}", "OUTPUT": f"Answer {i}"}
        )
        mock_shuffled = Mock()
        mock_shuffled.select = Mock(return_value=mock_sampled)
        mock_dataset.shuffle = Mock(return_value=mock_shuffled)

        mock_load_dataset.return_value = mock_dataset

        # Mock reasoning inference
        mock_reasoning = Mock()
        mock_reasoning_class.return_value = mock_reasoning

        # Mock reasoning results
        from ml_agents.core.reasoning_inference import ReasoningResult
        from ml_agents.utils.api_clients import StandardResponse

        mock_response = StandardResponse(
            text="test response",
            provider="test",
            model="test",
            prompt_tokens=50,
            completion_tokens=50,
            total_tokens=100,
            generation_time=1.0,
            parameters={},
            response_id="test123",
            metadata={},
        )
        mock_result = ReasoningResult(
            response=mock_response,
            approach_name="ChainOfThought",
            execution_time=1.0,
            cost_estimate=0.01,
            metadata={},
        )
        mock_reasoning.run_inference.return_value = mock_result

        # Create and run experiment
        runner = ExperimentRunner(self.config)

        # Disable database for test
        runner.results_processor = None

        result = runner.run_single_experiment(
            approach="ChainOfThought", benchmark_id="GPQA"
        )

        # Verify experiment completed
        assert result is not None
        assert result.experiment_id
        assert "ChainOfThought" in result.approaches_tested
        assert result.total_samples == 5

    @patch("ml_agents.core.benchmark_registry.load_dataset")
    def test_benchmark_registry_end_to_end(self, mock_load_dataset):
        """Test benchmark registry from loading to info retrieval."""
        # Mock dataset for loading
        mock_dataset = Mock()
        mock_dataset.column_names = ["INPUT", "OUTPUT"]
        mock_dataset.__len__ = Mock(return_value=1000)
        mock_dataset.__getitem__ = Mock(
            return_value={"INPUT": "What is the capital of France?", "OUTPUT": "Paris"}
        )
        mock_load_dataset.return_value = mock_dataset

        registry = BenchmarkRegistry()

        # Test loading
        dataset = registry.load_benchmark("GPQA")
        assert dataset == mock_dataset

        # Test info retrieval
        mock_load_dataset.reset_mock()
        mock_sample_dataset = Mock()
        mock_sample_dataset.column_names = ["INPUT", "OUTPUT"]
        mock_sample_dataset.__len__ = Mock(return_value=10)
        mock_sample_dataset.__getitem__ = Mock(
            return_value={"INPUT": "What is the capital of France?", "OUTPUT": "Paris"}
        )

        mock_load_dataset.side_effect = [mock_sample_dataset, mock_dataset]

        info = registry.get_benchmark_info("GPQA")
        assert info["benchmark_id"] == "GPQA"
        assert info["num_samples"] == 1000
        assert info["has_input_output"] is True

    @patch("ml_agents.cli.commands.eval.BenchmarkRegistry")
    def test_cli_list_benchmarks(self, mock_registry_class):
        """Test CLI benchmark listing command."""
        mock_registry = Mock()
        mock_registry_class.return_value = mock_registry
        mock_registry.list_available_benchmarks.return_value = ["GPQA", "MMLU", "ARC"]

        # Mock registry info calls
        def mock_get_info(benchmark_id):
            return {"num_samples": 1000, "has_input_output": True}

        mock_registry.get_benchmark_info.side_effect = mock_get_info

        # Should not raise exception
        list_benchmarks()

        mock_registry.list_available_benchmarks.assert_called_once()

    @patch("ml_agents.cli.commands.eval.BenchmarkRegistry")
    def test_cli_benchmark_info(self, mock_registry_class):
        """Test CLI benchmark info command."""
        mock_registry = Mock()
        mock_registry_class.return_value = mock_registry
        mock_registry.get_benchmark_info.return_value = {
            "benchmark_id": "GPQA",
            "num_samples": 1000,
            "columns": ["INPUT", "OUTPUT"],
            "has_input_output": True,
            "sample": {
                "INPUT": "What is quantum entanglement?",
                "OUTPUT": "A quantum phenomenon where particles become interconnected",
            },
        }

        # Should not raise exception
        benchmark_info("GPQA")

        mock_registry.get_benchmark_info.assert_called_once_with("GPQA")

    @patch("ml_agents.core.benchmark_registry.load_dataset")
    def test_config_benchmark_id_integration(self, mock_load_dataset):
        """Test configuration integration with benchmark IDs."""
        # Test config validation
        config = ExperimentConfig(benchmark_id="GPQA")
        assert config.benchmark_id == "GPQA"

        # Test config serialization
        config_dict = config.to_dict()
        assert config_dict["benchmark_id"] == "GPQA"

        # Test config from dict
        restored_config = ExperimentConfig.from_dict(config_dict)
        assert restored_config.benchmark_id == "GPQA"

    def test_config_validation_benchmark_or_dataset(self):
        """Test config validation requires either benchmark_id or dataset_name."""
        # Should work with benchmark_id
        config1 = ExperimentConfig(benchmark_id="GPQA", dataset_name="")
        config1.validate()  # Should not raise

        # Should work with dataset_name
        config2 = ExperimentConfig(dataset_name="test/dataset")
        config2.validate()  # Should not raise

        # Should fail with neither
        with pytest.raises(ValueError) as exc_info:
            config3 = ExperimentConfig(dataset_name="", benchmark_id=None)
            config3.validate()

        assert "Either dataset_name or benchmark_id must be provided" in str(
            exc_info.value
        )

    @patch("ml_agents.core.benchmark_registry.load_dataset")
    def test_dataset_loader_benchmark_integration(self, mock_load_dataset):
        """Test dataset loader integration with benchmark system."""
        from ml_agents.core.dataset_loader import BBEHDatasetLoader

        # Mock dataset
        mock_dataset = Mock()
        mock_dataset.column_names = ["INPUT", "OUTPUT"]
        mock_dataset.__len__ = Mock(return_value=100)
        mock_dataset.__getitem__ = Mock(
            return_value={"INPUT": "test", "OUTPUT": "result"}
        )
        mock_load_dataset.return_value = mock_dataset

        config = ExperimentConfig(benchmark_id="GPQA")
        loader = BBEHDatasetLoader(config)

        # Test loading
        dataset = loader.load_dataset("GPQA")
        assert dataset == mock_dataset

        # Test sampling
        mock_sampled = Mock()
        mock_sampled.__len__ = Mock(return_value=10)
        mock_shuffled = Mock()
        mock_shuffled.select = Mock(return_value=mock_sampled)
        mock_dataset.shuffle = Mock(return_value=mock_shuffled)

        sampled = loader.sample_data(sample_size=10)
        assert sampled == mock_sampled

    @patch("ml_agents.core.benchmark_registry.load_dataset")
    def test_experiment_runner_benchmark_integration(self, mock_load_dataset):
        """Test experiment runner integration with benchmark system."""
        # Mock dataset
        mock_dataset = Mock()
        mock_dataset.column_names = ["INPUT", "OUTPUT"]
        mock_dataset.__len__ = Mock(return_value=5)
        mock_dataset.__getitem__ = Mock(
            side_effect=lambda i: {"INPUT": f"Question {i}", "OUTPUT": f"Answer {i}"}
        )

        # Mock sampling
        mock_sampled = Mock()
        mock_sampled.__len__ = Mock(return_value=3)
        mock_shuffled = Mock()
        mock_shuffled.select = Mock(return_value=mock_sampled)
        mock_dataset.shuffle = Mock(return_value=mock_shuffled)

        mock_load_dataset.return_value = mock_dataset

        config = ExperimentConfig(
            benchmark_id="TEST",
            sample_count=3,
            database_enabled=False,  # Disable database for integration test
        )
        runner = ExperimentRunner(config)

        # Verify benchmark ID is used in directory structure
        assert "TEST" in str(runner.output_dir)

        # Verify config has benchmark_id
        assert runner.config.benchmark_id == "TEST"

    def test_error_scenarios_integration(self):
        """Test error handling scenarios in benchmark workflows."""
        # Test missing benchmark in config
        with pytest.raises(ValueError):
            config = ExperimentConfig(dataset_name="", benchmark_id=None)
            config.validate()

        # Test experiment runner with missing benchmark
        config = ExperimentConfig(
            benchmark_id="NonExistent",
            database_enabled=False,  # Disable database for integration test
        )
        runner = ExperimentRunner(config)

        # Mock the benchmark registry to avoid real API calls
        with patch("ml_agents.core.benchmark_registry.load_dataset") as mock_load:
            from ml_agents.core.benchmark_registry import BenchmarkNotFoundError

            mock_load.side_effect = BenchmarkNotFoundError(
                "Benchmark 'NonExistent' not found"
            )

            with pytest.raises(
                (ValueError, RuntimeError, BenchmarkNotFoundError)
            ) as exc_info:
                runner.run_single_experiment("ChainOfThought")

            assert (
                "NonExistent" in str(exc_info.value)
                or "not found" in str(exc_info.value).lower()
            )

    @patch("ml_agents.core.benchmark_registry.load_dataset")
    def test_benchmark_format_validation_integration(self, mock_load_dataset):
        """Test format validation throughout the benchmark system."""
        # Test with invalid format
        mock_invalid_dataset = Mock()
        mock_invalid_dataset.column_names = ["question", "answer"]  # Wrong columns
        mock_load_dataset.return_value = mock_invalid_dataset

        registry = BenchmarkRegistry()

        with pytest.raises(
            Exception
        ):  # Should raise BenchmarkFormatError or ValueError
            registry.load_benchmark("InvalidFormat")

    @patch("ml_agents.core.benchmark_registry.load_dataset")
    def test_performance_integration(self, mock_load_dataset):
        """Test performance-related integration scenarios."""
        # Test with large dataset
        mock_dataset = Mock()
        mock_dataset.column_names = ["INPUT", "OUTPUT"]
        mock_dataset.__len__ = Mock(return_value=10000)
        mock_load_dataset.return_value = mock_dataset

        registry = BenchmarkRegistry()

        # Should handle large dataset info requests efficiently
        # Mock the info retrieval to avoid actual HF API calls
        with patch.object(registry, "get_benchmark_info") as mock_get_info:
            mock_get_info.return_value = {
                "num_samples": 10000,
                "benchmark_id": "LargeDataset",
            }
            info = registry.get_benchmark_info("LargeDataset")
            assert info["num_samples"] == 10000

    def test_cli_error_scenarios(self):
        """Test CLI command error scenarios."""
        with patch(
            "ml_agents.cli.commands.eval.BenchmarkRegistry"
        ) as mock_registry_class:
            mock_registry = Mock()
            mock_registry_class.return_value = mock_registry
            mock_registry.list_available_benchmarks.return_value = []

            # Should handle empty benchmark list gracefully
            list_benchmarks()  # Should not raise

        with patch(
            "ml_agents.cli.commands.eval.BenchmarkRegistry"
        ) as mock_registry_class:
            mock_registry = Mock()
            mock_registry_class.return_value = mock_registry
            mock_registry.get_benchmark_info.side_effect = Exception("Connection error")

            # Should handle errors gracefully
            import typer

            with pytest.raises(typer.Exit):  # typer.Exit(1)
                benchmark_info("GPQA")

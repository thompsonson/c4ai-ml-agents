"""Dataset loader for centralized benchmark evaluation."""

from typing import Any, Dict, Optional

from datasets import Dataset

from ml_agents.config import ExperimentConfig
from ml_agents.core.benchmark_registry import BenchmarkRegistry
from ml_agents.utils.logging_config import get_logger

logger = get_logger(__name__)


class BBEHDatasetLoader:
    """Loads and manages benchmark datasets from centralized HuggingFace repository."""

    def __init__(self, config: ExperimentConfig) -> None:
        """Initialize dataset loader with experiment configuration.

        Args:
            config: Experiment configuration containing benchmark settings
        """
        self.config = config
        self.benchmark_registry = BenchmarkRegistry()
        self.sample_count = config.sample_count
        self._dataset: Optional[Dataset] = None

        logger.info(
            f"Initialized BBEHDatasetLoader with sample_count: {self.sample_count}"
        )

    def load_dataset(self, benchmark_id: str, split: str = "train") -> Dataset:
        """Load dataset from benchmark registry.

        Args:
            benchmark_id: Benchmark identifier
            split: Dataset split (ignored for benchmarks, kept for compatibility)

        Returns:
            Loaded dataset with INPUT/OUTPUT columns

        Raises:
            BenchmarkNotFoundError: If benchmark not found
            BenchmarkFormatError: If benchmark format is invalid
        """
        logger.info(f"Loading benchmark: {benchmark_id}")

        # Load dataset from benchmark registry
        dataset = self.benchmark_registry.load_benchmark(benchmark_id)

        # Validate format (should already be validated by registry)
        self.validate_format(dataset)

        self._dataset = dataset
        logger.info(
            f"Successfully loaded benchmark {benchmark_id} with {len(dataset)} examples"
        )
        return dataset

    def validate_format(self, dataset: Dataset) -> None:
        """Validate that dataset has required INPUT/OUTPUT format.

        Args:
            dataset: Dataset to validate

        Raises:
            ValueError: If dataset format is invalid
        """
        logger.debug("Validating dataset format")

        if len(dataset) == 0:
            raise ValueError("Dataset is empty")

        # Check for required INPUT/OUTPUT columns
        columns = set(dataset.column_names)

        if "INPUT" not in columns:
            raise ValueError(
                f"Dataset must have INPUT column. Found columns: {list(columns)}"
            )

        if "OUTPUT" not in columns:
            raise ValueError(
                f"Dataset must have OUTPUT column. Found columns: {list(columns)}"
            )

        # Validate that INPUT and OUTPUT have content
        example = dataset[0]
        if not example.get("INPUT"):
            raise ValueError("INPUT column contains empty values")

        if not example.get("OUTPUT"):
            raise ValueError("OUTPUT column contains empty values")

        logger.info("Dataset validation passed - INPUT/OUTPUT format confirmed")

    def sample_data(
        self,
        dataset: Optional[Dataset] = None,
        sample_size: Optional[int] = None,
        random_seed: int = 42,
    ) -> Dataset:
        """Sample data from the dataset.

        Args:
            dataset: Dataset to sample from. If None, uses loaded dataset
            sample_size: Number of samples. If None, uses config sample_count
            random_seed: Random seed for reproducible sampling

        Returns:
            Sampled dataset

        Raises:
            ValueError: If no dataset is available or sample size is invalid
        """
        if dataset is None:
            if self._dataset is None:
                raise ValueError("No dataset loaded. Call load_dataset() first.")
            dataset = self._dataset

        if sample_size is None:
            sample_size = self.sample_count

        if sample_size <= 0:
            raise ValueError("Sample size must be positive")

        total_examples = len(dataset)

        if sample_size >= total_examples:
            logger.warning(
                f"Sample size ({sample_size}) >= dataset size ({total_examples}). "
                f"Returning full dataset."
            )
            return dataset

        logger.info(f"Sampling {sample_size} examples from {total_examples}")

        # Create shuffled indices for sampling
        sampled_dataset = dataset.shuffle(seed=random_seed).select(range(sample_size))

        logger.info(f"Successfully sampled {len(sampled_dataset)} examples")
        return sampled_dataset

    def get_dataset_info(self, dataset: Optional[Dataset] = None) -> Dict[str, Any]:
        """Get information about the dataset.

        Args:
            dataset: Dataset to analyze. If None, uses loaded dataset

        Returns:
            Dictionary containing dataset information
        """
        if dataset is None:
            if self._dataset is None:
                return {"error": "No dataset loaded"}
            dataset = self._dataset

        info = {
            "size": len(dataset),
            "columns": list(dataset.column_names),
            "features": {
                name: str(feature) for name, feature in dataset.features.items()
            },
        }

        # Add sample of first few examples
        if len(dataset) > 0:
            sample_size = min(3, len(dataset))
            info["sample_examples"] = [dataset[i] for i in range(sample_size)]

        return info

    def get_input_column_name(self) -> str:
        """Get the name of the input column for this dataset.

        Returns:
            Column name for input data (always 'INPUT' for benchmarks)
        """
        return "INPUT"

    def get_output_column_name(self) -> str:
        """Get the name of the output column for this dataset.

        Returns:
            Column name for output data (always 'OUTPUT' for benchmarks)
        """
        return "OUTPUT"

    def list_available_benchmarks(self) -> list[str]:
        """List all available benchmarks.

        Returns:
            List of available benchmark IDs
        """
        return self.benchmark_registry.list_available_benchmarks()

    def get_benchmark_info(self, benchmark_id: str) -> Dict[str, Any]:
        """Get information about a specific benchmark.

        Args:
            benchmark_id: Benchmark identifier

        Returns:
            Dictionary containing benchmark metadata
        """
        return self.benchmark_registry.get_benchmark_info(benchmark_id)

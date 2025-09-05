"""Local test dataset support for Phase 14 experiments."""

from typing import Optional

from datasets import Dataset

from ml_agents.core.phase14_test_data import create_local_test_dataset
from ml_agents.utils.logging_config import get_logger

logger = get_logger(__name__)


class LocalTestDataset:
    """Provides LOCAL_TEST dataset for Phase 14 experiments."""

    DATASET_ID = "LOCAL_TEST"
    DATASET_ID_LARGE = "LOCAL_TEST_LARGE"

    @classmethod
    def is_local_test_dataset(cls, dataset_id: str) -> bool:
        """Check if dataset ID is a local test dataset.

        Args:
            dataset_id: Dataset identifier to check

        Returns:
            True if this is a local test dataset
        """
        return dataset_id in [cls.DATASET_ID, cls.DATASET_ID_LARGE]

    @classmethod
    def load_dataset(
        cls, dataset_id: str, sample_size: Optional[int] = None
    ) -> Dataset:
        """Load local test dataset.

        Args:
            dataset_id: Dataset identifier (LOCAL_TEST or LOCAL_TEST_LARGE)
            sample_size: Optional sample size limit

        Returns:
            Dataset with INPUT/OUTPUT columns

        Raises:
            ValueError: If dataset_id is not a local test dataset
        """
        if not cls.is_local_test_dataset(dataset_id):
            raise ValueError(f"Unknown local test dataset: {dataset_id}")

        logger.info(f"Loading local test dataset: {dataset_id}")

        # Get the full dataset
        data = create_local_test_dataset()["train"]

        # For LOCAL_TEST_LARGE, duplicate questions to create larger dataset
        if dataset_id == cls.DATASET_ID_LARGE:
            # Create 500 samples by cycling through questions
            large_data = []
            question_count = len(data)
            for i in range(500):
                question = data[i % question_count].copy()
                question["id"] = f"LOCAL_TEST_LARGE_{i+1:03d}"
                large_data.append(question)
            data = large_data

        # Apply sample size limit if specified
        if sample_size and sample_size < len(data):
            data = data[:sample_size]
            logger.info(f"Limited dataset to {sample_size} samples")

        # Create Dataset object
        dataset = Dataset.from_list(data)

        logger.info(f"Loaded {len(dataset)} samples from {dataset_id}")
        return dataset

    @classmethod
    def get_dataset_info(cls, dataset_id: str) -> dict:
        """Get information about local test dataset.

        Args:
            dataset_id: Dataset identifier

        Returns:
            Dictionary with dataset metadata
        """
        if dataset_id == cls.DATASET_ID:
            return {
                "id": cls.DATASET_ID,
                "name": "Local Test Dataset",
                "description": "Phase 14 test questions for reasoning approach comparison",
                "size": 20,  # 5 basic + 5 reasoning + 10 extended
                "format": "INPUT/OUTPUT",
                "source": "phase14_test_data.py",
            }
        elif dataset_id == cls.DATASET_ID_LARGE:
            return {
                "id": cls.DATASET_ID_LARGE,
                "name": "Local Test Dataset (Large)",
                "description": "Extended dataset for scale testing (500 samples)",
                "size": 500,
                "format": "INPUT/OUTPUT",
                "source": "phase14_test_data.py (cycled)",
            }
        else:
            raise ValueError(f"Unknown local test dataset: {dataset_id}")

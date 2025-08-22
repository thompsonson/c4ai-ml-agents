"""Dataset loader for BBEH evaluation datasets."""

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from datasets import Dataset, load_dataset

from ml_agents.config import ExperimentConfig
from ml_agents.utils.logging_config import get_logger

logger = get_logger(__name__)


class BBEHDatasetLoader:
    """Loads and manages BBEH evaluation datasets from HuggingFace."""

    def __init__(self, config: ExperimentConfig) -> None:
        """Initialize dataset loader with experiment configuration.

        Args:
            config: Experiment configuration containing dataset settings
        """
        self.config = config
        self.dataset_name = config.dataset_name
        self.sample_count = config.sample_count
        self._dataset: Optional[Dataset] = None

        # Column mappings discovered during validation
        self._input_column: Optional[str] = None
        self._output_column: Optional[str] = None

        # Get cache directory from environment or use XDG default
        cache_base = os.getenv(
            "ML_AGENTS_CACHE_DIR", str(Path.home() / ".cache" / "ml-agents")
        )
        self._cache_dir = Path(cache_base) / "datasets"
        self._cache_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Initialized BBEHDatasetLoader for dataset: {self.dataset_name}")

    def load_dataset(self, split: str = "train") -> Dataset:
        """Load dataset from HuggingFace.

        Args:
            split: Dataset split to load (default: "train")

        Returns:
            Loaded dataset

        Raises:
            ValueError: If dataset cannot be loaded or is invalid
            RuntimeError: If HuggingFace API fails
        """
        try:
            logger.info(f"Loading dataset {self.dataset_name}, split: {split}")

            # Check cache first
            cached_dataset = self._load_from_cache(split)
            if cached_dataset is not None:
                logger.info("Loaded dataset from cache")
                self._dataset = cached_dataset
                # Validate format and discover column mappings for cached dataset
                self.validate_format(cached_dataset)
                return cached_dataset

            # Load from HuggingFace
            logger.info("Loading dataset from HuggingFace Hub")
            try:
                dataset = load_dataset(self.dataset_name, split=split)
            except ValueError as ve:
                if "Unknown split" in str(ve):
                    # Try to get available splits and suggest alternatives
                    try:
                        # Load dataset info to get available splits
                        dataset_info = load_dataset(self.dataset_name, split=None)
                        available_splits = list(dataset_info.keys())
                        logger.warning(
                            f"Requested split '{split}' not found. Available splits: {available_splits}"
                        )

                        # Try fallback splits in order of preference
                        fallback_splits = ["train", "test", "validation", "val"]
                        for fallback_split in fallback_splits:
                            if (
                                fallback_split in available_splits
                                and fallback_split != split
                            ):
                                logger.info(
                                    f"Using fallback split '{fallback_split}' instead of '{split}'"
                                )
                                dataset = dataset_info[fallback_split]
                                break
                        else:
                            # No fallback worked, use the first available split
                            if available_splits:
                                first_split = available_splits[0]
                                logger.info(
                                    f"Using first available split '{first_split}' instead of '{split}'"
                                )
                                dataset = dataset_info[first_split]
                            else:
                                raise ValueError(
                                    f"No splits available in dataset {self.dataset_name}"
                                )
                    except Exception:
                        # If split detection fails, re-raise original error
                        raise ve
                else:
                    raise ve

            if not isinstance(dataset, Dataset):
                raise ValueError(f"Expected Dataset, got {type(dataset)}")

            # Validate dataset format
            self.validate_format(dataset)

            # Cache the dataset
            self._save_to_cache(dataset, split)

            self._dataset = dataset
            logger.info(f"Successfully loaded dataset with {len(dataset)} examples")
            return dataset

        except Exception as e:
            error_msg = f"Failed to load dataset {self.dataset_name}: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e

    def validate_format(self, dataset: Dataset) -> None:
        """Validate that dataset has required format.

        Args:
            dataset: Dataset to validate

        Raises:
            ValueError: If dataset format is invalid
        """
        logger.debug("Validating dataset format")

        if len(dataset) == 0:
            raise ValueError("Dataset is empty")

        # Get the first example to check structure
        example = dataset[0]

        # Check for required columns - handle both 'input' and other common variations
        required_cols = set()
        has_input = False

        # Common input column names
        input_variations = ["input", "question", "prompt", "text", "query"]

        for col_name in input_variations:
            if col_name in example:
                has_input = True
                self._input_column = col_name
                logger.debug(f"Discovered input column: '{col_name}'")
                break

        if not has_input:
            # If no standard input column, check if we have any text-like columns
            text_cols = [
                k
                for k, v in example.items()
                if isinstance(v, str) and len(v.strip()) > 0
            ]
            if not text_cols:
                raise ValueError(
                    f"Dataset must contain at least one text input column. "
                    f"Available columns: {list(example.keys())}"
                )
            else:
                logger.warning(
                    f"No standard input column found. Available text columns: {text_cols}"
                )

        # Check for expected answer column (optional but recommended)
        answer_variations = ["answer", "target", "label", "output", "expected"]
        has_answer = False
        for col_name in answer_variations:
            if col_name in example:
                has_answer = True
                self._output_column = col_name
                logger.debug(f"Discovered output column: '{col_name}'")
                break

        if not has_answer:
            logger.warning(
                f"No standard answer column found. This may affect evaluation. "
                f"Available columns: {list(example.keys())}"
            )

        logger.info(f"Dataset validation passed. Columns: {list(example.keys())}")

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
            "name": self.dataset_name,
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

    def _get_cache_path(self, split: str) -> Path:
        """Generate cache path for dataset.

        Args:
            split: Dataset split name

        Returns:
            Path to cache file
        """
        # Create a hash of dataset name and split for cache key
        cache_key = hashlib.md5(
            f"{self.dataset_name}_{split}".encode(), usedforsecurity=False
        ).hexdigest()  # nosec B324
        return self._cache_dir / f"{cache_key}.json"

    def _load_from_cache(self, split: str) -> Optional[Dataset]:
        """Load dataset from cache.

        Args:
            split: Dataset split name

        Returns:
            Cached dataset or None if not found
        """
        cache_path = self._get_cache_path(split)

        if not cache_path.exists():
            return None

        try:
            logger.debug(f"Loading dataset from cache: {cache_path}")
            with open(cache_path, "r") as f:
                data = json.load(f)

            # Convert back to Dataset
            df = pd.DataFrame(data["examples"])
            dataset = Dataset.from_pandas(df)

            logger.debug(f"Successfully loaded {len(dataset)} examples from cache")
            return dataset

        except Exception as e:
            logger.warning(f"Failed to load from cache: {e}")
            # Remove corrupted cache file
            cache_path.unlink(missing_ok=True)
            return None

    def _save_to_cache(self, dataset: Dataset, split: str) -> None:
        """Save dataset to cache.

        Args:
            dataset: Dataset to cache
            split: Dataset split name
        """
        cache_path = self._get_cache_path(split)

        try:
            logger.debug(f"Saving dataset to cache: {cache_path}")

            # Convert dataset to serializable format
            cache_data = {
                "dataset_name": self.dataset_name,
                "split": split,
                "size": len(dataset),
                "examples": dataset.to_dict(),
            }

            with open(cache_path, "w") as f:
                json.dump(cache_data, f, ensure_ascii=False, indent=2)

            logger.debug(f"Successfully cached {len(dataset)} examples")

        except Exception as e:
            logger.warning(f"Failed to save to cache: {e}")

    def clear_cache(self) -> None:
        """Clear all cached datasets."""
        try:
            cache_files = list(self._cache_dir.glob("*.json"))
            for cache_file in cache_files:
                cache_file.unlink()

            logger.info(f"Cleared {len(cache_files)} cache files")

        except Exception as e:
            logger.warning(f"Failed to clear cache: {e}")

    def get_input_column_name(self) -> str:
        """Get the name of the input column for this dataset.

        Returns:
            Column name for input data, defaults to 'input' if not discovered
        """
        return self._input_column or "input"

    def get_output_column_name(self) -> str:
        """Get the name of the output column for this dataset.

        Returns:
            Column name for output/answer data, defaults to 'output' if not discovered
        """
        return self._output_column or "output"

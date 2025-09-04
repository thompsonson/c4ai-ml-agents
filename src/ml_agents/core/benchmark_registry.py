"""Centralized benchmark repository access for HuggingFace datasets."""

import re
from typing import Any, Dict, List

import pandas as pd
from datasets import Dataset, load_dataset

from ml_agents.utils.logging_config import get_logger

logger = get_logger(__name__)


class BenchmarkNotFoundError(Exception):
    """Raised when benchmark ID cannot be found."""

    pass


class BenchmarkFormatError(Exception):
    """Raised when benchmark CSV format is invalid."""

    pass


class BenchmarkRegistry:
    """Manages centralized benchmark repository access."""

    REPOSITORY = "c4ai-ml-agents/benchmarks-base"

    def __init__(self):
        """Initialize benchmark registry."""
        self._available_benchmarks: List[str] = []
        self._benchmarks_loaded = False
        logger.info(f"Initialized BenchmarkRegistry for repository: {self.REPOSITORY}")

    def load_benchmark(self, benchmark_id: str) -> Dataset:
        """Load benchmark directly from HuggingFace repository.

        Args:
            benchmark_id: Benchmark identifier

        Returns:
            Dataset with INPUT/OUTPUT columns

        Raises:
            BenchmarkNotFoundError: If benchmark not found
            BenchmarkFormatError: If benchmark format is invalid
        """
        try:
            logger.info(f"Loading benchmark: {benchmark_id}")

            # Load dataset from HuggingFace repository
            # Since we can enumerate contents, we'll load by benchmark_id as filename
            dataset = load_dataset(
                self.REPOSITORY, data_files=f"{benchmark_id}.csv", split="train"
            )

            # Validate that dataset has INPUT/OUTPUT columns
            if not self._validate_benchmark_format(dataset):
                raise BenchmarkFormatError(
                    f"Benchmark '{benchmark_id}' must have INPUT and OUTPUT columns"
                )

            logger.info(
                f"Successfully loaded benchmark: {benchmark_id} ({len(dataset)} samples)"
            )
            return dataset

        except FileNotFoundError as e:
            logger.error(f"Benchmark '{benchmark_id}' not found in repository")
            raise BenchmarkNotFoundError(
                f"Benchmark '{benchmark_id}' not found in repository"
            ) from e
        except Exception as e:
            logger.error(f"Error loading benchmark '{benchmark_id}': {str(e)}")
            raise BenchmarkFormatError(
                f"Invalid benchmark format for '{benchmark_id}': {str(e)}"
            ) from e

    def list_available_benchmarks(self) -> List[str]:
        """List all available benchmark IDs from repository.

        Returns:
            Sorted list of available benchmark identifiers
        """
        if not self._benchmarks_loaded:
            self._load_benchmark_list()

        return sorted(self._available_benchmarks)

    def get_benchmark_info(self, benchmark_id: str) -> Dict[str, Any]:
        """Get metadata for specific benchmark.

        Args:
            benchmark_id: Benchmark identifier

        Returns:
            Dictionary with benchmark metadata

        Raises:
            BenchmarkNotFoundError: If benchmark not found
        """
        try:
            # Load just the first few rows to get schema info
            dataset = load_dataset(
                self.REPOSITORY, data_files=f"{benchmark_id}.csv", split="train[:10]"
            )

            info = {
                "benchmark_id": benchmark_id,
                "columns": list(dataset.column_names),
                "num_samples": len(
                    load_dataset(
                        self.REPOSITORY, data_files=f"{benchmark_id}.csv", split="train"
                    )
                ),
                "has_input_output": self._validate_benchmark_format(dataset),
            }

            # Add sample data for inspection
            if len(dataset) > 0:
                sample = dataset[0]
                info["sample"] = {
                    "INPUT": sample.get("INPUT", "N/A"),
                    "OUTPUT": sample.get("OUTPUT", "N/A"),
                }

            return info

        except FileNotFoundError as e:
            raise BenchmarkNotFoundError(
                f"Benchmark '{benchmark_id}' not found in repository"
            ) from e
        except Exception as e:
            logger.error(f"Error getting info for benchmark '{benchmark_id}': {str(e)}")
            raise BenchmarkFormatError(
                f"Error accessing benchmark '{benchmark_id}': {str(e)}"
            ) from e

    def _load_benchmark_list(self) -> None:
        """Load list of available benchmarks from repository."""
        try:
            logger.info("Loading available benchmarks from repository")

            # Load the repository to enumerate available files
            # We'll get the dataset info to see what files are available
            repo_info = load_dataset(self.REPOSITORY, split=None)

            # Extract benchmark IDs from available files
            # This assumes files are named like "benchmark_id.csv"
            self._available_benchmarks = []

            # If the repository contains multiple CSV files, extract their names
            # For now, we'll implement a fallback that attempts to list common benchmarks
            # This would need to be updated once we can properly enumerate the repo

            # Fallback: try to detect available benchmarks by attempting to load common ones
            potential_benchmarks = [
                "GPQA",
                "MMLU",
                "HellaSwag",
                "ARC",
                "TruthfulQA",
                "GSM8K",
                "HumanEval",
                "DROP",
                "SQuAD",
                "CommonSenseQA",
            ]

            for benchmark_id in potential_benchmarks:
                try:
                    # Try to load just the info to see if it exists
                    load_dataset(
                        self.REPOSITORY,
                        data_files=f"{benchmark_id}.csv",
                        split="train[:1]",
                    )
                    self._available_benchmarks.append(benchmark_id)
                    logger.debug(f"Found benchmark: {benchmark_id}")
                except:
                    # Benchmark doesn't exist, skip it
                    pass

            self._benchmarks_loaded = True
            logger.info(f"Found {len(self._available_benchmarks)} available benchmarks")

        except Exception as e:
            logger.error(f"Error loading benchmark list: {str(e)}")
            # Set empty list as fallback
            self._available_benchmarks = []
            self._benchmarks_loaded = True

    def _validate_benchmark_format(self, dataset: Dataset) -> bool:
        """Validate that dataset has required INPUT/OUTPUT columns.

        Args:
            dataset: Dataset to validate

        Returns:
            True if dataset has INPUT and OUTPUT columns
        """
        columns = set(dataset.column_names)
        has_input_output = "INPUT" in columns and "OUTPUT" in columns

        if not has_input_output:
            logger.warning(
                f"Dataset missing required columns. Found: {list(columns)}, Required: [INPUT, OUTPUT]"
            )

        return has_input_output

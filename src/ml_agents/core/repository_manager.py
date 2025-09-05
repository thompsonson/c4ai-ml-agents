"""Repository manager for flexible CSV file access from HuggingFace datasets."""

from typing import Any, Dict, List, Optional

import pandas as pd
from datasets import Dataset
from huggingface_hub import hf_hub_download, list_repo_files

from ml_agents.utils.logging_config import get_logger

logger = get_logger(__name__)


class RepositoryManagerError(Exception):
    """Base exception for repository manager errors."""

    pass


class RepositoryNotFoundError(RepositoryManagerError):
    """Raised when repository cannot be found."""

    pass


class FileNotFoundError(RepositoryManagerError):
    """Raised when file cannot be found in repository."""

    pass


class RepositoryManager:
    """Manages CSV file access from HuggingFace dataset repositories."""

    DEFAULT_REPO = "c4ai-ml-agents/benchmarks-base"

    def __init__(self) -> None:
        """Initialize repository manager."""
        logger.info(
            f"Initialized RepositoryManager with default repository: {self.DEFAULT_REPO}"
        )

    def list_csv_files(self, repo_id: Optional[str] = None) -> List[str]:
        """List all CSV files in repository.

        Args:
            repo_id: Repository ID (uses default if None)

        Returns:
            List of CSV filenames

        Raises:
            RepositoryNotFoundError: If repository cannot be accessed
        """
        repo_id = repo_id or self.DEFAULT_REPO

        try:
            logger.debug(f"Listing CSV files in repository: {repo_id}")
            files = list_repo_files(repo_id, repo_type="dataset")
            csv_files = [f for f in files if f.endswith(".csv")]

            logger.info(f"Found {len(csv_files)} CSV files in {repo_id}")
            return sorted(csv_files)

        except Exception as e:
            logger.error(f"Error listing files in repository '{repo_id}': {str(e)}")
            raise RepositoryNotFoundError(
                f"Cannot access repository '{repo_id}': {str(e)}"
            ) from e

    def load_csv_file(
        self, filename: str, repo_id: Optional[str] = None
    ) -> pd.DataFrame:
        """Download and load specific CSV file.

        Args:
            filename: CSV filename to load
            repo_id: Repository ID (uses default if None)

        Returns:
            DataFrame with CSV contents

        Raises:
            FileNotFoundError: If file not found in repository
            RepositoryNotFoundError: If repository cannot be accessed
        """
        repo_id = repo_id or self.DEFAULT_REPO

        try:
            logger.debug(f"Loading CSV file '{filename}' from repository: {repo_id}")

            # Download file from HuggingFace Hub
            file_path = hf_hub_download(repo_id, filename, repo_type="dataset")

            # Load CSV file
            df = pd.read_csv(file_path)

            logger.info(
                f"Successfully loaded '{filename}' from {repo_id}: {len(df)} rows, "
                f"{len(df.columns)} columns"
            )
            return df

        except FileNotFoundError as e:
            logger.error(f"File '{filename}' not found in repository '{repo_id}'")
            raise FileNotFoundError(
                f"File '{filename}' not found in repository '{repo_id}'"
            ) from e
        except Exception as e:
            logger.error(
                f"Error loading file '{filename}' from repository '{repo_id}': {str(e)}"
            )
            raise RepositoryManagerError(
                f"Error loading file '{filename}' from repository '{repo_id}': {str(e)}"
            ) from e

    def load_csv_as_dataset(
        self, filename: str, repo_id: Optional[str] = None
    ) -> Dataset:
        """Load CSV file as HuggingFace Dataset.

        Args:
            filename: CSV filename to load
            repo_id: Repository ID (uses default if None)

        Returns:
            HuggingFace Dataset object

        Raises:
            FileNotFoundError: If file not found in repository
            RepositoryManagerError: If conversion fails
        """
        try:
            df = self.load_csv_file(filename, repo_id)
            dataset = Dataset.from_pandas(df)

            logger.info(f"Converted '{filename}' to HuggingFace Dataset format")
            return dataset

        except Exception as e:
            logger.error(f"Error converting '{filename}' to Dataset: {str(e)}")
            raise RepositoryManagerError(
                f"Error converting '{filename}' to Dataset: {str(e)}"
            ) from e

    def get_file_info(
        self, filename: str, repo_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get basic information about CSV file.

        Args:
            filename: CSV filename to analyze
            repo_id: Repository ID (uses default if None)

        Returns:
            Dictionary containing file metadata

        Raises:
            FileNotFoundError: If file not found in repository
        """
        repo_id = repo_id or self.DEFAULT_REPO

        try:
            df = self.load_csv_file(filename, repo_id)

            info = {
                "filename": filename,
                "repo_id": repo_id,
                "num_rows": len(df),
                "columns": list(df.columns),
                "has_input_output": "INPUT" in df.columns and "OUTPUT" in df.columns,
                "sample_data": {},
            }

            # Add sample data if available
            if len(df) > 0:
                # Show first few rows (up to 3) as sample
                sample_size = min(3, len(df))
                sample_df = df.head(sample_size)
                info["sample_data"] = sample_df.to_dict("records")

            return info

        except Exception as e:
            logger.error(f"Error getting info for file '{filename}': {str(e)}")
            raise

    def file_exists(self, filename: str, repo_id: Optional[str] = None) -> bool:
        """Check if file exists in repository.

        Args:
            filename: CSV filename to check
            repo_id: Repository ID (uses default if None)

        Returns:
            True if file exists, False otherwise
        """
        try:
            csv_files = self.list_csv_files(repo_id)
            return filename in csv_files
        except RepositoryNotFoundError:
            return False

    def validate_csv_format(
        self, filename: str, repo_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Validate that CSV file has required INPUT/OUTPUT format.

        Args:
            filename: CSV filename to validate
            repo_id: Repository ID (uses default if None)

        Returns:
            Dictionary containing validation results

        Raises:
            FileNotFoundError: If file not found in repository
        """
        try:
            df = self.load_csv_file(filename, repo_id)

            validation = {
                "filename": filename,
                "repo_id": repo_id or self.DEFAULT_REPO,
                "is_valid": True,
                "errors": [],
                "warnings": [],
                "info": {
                    "num_rows": len(df),
                    "columns": list(df.columns),
                    "has_input": "INPUT" in df.columns,
                    "has_output": "OUTPUT" in df.columns,
                },
            }

            # Check for required columns
            if "INPUT" not in df.columns:
                validation["errors"].append("Missing required 'INPUT' column")
                validation["is_valid"] = False

            if "OUTPUT" not in df.columns:
                validation["errors"].append("Missing required 'OUTPUT' column")
                validation["is_valid"] = False

            # Check for empty data
            if len(df) == 0:
                validation["errors"].append("File is empty")
                validation["is_valid"] = False

            # Check for empty columns (if columns exist)
            if validation["is_valid"] and len(df) > 0:
                if df["INPUT"].isna().all():
                    validation["errors"].append("INPUT column is empty")
                    validation["is_valid"] = False

                if df["OUTPUT"].isna().all():
                    validation["errors"].append("OUTPUT column is empty")
                    validation["is_valid"] = False

            logger.info(
                f"Validation for '{filename}': {'PASSED' if validation['is_valid'] else 'FAILED'}"
            )
            return validation

        except Exception as e:
            logger.error(f"Error validating file '{filename}': {str(e)}")
            raise

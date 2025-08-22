"""Dataset uploader for sharing processed datasets to HuggingFace Hub."""

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from datasets import load_dataset  # Only needed for list_organization_datasets
from huggingface_hub import DatasetCard, DatasetCardData, HfApi, login
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from ml_agents.utils.logging_config import get_logger

logger = get_logger(__name__)
console = Console()


class DatasetUploader:
    """Upload processed datasets to HuggingFace Hub with automated metadata generation."""

    def __init__(self, org_name: str = "c4ai-ml-agents"):
        """Initialize dataset uploader.

        Args:
            org_name: Organization name for dataset uploads
        """
        self.org_name = org_name
        self.api = HfApi()
        self._authenticated = False

    def authenticate(self) -> bool:
        """Authenticate with HuggingFace Hub using token.

        Returns:
            True if authentication successful, False otherwise
        """
        if self._authenticated:
            return True

        try:
            # Try to get token from environment
            token = os.getenv("HF_TOKEN")

            if not token:
                console.print(
                    "[yellow]HF_TOKEN environment variable not found.[/yellow]"
                )
                console.print("Please set HF_TOKEN or enter your HuggingFace token:")
                token = input("HF Token: ").strip()

            if not token:
                logger.error("No HuggingFace token provided")
                return False

            # Login with token
            login(token=token)

            # Verify authentication by getting user info
            user_info = self.api.whoami()
            console.print(f"[green]✅ Authenticated as: {user_info['name']}[/green]")

            self._authenticated = True
            return True

        except Exception as e:
            logger.error(f"Authentication failed: {e}")
            console.print(f"[red]❌ Authentication failed: {e}[/red]")
            return False

    def validate_processed_file(self, processed_file: str) -> Dict[str, Any]:
        """Validate processed dataset file format and content.

        Args:
            processed_file: Path to processed dataset file

        Returns:
            Validation results dictionary

        Raises:
            ValueError: If file format is invalid
            FileNotFoundError: If file doesn't exist
        """
        file_path = Path(processed_file)
        if not file_path.exists():
            raise FileNotFoundError(f"Processed file not found: {processed_file}")

        validation_results = {
            "file_path": str(file_path),
            "file_size_mb": file_path.stat().st_size / (1024 * 1024),
            "format": "unknown",
            "sample_count": 0,
            "has_input_output_schema": False,
            "validation_passed": False,
            "issues": [],
        }

        try:
            # Determine file format and load
            if processed_file.endswith(".json"):
                with open(processed_file, "r") as f:
                    data = json.load(f)

                validation_results["format"] = "json"

                # Check if it's list of records format
                if isinstance(data, list) and len(data) > 0:
                    validation_results["sample_count"] = len(data)
                    first_record = data[0]

                    if (
                        isinstance(first_record, dict)
                        and "INPUT" in first_record
                        and "OUTPUT" in first_record
                    ):
                        validation_results["has_input_output_schema"] = True
                    else:
                        validation_results["issues"].append(
                            "Records don't have INPUT/OUTPUT schema"
                        )

                elif isinstance(data, dict):
                    validation_results["issues"].append(
                        "Expected list of records, got dictionary"
                    )
                else:
                    validation_results["issues"].append("Invalid JSON structure")

            elif processed_file.endswith(".csv"):
                import pandas as pd

                df = pd.read_csv(processed_file)

                validation_results["format"] = "csv"
                validation_results["sample_count"] = len(df)

                if "INPUT" in df.columns and "OUTPUT" in df.columns:
                    validation_results["has_input_output_schema"] = True
                else:
                    validation_results["issues"].append(
                        "CSV doesn't have INPUT/OUTPUT columns"
                    )

            else:
                raise ValueError(f"Unsupported file format: {file_path.suffix}")

            # Check for empty dataset
            if validation_results["sample_count"] == 0:
                validation_results["issues"].append("Dataset is empty")

            # Check file size (warn if > 100MB)
            if validation_results["file_size_mb"] > 100:
                validation_results["issues"].append(
                    f"Large file size: {validation_results['file_size_mb']:.1f}MB"
                )

            # Set validation status
            validation_results["validation_passed"] = (
                validation_results["has_input_output_schema"]
                and validation_results["sample_count"] > 0
                and not any(
                    "Invalid" in issue or "Expected" in issue
                    for issue in validation_results["issues"]
                )
            )

            return validation_results

        except Exception as e:
            logger.error(f"Failed to validate processed file: {e}")
            raise ValueError(f"File validation failed: {e}")

    def _generate_dataset_card(
        self,
        source_dataset: str,
        target_name: str,
        config: Optional[str] = None,
        description: Optional[str] = None,
        sample_count: int = 0,
        transformation_rules: Optional[Dict[str, Any]] = None,
        available_files: Optional[List[str]] = None,
    ) -> str:
        """Generate dataset card content with metadata.

        Args:
            source_dataset: Original dataset name/URL
            target_name: Target dataset name for upload
            config: Dataset configuration used (if any)
            description: Custom description
            sample_count: Number of samples in dataset

        Returns:
            Dataset card content as string
        """
        # Build source dataset URL
        if source_dataset.startswith("http"):
            source_url = source_dataset
        else:
            source_url = f"https://huggingface.co/datasets/{source_dataset}"

        # Default description if not provided
        if not description:
            description = f"Processed version of {source_dataset} in standardized INPUT/OUTPUT format for ML Agents reasoning evaluation."

        # Build tags and categories
        tags = ["question-answering", "reasoning", "ml-agents", "standardized-format"]
        if config:
            tags.append(f"config-{config}")

        card_content = f"""---
dataset_info:
  features:
  - name: INPUT
    dtype: string
  - name: OUTPUT
    dtype: string
  splits:
  - name: train
    num_examples: {sample_count}
task_categories:
- question-answering
tags:
{chr(10).join(f'- {tag}' for tag in tags)}
source_datasets:
- {source_dataset}
language:
- en
size_categories:
- {'1K<n<10K' if sample_count < 10000 else '10K<n<100K' if sample_count < 100000 else '100K<n<1M'}
---

# {target_name}

{description}

## Dataset Information

- **Source Dataset**: [{source_dataset}]({source_url})
- **Configuration**: {config if config else 'Default'}
- **Samples**: {sample_count:,}
- **Format**: Standardized INPUT/OUTPUT schema
- **Processed**: {datetime.now(UTC).strftime('%Y-%m-%d')}

## Schema

Each record contains:
- **INPUT**: The question, prompt, or input text
- **OUTPUT**: The expected answer or output

## Files Included

{self._format_file_list(available_files) if available_files else '- No files specified'}

## Transformation Rules

{self._format_transformation_rules(transformation_rules) if transformation_rules else 'No transformation rules available.'}

## Usage

```python
from datasets import load_dataset

# Load the dataset
dataset = load_dataset("c4ai-ml-agents/{target_name}")

# Access samples
for sample in dataset["train"]:
    print(f"Input: {{sample['INPUT']}}")
    print(f"Output: {{sample['OUTPUT']}}")
    break
```

## License

This processed dataset maintains the same license as the original source dataset. Please refer to the [source dataset]({source_url}) for licensing information.
"""
        return card_content

    def _format_transformation_rules(self, rules: Dict[str, Any]) -> str:
        """Format transformation rules for display in dataset card.

        Args:
            rules: Transformation rules dictionary

        Returns:
            Formatted rules as markdown string
        """
        if not rules:
            return "No transformation rules available."

        return f"""```json
{json.dumps(rules, indent=2)}
```"""

    def _format_file_list(self, files: List[str]) -> str:
        """Format list of files for display in dataset card.

        Args:
            files: List of file names

        Returns:
            Formatted file list as markdown string
        """
        if not files:
            return "- No files available"

        file_descriptions = {
            "json": "Main dataset file",
            "_analysis.json": "Schema analysis results",
            "_rules.json": "Transformation rules used",
            ".csv": "CSV format",
        }

        lines = []
        for file in files:
            # Find matching description
            desc = "Data file"
            for pattern, description in file_descriptions.items():
                if file.endswith(pattern):
                    desc = description
                    break
            lines.append(f"- `{file}` - {desc}")

        return "\n".join(lines)

    def upload_dataset(
        self,
        processed_file: str,
        source_dataset: str,
        target_name: str,
        config: Optional[str] = None,
        description: Optional[str] = None,
    ) -> str:
        """Upload processed dataset to HuggingFace Hub.

        Args:
            processed_file: Path to processed dataset file
            source_dataset: Original dataset name/URL for attribution
            target_name: Target name for the uploaded dataset
            config: Dataset configuration that was used (optional)
            description: Custom description (optional)

        Returns:
            Repository ID of uploaded dataset

        Raises:
            ValueError: If validation fails or authentication fails
            Exception: If upload fails
        """
        logger.info(
            f"Starting upload of {processed_file} to {self.org_name}/{target_name}"
        )

        # Authenticate
        if not self.authenticate():
            raise ValueError("HuggingFace authentication failed")

        # Validate processed file
        console.print("[yellow]🔍 Validating processed dataset...[/yellow]")
        validation_results = self.validate_processed_file(processed_file)

        if not validation_results["validation_passed"]:
            issues = "\n".join(f"  - {issue}" for issue in validation_results["issues"])
            raise ValueError(f"Dataset validation failed:\n{issues}")

        console.print(
            f"[green]✅ Validation passed - {validation_results['sample_count']} samples[/green]"
        )

        # Get base name and directory for related files
        base_path = Path(processed_file)
        base_name = base_path.stem  # e.g., "MilaWang_SpatialEval_tqa"
        base_dir = base_path.parent

        logger.info(f"Base name: {base_name}, Base dir: {base_dir}")

        # Load transformation rules if available
        rules_path = base_dir / f"{base_name}_rules.json"
        transformation_rules = None
        if rules_path.exists():
            try:
                with open(rules_path, "r") as f:
                    transformation_rules = json.load(f)
                logger.info(f"Loaded transformation rules from {rules_path}")
            except Exception as e:
                logger.warning(f"Could not load transformation rules: {e}")

        # Prepare for upload
        repo_id = f"{self.org_name}/{target_name}"

        # Check which files exist to include in dataset card
        available_files = []
        files_to_check = [
            (f"{base_name}.json", f"{target_name}.json"),
            (f"{base_name}_analysis.json", f"{target_name}_analysis.json"),
            (f"{base_name}_rules.json", f"{target_name}_rules.json"),
            (f"{base_name}.csv", f"{target_name}.csv"),
        ]

        for source_file, target_file in files_to_check:
            if (base_dir / source_file).exists():
                available_files.append(target_file)

        # Generate dataset card with list of available files
        console.print("[yellow]📝 Generating dataset card...[/yellow]")
        card_content = self._generate_dataset_card(
            source_dataset=source_dataset,
            target_name=target_name,
            config=config,
            description=description,
            sample_count=validation_results["sample_count"],
            transformation_rules=transformation_rules,
            available_files=available_files,
        )

        # Upload to hub
        console.print(f"[yellow]⬆️ Uploading to {repo_id}...[/yellow]")

        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                upload_task = progress.add_task("Preparing upload...", total=None)

                # Create repository if it doesn't exist
                logger.info(f"Creating repository: {repo_id}")
                self.api.create_repo(
                    repo_id, repo_type="dataset", private=False, exist_ok=True
                )
                progress.update(
                    upload_task, description="Repository created/verified..."
                )

                # Define files to upload with their target names
                files_to_upload = [
                    # Required files
                    (f"{base_name}.json", f"{target_name}.json"),
                    (f"{base_name}_analysis.json", f"{target_name}_analysis.json"),
                    (f"{base_name}_rules.json", f"{target_name}_rules.json"),
                    # Optional files
                    (f"{base_name}.csv", f"{target_name}.csv"),
                ]

                # Upload each file if it exists
                uploaded_files = []
                failed_uploads = []
                for source_file, target_file in files_to_upload:
                    source_path = base_dir / source_file
                    if source_path.exists():
                        progress.update(
                            upload_task, description=f"Uploading {target_file}..."
                        )
                        logger.info(f"Uploading: {source_path} → {target_file}")

                        try:
                            self.api.upload_file(
                                path_or_fileobj=str(source_path),
                                path_in_repo=target_file,
                                repo_id=repo_id,
                                repo_type="dataset",
                            )
                            uploaded_files.append(target_file)
                            console.print(f"[green]✅ Uploaded: {target_file}[/green]")
                        except Exception as e:
                            logger.error(f"Failed to upload {target_file}: {e}")
                            failed_uploads.append((target_file, str(e)))
                            console.print(
                                f"[red]❌ Failed to upload: {target_file} - {e}[/red]"
                            )
                    else:
                        logger.info(f"Skipping {source_file} (not found)")

                # Upload dataset card
                progress.update(upload_task, description="Uploading dataset card...")
                card = DatasetCard(card_content)
                card.push_to_hub(repo_id)
                console.print(f"[green]✅ Uploaded: README.md[/green]")

                progress.update(upload_task, description="Upload complete!")

        except Exception as e:
            logger.error(f"Upload failed: {e}")
            raise Exception(f"Failed to upload dataset to {repo_id}: {e}")

        # Display success message
        if uploaded_files:
            console.print(
                f"\n[green]✅ Upload successful! Dataset available at: https://huggingface.co/datasets/{repo_id}[/green]"
            )
            console.print(
                f"[green]   - Samples: {validation_results['sample_count']:,}[/green]"
            )
            console.print(f"[green]   - Format: INPUT/OUTPUT schema[/green]")
            console.print(
                f"[green]   - Files uploaded: {len(uploaded_files) + 1}[/green]"
            )  # +1 for README

            # List uploaded files
            console.print(f"\n[bold blue]📁 Uploaded Files:[/bold blue]")
            for file in uploaded_files:
                console.print(f"   - {file}")
            console.print("   - README.md")

            # Show failed uploads if any
            if failed_uploads:
                console.print(f"\n[yellow]⚠️ Some files failed to upload:[/yellow]")
                for file, error in failed_uploads:
                    console.print(f"   - {file}: {error}")
        else:
            console.print(
                f"\n[red]❌ Upload failed - no files were uploaded successfully[/red]"
            )

        logger.info(f"Successfully uploaded dataset to {repo_id}")
        return repo_id

    def list_organization_datasets(self) -> list:
        """List all datasets in the organization.

        Returns:
            List of dataset information dictionaries
        """
        if not self.authenticate():
            return []

        try:
            datasets = self.api.list_datasets(author=self.org_name)
            return [
                {"id": ds.id, "downloads": ds.downloads, "likes": ds.likes}
                for ds in datasets
            ]
        except Exception as e:
            logger.error(f"Failed to list organization datasets: {e}")
            return []

    def delete_dataset(self, dataset_name: str) -> bool:
        """Delete a dataset from the organization (admin only).

        Args:
            dataset_name: Name of dataset to delete

        Returns:
            True if successful, False otherwise
        """
        if not self.authenticate():
            return False

        repo_id = f"{self.org_name}/{dataset_name}"
        try:
            self.api.delete_repo(repo_id, repo_type="dataset")
            console.print(f"[green]✅ Deleted dataset: {repo_id}[/green]")
            return True
        except Exception as e:
            logger.error(f"Failed to delete dataset {repo_id}: {e}")
            console.print(f"[red]❌ Failed to delete dataset: {e}[/red]")
            return False

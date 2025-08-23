"""Preprocessing commands for dataset standardization."""

import json
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.progress import track
from rich.table import Table

from ml_agents.cli.display import (
    display_error,
    display_info,
    display_success,
    display_warning,
)
from ml_agents.utils.logging_config import get_logger

console = Console()
logger = get_logger(__name__)

# Default preprocessing output directory
PREPROCESSING_OUTPUT_DIR = Path("./outputs/preprocessing")


def ensure_preprocessing_output_dir() -> Path:
    """Ensure preprocessing output directory exists and return the path."""
    PREPROCESSING_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return PREPROCESSING_OUTPUT_DIR


def preprocess_list_unprocessed(
    benchmark_csv: str = typer.Option(
        "./documentation/Tasks - Benchmarks.csv",
        "--benchmark-csv",
        "-b",
        help="Path to benchmark CSV file containing dataset information",
    ),
    output_format: str = typer.Option(
        "table", "--format", "-f", help="Output format: table, json, csv"
    ),
    output_file: Optional[str] = typer.Option(
        None, "--output", "-o", help="Save results to file"
    ),
    db_path: Optional[str] = typer.Option(
        None,
        "--db-path",
        help="Database path (default: ./ml_agents_results.db)",
    ),
) -> None:
    """List datasets that haven't been preprocessed yet."""
    from ml_agents.core.dataset_preprocessor import DatasetPreprocessor

    db_path = db_path or "./ml_agents_results.db"
    display_info(f"Scanning for unprocessed datasets in: {benchmark_csv}")

    try:
        preprocessor = DatasetPreprocessor(benchmark_csv, db_path)
        unprocessed = preprocessor.get_unprocessed_datasets()

        if not unprocessed:
            display_info("No unprocessed datasets found")
            return

        # Display results
        if output_format == "table":
            table = Table(title=f"Unprocessed Datasets ({len(unprocessed)} found)")
            table.add_column("Name", style="cyan")
            table.add_column("Task Type", style="yellow")
            table.add_column("Status", style="red")
            table.add_column("Description", style="dim", max_width=50)

            for dataset in unprocessed:
                table.add_row(
                    dataset["name"],
                    dataset.get("task_type", "unknown"),
                    dataset["status"],
                    (
                        dataset.get("description", "")[:47] + "..."
                        if len(dataset.get("description", "")) > 50
                        else dataset.get("description", "")
                    ),
                )

            console.print(table)

        elif output_format == "json":
            result = json.dumps(unprocessed, indent=2)
            if output_file:
                with open(output_file, "w") as f:
                    f.write(result)
                display_success(f"Results saved to: {output_file}")
            else:
                console.print(result)

        elif output_format == "csv":
            import pandas as pd

            df = pd.DataFrame(unprocessed)
            if output_file:
                df.to_csv(output_file, index=False)
                display_success(f"Results saved to: {output_file}")
            else:
                console.print(df.to_csv(index=False))

        display_info(f"Total unprocessed datasets: {len(unprocessed)}")

    except Exception as e:
        display_error(f"Failed to list unprocessed datasets: {e}")
        raise typer.Exit(1)


def preprocess_inspect(
    dataset: str = typer.Argument(
        ..., help="Dataset name or HuggingFace URL to inspect"
    ),
    sample_size: int = typer.Option(
        100,
        "--samples",
        "-n",
        help="Number of samples to analyze for pattern detection",
    ),
    output_file: Optional[str] = typer.Option(
        None,
        "--output",
        "-o",
        help="Save inspection results to JSON file (default: ./outputs/preprocessing/<dataset>_analysis.json)",
    ),
    config: Optional[str] = typer.Option(
        None,
        "--config",
        "-c",
        help="Dataset configuration name (for datasets with multiple configs)",
    ),
) -> None:
    """Inspect dataset schema and detect input/output patterns."""
    from ml_agents.core.dataset_preprocessor import DatasetPreprocessor

    display_info(
        f"Inspecting dataset schema: {dataset}"
        + (f" (config: {config})" if config else "")
    )

    try:
        preprocessor = DatasetPreprocessor()
        schema_info = preprocessor.inspect_dataset_schema(dataset, sample_size, config)

        # Display basic information
        console.print(f"\n📊 [bold blue]Dataset Schema Analysis[/bold blue]")
        console.print(f"Dataset: [cyan]{schema_info['dataset_name']}[/cyan]")
        console.print(
            f"Total samples: [yellow]{schema_info['total_samples']:,}[/yellow]"
        )
        console.print(
            f"Analyzed samples: [yellow]{schema_info['sample_size_analyzed']:,}[/yellow]"
        )
        console.print(f"Columns: [green]{len(schema_info['columns'])}[/green]")

        # Display columns and types
        columns_table = Table(title="Columns")
        columns_table.add_column("Column", style="cyan")
        columns_table.add_column("Type", style="yellow")

        for col, col_type in schema_info["column_types"].items():
            columns_table.add_row(col, col_type)

        console.print(columns_table)

        # Display detected patterns
        patterns = schema_info["detected_patterns"]
        recommendation = patterns.get("recommended_pattern", {})

        console.print(f"\n🔍 [bold blue]Pattern Detection Results[/bold blue]")
        console.print(
            f"Recommended pattern: [green]{recommendation.get('type', 'unknown')}[/green]"
        )
        console.print(
            f"Confidence: [yellow]{recommendation.get('confidence', 0.0):.2f}[/yellow]"
        )

        if recommendation.get("input_fields"):
            console.print(
                f"Input fields: [cyan]{', '.join(recommendation['input_fields'])}[/cyan]"
            )
        if recommendation.get("output_field"):
            console.print(
                f"Output field: [cyan]{recommendation['output_field']}[/cyan]"
            )
        if recommendation.get("reasoning"):
            console.print(f"Reasoning: [dim]{recommendation['reasoning']}[/dim]")

        # Determine output file
        if not output_file:
            output_dir = ensure_preprocessing_output_dir()
            dataset_name = dataset.replace("/", "_").replace("\\", "_")
            if config:
                dataset_name += f"_{config}"
            output_file = str(output_dir / f"{dataset_name}_analysis.json")

        # Save detailed results
        from ml_agents.core.dataset_preprocessor import NumpyJSONEncoder

        with open(output_file, "w") as f:
            json.dump(schema_info, f, indent=2, cls=NumpyJSONEncoder)
        display_success(f"Detailed inspection results saved to: {output_file}")

        display_success("Dataset inspection completed")

    except Exception as e:
        display_error(f"Failed to inspect dataset: {e}")
        raise typer.Exit(1)


def preprocess_generate_rules(
    dataset: str = typer.Argument(..., help="Dataset name to generate rules for"),
    output: str = typer.Option(
        None,
        "--output",
        "-o",
        help="Output file for transformation rules (default: ./outputs/preprocessing/<dataset>_rules.json)",
    ),
    sample_size: int = typer.Option(
        100, "--samples", "-n", help="Number of samples to analyze"
    ),
    config: Optional[str] = typer.Option(
        None,
        "--config",
        "-c",
        help="Dataset configuration name (for datasets with multiple configs)",
    ),
) -> None:
    """Generate transformation rules for a dataset based on schema analysis."""
    from ml_agents.core.dataset_preprocessor import DatasetPreprocessor

    display_info(
        f"Generating transformation rules for: {dataset}"
        + (f" (config: {config})" if config else "")
    )

    try:
        preprocessor = DatasetPreprocessor()

        # First inspect the schema
        schema_info = preprocessor.inspect_dataset_schema(dataset, sample_size, config)

        # Generate transformation rules
        rules = preprocessor.generate_transformation_rules(schema_info)

        # Determine output file
        if not output:
            output_dir = ensure_preprocessing_output_dir()
            dataset_name = dataset.replace("/", "_").replace("\\", "_")
            if config:
                dataset_name += f"_{config}"
            output = str(output_dir / f"{dataset_name}_rules.json")

        # Save rules
        from ml_agents.core.dataset_preprocessor import NumpyJSONEncoder

        with open(output, "w") as f:
            json.dump(rules, f, indent=2, cls=NumpyJSONEncoder)

        # Display summary
        console.print(f"\n🔧 [bold blue]Transformation Rules Generated[/bold blue]")
        console.print(f"Dataset: [cyan]{rules['dataset_name']}[/cyan]")
        console.print(
            f"Transformation type: [yellow]{rules['transformation_type']}[/yellow]"
        )
        console.print(f"Confidence: [green]{rules['confidence']:.2f}[/green]")
        console.print(f"Input format: [cyan]{rules['input_format']}[/cyan]")
        console.print(f"Rules saved to: [green]{output}[/green]")

        display_success("Transformation rules generated successfully")

    except Exception as e:
        display_error(f"Failed to generate transformation rules: {e}")
        raise typer.Exit(1)


def preprocess_transform(
    dataset: str = typer.Argument(..., help="Dataset name to transform"),
    rules: str = typer.Argument(..., help="Path to transformation rules JSON file"),
    output: str = typer.Option(
        None,
        "--output",
        "-o",
        help="Output path for standardized dataset (default: ./outputs/preprocessing/<dataset>.json)",
    ),
    validate: bool = typer.Option(
        True, "--validate/--no-validate", help="Validate transformation integrity"
    ),
    config: Optional[str] = typer.Option(
        None,
        "--config",
        "-c",
        help="Dataset configuration name (for datasets with multiple configs)",
    ),
) -> None:
    """Apply transformation rules to convert dataset to {INPUT, OUTPUT} format."""
    from ml_agents.core.dataset_preprocessor import DatasetPreprocessor

    display_info(
        f"Transforming dataset: {dataset}" + (f" (config: {config})" if config else "")
    )

    try:
        # Load transformation rules
        with open(rules, "r") as f:
            transformation_rules = json.load(f)

        display_info(f"Loaded transformation rules from: {rules}")

        preprocessor = DatasetPreprocessor()

        # Load original dataset for comparison if validation requested
        original_dataset = None
        if validate:
            from datasets import get_dataset_config_info, load_dataset

            # Determine available split
            try:
                if config:
                    dataset_info = get_dataset_config_info(dataset, config_name=config)
                else:
                    dataset_info = get_dataset_config_info(dataset)

                available_splits = list(dataset_info.splits.keys())

                if "train" in available_splits:
                    split_to_use = "train"
                elif "test" in available_splits:
                    split_to_use = "test"
                else:
                    split_to_use = available_splits[0]
            except Exception:
                split_to_use = "train"

            if config:
                original_dataset = load_dataset(dataset, config, split=split_to_use)
            else:
                original_dataset = load_dataset(dataset, split=split_to_use)

        # Apply transformation
        transformed_dataset = preprocessor.apply_transformation(
            dataset, transformation_rules, config
        )

        # Determine output path
        if not output:
            output_dir = ensure_preprocessing_output_dir()
            dataset_name = dataset.replace("/", "_").replace("\\", "_")
            if config:
                dataset_name += f"_{config}"
            output = str(output_dir / f"{dataset_name}.json")

        # Export standardized dataset
        preprocessor.export_standardized(transformed_dataset, output)

        # Validate transformation if requested
        if validate and original_dataset:
            display_info("Validating transformation integrity...")
            validation_results = preprocessor.validate_transformation(
                original_dataset, transformed_dataset
            )

            if validation_results["validation_passed"]:
                display_success("✅ Transformation validation passed")
            else:
                display_warning("⚠️  Transformation validation issues detected:")
                for issue in validation_results["issues"]:
                    console.print(f"   • {issue}")

            # Display validation metrics
            console.print(f"\n📊 [bold blue]Validation Metrics[/bold blue]")
            console.print(
                f"Original samples: [yellow]{validation_results['original_samples']:,}[/yellow]"
            )
            console.print(
                f"Transformed samples: [yellow]{validation_results['transformed_samples']:,}[/yellow]"
            )
            console.print(
                f"Empty inputs: [red]{validation_results['empty_inputs']}[/red]"
            )
            console.print(
                f"Empty outputs: [red]{validation_results['empty_outputs']}[/red]"
            )

        display_success(f"Dataset transformation completed: {output}")

    except Exception as e:
        display_error(f"Failed to transform dataset: {e}")
        raise typer.Exit(1)


def preprocess_batch(
    benchmark_csv: str = typer.Option(
        "./documentation/Tasks - Benchmarks.csv",
        "--benchmark-csv",
        "-b",
        help="Path to benchmark CSV file",
    ),
    output_dir: str = typer.Option(
        "./outputs/preprocessing",
        "--output-dir",
        "-o",
        help="Output directory for processed datasets",
    ),
    max_datasets: int = typer.Option(
        10, "--max", "-m", help="Maximum number of datasets to process"
    ),
    sample_size: int = typer.Option(
        100,
        "--samples",
        "-n",
        help="Number of samples to analyze for pattern detection",
    ),
    confidence_threshold: float = typer.Option(
        0.6,
        "--confidence",
        "-c",
        help="Minimum confidence threshold for automatic processing",
    ),
    config: Optional[str] = typer.Option(
        None,
        "--config",
        help="Dataset configuration name (applies to all datasets in batch)",
    ),
) -> None:
    """Batch process multiple unprocessed datasets."""
    from ml_agents.core.dataset_preprocessor import DatasetPreprocessor

    display_info(f"Starting batch preprocessing of datasets from: {benchmark_csv}")

    try:
        # Initialize preprocessor with database integration
        db_path = "./ml_agents_results.db"
        preprocessor = DatasetPreprocessor(benchmark_csv, db_path)
        unprocessed = preprocessor.get_unprocessed_datasets()

        if not unprocessed:
            display_info("No unprocessed datasets found")
            return

        # Limit processing
        datasets_to_process = unprocessed[:max_datasets]

        display_info(
            f"Processing {len(datasets_to_process)} of {len(unprocessed)} unprocessed datasets"
        )

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        successful = 0
        failed = 0

        for i, dataset_info in enumerate(
            track(datasets_to_process, description="Processing datasets...")
        ):
            dataset_name = dataset_info["name"]
            dataset_url = dataset_info.get("url", dataset_name)

            try:
                console.print(
                    f"\n🔄 [{i+1}/{len(datasets_to_process)}] Processing: [cyan]{dataset_name}[/cyan]"
                    + (f" (config: {config})" if config else "")
                )

                # Inspect schema
                schema_info = preprocessor.inspect_dataset_schema(
                    dataset_url, sample_size, config
                )

                # Generate rules
                rules = preprocessor.generate_transformation_rules(schema_info)

                # Check confidence threshold
                if rules["confidence"] < confidence_threshold:
                    display_warning(
                        f"Low confidence ({rules['confidence']:.2f}) - skipping automatic processing"
                    )
                    failed += 1
                    continue

                # Apply transformation
                from datasets import get_dataset_config_info, load_dataset

                # Determine available split
                try:
                    if config:
                        dataset_info = get_dataset_config_info(
                            dataset_url, config_name=config
                        )
                    else:
                        dataset_info = get_dataset_config_info(dataset_url)

                    available_splits = list(dataset_info.splits.keys())

                    if "train" in available_splits:
                        split_to_use = "train"
                    elif "test" in available_splits:
                        split_to_use = "test"
                    else:
                        split_to_use = available_splits[0]
                except Exception:
                    split_to_use = "train"

                if config:
                    original_dataset = load_dataset(
                        dataset_url, config, split=split_to_use
                    )
                else:
                    original_dataset = load_dataset(dataset_url, split=split_to_use)
                transformed_dataset = preprocessor.apply_transformation(
                    dataset_url, rules, config
                )

                # Export
                safe_name = dataset_name.replace("/", "_").replace("\\", "_")
                if config:
                    safe_name += f"_{config}"
                dataset_output_path = output_path / f"{safe_name}.json"
                preprocessor.export_standardized(
                    transformed_dataset, str(dataset_output_path)
                )

                # Validate transformation
                validation_results = preprocessor.validate_transformation(
                    original_dataset, transformed_dataset
                )

                # Save metadata to database
                preprocessor._save_preprocessing_metadata(
                    dataset_name=dataset_name,
                    dataset_url=dataset_url,
                    schema_info=schema_info,
                    rules=rules,
                    validation_results=validation_results,
                    output_path=str(dataset_output_path),
                )

                # Save rules alongside
                rules_path = output_path / f"{safe_name}_rules.json"
                from ml_agents.core.dataset_preprocessor import NumpyJSONEncoder

                with open(rules_path, "w") as f:
                    json.dump(rules, f, indent=2, cls=NumpyJSONEncoder)

                display_success(f"✅ Processed: {dataset_name}")
                successful += 1

            except Exception as dataset_error:
                display_error(f"Failed to process {dataset_name}: {dataset_error}")
                failed += 1
                continue

        # Summary
        console.print(f"\n📊 [bold blue]Batch Processing Summary[/bold blue]")
        console.print(f"Successful: [green]{successful}[/green]")
        console.print(f"Failed: [red]{failed}[/red]")
        console.print(f"Total processed: [yellow]{successful + failed}[/yellow]")
        console.print(f"Output directory: [cyan]{output_dir}[/cyan]")

        if successful > 0:
            display_success("Batch preprocessing completed with successes")
        else:
            display_warning("Batch preprocessing completed with no successes")

    except Exception as e:
        display_error(f"Failed to run batch preprocessing: {e}")
        raise typer.Exit(1)


def preprocess_upload(
    processed_file: str = typer.Argument(
        ...,
        help="Path to processed dataset JSON file (will also upload related _analysis.json, _rules.json, and .csv files)",
    ),
    source_dataset: str = typer.Option(
        ...,
        "--source-dataset",
        "-s",
        help="Original dataset name/URL for attribution (e.g., MilaWang/SpatialEval)",
    ),
    target_name: str = typer.Option(
        ...,
        "--target-name",
        "-t",
        help="Target name for uploaded dataset (will be uploaded to c4ai-ml-agents/<target-name>)",
    ),
    config: Optional[str] = typer.Option(
        None,
        "--config",
        "-c",
        help="Dataset configuration that was used during preprocessing",
    ),
    description: Optional[str] = typer.Option(
        None, "--description", "-d", help="Custom description for the dataset"
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Validate and prepare but don't actually upload"
    ),
) -> None:
    """Upload processed dataset and all related preprocessing files to HuggingFace Hub.

    This command uploads:
    - Main dataset JSON file
    - Schema analysis JSON file (*_analysis.json)
    - Transformation rules JSON file (*_rules.json)
    - CSV file if available (*.csv)
    - Generated README.md with metadata and transformation rules
    """
    from ml_agents.core.dataset_uploader import DatasetUploader

    display_info(f"Preparing to upload: {processed_file}")
    display_info(f"Source dataset: {source_dataset}")
    display_info(f"Target: c4ai-ml-agents/{target_name}")

    if config:
        display_info(f"Configuration: {config}")

    if description:
        display_info(f"Description: {description}")

    try:
        uploader = DatasetUploader(org_name="c4ai-ml-agents")

        # Validate the processed file first
        console.print("\n🔍 [bold blue]Validating processed dataset...[/bold blue]")
        validation_results = uploader.validate_processed_file(processed_file)

        if not validation_results["validation_passed"]:
            display_error("Dataset validation failed:")
            for issue in validation_results["issues"]:
                console.print(f"  [red]❌ {issue}[/red]")
            raise typer.Exit(1)

        # Display validation results
        console.print(f"[green]✅ Dataset validation passed[/green]")
        console.print(f"  - Format: {validation_results['format'].upper()}")
        console.print(f"  - Samples: {validation_results['sample_count']:,}")
        console.print(f"  - File size: {validation_results['file_size_mb']:.1f} MB")
        console.print(
            f"  - Schema: {'INPUT/OUTPUT' if validation_results['has_input_output_schema'] else 'Unknown'}"
        )

        if validation_results["issues"]:
            console.print("[yellow]⚠️ Warnings:[/yellow]")
            for issue in validation_results["issues"]:
                console.print(f"  [yellow]• {issue}[/yellow]")

        if dry_run:
            display_info("Dry run mode - skipping actual upload")
            console.print(
                f"\n[green]✅ Dry run successful! Dataset is ready for upload.[/green]"
            )
            console.print(
                f"[green]   Target: https://huggingface.co/datasets/c4ai-ml-agents/{target_name}[/green]"
            )
            console.print(
                f"[green]   Run without --dry-run to perform actual upload[/green]"
            )
            return

        # Confirm upload
        console.print(
            f"\n📤 [bold yellow]Ready to upload to HuggingFace Hub[/bold yellow]"
        )
        console.print(f"Target repository: [cyan]c4ai-ml-agents/{target_name}[/cyan]")

        confirm = typer.confirm("Proceed with upload?")
        if not confirm:
            display_info("Upload cancelled by user")
            return

        # Perform upload
        console.print("\n🚀 [bold green]Starting upload...[/bold green]")
        repo_id = uploader.upload_dataset(
            processed_file=processed_file,
            source_dataset=source_dataset,
            target_name=target_name,
            config=config,
            description=description,
        )

        display_success(
            f"✅ Dataset successfully uploaded to: https://huggingface.co/datasets/{repo_id}"
        )

        # Display usage instructions
        console.print("\n📋 [bold blue]Usage Instructions:[/bold blue]")
        console.print(f"[cyan]# Load in Python[/cyan]")
        console.print(f"from datasets import load_dataset")
        console.print(f"dataset = load_dataset('{repo_id}')")
        console.print()
        console.print(f"[cyan]# Use with ML Agents CLI[/cyan]")
        console.print(
            f"ml-agents run --custom-dataset {repo_id} --approach ChainOfThought"
        )

    except Exception as e:
        display_error(f"Failed to upload dataset: {e}")
        logger.error(f"Dataset upload failed: {e}")
        raise typer.Exit(1)

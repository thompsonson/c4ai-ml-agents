"""Dataset preprocessing for standardizing diverse benchmark datasets."""

import json
import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from datasets import Dataset, load_dataset

from ml_agents.core.database_manager import DatabaseConfig, DatabaseManager
from ml_agents.utils.logging_config import get_logger

logger = get_logger(__name__)


class NumpyJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles numpy arrays and other numpy types."""

    def default(self, obj):
        """Convert numpy objects to JSON serializable types."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif pd.isna(obj):
            return None
        return super().default(obj)


class DatasetPreprocessor:
    """Preprocesses diverse benchmark datasets to standardized {INPUT, OUTPUT} schema."""

    def __init__(
        self, benchmark_csv: Optional[str] = None, db_path: Optional[str] = None
    ) -> None:
        """Initialize dataset preprocessor.

        Args:
            benchmark_csv: Path to CSV file containing benchmark dataset information
            db_path: Path to database for tracking preprocessing metadata
        """
        self.benchmark_csv = benchmark_csv
        self._benchmark_data: Optional[pd.DataFrame] = None

        # Database integration
        if db_path:
            self.db_config = DatabaseConfig(db_path=db_path)
            self.db_manager = DatabaseManager(self.db_config)
        else:
            self.db_config = None
            self.db_manager = None

        # Common input field patterns (ordered by specificity)
        self.input_patterns = [
            "question",
            "input",
            "text",
            "sentence",
            "context",
            "prompt",
            "query",
            "problem",
            "statement",
            "passage",
            "content",
        ]

        # Common output field patterns (ordered by specificity - most complete first)
        self.output_patterns = [
            "full_answer",
            "complete_answer",
            "detailed_answer",
            "oracle_full_answer",
            "full_response",
            "complete_response",
            "detailed_response",
            "answer",
            "output",
            "label",
            "target",
            "response",
            "solution",
            "result",
            "ground_truth",
            "correct_answer",
            "expected",
        ]

        # Multi-field patterns for complex schemas
        self.multi_field_patterns = {
            "sentence_pair": ["sentence1", "sentence2"],
            "premise_hypothesis": ["premise", "hypothesis"],
            "context_question": ["context", "question"],
            "passage_question": ["passage", "question"],
            "story_question_choices": ["story", "question", "candidate_answers"],
            "conversation": ["conversation", "response"],
            "code_description": ["code", "description"],
            "instruction_input": ["instruction", "input"],
        }

    def get_unprocessed_datasets(self) -> List[Dict[str, Any]]:
        """Get list of datasets that haven't been preprocessed yet.

        Returns:
            List of dataset dictionaries with metadata
        """
        if not self.benchmark_csv:
            raise ValueError("benchmark_csv path not provided")

        if self._benchmark_data is None:
            self._benchmark_data = pd.read_csv(self.benchmark_csv)

        # Get processed datasets from database if available
        processed_datasets = set()
        if self.db_manager:
            try:
                with self.db_manager.get_connection() as conn:
                    cursor = conn.execute(
                        "SELECT dataset_name FROM dataset_preprocessing WHERE status = 'processed'"
                    )
                    processed_datasets = {row[0] for row in cursor.fetchall()}
            except Exception as e:
                logger.warning(f"Failed to query processed datasets from database: {e}")

        # Filter out already processed datasets
        unprocessed = []
        for _, row in self._benchmark_data.iterrows():
            dataset_name = row.get("dataset_name", row.get("name", "unknown"))

            # Skip if already processed
            if dataset_name in processed_datasets:
                continue

            dataset_info = {
                "name": dataset_name,
                "url": row.get("url", row.get("dataset_url", "")),
                "description": row.get("description", ""),
                "task_type": row.get("task_type", row.get("category", "unknown")),
                "status": "unprocessed",
            }
            unprocessed.append(dataset_info)

        logger.info(
            f"Found {len(unprocessed)} unprocessed datasets ({len(processed_datasets)} already processed)"
        )
        return unprocessed

    def inspect_dataset_schema(
        self, dataset_url: str, sample_size: int = 100, config: Optional[str] = None
    ) -> Dict[str, Any]:
        """Inspect dataset schema and detect input/output patterns.

        Args:
            dataset_url: HuggingFace dataset URL or name
            sample_size: Number of samples to analyze for pattern detection
            config: Dataset configuration name (for datasets with multiple configs)

        Returns:
            Dictionary containing schema information and detected patterns
        """
        logger.info(
            f"Inspecting schema for dataset: {dataset_url}"
            + (f" (config: {config})" if config else "")
        )

        try:
            # Load dataset with config support and flexible split handling
            from datasets import get_dataset_config_info

            # Try to get available splits
            try:
                if config:
                    dataset_info = get_dataset_config_info(
                        dataset_url, config_name=config
                    )
                    dataset_display_name = f"{dataset_url}:{config}"
                else:
                    dataset_info = get_dataset_config_info(dataset_url)
                    dataset_display_name = dataset_url

                available_splits = list(dataset_info.splits.keys())

                # Prefer 'train' split, but fallback to first available split
                if "train" in available_splits:
                    split_to_use = "train"
                elif "test" in available_splits:
                    split_to_use = "test"
                else:
                    split_to_use = available_splits[0]

                logger.info(
                    f"Using split '{split_to_use}' (available splits: {available_splits})"
                )

            except Exception:
                # Fallback to default 'train' split if we can't get info
                split_to_use = "train"

            # Load dataset with determined split
            if config:
                dataset = load_dataset(dataset_url, config, split=split_to_use)
            else:
                dataset = load_dataset(dataset_url, split=split_to_use)

            # Get sample for analysis
            sample_data = dataset.select(range(min(sample_size, len(dataset))))

            # Convert to pandas for easier analysis
            df = sample_data.to_pandas()

            schema_info = {
                "dataset_name": dataset_display_name,
                "dataset_url": dataset_url,
                "config": config,
                "total_samples": len(dataset),
                "columns": list(df.columns),
                "sample_size_analyzed": len(df),
                "column_types": {col: str(dtype) for col, dtype in df.dtypes.items()},
                "detected_patterns": self._detect_patterns(df),
                "sample_data": df.head(3).to_dict(
                    "records"
                ),  # Include few samples for reference
            }

            logger.info(f"Schema inspection complete for {dataset_url}")
            return schema_info

        except Exception as e:
            error_msg = str(e)

            # Enhanced error handling for config-related issues
            if "Config name is missing" in error_msg:
                # Try to extract available configs from error message
                try:
                    from datasets import get_dataset_config_names

                    available_configs = get_dataset_config_names(dataset_url)
                    enhanced_msg = f"Dataset '{dataset_url}' requires a configuration. Available configs: {available_configs}\n"
                    enhanced_msg += f"Use: ml-agents preprocess-inspect {dataset_url} --config <config_name>"
                    logger.error(
                        f"Configuration required for {dataset_url}: {enhanced_msg}"
                    )
                    raise ValueError(enhanced_msg)
                except ImportError:
                    pass

            logger.error(f"Failed to inspect dataset {dataset_url}: {e}")
            raise

    def _detect_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Auto-detect input/output fields from column names and content.

        Args:
            df: DataFrame containing dataset samples

        Returns:
            Dictionary with detected pattern information
        """
        columns = [col.lower() for col in df.columns]

        # Detect single input/output fields
        input_candidates = []
        output_candidates = []

        for col in df.columns:
            col_lower = col.lower()

            # Check input patterns
            for pattern in self.input_patterns:
                if pattern in col_lower:
                    confidence = self._calculate_pattern_confidence(
                        col_lower, pattern, df[col]
                    )
                    input_candidates.append((col, pattern, confidence))
                    break

            # Check output patterns
            for pattern in self.output_patterns:
                if pattern in col_lower:
                    confidence = self._calculate_pattern_confidence(
                        col_lower, pattern, df[col]
                    )
                    output_candidates.append((col, pattern, confidence))
                    break

        # Sort by confidence, then by completeness indicators
        input_candidates.sort(
            key=lambda x: (x[2], self._get_completeness_score(x[0])), reverse=True
        )
        output_candidates.sort(
            key=lambda x: (x[2], self._get_completeness_score(x[0])), reverse=True
        )

        # Detect multi-field patterns
        multi_field_matches = {}
        for pattern_name, required_fields in self.multi_field_patterns.items():
            matches = []
            for field in required_fields:
                for col in df.columns:
                    if field.lower() in col.lower():
                        matches.append(col)
                        break

            if len(matches) == len(required_fields):
                multi_field_matches[pattern_name] = matches

        # Analyze content types
        content_analysis = {}
        for col in df.columns:
            sample_values = df[col].dropna().head(10)
            content_analysis[col] = self._analyze_content_type(sample_values)

        return {
            "single_field_input_candidates": input_candidates,
            "single_field_output_candidates": output_candidates,
            "multi_field_patterns": multi_field_matches,
            "content_analysis": content_analysis,
            "recommended_pattern": self._recommend_pattern(
                input_candidates, output_candidates, multi_field_matches
            ),
        }

    def _calculate_pattern_confidence(
        self, column_name: str, pattern: str, content: pd.Series = None
    ) -> float:
        """Calculate confidence score for pattern match with content analysis.

        Args:
            column_name: Column name to analyze
            pattern: Pattern to match against
            content: Column content for additional scoring

        Returns:
            Confidence score between 0 and 1
        """
        base_confidence = 0.0

        # Exact match gets highest score
        if column_name == pattern:
            base_confidence = 1.0
        # Contains pattern gets medium score
        elif pattern in column_name:
            base_confidence = 0.8
        else:
            # Partial match using string similarity
            pattern_chars = set(pattern)
            column_chars = set(column_name)
            overlap = len(pattern_chars.intersection(column_chars))
            base_confidence = overlap / max(len(pattern_chars), len(column_chars)) * 0.6

        # Content-based adjustments
        if content is not None and len(content) > 0:
            content_boost = self._analyze_content_quality(content, pattern)
            base_confidence = min(1.0, base_confidence + content_boost)

        return base_confidence

    def _get_completeness_score(self, column_name: str) -> float:
        """Calculate completeness score based on field name indicators.

        Args:
            column_name: Column name to analyze

        Returns:
            Completeness score (higher = more complete)
        """
        col_lower = column_name.lower()
        score = 0.0

        # Bonus for completeness indicators
        if any(
            indicator in col_lower
            for indicator in ["full", "complete", "detailed", "comprehensive"]
        ):
            score += 0.3

        # Bonus for oracle/ground truth indicators
        if any(
            indicator in col_lower
            for indicator in ["oracle", "ground_truth", "reference"]
        ):
            score += 0.2

        # Bonus for longer, more descriptive names
        score += min(0.2, len(column_name) / 50.0)

        return score

    def _analyze_content_quality(self, content: pd.Series, pattern: str) -> float:
        """Analyze content quality for additional confidence boost.

        Args:
            content: Column content to analyze
            pattern: Pattern being matched

        Returns:
            Content quality boost (0.0 to 0.2)
        """
        if len(content) == 0:
            return 0.0

        sample_values = content.dropna().head(10)
        if len(sample_values) == 0:
            return 0.0

        str_values = sample_values.astype(str)
        avg_length = str_values.str.len().mean()

        boost = 0.0

        # Prefer fields with longer, more informative content
        if pattern in ["answer", "response", "output"]:
            # Longer answers likely more complete
            if avg_length > 20:
                boost += 0.1
            elif avg_length > 50:
                boost += 0.15

            # Check for structured answers (e.g., "A. Answer", "1. Response")
            structured_count = sum(
                1
                for val in str_values
                if any(
                    val.strip().startswith(prefix)
                    for prefix in ["A.", "B.", "C.", "D.", "1.", "2.", "("]
                )
            )
            if structured_count > len(str_values) * 0.5:
                boost += 0.1

        return min(0.2, boost)

    def _analyze_content_type(self, sample_values: pd.Series) -> Dict[str, Any]:
        """Analyze content type and characteristics of sample values.

        Args:
            sample_values: Sample values from a column

        Returns:
            Dictionary with content analysis
        """
        if len(sample_values) == 0:
            return {"type": "empty", "avg_length": 0, "samples": []}

        # Convert to strings for analysis
        str_values = sample_values.astype(str)

        # Calculate average length
        avg_length = str_values.str.len().mean()

        # Detect content type
        content_type = "text"

        # Check if numeric
        try:
            pd.to_numeric(sample_values)
            content_type = "numeric"
        except (ValueError, TypeError):
            pass

        # Check if boolean
        unique_values = set(str_values.str.lower())
        if unique_values.issubset({"true", "false", "1", "0", "yes", "no"}):
            content_type = "boolean"

        # Check if multiple choice (short options)
        if avg_length < 10 and len(unique_values) < len(sample_values) * 0.5:
            content_type = "categorical"

        return {
            "type": content_type,
            "avg_length": round(avg_length, 1),
            "unique_values": len(unique_values),
            "sample_values": list(str_values.head(3)),
        }

    def _recommend_pattern(
        self,
        input_candidates: List[Tuple],
        output_candidates: List[Tuple],
        multi_field_matches: Dict[str, List[str]],
    ) -> Dict[str, Any]:
        """Recommend the best transformation pattern based on detected patterns.

        Args:
            input_candidates: List of (column, pattern, confidence) tuples for inputs
            output_candidates: List of (column, pattern, confidence) tuples for outputs
            multi_field_matches: Dictionary of multi-field pattern matches

        Returns:
            Recommended transformation pattern
        """
        recommendation = {
            "type": "unknown",
            "confidence": 0.0,
            "input_fields": [],
            "output_field": None,
            "reasoning": "",
        }

        # Prefer multi-field patterns if they exist
        if multi_field_matches:
            best_multi_pattern = max(
                multi_field_matches.keys(), key=lambda x: len(multi_field_matches[x])
            )
            recommendation.update(
                {
                    "type": "multi_field",
                    "pattern_name": best_multi_pattern,
                    "input_fields": multi_field_matches[best_multi_pattern][
                        :-1
                    ],  # All but last
                    "output_field": multi_field_matches[best_multi_pattern][
                        -1
                    ],  # Last field as output
                    "confidence": 0.8,
                    "reasoning": f"Detected {best_multi_pattern} pattern with fields: {multi_field_matches[best_multi_pattern]}",
                }
            )

        # Fallback to single field pattern
        elif input_candidates and output_candidates:
            best_input = input_candidates[0]
            best_output = output_candidates[0]
            avg_confidence = (best_input[2] + best_output[2]) / 2

            recommendation.update(
                {
                    "type": "single_field",
                    "input_fields": [best_input[0]],
                    "output_field": best_output[0],
                    "confidence": avg_confidence,
                    "reasoning": f"Best single field mapping: {best_input[0]} -> {best_output[0]}",
                }
            )

        return recommendation

    def _format_multiple_choice_options(self, candidates: List[str]) -> str:
        """Format candidate answers as multiple choice options.

        Args:
            candidates: List of candidate answer strings

        Returns:
            Formatted string with options labeled A), B), C), etc.
        """
        if not candidates:
            return ""

        formatted_options = []
        for i, candidate in enumerate(candidates):
            # Convert index to letter (A, B, C, ...)
            option_letter = chr(65 + i)  # 65 is ASCII for 'A'
            # Clean up the candidate text
            clean_candidate = str(candidate).strip()
            formatted_options.append(f"{option_letter}) {clean_candidate}")

        return "\n".join(formatted_options)

    def generate_transformation_rules(self, schema: Dict[str, Any]) -> Dict[str, Any]:
        """Generate transformation rules based on detected patterns.

        Args:
            schema: Schema information from inspect_dataset_schema

        Returns:
            Transformation rules dictionary
        """
        detected_patterns = schema.get("detected_patterns", {})
        recommendation = detected_patterns.get("recommended_pattern", {})

        rules = {
            "dataset_name": schema["dataset_name"],
            "transformation_type": recommendation.get("type", "manual"),
            "confidence": recommendation.get("confidence", 0.0),
            "input_format": "single_field",
            "input_fields": recommendation.get("input_fields", []),
            "output_field": recommendation.get("output_field"),
            "field_separator": "\n\n",
            "field_labels": {},
            "preprocessing_steps": [],
        }

        # Set up field labels and format based on pattern type
        if recommendation.get("type") == "multi_field":
            rules["input_format"] = "multi_field"

            # Create labels for each input field
            for field in rules["input_fields"]:
                clean_label = field.replace("_", " ").upper()
                rules["field_labels"][field] = f"{clean_label}:"

            # Add pattern-specific formatting
            pattern_name = recommendation.get("pattern_name", "")
            if pattern_name == "sentence_pair":
                rules["field_labels"] = {
                    rules["input_fields"][0]: "SENTENCE 1:",
                    rules["input_fields"][1]: "SENTENCE 2:",
                }
            elif pattern_name == "context_question":
                rules["field_labels"] = {
                    rules["input_fields"][0]: "CONTEXT:",
                    rules["input_fields"][1]: "QUESTION:",
                }
            elif pattern_name == "story_question_choices":
                rules["field_labels"] = {
                    "story": "STORY:",
                    "question": "QUESTION:",
                    "candidate_answers": "OPTIONS:",
                }
                # Add preprocessing step for answer resolution
                rules["preprocessing_steps"] = ["resolve_answer_index"]

        logger.info(
            f"Generated transformation rules for {schema['dataset_name']} with confidence {rules['confidence']:.2f}"
        )
        return rules

    def apply_transformation(
        self, dataset_path: str, rules: Dict[str, Any], config: Optional[str] = None
    ) -> Dataset:
        """Apply transformation rules to convert dataset to {INPUT, OUTPUT} format.

        Args:
            dataset_path: Path to dataset to transform
            rules: Transformation rules from generate_transformation_rules
            config: Dataset configuration name (for datasets with multiple configs)

        Returns:
            Transformed dataset with INPUT/OUTPUT schema
        """
        logger.info(
            f"Applying transformation to dataset: {dataset_path}"
            + (f" (config: {config})" if config else "")
        )

        try:
            # Load dataset with config support and flexible split handling
            from datasets import get_dataset_config_info

            # Try to get available splits
            try:
                if config:
                    dataset_info = get_dataset_config_info(
                        dataset_path, config_name=config
                    )
                else:
                    dataset_info = get_dataset_config_info(dataset_path)

                available_splits = list(dataset_info.splits.keys())

                # Prefer 'train' split, but fallback to first available split
                if "train" in available_splits:
                    split_to_use = "train"
                elif "test" in available_splits:
                    split_to_use = "test"
                else:
                    split_to_use = available_splits[0]

            except Exception:
                # Fallback to default 'train' split if we can't get info
                split_to_use = "train"

            # Load dataset with determined split
            if config:
                dataset = load_dataset(dataset_path, config, split=split_to_use)
            else:
                dataset = load_dataset(dataset_path, split=split_to_use)

            def transform_example(example):
                """Transform a single example according to rules."""

                # Build INPUT field
                if rules["input_format"] == "multi_field":
                    input_parts = []
                    for field in rules["input_fields"]:
                        if field in example:
                            label = rules["field_labels"].get(
                                field, f"{field.upper()}:"
                            )

                            # Special formatting for candidate_answers field
                            if field == "candidate_answers" and isinstance(
                                example[field], list
                            ):
                                options_text = self._format_multiple_choice_options(
                                    example[field]
                                )
                                input_parts.append(f"{label}\n\n{options_text}")
                            else:
                                input_parts.append(f"{label}\n\n{example[field]}")
                        else:
                            logger.warning(f"Field {field} not found in example")

                    input_text = f"\n\n{rules['field_separator']}".join(input_parts)
                else:
                    # Single field input
                    input_field = (
                        rules["input_fields"][0] if rules["input_fields"] else None
                    )
                    if input_field and input_field in example:
                        input_text = str(example[input_field])
                    else:
                        logger.warning(
                            f"Input field {input_field} not found in example"
                        )
                        input_text = ""

                # Build OUTPUT field
                output_field = rules["output_field"]
                if output_field and output_field in example:
                    output_value = example[output_field]

                    # Check if we need to resolve answer index to text
                    if "resolve_answer_index" in rules.get("preprocessing_steps", []):
                        if (
                            isinstance(output_value, (int, np.integer))
                            and "candidate_answers" in example
                        ):
                            candidates = example["candidate_answers"]
                            if isinstance(candidates, list) and 0 <= output_value < len(
                                candidates
                            ):
                                output_text = str(candidates[output_value]).strip()
                            else:
                                logger.warning(
                                    f"Could not resolve answer index {output_value}"
                                )
                                output_text = str(output_value)
                        else:
                            output_text = str(output_value)
                    else:
                        output_text = str(output_value)
                else:
                    logger.warning(f"Output field {output_field} not found in example")
                    output_text = ""

                return {"INPUT": input_text, "OUTPUT": output_text}

            # Apply transformation
            transformed_dataset = dataset.map(
                transform_example, remove_columns=dataset.column_names
            )

            logger.info(
                f"Transformation complete. Dataset now has {len(transformed_dataset)} samples with INPUT/OUTPUT schema"
            )
            return transformed_dataset

        except Exception as e:
            logger.error(f"Failed to transform dataset {dataset_path}: {e}")
            raise

    def export_standardized(self, dataset: Dataset, output_path: str) -> None:
        """Export standardized dataset to specified path.

        Args:
            dataset: Transformed dataset to export
            output_path: Path where to save the standardized dataset
        """
        output_path_obj = Path(output_path)
        output_path_obj.parent.mkdir(parents=True, exist_ok=True)

        # Export as CSV for easy inspection
        if output_path.endswith(".csv"):
            df = dataset.to_pandas()
            df.to_csv(output_path, index=False)

        # Export as JSON for programmatic use
        elif output_path.endswith(".json"):
            # Convert to list of records format: [{"INPUT": "...", "OUTPUT": "..."}, ...]
            records = []
            for i in range(len(dataset)):
                records.append(
                    {"INPUT": dataset[i]["INPUT"], "OUTPUT": dataset[i]["OUTPUT"]}
                )
            with open(output_path, "w") as f:
                json.dump(records, f, indent=2, cls=NumpyJSONEncoder)

        # Default to arrow format for efficient loading
        else:
            dataset.save_to_disk(output_path)

        logger.info(f"Exported standardized dataset to: {output_path}")

    def validate_transformation(
        self, original_dataset: Dataset, transformed_dataset: Dataset
    ) -> Dict[str, Any]:
        """Validate that transformation preserved data integrity.

        Args:
            original_dataset: Original dataset before transformation
            transformed_dataset: Transformed dataset with INPUT/OUTPUT schema

        Returns:
            Validation results dictionary
        """
        validation_results = {
            "sample_count_preserved": len(original_dataset) == len(transformed_dataset),
            "original_samples": len(original_dataset),
            "transformed_samples": len(transformed_dataset),
            "empty_inputs": sum(
                1 for example in transformed_dataset if not example["INPUT"].strip()
            ),
            "empty_outputs": sum(
                1 for example in transformed_dataset if not example["OUTPUT"].strip()
            ),
            "validation_passed": True,
            "issues": [],
        }

        # Check for data loss
        if not validation_results["sample_count_preserved"]:
            validation_results["issues"].append(
                f"Sample count changed: {len(original_dataset)} -> {len(transformed_dataset)}"
            )
            validation_results["validation_passed"] = False

        # Check for empty fields
        if validation_results["empty_inputs"] > 0:
            validation_results["issues"].append(
                f"{validation_results['empty_inputs']} samples have empty INPUT fields"
            )

        if validation_results["empty_outputs"] > 0:
            validation_results["issues"].append(
                f"{validation_results['empty_outputs']} samples have empty OUTPUT fields"
            )

        # If more than 10% of samples have empty fields, mark as failed
        total_samples = len(transformed_dataset)
        if total_samples > 0:
            empty_ratio = (
                validation_results["empty_inputs"] + validation_results["empty_outputs"]
            ) / (2 * total_samples)
            if empty_ratio > 0.1:
                validation_results["validation_passed"] = False
                validation_results["issues"].append(
                    f"Too many empty fields: {empty_ratio:.1%} of all fields are empty"
                )

        logger.info(
            f"Validation complete: {'PASSED' if validation_results['validation_passed'] else 'FAILED'}"
        )
        if validation_results["issues"]:
            for issue in validation_results["issues"]:
                logger.warning(f"Validation issue: {issue}")

        return validation_results

    def _save_preprocessing_metadata(
        self,
        dataset_name: str,
        dataset_url: str,
        schema_info: Dict[str, Any],
        rules: Dict[str, Any],
        validation_results: Optional[Dict[str, Any]] = None,
        output_path: Optional[str] = None,
    ) -> str:
        """Save preprocessing metadata to database.

        Args:
            dataset_name: Name of the dataset
            dataset_url: URL or path to the dataset
            schema_info: Schema analysis results
            rules: Transformation rules applied
            validation_results: Validation results if available
            output_path: Path where processed dataset was saved

        Returns:
            Processing record ID
        """
        if not self.db_manager:
            logger.warning("Database manager not initialized - skipping metadata save")
            return ""

        record_id = str(uuid.uuid4())

        try:
            with self.db_manager.get_connection() as conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO dataset_preprocessing
                    (id, dataset_name, dataset_url, status, schema_analysis, transformation_rules,
                     confidence_score, original_samples, processed_samples, validation_results,
                     output_path, processed_at, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        record_id,
                        dataset_name,
                        dataset_url,
                        (
                            "processed"
                            if validation_results
                            and validation_results.get("validation_passed")
                            else "failed"
                        ),
                        json.dumps(schema_info, cls=NumpyJSONEncoder),
                        json.dumps(rules, cls=NumpyJSONEncoder),
                        rules.get("confidence", 0.0),
                        schema_info.get("total_samples", 0),
                        (
                            validation_results.get("transformed_samples", 0)
                            if validation_results
                            else 0
                        ),
                        (
                            json.dumps(validation_results, cls=NumpyJSONEncoder)
                            if validation_results
                            else None
                        ),
                        output_path,
                        datetime.utcnow().isoformat(),
                        datetime.utcnow().isoformat(),
                    ),
                )

            logger.info(
                f"Saved preprocessing metadata for {dataset_name} with ID: {record_id}"
            )
            return record_id

        except Exception as e:
            logger.error(f"Failed to save preprocessing metadata: {e}")
            return ""

    def get_preprocessing_status(self, dataset_name: str) -> Optional[Dict[str, Any]]:
        """Get preprocessing status for a dataset.

        Args:
            dataset_name: Name of the dataset to check

        Returns:
            Preprocessing status record or None if not found
        """
        if not self.db_manager:
            return None

        try:
            with self.db_manager.get_connection() as conn:
                cursor = conn.execute(
                    """
                    SELECT id, dataset_name, dataset_url, status, confidence_score,
                           original_samples, processed_samples, output_path, processed_at
                    FROM dataset_preprocessing
                    WHERE dataset_name = ?
                """,
                    (dataset_name,),
                )

                row = cursor.fetchone()
                if row:
                    return {
                        "id": row[0],
                        "dataset_name": row[1],
                        "dataset_url": row[2],
                        "status": row[3],
                        "confidence_score": row[4],
                        "original_samples": row[5],
                        "processed_samples": row[6],
                        "output_path": row[7],
                        "processed_at": row[8],
                    }

        except Exception as e:
            logger.error(f"Failed to get preprocessing status: {e}")

        return None

    def list_processed_datasets(
        self, status: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """List all processed datasets.

        Args:
            status: Filter by status (processed, failed, pending)

        Returns:
            List of processing records
        """
        if not self.db_manager:
            return []

        try:
            with self.db_manager.get_connection() as conn:
                if status:
                    cursor = conn.execute(
                        """
                        SELECT id, dataset_name, dataset_url, status, confidence_score,
                               original_samples, processed_samples, output_path, processed_at
                        FROM dataset_preprocessing
                        WHERE status = ?
                        ORDER BY processed_at DESC
                    """,
                        (status,),
                    )
                else:
                    cursor = conn.execute(
                        """
                        SELECT id, dataset_name, dataset_url, status, confidence_score,
                               original_samples, processed_samples, output_path, processed_at
                        FROM dataset_preprocessing
                        ORDER BY processed_at DESC
                    """
                    )

                records = []
                for row in cursor.fetchall():
                    records.append(
                        {
                            "id": row[0],
                            "dataset_name": row[1],
                            "dataset_url": row[2],
                            "status": row[3],
                            "confidence_score": row[4],
                            "original_samples": row[5],
                            "processed_samples": row[6],
                            "output_path": row[7],
                            "processed_at": row[8],
                        }
                    )

                return records

        except Exception as e:
            logger.error(f"Failed to list processed datasets: {e}")
            return []

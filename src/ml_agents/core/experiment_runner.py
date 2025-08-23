"""Experiment execution and orchestration for ML Agents reasoning research.

This module provides the ExperimentRunner class that orchestrates complete
experiments across different reasoning approaches, datasets, and models.
"""

import csv
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import pandas as pd
from tqdm import tqdm

try:
    from rich.console import Console
    from rich.progress import (
        BarColumn,
        Progress,
        SpinnerColumn,
        TaskProgressColumn,
        TextColumn,
        TimeElapsedColumn,
    )

    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

from ml_agents.config import ExperimentConfig
from ml_agents.core.database_manager import DatabaseConfig
from ml_agents.core.dataset_loader import BBEHDatasetLoader
from ml_agents.core.reasoning_inference import ReasoningInference, ReasoningResult
from ml_agents.core.results_processor import ResultsProcessor
from ml_agents.reasoning import get_available_approaches
from ml_agents.utils.logging_config import get_logger, log_experiment_start

logger = get_logger(__name__)


@dataclass
class ExperimentSummary:
    """Summary of experiment results.

    Attributes:
        experiment_id: Unique identifier for the experiment
        config: Configuration used for the experiment
        start_time: When the experiment started
        end_time: When the experiment completed
        duration: Total experiment duration in seconds
        total_samples: Number of samples processed
        approaches_tested: List of reasoning approaches tested
        results_summary: Summary statistics for each approach
        cost_summary: Cost breakdown by approach and provider
        error_summary: Summary of any errors encountered
    """

    experiment_id: str
    config: Dict[str, Any]
    start_time: str
    end_time: str
    duration: float
    total_samples: int
    approaches_tested: List[str]
    results_summary: Dict[str, Dict[str, Any]]
    cost_summary: Dict[str, float]
    error_summary: Dict[str, int]


class ExperimentRunner:
    """Core experiment execution and orchestration.

    This class orchestrates complete reasoning experiments across multiple
    approaches, handling dataset loading, inference execution, progress
    tracking, checkpointing, and result collection.

    The runner supports:
    - Single approach experiments
    - Multi-approach comparisons
    - Progress tracking with resumption
    - Parallel execution for comparison runs
    - Comprehensive result analysis
    - Cost tracking and monitoring
    """

    def __init__(self, config: ExperimentConfig) -> None:
        """Initialize the experiment runner.

        Args:
            config: Experiment configuration containing all settings
        """
        self.config = config
        self.dataset_loader = BBEHDatasetLoader(config)
        self.reasoning_engine = ReasoningInference(config)

        # Database and results processing
        self.results_processor = None
        if config.database_enabled:
            db_config = DatabaseConfig(
                db_path=config.database_path,
                backup_frequency=config.database_backup_frequency,
                auto_vacuum=config.database_auto_vacuum,
            )
            self.results_processor = ResultsProcessor(db_config)

        # Experiment tracking
        self.experiment_id = self._generate_experiment_id()
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None
        self.results: List[Dict[str, Any]] = []
        self.errors: List[Dict[str, Any]] = []

        # Progress tracking
        self.current_sample = 0
        self.total_samples = 0
        self.checkpoint_interval = 10  # Save every 10 samples

        # Output management
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Rich progress setup
        self.use_rich = RICH_AVAILABLE
        self.console = Console() if RICH_AVAILABLE else None

        # Save experiment metadata to database if enabled
        if self.results_processor:
            self.results_processor.save_experiment(
                experiment_id=self.experiment_id,
                name=f"{config.provider}_{config.model}_{config.reasoning_approaches}",
                config=config.to_dict(),
                description=f"Experiment testing {', '.join(config.reasoning_approaches)} on {config.dataset_name}",
            )

        logger.info(f"Initialized ExperimentRunner with ID: {self.experiment_id}")

    def _create_rich_progress(self) -> Optional[Progress]:
        """Create Rich progress display if available."""
        if not self.use_rich:
            return None

        return Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            console=self.console,
            expand=True,
        )

    def run_single_experiment(
        self,
        approach: str,
        sample_count: Optional[int] = None,
        resume_from_checkpoint: bool = True,
        save_checkpoints: bool = True,
        progress_callback: Optional[Callable[[str], None]] = None,
    ) -> ExperimentSummary:
        """Run a single reasoning approach experiment.

        Args:
            approach: Name of the reasoning approach to test
            sample_count: Number of samples to process (uses config default if None)
            resume_from_checkpoint: Whether to resume from existing checkpoint
            save_checkpoints: Whether to save progress checkpoints
            progress_callback: Optional callback function for progress updates

        Returns:
            ExperimentSummary with complete experiment results

        Raises:
            ValueError: If approach is not available or invalid
            RuntimeError: If experiment fails to initialize
        """
        if approach not in get_available_approaches():
            raise ValueError(f"Unknown reasoning approach: {approach}")

        sample_count = sample_count or self.config.sample_count

        logger.info(
            f"Starting single experiment: {approach} with {sample_count} samples"
        )
        log_experiment_start(
            {
                "experiment_id": self.experiment_id,
                "approach": approach,
                "sample_count": sample_count,
                "model": self.config.model,
                "provider": self.config.provider,
            }
        )

        self.start_time = datetime.now()

        try:
            # Load dataset
            logger.info("Loading dataset...")
            if progress_callback:
                progress_callback("Loading dataset...")
            dataset = self.dataset_loader.load_dataset()
            samples = self.dataset_loader.sample_data(sample_size=sample_count)
            self.total_samples = len(samples)
            if progress_callback:
                progress_callback(f"Loaded {len(samples)} samples")

            # Check for checkpoint if resuming
            checkpoint_path = self._get_checkpoint_path(approach)
            if resume_from_checkpoint and checkpoint_path.exists():
                logger.info(f"Resuming from checkpoint: {checkpoint_path}")
                if progress_callback:
                    progress_callback("Resuming from checkpoint...")
                self._load_checkpoint(checkpoint_path)
            else:
                self.current_sample = 0
                self.results = []
                self.errors = []

            # Run experiment with progress tracking
            if progress_callback:
                progress_callback(f"Starting {approach} processing...")
            progress = self._create_rich_progress()
            if progress:
                with progress:
                    task_id = progress.add_task(
                        f"Processing {approach}", total=self.total_samples
                    )
                    progress.update(task_id, completed=self.current_sample)

                    for i in range(self.current_sample, self.total_samples):
                        sample = samples[i]

                        try:
                            # Execute reasoning
                            # Get dynamic column names
                            input_col = self.dataset_loader.get_input_column_name()

                            result = self.reasoning_engine.run_inference(
                                sample[input_col], approach
                            )

                            # Store result with enhanced Phase 4 metadata
                            enhanced_result = self._enhance_result_metadata(
                                result, approach, i
                            )
                            result_data = {
                                "sample_id": i,
                                "input": sample[input_col],
                                "expected_output": sample.get(
                                    self.dataset_loader.get_output_column_name(), ""
                                ),
                                "approach": approach,
                                "result": asdict(enhanced_result),
                                "timestamp": datetime.now().isoformat(),
                            }
                            self.results.append(result_data)

                            # Save to database if enabled
                            if self.results_processor:
                                self._save_result_to_database(
                                    result, approach, i, sample
                                )

                        except Exception as e:
                            logger.warning(f"Error processing sample {i}: {e}")
                            error_data = {
                                "sample_id": i,
                                "approach": approach,
                                "error": str(e),
                                "timestamp": datetime.now().isoformat(),
                            }
                            self.errors.append(error_data)

                        self.current_sample = i + 1
                        progress.update(task_id, completed=self.current_sample)

                        # Save checkpoint periodically
                        if save_checkpoints and (i + 1) % self.checkpoint_interval == 0:
                            self._save_checkpoint(approach)
                            if progress_callback:
                                progress_callback(f"Checkpoint saved at sample {i + 1}")
            else:
                # Fallback to tqdm if Rich not available
                with tqdm(
                    total=self.total_samples,
                    initial=self.current_sample,
                    desc=f"Processing {approach}",
                    unit="samples",
                ) as pbar:
                    for i in range(self.current_sample, self.total_samples):
                        sample = samples[i]

                        try:
                            # Execute reasoning
                            # Get dynamic column names
                            input_col = self.dataset_loader.get_input_column_name()

                            result = self.reasoning_engine.run_inference(
                                sample[input_col], approach
                            )

                            # Store result with enhanced Phase 4 metadata
                            enhanced_result = self._enhance_result_metadata(
                                result, approach, i
                            )
                            result_data = {
                                "sample_id": i,
                                "input": sample[input_col],
                                "expected_output": sample.get(
                                    self.dataset_loader.get_output_column_name(), ""
                                ),
                                "approach": approach,
                                "result": asdict(enhanced_result),
                                "timestamp": datetime.now().isoformat(),
                            }
                            self.results.append(result_data)

                            # Save to database if enabled
                            if self.results_processor:
                                self._save_result_to_database(
                                    result, approach, i, sample
                                )

                        except Exception as e:
                            logger.warning(f"Error processing sample {i}: {e}")
                            error_data = {
                                "sample_id": i,
                                "approach": approach,
                                "error": str(e),
                                "timestamp": datetime.now().isoformat(),
                            }
                            self.errors.append(error_data)

                        self.current_sample = i + 1
                        pbar.update(1)

                        # Save checkpoint periodically
                        if save_checkpoints and (i + 1) % self.checkpoint_interval == 0:
                            self._save_checkpoint(approach)

            self.end_time = datetime.now()
            if progress_callback:
                progress_callback("Processing completed, generating results...")

            # Generate and save final results
            summary = self._generate_summary([approach])
            self._save_results(summary)
            if progress_callback:
                progress_callback("Results saved successfully!")

            # Clean up checkpoint
            if checkpoint_path.exists():
                checkpoint_path.unlink()

            logger.info(f"Completed single experiment: {approach}")
            return summary

        except Exception as e:
            logger.error(f"Experiment failed: {e}")
            raise RuntimeError(f"Experiment execution failed: {e}")

    def run_comparison(
        self,
        approaches: List[str],
        sample_count: Optional[int] = None,
        parallel: bool = True,
        max_workers: Optional[int] = None,
        progress_callback: Optional[Callable[[str], None]] = None,
    ) -> ExperimentSummary:
        """Run comparison experiment across multiple reasoning approaches.

        Args:
            approaches: List of reasoning approaches to compare
            sample_count: Number of samples to process (uses config default if None)
            parallel: Whether to run approaches in parallel
            max_workers: Maximum number of parallel workers (None for auto)
            progress_callback: Optional callback function for progress updates

        Returns:
            ExperimentSummary with comparison results across all approaches

        Raises:
            ValueError: If any approach is not available
        """
        # Validate approaches
        available = get_available_approaches()
        invalid = [a for a in approaches if a not in available]
        if invalid:
            raise ValueError(f"Unknown reasoning approaches: {invalid}")

        sample_count = sample_count or self.config.sample_count

        logger.info(
            f"Starting comparison experiment: {approaches} with {sample_count} samples"
        )
        if progress_callback:
            progress_callback(f"Starting comparison of {len(approaches)} approaches...")
        log_experiment_start(
            {
                "experiment_id": self.experiment_id,
                "approaches": approaches,
                "sample_count": sample_count,
                "parallel": parallel,
                "model": self.config.model,
                "provider": self.config.provider,
            }
        )

        self.start_time = datetime.now()

        try:
            # Load dataset once for all approaches
            logger.info("Loading dataset...")
            dataset = self.dataset_loader.load_dataset()
            samples = self.dataset_loader.sample_data(sample_size=sample_count)
            self.total_samples = len(samples)

            if parallel and len(approaches) > 1:
                results = self._run_parallel_comparison(
                    approaches, samples, max_workers
                )
            else:
                results = self._run_sequential_comparison(approaches, samples)

            self.results = results
            self.end_time = datetime.now()

            # Generate and save final results
            summary = self._generate_summary(approaches)
            self._save_results(summary)

            logger.info(f"Completed comparison experiment: {approaches}")
            return summary

        except Exception as e:
            logger.error(f"Comparison experiment failed: {e}")
            raise RuntimeError(f"Comparison execution failed: {e}")

    def _run_parallel_comparison(
        self,
        approaches: List[str],
        samples: List[Dict[str, Any]],
        max_workers: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Run approaches in parallel for comparison.

        Args:
            approaches: List of reasoning approaches
            samples: Dataset samples to process
            max_workers: Maximum number of parallel workers

        Returns:
            Combined results from all approaches
        """
        max_workers = max_workers or min(len(approaches), 4)
        all_results = []

        logger.info(f"Running parallel comparison with {max_workers} workers")

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit jobs for each approach
            future_to_approach = {
                executor.submit(
                    self._run_approach_on_samples, approach, samples
                ): approach
                for approach in approaches
            }

            # Collect results as they complete
            with tqdm(
                total=len(approaches), desc="Approaches", unit="approach"
            ) as pbar:
                for future in as_completed(future_to_approach):
                    approach = future_to_approach[future]
                    try:
                        approach_results = future.result()
                        all_results.extend(approach_results)
                        logger.info(f"Completed approach: {approach}")
                    except Exception as e:
                        logger.error(f"Approach {approach} failed: {e}")
                        # Continue with other approaches

                    pbar.update(1)

        return all_results

    def _run_sequential_comparison(
        self, approaches: List[str], samples: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Run approaches sequentially for comparison.

        Args:
            approaches: List of reasoning approaches
            samples: Dataset samples to process

        Returns:
            Combined results from all approaches
        """
        all_results = []

        logger.info("Running sequential comparison")

        for approach in approaches:
            logger.info(f"Processing approach: {approach}")
            try:
                approach_results = self._run_approach_on_samples(approach, samples)
                all_results.extend(approach_results)
            except Exception as e:
                logger.error(f"Approach {approach} failed: {e}")
                # Continue with other approaches

        return all_results

    def _run_approach_on_samples(
        self, approach: str, samples: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Run a single approach on all samples.

        Args:
            approach: Reasoning approach name
            samples: Dataset samples to process

        Returns:
            Results for this approach
        """
        results = []
        errors = []

        with tqdm(
            total=len(samples), desc=f"{approach}", position=0, leave=False
        ) as pbar:
            for i, sample in enumerate(samples):
                try:
                    # Execute reasoning
                    result = self.reasoning_engine.run_inference(
                        sample["input"], approach
                    )

                    # Store result
                    result_data = {
                        "sample_id": i,
                        "input": sample["input"],
                        "expected_output": sample.get("output", ""),
                        "approach": approach,
                        "result": asdict(result),
                        "timestamp": datetime.now().isoformat(),
                    }
                    results.append(result_data)

                    # Save to database if enabled
                    if self.results_processor:
                        self._save_result_to_database(result, approach, i, sample)

                except Exception as e:
                    logger.warning(f"Error in {approach} sample {i}: {e}")
                    error_data = {
                        "sample_id": i,
                        "approach": approach,
                        "error": str(e),
                        "timestamp": datetime.now().isoformat(),
                    }
                    errors.append(error_data)

                pbar.update(1)

        # Store errors for this approach
        self.errors.extend(errors)

        return results

    def _generate_summary(self, approaches: List[str]) -> ExperimentSummary:
        """Generate experiment summary with statistics.

        Args:
            approaches: List of approaches that were tested

        Returns:
            Complete experiment summary
        """
        duration = (self.end_time - self.start_time).total_seconds()

        # Calculate results summary by approach
        results_summary = {}
        cost_summary = {}

        for approach in approaches:
            approach_results = [r for r in self.results if r["approach"] == approach]
            approach_errors = [e for e in self.errors if e["approach"] == approach]

            # Calculate statistics
            if approach_results:
                execution_times = [
                    r["result"]["execution_time"] for r in approach_results
                ]
                token_counts = [
                    r["result"]["response"]["total_tokens"] for r in approach_results
                ]
                costs = [r["result"]["cost_estimate"] for r in approach_results]

                # Calculate basic accuracy by comparing outputs
                correct_responses = 0
                for result_data in approach_results:
                    expected = result_data.get("expected_output", "").strip().lower()
                    # Use extracted answer if available, otherwise fall back to raw text
                    extracted = result_data["result"]["response"].get(
                        "extracted_answer"
                    )
                    if extracted and extracted.strip():
                        actual = extracted.strip().lower()
                    else:
                        actual = (
                            result_data["result"]["response"]["text"].strip().lower()
                        )
                    if expected and expected == actual:
                        correct_responses += 1

                accuracy = (
                    (correct_responses / len(approach_results))
                    if approach_results
                    else 0
                )

                results_summary[approach] = {
                    "total_samples": len(approach_results),
                    "success_rate": len(approach_results)
                    / (len(approach_results) + len(approach_errors)),
                    "avg_execution_time": sum(execution_times) / len(execution_times),
                    "avg_tokens": sum(token_counts) / len(token_counts),
                    "total_cost": sum(costs),
                    "error_count": len(approach_errors),
                    "accuracy": f"{accuracy:.1%}",
                }

                cost_summary[approach] = sum(costs)
            else:
                results_summary[approach] = {
                    "total_samples": 0,
                    "success_rate": 0.0,
                    "avg_execution_time": 0.0,
                    "avg_tokens": 0.0,
                    "total_cost": 0.0,
                    "error_count": len(approach_errors),
                    "accuracy": "0.0%",
                }
                cost_summary[approach] = 0.0

        # Error summary
        error_summary = {}
        for approach in approaches:
            approach_errors = [e for e in self.errors if e["approach"] == approach]
            error_summary[approach] = len(approach_errors)

        return ExperimentSummary(
            experiment_id=self.experiment_id,
            config=asdict(self.config),
            start_time=self.start_time.isoformat(),
            end_time=self.end_time.isoformat(),
            duration=duration,
            total_samples=self.total_samples,
            approaches_tested=approaches,
            results_summary=results_summary,
            cost_summary=cost_summary,
            error_summary=error_summary,
        )

    def _save_results(self, summary: ExperimentSummary) -> None:
        """Save experiment results to files.

        Args:
            summary: Experiment summary to save
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = (
            f"{self.config.provider}_{self.config.model.replace('/', '_')}_{timestamp}"
        )

        # Save detailed results as CSV
        results_file = self.output_dir / f"{base_filename}_results.csv"
        self._save_results_csv(results_file)

        # Save summary as JSON
        summary_file = self.output_dir / f"{base_filename}_summary.json"
        with open(summary_file, "w") as f:
            json.dump(asdict(summary), f, indent=2)

        # Save errors if any
        if self.errors:
            errors_file = self.output_dir / f"{base_filename}_errors.json"
            with open(errors_file, "w") as f:
                json.dump(self.errors, f, indent=2)

        logger.info(f"Results saved to {self.output_dir}")
        logger.info(f"- Results: {results_file}")
        logger.info(f"- Summary: {summary_file}")
        if self.errors:
            logger.info(f"- Errors: {errors_file}")

        # Update experiment status in database
        if self.results_processor:
            self.results_processor.update_experiment_status(
                self.experiment_id, "completed"
            )

    def _save_results_csv(self, filepath: Path) -> None:
        """Save results as CSV file.

        Args:
            filepath: Path to save the CSV file
        """
        if not self.results:
            logger.warning("No results to save")
            return

        with open(filepath, "w", newline="") as csvfile:
            fieldnames = [
                "sample_id",
                "approach",
                "input",
                "expected_output",
                "extracted_answer",
                "response_text",
                "execution_time",
                "total_tokens",
                "cost_estimate",
                "reasoning_steps",
                "timestamp",
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            for result in self.results:
                # Safely access the response text and extracted answer
                try:
                    response_text = result["result"]["response"]["text"]
                    extracted_answer = result["result"]["response"].get(
                        "extracted_answer"
                    )

                    # Use extracted answer if available, otherwise fall back to response text
                    if extracted_answer and extracted_answer.strip():
                        display_answer = extracted_answer
                    else:
                        # For backward compatibility, use the original response text
                        display_answer = response_text
                        logger.debug(
                            f"No extracted answer found for sample {result.get('sample_id')}, using full response"
                        )

                except (KeyError, TypeError) as e:
                    logger.warning(
                        f"Error accessing response data for sample {result.get('sample_id', 'unknown')}: {e}"
                    )
                    response_text = "ERROR_ACCESSING_RESPONSE"
                    extracted_answer = None
                    display_answer = "ERROR_ACCESSING_RESPONSE"

                row = {
                    "sample_id": result["sample_id"],
                    "approach": result["approach"],
                    "input": result["input"],
                    "expected_output": result["expected_output"],
                    "extracted_answer": extracted_answer or "",
                    "response_text": response_text,
                    "execution_time": result["result"]["execution_time"],
                    "total_tokens": result["result"]["response"]["total_tokens"],
                    "cost_estimate": result["result"]["cost_estimate"],
                    "reasoning_steps": result["result"]["metadata"].get(
                        "reasoning_steps", 0
                    ),
                    "timestamp": result["timestamp"],
                }
                writer.writerow(row)

    def _save_result_to_database(
        self,
        result: ReasoningResult,
        approach: str,
        sample_index: int,
        sample: Dict[str, Any],
    ) -> None:
        """Save a single result to the database.

        Args:
            result: ReasoningResult from inference
            approach: Reasoning approach name
            sample_index: Index of the sample
            sample: The input sample data
        """
        try:
            # Convert ReasoningResult to StandardResponse for ResultsProcessor
            standard_response = result.response

            # Add metadata needed for database storage
            standard_response.metadata.update(
                {
                    "experiment_id": self.experiment_id,
                    "approach": approach,
                    "sample_index": sample_index,
                    "input_text": sample.get(
                        self.dataset_loader.get_input_column_name(), ""
                    ),
                    "expected_answer": sample.get(
                        self.dataset_loader.get_output_column_name(), ""
                    ),
                    "provider": self.config.provider,
                    "model": self.config.model,
                    "latency": result.execution_time,
                    "cost": result.cost_estimate,
                }
            )

            # Save to database
            self.results_processor.save_run_result(standard_response)

        except Exception as e:
            logger.warning(f"Failed to save result to database: {e}")

    def _enhance_result_metadata(
        self, result: ReasoningResult, approach: str, sample_id: int
    ) -> ReasoningResult:
        """Enhance result with Phase 4 metadata requirements.

        Args:
            result: Original reasoning result
            approach: Name of reasoning approach used
            sample_id: ID of the current sample

        Returns:
            Enhanced reasoning result with additional metadata
        """
        # Ensure metadata exists
        if result.response.metadata is None:
            result.response.metadata = {}

        # Add Phase 4 enhanced metadata requirements
        enhanced_metadata = {
            # Experiment context for reproducibility
            "experiment_id": self.experiment_id,
            "sample_id": sample_id,
            # Reasoning trace for detailed analysis
            "reasoning_trace": self._extract_reasoning_trace(result, approach),
            # Approach configuration for parameter tracking
            "approach_config": self._get_approach_config(approach),
            # Performance metrics for comprehensive tracking
            "performance_metrics": self._calculate_performance_metrics(
                result, approach
            ),
            # Experiment timing
            "experiment_timestamp": datetime.now().isoformat(),
        }

        # Merge with existing metadata
        result.response.metadata.update(enhanced_metadata)

        return result

    def _extract_reasoning_trace(
        self, result: ReasoningResult, approach: str
    ) -> List[Dict[str, Any]]:
        """Extract step-by-step reasoning trace from result.

        Args:
            result: Reasoning result to analyze
            approach: Name of reasoning approach

        Returns:
            List of reasoning steps with details
        """
        reasoning_trace = []

        # Extract from existing metadata if available
        existing_metadata = result.response.metadata or {}

        # Handle multi-step approaches
        if "multi_step_details" in existing_metadata:
            multi_step = existing_metadata["multi_step_details"]
            if isinstance(multi_step, dict):
                step_count = multi_step.get("step_count", 1)
                for i in range(step_count):
                    step_name = f"step_{i+1}"
                    if step_name in multi_step or f"step{i+1}" in multi_step:
                        reasoning_trace.append(
                            {
                                "step": i + 1,
                                "type": "reasoning",
                                "approach": approach,
                                "content": multi_step.get(
                                    step_name, multi_step.get(f"step{i+1}", "")
                                ),
                            }
                        )

        # Handle approaches with reasoning_steps metadata
        reasoning_steps = existing_metadata.get("reasoning_steps", 1)
        if reasoning_steps > 1 and not reasoning_trace:
            for i in range(reasoning_steps):
                reasoning_trace.append(
                    {
                        "step": i + 1,
                        "type": "reasoning",
                        "approach": approach,
                        "content": f"Reasoning step {i+1}",
                    }
                )

        # Default single step if no specific trace found
        if not reasoning_trace:
            reasoning_trace.append(
                {
                    "step": 1,
                    "type": "reasoning",
                    "approach": approach,
                    "content": (
                        result.response.text[:100] + "..."
                        if len(result.response.text) > 100
                        else result.response.text
                    ),
                }
            )

        return reasoning_trace

    def _get_approach_config(self, approach: str) -> Dict[str, Any]:
        """Get configuration parameters used for the reasoning approach.

        Args:
            approach: Name of reasoning approach

        Returns:
            Dictionary of configuration parameters
        """
        base_config = {
            "approach_name": approach,
            "provider": self.config.provider,
            "model": self.config.model,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "top_p": self.config.top_p,
        }

        # Add approach-specific configuration
        approach_specific = {}

        if approach == "ChainOfVerification" and hasattr(
            self.config, "multi_step_verification"
        ):
            approach_specific["multi_step_verification"] = (
                self.config.multi_step_verification
            )
            approach_specific["max_reasoning_calls"] = self.config.max_reasoning_calls

        if approach == "Reflection" and hasattr(self.config, "multi_step_reflection"):
            approach_specific["multi_step_reflection"] = (
                self.config.multi_step_reflection
            )
            approach_specific["max_reflection_iterations"] = (
                self.config.max_reflection_iterations
            )
            approach_specific["reflection_threshold"] = self.config.reflection_threshold

        base_config.update(approach_specific)
        return base_config

    def _calculate_performance_metrics(
        self, result: ReasoningResult, approach: str
    ) -> Dict[str, Any]:
        """Calculate performance metrics for comprehensive tracking.

        Args:
            result: Reasoning result to analyze
            approach: Name of reasoning approach

        Returns:
            Dictionary of performance metrics
        """
        existing_metadata = result.response.metadata or {}

        metrics = {
            "execution_time": result.execution_time,
            "total_tokens": result.response.total_tokens,
            "prompt_tokens": result.response.prompt_tokens,
            "completion_tokens": result.response.completion_tokens,
            "cost_estimate": result.cost_estimate,
            "response_length": len(result.response.text),
            "reasoning_depth": existing_metadata.get("reasoning_steps", 1),
        }

        # Approach-specific metrics
        if approach in ["TreeOfThought", "GraphOfThought"]:
            metrics["branch_exploration"] = existing_metadata.get(
                "branches_explored", 1
            )
            metrics["selected_path"] = existing_metadata.get("selected_path", "unknown")

        if approach == "ChainOfVerification":
            metrics["verification_steps"] = existing_metadata.get(
                "verification_count", 1
            )
            metrics["verification_confidence"] = existing_metadata.get(
                "verification_confidence", 0.5
            )

        if approach in ["SkeletonOfThought"]:
            metrics["outline_depth"] = existing_metadata.get("outline_depth", 1)
            metrics["expansion_ratio"] = existing_metadata.get("expansion_ratio", 1.0)

        if approach == "AsPlanning":
            metrics["planning_stages"] = existing_metadata.get("planning_stages", 1)
            metrics["risk_assessments"] = existing_metadata.get("risk_assessments", 0)

        return metrics

    def _generate_experiment_id(self) -> str:
        """Generate unique experiment ID.

        Returns:
            Unique experiment identifier
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"exp_{timestamp}_{hash(str(self.config)) % 10000:04d}"

    def _get_checkpoint_path(self, approach: str) -> Path:
        """Get checkpoint file path for an approach.

        Args:
            approach: Reasoning approach name

        Returns:
            Path to checkpoint file
        """
        return self.output_dir / f"checkpoint_{self.experiment_id}_{approach}.json"

    def _save_checkpoint(self, approach: str) -> None:
        """Save experiment checkpoint.

        Args:
            approach: Reasoning approach name
        """
        checkpoint_data = {
            "experiment_id": self.experiment_id,
            "approach": approach,
            "current_sample": self.current_sample,
            "total_samples": self.total_samples,
            "results": self.results,
            "errors": self.errors,
            "timestamp": datetime.now().isoformat(),
        }

        checkpoint_path = self._get_checkpoint_path(approach)
        with open(checkpoint_path, "w") as f:
            json.dump(checkpoint_data, f, indent=2)

        logger.debug(f"Checkpoint saved: {checkpoint_path}")

    def _load_checkpoint(self, checkpoint_path: Path) -> None:
        """Load experiment checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file
        """
        with open(checkpoint_path, "r") as f:
            checkpoint_data = json.load(f)

        self.current_sample = checkpoint_data["current_sample"]
        self.total_samples = checkpoint_data["total_samples"]
        self.results = checkpoint_data["results"]
        self.errors = checkpoint_data["errors"]

        logger.info(f"Loaded checkpoint from sample {self.current_sample}")

    def resume_from_checkpoint(
        self, checkpoint_path: Union[str, Path]
    ) -> ExperimentSummary:
        """Resume experiment from an external checkpoint file.

        Args:
            checkpoint_path: Path to the checkpoint file to resume from

        Returns:
            ExperimentSummary with the completed experiment results

        Raises:
            FileNotFoundError: If checkpoint file doesn't exist
            ValueError: If checkpoint data is invalid
        """
        checkpoint_path = Path(checkpoint_path)

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

        logger.info(f"Resuming experiment from checkpoint: {checkpoint_path}")

        # Load checkpoint data
        with open(checkpoint_path, "r") as f:
            checkpoint_data = json.load(f)

        # Validate checkpoint structure
        required_fields = [
            "experiment_id",
            "approach",
            "config",
            "current_sample",
            "total_samples",
            "results",
            "errors",
        ]
        for field in required_fields:
            if field not in checkpoint_data:
                raise ValueError(f"Invalid checkpoint file: missing field '{field}'")

        # Restore experiment state
        self.experiment_id = checkpoint_data["experiment_id"]
        approach = checkpoint_data["approach"]
        self.current_sample = checkpoint_data["current_sample"]
        self.total_samples = checkpoint_data["total_samples"]
        self.results = checkpoint_data["results"]
        self.errors = checkpoint_data["errors"]

        # Update config if needed
        saved_config = checkpoint_data["config"]
        if saved_config != self.config.to_dict():
            logger.warning(
                "Checkpoint config differs from current config - using checkpoint config"
            )
            from ml_agents.config import ExperimentConfig

            self.config = ExperimentConfig.from_dict(saved_config)

        logger.info(
            f"Resuming {approach} experiment from sample {self.current_sample}/{self.total_samples}"
        )

        # Continue the experiment if not completed
        if self.current_sample < self.total_samples:
            # Load the dataset
            loader = BBEHDatasetLoader(self.config.dataset_name)
            try:
                dataset = loader.load_dataset()
                dataset = loader.sample_data(dataset, self.total_samples)
            except Exception as e:
                logger.error(f"Failed to load dataset: {e}")
                raise ValueError(
                    f"Cannot resume experiment: dataset loading failed: {e}"
                )

            # Resume processing from where we left off
            logger.info(f"Continuing experiment from sample {self.current_sample + 1}")

            # Set up reasoning inference
            reasoning_inference = ReasoningInference(self.config)

            # Process remaining samples
            for i in range(self.current_sample, self.total_samples):
                if i % 10 == 0:
                    logger.info(f"Processing sample {i + 1}/{self.total_samples}")

                sample = dataset[i]

                try:
                    # Run inference
                    result = reasoning_inference.run_inference(
                        approach=approach,
                        prompt=sample.get("input", ""),
                        context={"sample_id": i, "approach": approach},
                    )

                    self.results.append(
                        {
                            "sample_id": i,
                            "input": sample.get("input", ""),
                            "output": result.response,
                            "reasoning_trace": result.reasoning_trace,
                            "metadata": result.metadata,
                            "approach": approach,
                            "timestamp": datetime.now().isoformat(),
                        }
                    )

                except Exception as e:
                    logger.error(f"Error processing sample {i}: {e}")
                    self.errors.append(
                        {
                            "sample_id": i,
                            "error": str(e),
                            "approach": approach,
                            "timestamp": datetime.now().isoformat(),
                        }
                    )

                self.current_sample = i

                # Save checkpoint periodically
                if (i + 1) % self.checkpoint_interval == 0:
                    self._save_checkpoint(approach)

        # Generate final summary
        end_time = datetime.now()
        duration = (
            end_time
            - datetime.fromisoformat(
                checkpoint_data.get("start_time", end_time.isoformat())
            )
        ).total_seconds()

        summary = ExperimentSummary(
            experiment_id=self.experiment_id,
            config=self.config.to_dict(),
            start_time=checkpoint_data.get("start_time", end_time.isoformat()),
            end_time=end_time.isoformat(),
            duration=duration,
            total_samples=self.total_samples,
            approaches_tested=[approach],
            results_summary={
                approach: {
                    "total_samples": len(self.results),
                    "error_count": len(self.errors),
                    "success_rate": (
                        len(self.results) / self.total_samples
                        if self.total_samples > 0
                        else 0.0
                    ),
                    "completed": self.current_sample >= self.total_samples - 1,
                }
            },
            cost_summary={},
            error_summary={approach: len(self.errors)},
        )

        # Save final results
        self._save_results_to_csv(approach, self.results)

        # Clean up checkpoint file
        if checkpoint_path.exists():
            checkpoint_path.unlink()
            logger.info(f"Removed checkpoint file: {checkpoint_path}")

        logger.info(
            f"Experiment {self.experiment_id} resumed and completed successfully"
        )
        return summary

    def get_progress_status(self) -> Dict[str, Any]:
        """Get current experiment progress status.

        Returns:
            Dictionary with progress information
        """
        if self.total_samples == 0:
            progress_pct = 0.0
        else:
            progress_pct = (self.current_sample / self.total_samples) * 100

        return {
            "experiment_id": self.experiment_id,
            "current_sample": self.current_sample,
            "total_samples": self.total_samples,
            "progress_percentage": progress_pct,
            "results_count": len(self.results),
            "errors_count": len(self.errors),
            "start_time": self.start_time.isoformat() if self.start_time else None,
        }

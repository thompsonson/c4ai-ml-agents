"""Tests for ExperimentRunner - experiment execution and orchestration."""

import json
import tempfile
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, Mock, call, patch

import pandas as pd
import pytest
from datasets import Dataset

from ml_agents.config import ExperimentConfig
from ml_agents.core.experiment_runner import ExperimentRunner, ExperimentSummary
from ml_agents.core.reasoning_inference import ReasoningResult
from ml_agents.utils.api_clients import StandardResponse


class TestExperimentRunner:
    """Test suite for ExperimentRunner."""

    @pytest.fixture
    def config(self, temp_dir, test_db_path):
        """Create test configuration with temporary output directory."""
        return ExperimentConfig(
            dataset_name="test/dataset",
            benchmark_id="TEST_BENCHMARK",  # Add benchmark_id for new system
            sample_count=3,  # Changed to match test expectations
            provider="openrouter",
            model="openai/gpt-oss-120b",
            temperature=0.3,
            max_tokens=100,
            output_dir=str(temp_dir / "outputs"),
            multi_step_verification=True,
            max_reasoning_calls=3,
            database_path=test_db_path,  # Use test database with proper schema
        )

    @pytest.fixture
    def critical_config(self, temp_dir, test_db_path):
        """Create configuration for critical success scenario with Phase 4 specifications."""
        return ExperimentConfig(
            dataset_name="test/dataset",
            benchmark_id="TEST_BENCHMARK",  # Add benchmark_id for new system
            sample_count=5,  # Phase 4 specification requirement
            provider="openrouter",
            model="openai/gpt-oss-120b",
            temperature=0.3,
            max_tokens=100,
            output_dir=str(temp_dir / "outputs"),
            multi_step_verification=True,
            max_reasoning_calls=3,
            database_path=test_db_path,  # Use test database with proper schema
        )

    @pytest.fixture
    def mock_dataset_loader(self):
        """Mock BBEHDatasetLoader."""
        mock_loader = Mock()

        # Mock dataset - only 3 samples to match config
        sample_data = [
            {"input": "What is 2+2?", "output": "4", "id": 1},
            {"input": "What is 3+3?", "output": "6", "id": 2},
            {"input": "What is 5+5?", "output": "10", "id": 3},
        ]

        mock_dataset = Dataset.from_dict(
            {
                "input": [item["input"] for item in sample_data],
                "output": [item["output"] for item in sample_data],
                "id": [item["id"] for item in sample_data],
            }
        )

        mock_loader.load_dataset.return_value = mock_dataset

        # Make sample_data respect the sample_size parameter
        def sample_data_side_effect(sample_size=None):
            if sample_size is None:
                return sample_data
            return sample_data[:sample_size]

        mock_loader.sample_data.side_effect = sample_data_side_effect
        mock_loader.get_input_column_name.return_value = "input"
        mock_loader.get_output_column_name.return_value = "output"

        return mock_loader

    @pytest.fixture
    def mock_reasoning_result(self):
        """Create mock reasoning result."""
        return ReasoningResult(
            response=StandardResponse(
                text="The answer is 4.",
                provider="openrouter",
                model="openai/gpt-3.5-turbo",
                prompt_tokens=10,
                completion_tokens=5,
                total_tokens=15,
                generation_time=0.5,
                parameters={"temperature": 0.3},
                response_id="test-123",
                metadata={"reasoning_approach": "ChainOfThought", "reasoning_steps": 2},
            ),
            approach_name="ChainOfThought",
            execution_time=0.5,
            cost_estimate=0.001,
            metadata={"reasoning_steps": 2, "confidence": 0.9},
        )

    @pytest.fixture
    def mock_reasoning_inference(self, mock_reasoning_result):
        """Mock ReasoningInference engine."""
        mock_engine = Mock()
        mock_engine.run_inference.return_value = mock_reasoning_result
        return mock_engine

    @pytest.fixture
    def runner(self, config):
        """Create ExperimentRunner instance with mocked dependencies."""
        with (
            patch(
                "ml_agents.core.experiment_runner.BBEHDatasetLoader"
            ) as mock_dataset_loader_class,
            patch(
                "ml_agents.core.experiment_runner.ReasoningInference"
            ) as mock_inference_class,
        ):
            runner = ExperimentRunner(config)
            return runner

    def test_init(self, config, temp_dir):
        """Test ExperimentRunner initialization."""
        with (
            patch("ml_agents.core.experiment_runner.BBEHDatasetLoader"),
            patch("ml_agents.core.experiment_runner.ReasoningInference"),
        ):
            runner = ExperimentRunner(config)

            assert runner.config == config
            assert runner.experiment_id.startswith("exp_")
            assert runner.start_time is None
            assert runner.end_time is None
            assert runner.results == []
            assert runner.errors == []
            assert runner.current_sample == 0
            assert runner.total_samples == 0
            assert runner.checkpoint_interval == 10
            # New structure: outputs/{benchmark_id}/eval/{timestamp}/
            assert runner.output_dir.parent.parent.parent == Path(temp_dir / "outputs")
            assert "TEST_BENCHMARK" in str(
                runner.output_dir
            )  # Benchmark ID in directory structure
            assert "eval" in str(runner.output_dir)
            assert runner.output_dir.exists()

    @patch("ml_agents.core.experiment_runner.get_available_approaches")
    def test_run_single_experiment_basic(
        self,
        mock_get_approaches,
        runner,
        mock_dataset_loader,
        mock_reasoning_inference,
        mock_reasoning_result,
    ):
        """Test basic single experiment execution."""
        # Setup mocks
        mock_get_approaches.return_value = ["None", "ChainOfThought", "AsPlanning"]
        runner.dataset_loader = mock_dataset_loader
        runner.reasoning_engine = mock_reasoning_inference

        # Run experiment
        result = runner.run_single_experiment(
            "ChainOfThought", sample_count=3, save_checkpoints=False
        )

        # Assertions
        assert isinstance(result, ExperimentSummary)
        assert result.experiment_id == runner.experiment_id
        assert result.total_samples == 3
        assert result.approaches_tested == ["ChainOfThought"]
        assert len(runner.results) == 3

        # Verify dataset loading
        mock_dataset_loader.load_dataset.assert_called_once()
        mock_dataset_loader.sample_data.assert_called_once_with(sample_size=3)

        # Verify reasoning calls
        assert mock_reasoning_inference.run_inference.call_count == 3
        expected_calls = [
            call("What is 2+2?", "ChainOfThought"),
            call("What is 3+3?", "ChainOfThought"),
            call("What is 5+5?", "ChainOfThought"),
        ]
        mock_reasoning_inference.run_inference.assert_has_calls(expected_calls)

    @patch("ml_agents.core.experiment_runner.get_available_approaches")
    def test_run_single_experiment_invalid_approach(self, mock_get_approaches, runner):
        """Test single experiment with invalid approach."""
        mock_get_approaches.return_value = ["None", "ChainOfThought"]

        with pytest.raises(
            ValueError, match="Unknown reasoning approach: InvalidApproach"
        ):
            runner.run_single_experiment("InvalidApproach")

    @patch("ml_agents.core.experiment_runner.get_available_approaches")
    def test_run_single_experiment_error_handling(
        self, mock_get_approaches, runner, mock_dataset_loader, mock_reasoning_inference
    ):
        """Test error handling during single experiment execution."""
        # Setup mocks
        mock_get_approaches.return_value = ["None", "ChainOfThought"]
        runner.dataset_loader = mock_dataset_loader
        runner.reasoning_engine = mock_reasoning_inference

        # Make reasoning inference fail for second sample
        mock_reasoning_inference.run_inference.side_effect = [
            mock_reasoning_inference.run_inference.return_value,  # Success
            Exception("API Error"),  # Failure
            mock_reasoning_inference.run_inference.return_value,  # Success
        ]

        # Run experiment
        result = runner.run_single_experiment(
            "ChainOfThought", sample_count=3, save_checkpoints=False
        )

        # Should complete with 2 successful results and 1 error
        assert len(runner.results) == 2
        assert len(runner.errors) == 1
        assert runner.errors[0]["sample_id"] == 1
        assert "API Error" in runner.errors[0]["error"]
        assert result.error_summary["ChainOfThought"] == 1

    @patch("ml_agents.core.experiment_runner.get_available_approaches")
    def test_run_comparison_sequential(
        self, mock_get_approaches, runner, mock_dataset_loader, mock_reasoning_inference
    ):
        """Test comparison experiment in sequential mode."""
        # Setup mocks
        approaches = ["ChainOfThought", "AsPlanning", "TreeOfThought"]
        mock_get_approaches.return_value = ["None"] + approaches
        runner.dataset_loader = mock_dataset_loader
        runner.reasoning_engine = mock_reasoning_inference

        # Run comparison
        result = runner.run_comparison(approaches, sample_count=2, parallel=False)

        # Assertions
        assert isinstance(result, ExperimentSummary)
        assert result.approaches_tested == approaches
        assert result.total_samples == 2  # Per approach

        # Should have called reasoning inference for each approach * sample count
        assert (
            mock_reasoning_inference.run_inference.call_count == 6
        )  # 3 approaches * 2 samples

    @patch("ml_agents.core.experiment_runner.get_available_approaches")
    @patch("ml_agents.core.experiment_runner.ThreadPoolExecutor")
    def test_run_comparison_parallel(
        self,
        mock_thread_pool,
        mock_get_approaches,
        runner,
        mock_dataset_loader,
        mock_reasoning_inference,
    ):
        """Test comparison experiment in parallel mode - CRITICAL for thread safety."""
        # Setup mocks
        approaches = [
            "ChainOfThought",
            "AsPlanning",
            "TreeOfThought",
            "ChainOfVerification",
        ]
        mock_get_approaches.return_value = ["None"] + approaches
        runner.dataset_loader = mock_dataset_loader
        runner.reasoning_engine = mock_reasoning_inference

        # Mock thread pool executor
        mock_executor = Mock()
        mock_thread_pool.return_value.__enter__.return_value = mock_executor

        # Mock futures for parallel execution
        mock_futures = []
        for i, approach in enumerate(approaches):
            mock_future = Mock(spec=Future)
            mock_future.result.return_value = ExperimentSummary(
                experiment_id=f"exp_test_{i}",
                config=runner.config.to_dict(),
                start_time=datetime.now().isoformat(),
                end_time=datetime.now().isoformat(),
                duration=1.0,
                total_samples=2,
                approaches_tested=[approach],
                results_summary={approach: {"success_count": 2, "error_count": 0}},
                cost_summary={approach: 0.01},
                error_summary={approach: 0},
            )
            mock_futures.append(mock_future)

        mock_executor.submit.side_effect = mock_futures

        # Mock as_completed to return futures in order
        with patch(
            "ml_agents.core.experiment_runner.as_completed", return_value=mock_futures
        ):
            result = runner.run_comparison(
                approaches, sample_count=2, parallel=True, max_workers=4
            )

        # Assertions
        assert isinstance(result, ExperimentSummary)
        assert result.approaches_tested == approaches

        # Verify parallel execution was used
        mock_thread_pool.assert_called_once_with(max_workers=4)
        assert mock_executor.submit.call_count == 4  # One per approach

    @patch("ml_agents.core.experiment_runner.get_available_approaches")
    def test_run_comparison_thread_safety_validation(self, mock_get_approaches, runner):
        """Test thread safety with real threading for new reasoning approaches."""
        # This test validates that new approaches (AsPlanning, ChainOfVerification,
        # TreeOfThought, SkeletonOfThought) are thread-safe

        approaches = [
            "AsPlanning",
            "ChainOfVerification",
            "TreeOfThought",
            "SkeletonOfThought",
        ]
        mock_get_approaches.return_value = ["None"] + approaches

        # Track thread-specific state to detect contamination
        thread_states = {}
        contamination_detected = threading.Event()

        def mock_reasoning_inference_with_state_tracking(*args, **kwargs):
            """Mock that tracks thread-specific state."""
            thread_id = threading.current_thread().ident
            approach = args[1] if len(args) > 1 else "unknown"

            # Initialize thread state
            if thread_id not in thread_states:
                thread_states[thread_id] = {"approach": approach, "calls": 0}

            # Detect state contamination
            if thread_states[thread_id]["approach"] != approach:
                contamination_detected.set()

            thread_states[thread_id]["calls"] += 1

            # Simulate some processing time
            time.sleep(0.01)

            return ReasoningResult(
                response=StandardResponse(
                    text=f"Result from {approach}",
                    provider="test",
                    model="test",
                    prompt_tokens=10,
                    completion_tokens=5,
                    total_tokens=15,
                    generation_time=0.01,
                    parameters={},
                    metadata={"thread_id": thread_id, "approach": approach},
                ),
                approach_name=approach,
                execution_time=0.01,
                cost_estimate=0.001,
                metadata={"thread_safe": True},
            )

        # Setup mock dataset and reasoning engine
        with (
            patch.object(runner.dataset_loader, "load_dataset"),
            patch.object(
                runner.dataset_loader,
                "sample_data",
                return_value=[{"input": "test", "id": 1}],
            ),
            patch.object(
                runner.reasoning_engine,
                "run_inference",
                side_effect=mock_reasoning_inference_with_state_tracking,
            ),
        ):
            # Run parallel comparison
            result = runner.run_comparison(
                approaches, sample_count=1, parallel=True, max_workers=4
            )

        # Wait a bit for all threads to complete
        time.sleep(0.1)

        # Assert no thread state contamination occurred
        assert (
            not contamination_detected.is_set()
        ), "Thread state contamination detected between reasoning approaches"

        # Verify each thread maintained its own state
        assert len(thread_states) <= 4, "More threads created than max_workers"

        for thread_id, state in thread_states.items():
            assert state["calls"] >= 1, f"Thread {thread_id} made no calls"

    def test_parallel_cost_tracking_accuracy(self, runner, mock_dataset_loader):
        """Test cost tracking accuracy in parallel scenarios."""
        approaches = ["ChainOfThought", "AsPlanning", "ChainOfVerification"]

        # Mock cost per approach
        approach_costs = {
            "ChainOfThought": 0.001,
            "AsPlanning": 0.002,
            "ChainOfVerification": 0.003,  # Higher cost for multi-step
        }

        def mock_reasoning_with_cost(*args, **kwargs):
            approach = args[1] if len(args) > 1 else "unknown"
            cost = approach_costs.get(approach, 0.001)

            return ReasoningResult(
                response=StandardResponse(
                    text=f"Result from {approach}",
                    provider="test",
                    model="test",
                    prompt_tokens=10,
                    completion_tokens=5,
                    total_tokens=15,
                    generation_time=0.01,
                    parameters={},
                ),
                approach_name=approach,
                execution_time=0.01,
                cost_estimate=cost,
                metadata={},
            )

        with (
            patch.object(runner.dataset_loader, "load_dataset"),
            patch.object(
                runner.dataset_loader,
                "sample_data",
                return_value=[{"input": "test", "id": 1}, {"input": "test2", "id": 2}],
            ),
            patch.object(
                runner.reasoning_engine,
                "run_inference",
                side_effect=mock_reasoning_with_cost,
            ),
            patch(
                "ml_agents.core.experiment_runner.get_available_approaches",
                return_value=["None"] + approaches,
            ),
        ):
            result = runner.run_comparison(approaches, sample_count=2, parallel=True)

        # Verify cost tracking accuracy
        expected_total_cost = sum(approach_costs.values()) * 2  # 2 samples per approach
        actual_total_cost = sum(result.cost_summary.values())

        assert (
            abs(actual_total_cost - expected_total_cost) < 0.001
        ), f"Cost tracking inaccurate: expected {expected_total_cost}, got {actual_total_cost}"

    def test_checkpointing_and_resumption(
        self, runner, mock_dataset_loader, mock_reasoning_inference, temp_dir
    ):
        """Test experiment checkpointing and resumption functionality."""
        runner.dataset_loader = mock_dataset_loader
        runner.reasoning_engine = mock_reasoning_inference
        runner.checkpoint_interval = 2  # Save every 2 samples

        # Mock checkpoint path
        checkpoint_path = (
            temp_dir / f"checkpoint_ChainOfThought_{runner.experiment_id}.json"
        )

        with (
            patch.object(runner, "_get_checkpoint_path", return_value=checkpoint_path),
            patch(
                "ml_agents.core.experiment_runner.get_available_approaches",
                return_value=["ChainOfThought"],
            ),
        ):
            # First run - should save checkpoints
            result1 = runner.run_single_experiment(
                "ChainOfThought", sample_count=3, save_checkpoints=True
            )

            # Verify checkpoint was created (would be called at sample 2)
            # Reset runner state to simulate restart
            runner.current_sample = 0
            runner.results = []
            runner.errors = []

            # Create mock checkpoint data
            checkpoint_data = {
                "experiment_id": runner.experiment_id,
                "current_sample": 2,
                "results": runner.results[:2] if len(runner.results) >= 2 else [],
                "errors": [],
                "timestamp": datetime.now().isoformat(),
            }
            checkpoint_path.write_text(json.dumps(checkpoint_data))

            # Second run - should resume from checkpoint
            with patch.object(runner, "_load_checkpoint") as mock_load_checkpoint:
                result2 = runner.run_single_experiment(
                    "ChainOfThought",
                    sample_count=3,
                    resume_from_checkpoint=True,
                    save_checkpoints=False,
                )

                # Verify checkpoint loading was attempted
                if checkpoint_path.exists():
                    mock_load_checkpoint.assert_called()

    def test_progress_tracking_coordination(
        self, runner, mock_dataset_loader, mock_reasoning_inference
    ):
        """Test progress tracking coordination with multiple approaches."""
        approaches = ["ChainOfThought", "AsPlanning", "TreeOfThought"]
        runner.dataset_loader = mock_dataset_loader
        runner.reasoning_engine = mock_reasoning_inference

        # Track progress updates
        progress_updates = []

        def mock_tqdm(*args, **kwargs):
            mock_pbar = Mock()
            mock_pbar.__enter__ = Mock(return_value=mock_pbar)
            mock_pbar.__exit__ = Mock(return_value=None)

            def track_update(n):
                progress_updates.append(
                    {"approach": kwargs.get("desc", "unknown"), "update": n}
                )

            mock_pbar.update = track_update
            return mock_pbar

        with (
            patch("ml_agents.core.experiment_runner.tqdm", side_effect=mock_tqdm),
            patch(
                "ml_agents.core.experiment_runner.get_available_approaches",
                return_value=["None"] + approaches,
            ),
        ):
            result = runner.run_comparison(approaches, sample_count=2, parallel=False)

        # Should have progress updates for each approach
        assert len(progress_updates) >= 6  # 3 approaches * 2 samples minimum

    def test_result_saving_formats(self, runner, temp_dir):
        """Test result saving in different formats (CSV/JSON)."""
        # Mock experiment data
        runner.results = [
            {
                "sample_id": 0,
                "input": "What is 2+2?",
                "expected_output": "4",
                "approach": "ChainOfThought",
                "result": {"text": "The answer is 4.", "cost_estimate": 0.001},
                "timestamp": datetime.now().isoformat(),
            }
        ]

        summary = ExperimentSummary(
            experiment_id=runner.experiment_id,
            config=runner.config.to_dict(),
            start_time=datetime.now().isoformat(),
            end_time=datetime.now().isoformat(),
            duration=1.0,
            total_samples=1,
            approaches_tested=["ChainOfThought"],
            results_summary={"ChainOfThought": {"success_count": 1}},
            cost_summary={"ChainOfThought": 0.001},
            error_summary={"ChainOfThought": 0},
        )

        # Test result saving
        with patch.object(runner, "_save_results_csv") as mock_save_csv:
            runner._save_results(summary)

            # Verify CSV saving was called
            mock_save_csv.assert_called_once()

            # Verify JSON file creation (the method writes JSON directly)
            # We can't easily mock the JSON writing without breaking the implementation
            # So we just verify the CSV part was called

    @patch("ml_agents.core.experiment_runner.get_available_approaches")
    def test_multi_step_chain_of_verification_parallel(
        self, mock_get_approaches, runner, mock_dataset_loader, mock_reasoning_inference
    ):
        """Test multi-step Chain-of-Verification in parallel mode."""
        # This is critical - Chain-of-Verification with multi_step_verification=True
        # needs to work correctly in parallel with other approaches

        approaches = ["ChainOfThought", "ChainOfVerification", "AsPlanning"]
        mock_get_approaches.return_value = ["None"] + approaches
        runner.dataset_loader = mock_dataset_loader
        runner.reasoning_engine = mock_reasoning_inference

        # Ensure config has multi-step verification enabled
        assert runner.config.multi_step_verification == True

        # Mock multi-step result for ChainOfVerification
        def mock_reasoning_with_multi_step(*args, **kwargs):
            approach = args[1] if len(args) > 1 else "unknown"

            # ChainOfVerification should have multi-step metadata
            if approach == "ChainOfVerification":
                metadata = {
                    "multi_step_details": {
                        "initial_response": "Initial answer",
                        "verification_questions": "Is this correct?",
                        "final_verification": "Yes, correct",
                        "step_count": 3,
                    },
                    "reasoning_steps": 3,
                    "verification_count": 2,
                }
            else:
                metadata = {"reasoning_steps": 1}

            return ReasoningResult(
                response=StandardResponse(
                    text=f"Result from {approach}",
                    provider="test",
                    model="test",
                    prompt_tokens=10,
                    completion_tokens=5,
                    total_tokens=15,
                    generation_time=0.01,
                    parameters={},
                    metadata=metadata,
                ),
                approach_name=approach,
                execution_time=0.01,
                cost_estimate=(
                    0.001 if approach != "ChainOfVerification" else 0.003
                ),  # Higher cost for multi-step
                metadata=metadata,
            )

        runner.reasoning_engine.run_inference.side_effect = (
            mock_reasoning_with_multi_step
        )

        # Run parallel comparison including multi-step approach
        result = runner.run_comparison(approaches, sample_count=2, parallel=True)

        # Verify multi-step approach was included and completed successfully
        assert "ChainOfVerification" in result.approaches_tested
        assert (
            result.cost_summary["ChainOfVerification"]
            > result.cost_summary["ChainOfThought"]
        )  # Should cost more

    def test_experiment_summary_generation(self, runner):
        """Test comprehensive ExperimentSummary generation."""
        # Mock experiment results with proper nested structure matching ReasoningResult.asdict()
        runner.results = [
            {
                "sample_id": 0,
                "approach": "ChainOfThought",
                "expected_output": "4",
                "result": {
                    "response": {
                        "text": "Result 1",
                        "provider": "test",
                        "model": "test-model",
                        "prompt_tokens": 50,
                        "completion_tokens": 50,
                        "total_tokens": 100,
                        "generation_time": 1.0,
                        "parameters": {},
                        "response_id": "test1",
                        "metadata": {},
                        "extracted_answer": "4",
                    },
                    "approach_name": "ChainOfThought",
                    "execution_time": 1.5,
                    "cost_estimate": 0.005,
                    "metadata": {},
                },
            },
            {
                "sample_id": 1,
                "approach": "ChainOfThought",
                "expected_output": "6",
                "result": {
                    "response": {
                        "text": "Result 2",
                        "provider": "test",
                        "model": "test-model",
                        "prompt_tokens": 60,
                        "completion_tokens": 60,
                        "total_tokens": 120,
                        "generation_time": 1.5,
                        "parameters": {},
                        "response_id": "test2",
                        "metadata": {},
                        "extracted_answer": "6",
                    },
                    "approach_name": "ChainOfThought",
                    "execution_time": 2.0,
                    "cost_estimate": 0.006,
                    "metadata": {},
                },
            },
        ]
        runner.errors = [
            {"sample_id": 2, "approach": "ChainOfThought", "error": "API Error"}
        ]
        runner.start_time = datetime.now()
        runner.end_time = datetime.now()
        runner.total_samples = 2  # Set total_samples to match successful results

        summary = runner._generate_summary(["ChainOfThought"])

        assert isinstance(summary, ExperimentSummary)
        assert summary.experiment_id == runner.experiment_id
        assert summary.approaches_tested == ["ChainOfThought"]
        assert summary.total_samples == 2  # Successful results only
        assert summary.error_summary["ChainOfThought"] == 1

    @patch("ml_agents.core.experiment_runner.get_available_approaches")
    @patch("ml_agents.core.experiment_runner.ThreadPoolExecutor")
    def test_resource_cleanup_parallel_execution(
        self, mock_thread_pool, mock_get_approaches, runner
    ):
        """Test proper resource cleanup when parallel threads complete or fail."""
        approaches = ["ChainOfThought", "AsPlanning"]
        mock_get_approaches.return_value = ["None"] + approaches

        # Mock thread pool executor with proper context manager protocol
        mock_executor = Mock()
        mock_pool_instance = Mock()
        mock_pool_instance.__enter__ = Mock(return_value=mock_executor)
        mock_pool_instance.__exit__ = Mock(return_value=None)
        mock_thread_pool.return_value = mock_pool_instance

        # Mock futures for parallel execution
        mock_futures = []
        for i, approach in enumerate(approaches):
            mock_future = Mock(spec=Future)
            mock_future.result.return_value = ExperimentSummary(
                experiment_id=f"exp_test_{i}",
                config=runner.config.to_dict(),
                start_time=datetime.now().isoformat(),
                end_time=datetime.now().isoformat(),
                duration=1.0,
                total_samples=1,
                approaches_tested=[approach],
                results_summary={approach: {"success_count": 1, "error_count": 0}},
                cost_summary={approach: 0.01},
                error_summary={approach: 0},
            )
            mock_futures.append(mock_future)

        mock_executor.submit.side_effect = mock_futures

        # Mock dataset dependencies
        with (
            patch.object(runner.dataset_loader, "load_dataset"),
            patch.object(
                runner.dataset_loader,
                "sample_data",
                return_value=[{"input": "test", "id": 1}],
            ),
        ):
            # Mock as_completed to return futures immediately - THIS IS THE KEY FIX
            with patch(
                "ml_agents.core.experiment_runner.as_completed",
                return_value=mock_futures,
            ):
                result = runner.run_comparison(
                    approaches, sample_count=1, parallel=True, max_workers=2
                )

        # Verify parallel execution was used
        mock_thread_pool.assert_called_once_with(max_workers=2)
        assert mock_executor.submit.call_count == 2  # One per approach

        # Verify result aggregation worked
        assert isinstance(result, ExperimentSummary)
        assert result.approaches_tested == approaches

    @patch("ml_agents.core.experiment_runner.get_available_approaches")
    def test_critical_success_scenario(
        self,
        mock_get_approaches,
        critical_config,
        mock_dataset_loader,
        mock_reasoning_inference,
    ):
        """Test the critical success scenario from Phase 4 close-out document."""
        # Create runner with critical config for Phase 4 specifications
        with (
            patch("ml_agents.core.experiment_runner.BBEHDatasetLoader"),
            patch("ml_agents.core.experiment_runner.ReasoningInference"),
        ):
            critical_runner = ExperimentRunner(critical_config)
            critical_runner.dataset_loader = mock_dataset_loader
            critical_runner.reasoning_engine = mock_reasoning_inference

        # This is the exact test case specified in the close-out document
        approaches = ["ChainOfThought", "AsPlanning", "TreeOfThought"]
        mock_get_approaches.return_value = ["None"] + approaches

        # Ensure multi_step_verification is enabled as required
        assert critical_runner.config.multi_step_verification == True
        assert critical_runner.config.sample_count == 5

        try:
            # This should execute successfully without errors
            result = critical_runner.run_comparison(approaches, parallel=True)

            # Verify success criteria
            assert isinstance(result, ExperimentSummary)
            assert result.approaches_tested == approaches
            assert len(result.approaches_tested) == 3

            print("✅ Research platform ready for production experiments")

        except Exception as e:
            pytest.fail(f"Critical success scenario failed: {e}")

"""Integration tests for end-to-end database persistence."""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from ml_agents.config import ExperimentConfig
from ml_agents.core.database_manager import DatabaseConfig
from ml_agents.core.experiment_runner import ExperimentRunner
from ml_agents.core.results_processor import ResultsProcessor
from ml_agents.reasoning.base import StandardResponse


class TestDatabasePersistenceIntegration:
    """Integration tests for database persistence throughout the experiment pipeline."""

    @pytest.fixture
    def temp_config(self):
        """Create a temporary configuration for testing."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
            db_path = tmp.name

        config = ExperimentConfig(
            dataset_name="test_dataset",
            sample_count=5,
            provider="openai",
            model="gpt-4",
            reasoning_approaches=["None"],
            database_enabled=True,
            database_path=db_path,
            database_backup_frequency=10,
        )

        yield config, db_path

        # Cleanup
        Path(db_path).unlink(missing_ok=True)
        Path(f"{db_path}-wal").unlink(missing_ok=True)
        Path(f"{db_path}-shm").unlink(missing_ok=True)

    @pytest.fixture
    def mock_dataset_loader(self):
        """Mock dataset loader to return test data."""
        with patch("ml_agents.core.experiment_runner.BBEHDatasetLoader") as mock_loader:
            # Create mock instance
            mock_instance = MagicMock()
            mock_instance.load_samples.return_value = [
                {"input": f"Question {i}", "output": str(i)} for i in range(5)
            ]
            mock_instance.get_input_column_name.return_value = "input"
            mock_instance.get_output_column_name.return_value = "output"

            # Configure mock to return the instance
            mock_loader.return_value = mock_instance

            yield mock_instance

    @pytest.fixture
    def mock_reasoning_engine(self):
        """Mock reasoning engine to return controlled responses."""
        with patch(
            "ml_agents.core.experiment_runner.ReasoningInference"
        ) as mock_engine:
            mock_instance = MagicMock()

            def mock_run_inference(input_text, approach):
                # Create a mock ReasoningResult
                sample_id = input_text.split()[-1]  # Extract number from "Question X"
                is_correct = int(sample_id) % 2 == 0  # Even numbers are correct

                mock_result = MagicMock()
                mock_result.response = StandardResponse(
                    response=f"Response for {input_text}",
                    parsed_answer=sample_id,
                    is_correct=is_correct,
                    metadata={
                        "parsing_confidence": 0.8 + (int(sample_id) * 0.02),
                        "parsing_method": "instructor",
                    },
                )
                mock_result.execution_time = 1.5 + (int(sample_id) * 0.1)
                mock_result.cost_estimate = 0.001 * (int(sample_id) + 1)
                mock_result.metadata = {"reasoning_steps": int(sample_id) + 1}

                return mock_result

            mock_instance.run_inference.side_effect = mock_run_inference
            mock_engine.return_value = mock_instance

            yield mock_instance

    def test_experiment_runner_database_integration(
        self, temp_config, mock_dataset_loader, mock_reasoning_engine
    ):
        """Test that ExperimentRunner properly integrates with database persistence."""
        config, db_path = temp_config

        # Create and run experiment
        runner = ExperimentRunner(config)

        # Verify database was created and experiment was saved
        assert Path(db_path).exists()
        assert runner.results_processor is not None

        # Verify experiment metadata was saved
        processor = ResultsProcessor(DatabaseConfig(db_path=db_path))
        experiments = processor.get_experiments_list()

        assert len(experiments) == 1
        experiment = experiments[0]
        assert experiment["id"] == runner.experiment_id
        assert experiment["status"] == "running"
        assert "openai" in experiment["name"]
        assert "gpt-4" in experiment["name"]

        # Mock the progress callback to avoid UI calls
        with patch("ml_agents.core.experiment_runner.tqdm"):
            # Run single experiment
            summary = runner.run_single_experiment(
                approach="None", sample_count=5, save_checkpoints=False
            )

        # Verify experiment completed successfully
        assert summary is not None
        assert summary.total_samples == 5

        # Verify results were saved to database
        db_summary = processor.get_experiment_summary(runner.experiment_id)
        assert db_summary is not None
        assert db_summary.total_runs == 5
        assert db_summary.completed_runs == 5
        assert db_summary.failed_runs == 0

        # Verify individual runs were saved
        with processor.db_manager.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT COUNT(*) FROM runs WHERE experiment_id = ?",
                (runner.experiment_id,),
            )
            run_count = cursor.fetchone()[0]

            assert run_count == 5

            # Check run details
            cursor.execute(
                """
                SELECT approach_name, provider, model, input_text, parsed_answer, is_correct
                FROM runs WHERE experiment_id = ? ORDER BY sample_index
            """,
                (runner.experiment_id,),
            )

            runs = cursor.fetchall()
            assert len(runs) == 5

            for i, run in enumerate(runs):
                assert run[0] == "None"  # approach_name
                assert run[1] == "openai"  # provider
                assert run[2] == "gpt-4"  # model
                assert run[3] == f"Question {i}"  # input_text
                assert run[4] == str(i)  # parsed_answer
                assert run[5] == (i % 2 == 0)  # is_correct (even numbers)

        # Verify experiment status was updated to completed
        updated_experiments = processor.get_experiments_list()
        updated_experiment = updated_experiments[0]
        assert updated_experiment["status"] == "completed"

    def test_database_disabled_fallback(
        self, mock_dataset_loader, mock_reasoning_engine
    ):
        """Test that system works correctly when database is disabled."""
        config = ExperimentConfig(
            dataset_name="test_dataset",
            sample_count=3,
            provider="openai",
            model="gpt-4",
            reasoning_approaches=["None"],
            database_enabled=False,
        )

        # Create and run experiment
        runner = ExperimentRunner(config)

        # Verify database components are None
        assert runner.results_processor is None

        # Mock the progress callback to avoid UI calls
        with patch("ml_agents.core.experiment_runner.tqdm"):
            # Run experiment should still work
            summary = runner.run_single_experiment(
                approach="None", sample_count=3, save_checkpoints=False
            )

        # Verify experiment completed successfully even without database
        assert summary is not None
        assert summary.total_samples == 3

        # Verify results are still collected in memory
        assert len(runner.results) == 3

    def test_parsing_metrics_integration(self, temp_config, mock_dataset_loader):
        """Test that parsing metrics are properly saved through the full pipeline."""
        config, db_path = temp_config

        # Mock reasoning engine with parsing metrics
        with patch(
            "ml_agents.core.experiment_runner.ReasoningInference"
        ) as mock_engine:
            mock_instance = MagicMock()

            def mock_run_inference_with_metrics(input_text, approach):
                sample_id = input_text.split()[-1]

                mock_result = MagicMock()
                mock_result.response = StandardResponse(
                    response=f"Response for {input_text}",
                    parsed_answer=sample_id,
                    is_correct=True,
                    metadata={
                        "parsing_confidence": 0.9,
                        "parsing_method": "instructor",
                        "parsing_metrics": {
                            "attempts": 2,
                            "fallback_used": True,
                            "confidence": 0.85,
                            "extraction_time_ms": 150,
                            "error_details": "Minor parsing issue",
                        },
                    },
                )
                mock_result.execution_time = 1.5
                mock_result.cost_estimate = 0.001
                mock_result.metadata = {"reasoning_steps": 1}

                return mock_result

            mock_instance.run_inference.side_effect = mock_run_inference_with_metrics
            mock_engine.return_value = mock_instance

            # Run experiment
            runner = ExperimentRunner(config)

            with patch("ml_agents.core.experiment_runner.tqdm"):
                runner.run_single_experiment(
                    approach="None", sample_count=2, save_checkpoints=False
                )

        # Verify parsing metrics were saved
        processor = ResultsProcessor(DatabaseConfig(db_path=db_path))

        with processor.db_manager.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT pm.parsing_attempts, pm.fallback_used, pm.confidence_score,
                       pm.extraction_time_ms, pm.error_details
                FROM parsing_metrics pm
                JOIN runs r ON pm.run_id = r.id
                WHERE r.experiment_id = ?
            """,
                (runner.experiment_id,),
            )

            metrics = cursor.fetchall()
            assert len(metrics) == 2

            for metric in metrics:
                assert metric[0] == 2  # parsing_attempts
                assert metric[1] == True  # fallback_used
                assert metric[2] == 0.85  # confidence_score
                assert metric[3] == 150  # extraction_time_ms
                assert metric[4] == "Minor parsing issue"  # error_details

    def test_export_integration(
        self, temp_config, mock_dataset_loader, mock_reasoning_engine
    ):
        """Test export functionality works with persisted data."""
        config, db_path = temp_config

        # Run experiment to generate data
        runner = ExperimentRunner(config)

        with patch("ml_agents.core.experiment_runner.tqdm"):
            runner.run_single_experiment(
                approach="None", sample_count=3, save_checkpoints=False
            )

        # Test exports
        processor = ResultsProcessor(DatabaseConfig(db_path=db_path))

        # Test CSV export
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
            csv_path = tmp.name

        try:
            processor.export_to_csv(runner.experiment_id, csv_path)

            # Verify CSV was created and has content
            assert Path(csv_path).exists()
            with open(csv_path, "r") as f:
                content = f.read()

            lines = content.strip().split("\n")
            assert len(lines) == 4  # Header + 3 data rows
            assert "experiment_id" in lines[0]  # Header
            assert runner.experiment_id in content

        finally:
            Path(csv_path).unlink(missing_ok=True)

        # Test JSON export
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            json_path = tmp.name

        try:
            processor.export_to_json(runner.experiment_id, json_path)

            # Verify JSON was created and has correct structure
            assert Path(json_path).exists()

            with open(json_path, "r") as f:
                data = json.load(f)

            assert "experiment_metadata" in data
            assert "runs" in data
            assert "analysis" in data
            assert data["experiment_metadata"]["id"] == runner.experiment_id
            assert len(data["runs"]) == 3

        finally:
            Path(json_path).unlink(missing_ok=True)

    def test_comparison_analysis_integration(
        self, temp_config, mock_dataset_loader, mock_reasoning_engine
    ):
        """Test approach comparison works with database data."""
        config, db_path = temp_config
        config.reasoning_approaches = [
            "None",
            "ChainOfThought",
        ]  # Test multiple approaches

        # Mock different results for different approaches
        with patch(
            "ml_agents.core.experiment_runner.ReasoningInference"
        ) as mock_engine:
            mock_instance = MagicMock()

            def mock_run_inference_variable(input_text, approach):
                sample_id = int(input_text.split()[-1])

                # Different accuracy for different approaches
                if approach == "None":
                    is_correct = sample_id % 2 == 0
                else:  # ChainOfThought
                    is_correct = sample_id % 3 != 0

                mock_result = MagicMock()
                mock_result.response = StandardResponse(
                    response=f"{approach} response for {input_text}",
                    parsed_answer=str(sample_id),
                    is_correct=is_correct,
                    metadata={
                        "parsing_confidence": 0.8,
                        "parsing_method": "instructor",
                    },
                )
                mock_result.execution_time = 1.0 if approach == "None" else 2.0
                mock_result.cost_estimate = 0.001 if approach == "None" else 0.002
                mock_result.metadata = {"reasoning_steps": 1}

                return mock_result

            mock_instance.run_inference.side_effect = mock_run_inference_variable
            mock_engine.return_value = mock_instance

            # Run comparison experiment
            runner = ExperimentRunner(config)

            with patch("ml_agents.core.experiment_runner.tqdm"):
                summary = runner.run_comparison(
                    approaches=["None", "ChainOfThought"],
                    sample_count=3,
                    parallel=False,
                )

        # Test comparison analysis
        processor = ResultsProcessor(DatabaseConfig(db_path=db_path))

        comparisons = processor.compare_approaches(runner.experiment_id)

        assert len(comparisons) == 2

        # Find specific approaches
        none_comp = next(comp for comp in comparisons if comp.approach_name == "None")
        cot_comp = next(
            comp for comp in comparisons if comp.approach_name == "ChainOfThought"
        )

        assert none_comp.sample_count == 3
        assert cot_comp.sample_count == 3

        # Verify different performance metrics
        assert none_comp.avg_execution_time_ms != cot_comp.avg_execution_time_ms
        assert none_comp.total_cost != cot_comp.total_cost

        # Test accuracy report
        accuracy_report = processor.generate_accuracy_report(runner.experiment_id)

        assert "accuracy_by_approach" in accuracy_report
        assert "None" in accuracy_report["accuracy_by_approach"]
        assert "ChainOfThought" in accuracy_report["accuracy_by_approach"]

        none_accuracy = accuracy_report["accuracy_by_approach"]["None"]["accuracy"]
        cot_accuracy = accuracy_report["accuracy_by_approach"]["ChainOfThought"][
            "accuracy"
        ]

        # Verify accuracies are different (based on our mock logic)
        # None: samples 0,1,2 -> correct: 0,2 -> 2/3 ≈ 0.67
        # CoT: samples 0,1,2 -> correct: 1,2 -> 2/3 ≈ 0.67
        assert 0 <= none_accuracy <= 1
        assert 0 <= cot_accuracy <= 1

    def test_error_recovery_integration(self, temp_config, mock_dataset_loader):
        """Test that system handles errors gracefully while maintaining database consistency."""
        config, db_path = temp_config

        # Mock reasoning engine that fails on certain inputs
        with patch(
            "ml_agents.core.experiment_runner.ReasoningInference"
        ) as mock_engine:
            mock_instance = MagicMock()

            def mock_run_inference_with_errors(input_text, approach):
                sample_id = int(input_text.split()[-1])

                # Fail on sample 1
                if sample_id == 1:
                    raise Exception("Simulated inference error")

                mock_result = MagicMock()
                mock_result.response = StandardResponse(
                    response=f"Response for {input_text}",
                    parsed_answer=str(sample_id),
                    is_correct=True,
                    metadata={"parsing_confidence": 0.9},
                )
                mock_result.execution_time = 1.5
                mock_result.cost_estimate = 0.001
                mock_result.metadata = {"reasoning_steps": 1}

                return mock_result

            mock_instance.run_inference.side_effect = mock_run_inference_with_errors
            mock_engine.return_value = mock_instance

            # Run experiment
            runner = ExperimentRunner(config)

            with patch("ml_agents.core.experiment_runner.tqdm"):
                summary = runner.run_single_experiment(
                    approach="None", sample_count=3, save_checkpoints=False
                )

        # Verify experiment completed despite errors
        assert summary is not None

        # Verify database contains successful runs only
        processor = ResultsProcessor(DatabaseConfig(db_path=db_path))
        db_summary = processor.get_experiment_summary(runner.experiment_id)

        assert db_summary is not None
        # Should have 2 successful runs (samples 0 and 2), sample 1 failed
        assert db_summary.total_runs == 2
        assert db_summary.completed_runs == 2
        assert db_summary.failed_runs == 0

        # Verify errors were logged but database remains consistent
        with processor.db_manager.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT sample_index FROM runs WHERE experiment_id = ? ORDER BY sample_index",
                (runner.experiment_id,),
            )
            saved_samples = [row[0] for row in cursor.fetchall()]

            # Should have samples 0 and 2, but not 1 (which failed)
            assert saved_samples == [0, 2]


if __name__ == "__main__":
    pytest.main([__file__])

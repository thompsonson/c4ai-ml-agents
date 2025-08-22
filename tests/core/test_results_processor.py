"""Tests for results_processor module."""

import json
import tempfile
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.core.database_manager import DatabaseConfig
from src.core.results_processor import (
    ApproachComparison,
    ExperimentSummary,
    ResultsProcessor,
)
from src.reasoning.base import StandardResponse


class TestExperimentSummary:
    """Test ExperimentSummary dataclass."""

    def test_experiment_summary_creation(self):
        """Test ExperimentSummary can be created with all fields."""
        summary = ExperimentSummary(
            experiment_id="test_exp",
            total_runs=100,
            completed_runs=95,
            failed_runs=5,
            accuracy=0.85,
            avg_execution_time_ms=1500.0,
            total_cost=0.50,
            approaches_tested=["CoT", "PoT"],
            models_used=["gpt-4"],
            parsing_success_rate=0.92,
            created_at=datetime.now(),
            last_updated=datetime.now(),
        )

        assert summary.experiment_id == "test_exp"
        assert summary.total_runs == 100
        assert summary.accuracy == 0.85
        assert len(summary.approaches_tested) == 2


class TestApproachComparison:
    """Test ApproachComparison dataclass."""

    def test_approach_comparison_creation(self):
        """Test ApproachComparison can be created with all fields."""
        comparison = ApproachComparison(
            approach_name="Chain-of-Thought",
            accuracy=0.85,
            avg_execution_time_ms=1500.0,
            total_cost=0.25,
            sample_count=50,
            parsing_success_rate=0.92,
            confidence_avg=0.78,
            failure_rate=0.04,
        )

        assert comparison.approach_name == "Chain-of-Thought"
        assert comparison.accuracy == 0.85
        assert comparison.sample_count == 50


class TestResultsProcessor:
    """Test ResultsProcessor class."""

    @pytest.fixture
    def temp_processor(self):
        """Create a temporary ResultsProcessor for testing."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
            db_path = tmp.name

        config = DatabaseConfig(db_path=db_path)
        processor = ResultsProcessor(config)

        yield processor, db_path

        # Cleanup
        Path(db_path).unlink(missing_ok=True)
        # Also cleanup WAL files
        Path(f"{db_path}-wal").unlink(missing_ok=True)
        Path(f"{db_path}-shm").unlink(missing_ok=True)

    @pytest.fixture
    def sample_experiment_data(self, temp_processor):
        """Create sample experiment data for testing."""
        processor, db_path = temp_processor

        # Create sample experiment
        experiment_id = "test_experiment_123"
        config = {
            "provider": "openai",
            "model": "gpt-4",
            "reasoning_approaches": ["CoT", "PoT"],
            "sample_count": 10,
        }

        processor.save_experiment(
            experiment_id=experiment_id,
            name="Test Experiment",
            config=config,
            description="A test experiment for unit testing",
        )

        # Create sample runs
        for i in range(10):
            approach = "CoT" if i < 5 else "PoT"
            is_correct = i % 3 != 0  # 2/3 correct

            response = StandardResponse(
                text=f"Answer {i}",
                provider="openai",
                model="gpt-4",
                prompt_tokens=10,
                completion_tokens=5,
                total_tokens=15,
                generation_time=1.5 + (i * 0.1),
                parameters={},
                extracted_answer=str(i),
                metadata={
                    "experiment_id": experiment_id,
                    "approach": approach,
                    "sample_index": i,
                    "input_text": f"Question {i}",
                    "expected_answer": str(i),
                    "provider": "openai",
                    "model": "gpt-4",
                    "latency": 1.5 + (i * 0.1),
                    "cost": 0.001 * (i + 1),
                    "parsing_confidence": 0.8 + (i * 0.02),
                    "parsing_method": "instructor",
                },
            )

            processor.save_run_result(response)

        return processor, experiment_id, config

    def test_initialization(self, temp_processor):
        """Test ResultsProcessor initialization."""
        processor, db_path = temp_processor

        assert processor.db_config.db_path == db_path
        assert processor.db_manager is not None

        # Verify database was initialized
        assert Path(db_path).exists()

    def test_save_experiment(self, temp_processor):
        """Test saving experiment metadata."""
        processor, db_path = temp_processor

        experiment_id = "test_exp_123"
        config = {"model": "gpt-4", "approach": "CoT"}

        processor.save_experiment(
            experiment_id=experiment_id,
            name="Test Experiment",
            config=config,
            description="Test description",
        )

        # Verify experiment was saved
        with processor.db_manager.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM experiments WHERE id = ?", (experiment_id,))
            row = cursor.fetchone()

            assert row is not None
            assert row["id"] == experiment_id
            assert row["name"] == "Test Experiment"
            assert row["description"] == "Test description"
            assert json.loads(row["config_json"]) == config
            assert row["status"] == "running"

    def test_save_run_result(self, temp_processor):
        """Test saving run results."""
        processor, db_path = temp_processor

        # First create an experiment
        experiment_id = "test_exp"
        processor.save_experiment(experiment_id, "Test", {})

        # Create and save a run result
        response = StandardResponse(
            text="The answer is 42",
            provider="openai",
            model="gpt-4",
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
            generation_time=1.5,
            parameters={},
            extracted_answer="42",
            metadata={
                "experiment_id": experiment_id,
                "approach": "CoT",
                "sample_index": 0,
                "input_text": "What is the answer?",
                "expected_answer": "42",
                "provider": "openai",
                "model": "gpt-4",
                "latency": 1.5,
                "cost": 0.001,
                "parsing_confidence": 0.95,
                "parsing_method": "instructor",
                "parsing_metrics": {
                    "attempts": 1,
                    "fallback_used": False,
                    "confidence": 0.95,
                    "extraction_time_ms": 100,
                },
            },
        )

        processor.save_run_result(response)

        # Verify run was saved
        with processor.db_manager.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM runs WHERE experiment_id = ?", (experiment_id,)
            )
            row = cursor.fetchone()

            assert row is not None
            assert row["experiment_id"] == experiment_id
            assert row["approach_name"] == "CoT"
            assert row["raw_output"] == "The answer is 42"
            assert row["parsed_answer"] == "42"
            assert row["is_correct"] == True
            assert row["provider"] == "openai"
            assert row["model"] == "gpt-4"

            # Verify parsing metrics were saved
            cursor.execute(
                "SELECT * FROM parsing_metrics WHERE run_id = ?", (row["id"],)
            )
            metrics_row = cursor.fetchone()

            assert metrics_row is not None
            assert metrics_row["parsing_attempts"] == 1
            assert metrics_row["fallback_used"] == False
            assert metrics_row["confidence_score"] == 0.95

    def test_update_experiment_status(self, temp_processor):
        """Test updating experiment status."""
        processor, db_path = temp_processor

        experiment_id = "test_exp"
        processor.save_experiment(experiment_id, "Test", {})

        # Update status
        processor.update_experiment_status(experiment_id, "completed")

        # Verify status was updated
        with processor.db_manager.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT status FROM experiments WHERE id = ?", (experiment_id,)
            )
            status = cursor.fetchone()[0]

            assert status == "completed"

    def test_get_experiment_summary(self, sample_experiment_data):
        """Test getting experiment summary."""
        processor, experiment_id, config = sample_experiment_data

        summary = processor.get_experiment_summary(experiment_id)

        assert summary is not None
        assert isinstance(summary, ExperimentSummary)
        assert summary.experiment_id == experiment_id
        assert summary.total_runs == 10
        assert summary.completed_runs == 10
        assert summary.failed_runs == 0
        assert 0 < summary.accuracy < 1  # Should be around 2/3
        assert summary.avg_execution_time_ms > 0
        assert summary.total_cost > 0
        assert "CoT" in summary.approaches_tested
        assert "PoT" in summary.approaches_tested
        assert "gpt-4" in summary.models_used

    def test_get_experiment_summary_nonexistent(self, temp_processor):
        """Test getting summary for non-existent experiment."""
        processor, db_path = temp_processor

        summary = processor.get_experiment_summary("nonexistent")
        assert summary is None

    def test_compare_approaches(self, sample_experiment_data):
        """Test comparing reasoning approaches."""
        processor, experiment_id, config = sample_experiment_data

        comparisons = processor.compare_approaches(experiment_id)

        assert len(comparisons) == 2  # CoT and PoT
        assert all(isinstance(comp, ApproachComparison) for comp in comparisons)

        # Find CoT and PoT comparisons
        cot_comp = next(comp for comp in comparisons if comp.approach_name == "CoT")
        pot_comp = next(comp for comp in comparisons if comp.approach_name == "PoT")

        assert cot_comp.sample_count == 5
        assert pot_comp.sample_count == 5
        assert cot_comp.accuracy >= 0.0
        assert pot_comp.accuracy >= 0.0

    def test_compare_approaches_filtered(self, sample_experiment_data):
        """Test comparing specific approaches."""
        processor, experiment_id, config = sample_experiment_data

        comparisons = processor.compare_approaches(experiment_id, approaches=["CoT"])

        assert len(comparisons) == 1
        assert comparisons[0].approach_name == "CoT"
        assert comparisons[0].sample_count == 5

    def test_generate_accuracy_report(self, sample_experiment_data):
        """Test generating accuracy report."""
        processor, experiment_id, config = sample_experiment_data

        report = processor.generate_accuracy_report(experiment_id)

        assert "experiment_id" in report
        assert "accuracy_by_approach" in report
        assert "accuracy_by_model" in report
        assert "parsing_effectiveness" in report
        assert "generated_at" in report

        assert report["experiment_id"] == experiment_id
        assert "CoT" in report["accuracy_by_approach"]
        assert "PoT" in report["accuracy_by_approach"]
        assert "gpt-4" in report["accuracy_by_model"]

        # Check structure of approach data
        cot_data = report["accuracy_by_approach"]["CoT"]
        assert "total" in cot_data
        assert "correct" in cot_data
        assert "accuracy" in cot_data
        assert cot_data["total"] == 5

    def test_identify_failure_patterns(self, sample_experiment_data):
        """Test identifying failure patterns."""
        processor, experiment_id, config = sample_experiment_data

        patterns = processor.identify_failure_patterns(experiment_id)

        assert isinstance(patterns, list)
        # Since we have some incorrect answers, we should have patterns
        if patterns:
            pattern = patterns[0]
            assert "failure_type" in pattern
            assert "approach" in pattern
            assert "model" in pattern
            assert "count" in pattern
            assert "examples" in pattern

    def test_export_to_csv(self, sample_experiment_data):
        """Test exporting to CSV format."""
        processor, experiment_id, config = sample_experiment_data

        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
            output_path = tmp.name

        try:
            processor.export_to_csv(experiment_id, output_path)

            # Verify file was created and has content
            assert Path(output_path).exists()

            with open(output_path, "r") as f:
                content = f.read()

            # Check that CSV has expected structure
            lines = content.strip().split("\n")
            assert len(lines) > 1  # Header + data rows

            # Check header
            header = lines[0]
            assert "experiment_id" in header
            assert "approach_name" in header
            assert "is_correct" in header

            # Check we have data rows
            assert len(lines) == 11  # Header + 10 data rows

        finally:
            Path(output_path).unlink(missing_ok=True)

    def test_export_to_json(self, sample_experiment_data):
        """Test exporting to JSON format."""
        processor, experiment_id, config = sample_experiment_data

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            output_path = tmp.name

        try:
            processor.export_to_json(experiment_id, output_path)

            # Verify file was created and has valid JSON
            assert Path(output_path).exists()

            with open(output_path, "r") as f:
                data = json.load(f)

            # Check structure
            assert "experiment_metadata" in data
            assert "summary_statistics" in data
            assert "runs" in data
            assert "analysis" in data

            assert data["experiment_metadata"]["id"] == experiment_id
            assert len(data["runs"]) == 10
            assert "accuracy_report" in data["analysis"]
            assert "approach_comparison" in data["analysis"]

        finally:
            Path(output_path).unlink(missing_ok=True)

    def test_export_to_json_include_raw(self, sample_experiment_data):
        """Test exporting to JSON with raw output included."""
        processor, experiment_id, config = sample_experiment_data

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            output_path = tmp.name

        try:
            processor.export_to_json(
                experiment_id, output_path, include_raw_output=True
            )

            with open(output_path, "r") as f:
                data = json.load(f)

            # Check that raw output is included
            first_run = data["runs"][0]
            assert "raw_output" in first_run
            assert "Answer" in first_run["raw_output"]  # Should contain actual content

        finally:
            Path(output_path).unlink(missing_ok=True)

    def test_export_to_json_exclude_raw(self, sample_experiment_data):
        """Test exporting to JSON with raw output excluded."""
        processor, experiment_id, config = sample_experiment_data

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            output_path = tmp.name

        try:
            processor.export_to_json(
                experiment_id, output_path, include_raw_output=False
            )

            with open(output_path, "r") as f:
                data = json.load(f)

            # Check that raw output is truncated
            first_run = data["runs"][0]
            assert "raw_output" in first_run
            assert (
                "[" in first_run["raw_output"] and "chars]" in first_run["raw_output"]
            )

        finally:
            Path(output_path).unlink(missing_ok=True)

    @patch("openpyxl.Workbook")
    def test_export_to_excel(self, mock_workbook, sample_experiment_data):
        """Test exporting to Excel format."""
        processor, experiment_id, config = sample_experiment_data

        # Mock the workbook and worksheet
        mock_wb = MagicMock()
        mock_workbook.return_value = mock_wb
        mock_sheet = MagicMock()
        mock_wb.create_sheet.return_value = mock_sheet

        with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as tmp:
            output_path = tmp.name

        try:
            processor.export_to_excel([experiment_id], output_path)

            # Verify workbook methods were called
            mock_workbook.assert_called_once()
            mock_wb.save.assert_called_once_with(output_path)

            # Verify sheets were created
            assert mock_wb.create_sheet.call_count >= 2  # Summary + experiment sheet

        finally:
            Path(output_path).unlink(missing_ok=True)

    def test_get_experiments_list(self, sample_experiment_data):
        """Test getting list of experiments."""
        processor, experiment_id, config = sample_experiment_data

        experiments = processor.get_experiments_list()

        assert len(experiments) == 1
        experiment = experiments[0]

        assert experiment["id"] == experiment_id
        assert experiment["name"] == "Test Experiment"
        assert experiment["status"] == "running"
        assert "config" in experiment
        assert experiment["config"]["provider"] == "openai"

    def test_get_experiments_list_filtered(self, sample_experiment_data):
        """Test getting filtered list of experiments."""
        processor, experiment_id, config = sample_experiment_data

        # Update experiment status
        processor.update_experiment_status(experiment_id, "completed")

        # Test filtering
        running_experiments = processor.get_experiments_list(status="running")
        completed_experiments = processor.get_experiments_list(status="completed")

        assert len(running_experiments) == 0
        assert len(completed_experiments) == 1
        assert completed_experiments[0]["id"] == experiment_id

    def test_get_run_details(self, sample_experiment_data):
        """Test getting detailed run information."""
        processor, experiment_id, config = sample_experiment_data

        # Get a run ID
        with processor.db_manager.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id FROM runs LIMIT 1")
            run_id = cursor.fetchone()[0]

        details = processor.get_run_details(run_id)

        assert details is not None
        assert details["id"] == run_id
        assert details["experiment_id"] == experiment_id
        assert "metadata" in details
        assert details["approach_name"] in ["CoT", "PoT"]

    def test_get_run_details_nonexistent(self, temp_processor):
        """Test getting details for non-existent run."""
        processor, db_path = temp_processor

        details = processor.get_run_details("nonexistent_run")
        assert details is None

    def test_cleanup_old_experiments(self, sample_experiment_data):
        """Test cleaning up old experiments."""
        processor, experiment_id, config = sample_experiment_data

        # Set experiment to be old (simulate by updating created_at)
        with processor.db_manager.get_connection() as conn:
            conn.execute(
                """
                UPDATE experiments
                SET created_at = datetime('now', '-31 days')
                WHERE id = ?
            """,
                (experiment_id,),
            )
            conn.commit()

        # Cleanup experiments older than 30 days
        deleted_count = processor.cleanup_old_experiments(days=30)

        assert deleted_count == 1

        # Verify experiment was deleted
        summary = processor.get_experiment_summary(experiment_id)
        assert summary is None

    def test_save_parsing_metrics_standalone(self, temp_processor):
        """Test saving parsing metrics independently."""
        processor, db_path = temp_processor

        # Create experiment and run first
        experiment_id = "test_exp"
        processor.save_experiment(experiment_id, "Test", {})

        response = StandardResponse(
            text="Answer",
            provider="openai",
            model="gpt-4",
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
            generation_time=1.5,
            parameters={},
            extracted_answer="42",
            metadata={
                "experiment_id": experiment_id,
                "approach": "CoT",
                "sample_index": 0,
                "input_text": "Question",
                "expected_answer": "42",
                "provider": "openai",
                "model": "gpt-4",
                "latency": 1.5,
                "cost": 0.001,
            },
        )

        processor.save_run_result(response)

        # Get the run ID
        with processor.db_manager.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id FROM runs LIMIT 1")
            run_id = cursor.fetchone()[0]

        # Save additional parsing metrics
        metrics = {
            "attempts": 2,
            "fallback_used": True,
            "confidence": 0.75,
            "extraction_time_ms": 200,
            "error_details": "Some error",
        }

        processor.save_parsing_metrics(run_id, metrics)

        # Verify metrics were saved/updated
        with processor.db_manager.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM parsing_metrics WHERE run_id = ?", (run_id,))
            row = cursor.fetchone()

            assert row is not None
            assert row["parsing_attempts"] == 2
            assert row["fallback_used"] == True
            assert row["confidence_score"] == 0.75
            assert row["extraction_time_ms"] == 200
            assert row["error_details"] == "Some error"

    def test_error_handling(self, temp_processor):
        """Test error handling in various scenarios."""
        processor, db_path = temp_processor

        # Test with invalid experiment ID in run result
        response = StandardResponse(
            text="Answer",
            provider="openai",
            model="gpt-4",
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
            generation_time=1.5,
            parameters={},
            extracted_answer="42",
            metadata={
                "experiment_id": "nonexistent_experiment",
                "approach": "CoT",
                "sample_index": 0,
                "provider": "openai",
                "model": "gpt-4",
            },
        )

        # This should handle the error gracefully
        processor.save_run_result(response)

        # Test export with non-existent experiment
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
            output_path = tmp.name

        try:
            # Should not raise exception, but file might be empty or have headers only
            processor.export_to_csv("nonexistent", output_path)
        finally:
            Path(output_path).unlink(missing_ok=True)

    def test_concurrent_operations(self, sample_experiment_data):
        """Test concurrent database operations."""
        processor, experiment_id, config = sample_experiment_data

        # Create multiple processors (simulating concurrent access)
        processor2 = ResultsProcessor(processor.db_config)

        # Both should be able to read data
        summary1 = processor.get_experiment_summary(experiment_id)
        summary2 = processor2.get_experiment_summary(experiment_id)

        assert summary1 is not None
        assert summary2 is not None
        assert summary1.experiment_id == summary2.experiment_id
        assert summary1.total_runs == summary2.total_runs


if __name__ == "__main__":
    pytest.main([__file__])

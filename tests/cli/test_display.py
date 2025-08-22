"""Tests for CLI display utilities."""

from unittest.mock import Mock, patch

import pytest
from rich.console import Console
from rich.table import Table

from ml_agents.cli.display import (
    create_cost_summary_table,
    create_experiment_table,
    create_progress_display,
    display_banner,
    display_checkpoint_info,
    display_cost_warning,
    display_error,
    display_experiment_complete,
    display_experiment_start,
    display_info,
    display_success,
    display_validation_errors,
    display_warning,
    format_approach_name,
)


class TestDisplayFunctions:
    """Test basic display functions."""

    @patch("ml_agents.cli.display.console")
    def test_display_error(self, mock_console):
        """Test error message display."""
        display_error("Test error message")

        mock_console.print.assert_called_once()
        args, kwargs = mock_console.print.call_args
        assert "❌" in args[0]
        assert "Error:" in args[0]
        assert "Test error message" in args[0]

    @patch("ml_agents.cli.display.console")
    def test_display_warning(self, mock_console):
        """Test warning message display."""
        display_warning("Test warning message")

        mock_console.print.assert_called_once()
        args, kwargs = mock_console.print.call_args
        assert "⚠️" in args[0]
        assert "Warning:" in args[0]
        assert "Test warning message" in args[0]

    @patch("ml_agents.cli.display.console")
    def test_display_success(self, mock_console):
        """Test success message display."""
        display_success("Test success message")

        mock_console.print.assert_called_once()
        args, kwargs = mock_console.print.call_args
        assert "✅" in args[0]
        assert "Test success message" in args[0]

    @patch("ml_agents.cli.display.console")
    def test_display_info(self, mock_console):
        """Test info message display."""
        display_info("Test info message")

        mock_console.print.assert_called_once()
        args, kwargs = mock_console.print.call_args
        assert "ℹ️" in args[0]
        assert "Test info message" in args[0]

    @patch("ml_agents.cli.display.console")
    def test_display_banner(self, mock_console):
        """Test banner display."""
        display_banner()

        mock_console.print.assert_called_once()
        # Check that Panel was called with banner content
        args, kwargs = mock_console.print.call_args
        panel_content = args[0]
        # Panel object should contain ML Agents text
        assert hasattr(panel_content, "renderable")


class TestExperimentTable:
    """Test experiment results table creation."""

    def test_create_experiment_table_single_result(self):
        """Test creating table with single experiment result."""
        results = [
            {
                "approach": "ChainOfThought",
                "total_samples": 50,
                "accuracy": 0.85,
                "avg_time": 2.5,
                "total_cost": 1.25,
                "completed": True,
                "error_count": 0,
            }
        ]

        table = create_experiment_table(results)

        assert isinstance(table, Table)
        # Verify table has expected columns
        column_names = [column.header for column in table.columns]
        assert "Approach" in column_names
        assert "Samples" in column_names
        assert "Accuracy" in column_names
        assert "Avg Time" in column_names
        assert "Total Cost" in column_names
        assert "Status" in column_names

    def test_create_experiment_table_multiple_results(self):
        """Test creating table with multiple experiment results."""
        results = [
            {
                "approach": "ChainOfThought",
                "total_samples": 50,
                "accuracy": 0.85,
                "avg_time": 2.5,
                "total_cost": 1.25,
                "completed": True,
                "error_count": 0,
            },
            {
                "approach": "AsPlanning",
                "total_samples": 50,
                "accuracy": 0.78,
                "avg_time": 3.2,
                "total_cost": 1.80,
                "completed": True,
                "error_count": 2,
            },
            {
                "approach": "TreeOfThought",
                "total_samples": 25,
                "accuracy": None,
                "avg_time": None,
                "total_cost": 0.60,
                "completed": False,
                "error_count": 0,
            },
        ]

        table = create_experiment_table(results)

        assert isinstance(table, Table)
        # Table should have 3 rows (one for each result)
        assert len(table.rows) == 3

    def test_create_experiment_table_with_errors(self):
        """Test table creation with error conditions."""
        results = [
            {
                "approach": "ChainOfThought",
                "total_samples": 50,
                "accuracy": 0.80,
                "avg_time": 2.1,
                "total_cost": 1.10,
                "completed": True,
                "error_count": 5,  # Has errors
            }
        ]

        table = create_experiment_table(results)

        assert isinstance(table, Table)
        # Should show partial status due to errors
        assert len(table.rows) == 1

    def test_create_experiment_table_incomplete(self):
        """Test table creation with incomplete results."""
        results = [
            {
                "approach": "TreeOfThought",
                "total_samples": 30,
                "accuracy": None,
                "avg_time": None,
                "total_cost": 0.45,
                "completed": False,
                "error_count": 0,
            }
        ]

        table = create_experiment_table(results)

        assert isinstance(table, Table)
        # Should show running status
        assert len(table.rows) == 1

    def test_create_experiment_table_missing_fields(self):
        """Test table creation handles missing fields gracefully."""
        results = [
            {
                "approach": "ChainOfThought",
                # Missing most fields to test defaults
            }
        ]

        table = create_experiment_table(results)

        assert isinstance(table, Table)
        assert len(table.rows) == 1
        # Should use default values (0, N/A, etc.)


class TestCostSummaryTable:
    """Test cost summary table creation."""

    def test_create_cost_summary_table_single_provider(self):
        """Test creating cost summary table with single provider."""
        cost_breakdown = {
            "openrouter": {
                "gpt-3.5-turbo": {"requests": 50, "tokens": 25000, "cost": 1.25}
            }
        }

        table = create_cost_summary_table(cost_breakdown)

        assert isinstance(table, Table)
        # Should have provider row + total row
        assert len(table.rows) == 2

        # Check column headers
        column_names = [column.header for column in table.columns]
        assert "Provider" in column_names
        assert "Model" in column_names
        assert "Requests" in column_names
        assert "Tokens" in column_names
        assert "Cost" in column_names

    def test_create_cost_summary_table_multiple_providers(self):
        """Test creating cost summary table with multiple providers."""
        cost_breakdown = {
            "openrouter": {
                "gpt-3.5-turbo": {"requests": 30, "tokens": 15000, "cost": 0.75},
                "gpt-4": {"requests": 20, "tokens": 10000, "cost": 2.00},
            },
            "anthropic": {
                "claude-sonnet-4-20250514": {
                    "requests": 25,
                    "tokens": 12500,
                    "cost": 1.50,
                }
            },
        }

        table = create_cost_summary_table(cost_breakdown)

        assert isinstance(table, Table)
        # Should have 3 provider/model rows + 1 total row
        assert len(table.rows) == 4

    def test_create_cost_summary_table_empty(self):
        """Test creating cost summary table with no data."""
        cost_breakdown = {}

        table = create_cost_summary_table(cost_breakdown)

        assert isinstance(table, Table)
        # Should only have total row (showing $0.00)
        assert len(table.rows) == 1

    def test_create_cost_summary_table_malformed_data(self):
        """Test cost summary table handles malformed data."""
        cost_breakdown = {
            "openrouter": {
                "gpt-3.5-turbo": {
                    # Missing some fields
                    "requests": 30
                    # Missing tokens and cost
                }
            },
            "invalid_provider": "not_a_dict",  # Wrong structure
        }

        table = create_cost_summary_table(cost_breakdown)

        assert isinstance(table, Table)
        # Should handle malformed data gracefully


class TestProgressDisplay:
    """Test progress display creation."""

    def test_create_progress_display_single_approach(self):
        """Test creating progress display for single approach."""
        progress = create_progress_display(
            total_samples=50, approaches=["ChainOfThought"], parallel=False
        )

        # Should return Progress object
        assert progress is not None
        assert hasattr(progress, "add_task")

    def test_create_progress_display_multiple_approaches(self):
        """Test creating progress display for multiple approaches."""
        progress = create_progress_display(
            total_samples=100,
            approaches=["ChainOfThought", "AsPlanning", "TreeOfThought"],
            parallel=True,
        )

        assert progress is not None
        assert hasattr(progress, "add_task")

    def test_create_progress_display_large_dataset(self):
        """Test creating progress display for large dataset."""
        progress = create_progress_display(
            total_samples=10000, approaches=["ChainOfThought"], parallel=False
        )

        assert progress is not None


class TestExperimentDisplays:
    """Test experiment-specific display functions."""

    @patch("ml_agents.cli.display.console")
    def test_display_experiment_start(self, mock_console):
        """Test experiment start display."""
        config = {
            "dataset_name": "MrLight/bbeh-eval",
            "provider": "openrouter",
            "model": "gpt-3.5-turbo",
            "parallel": False,
        }
        approaches = ["ChainOfThought", "AsPlanning"]
        total_samples = 50

        display_experiment_start(config, approaches, total_samples)

        # Should have called print multiple times
        assert mock_console.print.call_count >= 2

    @patch("ml_agents.cli.display.console")
    def test_display_experiment_complete(self, mock_console):
        """Test experiment completion display."""
        display_experiment_complete(
            experiment_id="exp_12345",
            duration=125.5,
            total_cost=2.75,
            output_dir="./outputs",
        )

        # Should display completion info
        assert mock_console.print.call_count >= 2

        # Check that completion message was shown
        calls = [call for call in mock_console.print.call_args_list]
        completion_call = next(
            (call for call in calls if "Experiment Complete" in str(call)), None
        )
        assert completion_call is not None

    @patch("ml_agents.cli.display.console")
    def test_display_checkpoint_info(self, mock_console):
        """Test checkpoint info display."""
        display_checkpoint_info(
            checkpoint_file="/path/to/checkpoint.json",
            resume_from="2024-01-15 10:30:00",
        )

        assert mock_console.print.call_count >= 2

        # Should mention resuming from checkpoint
        calls = [call for call in mock_console.print.call_args_list]
        checkpoint_call = next(
            (call for call in calls if "checkpoint" in str(call[0]).lower()), None
        )
        assert checkpoint_call is not None


class TestValidationErrorsDisplay:
    """Test validation error display functions."""

    @patch("ml_agents.cli.display.console")
    def test_display_validation_errors_single(self, mock_console):
        """Test displaying single validation error."""
        errors = ["Temperature must be between 0.0 and 2.0"]

        display_validation_errors(errors)

        # Should show validation failed header and the error
        assert mock_console.print.call_count >= 2

        # Check that error message was displayed
        calls = [call for call in mock_console.print.call_args_list]
        header_call = next(
            (call for call in calls if "Validation Failed" in str(call)), None
        )
        assert header_call is not None

    @patch("ml_agents.cli.display.console")
    def test_display_validation_errors_multiple(self, mock_console):
        """Test displaying multiple validation errors."""
        errors = [
            "Temperature must be between 0.0 and 2.0",
            "Sample count must be at least 1",
            "Invalid reasoning approach: InvalidApproach",
        ]

        display_validation_errors(errors)

        # Should show header + 3 errors + footer
        assert mock_console.print.call_count >= 4


class TestCostWarning:
    """Test cost warning display and confirmation."""

    @patch("typer.confirm")
    @patch("ml_agents.cli.display.console")
    def test_display_cost_warning_under_threshold(self, mock_console, mock_confirm):
        """Test cost warning not shown for costs under threshold."""
        result = display_cost_warning(estimated_cost=5.0, threshold=10.0)

        assert result is True
        mock_console.print.assert_not_called()
        mock_confirm.assert_not_called()

    @patch("typer.confirm")
    @patch("ml_agents.cli.display.console")
    def test_display_cost_warning_over_threshold_confirmed(
        self, mock_console, mock_confirm
    ):
        """Test cost warning shown and confirmed for high costs."""
        mock_confirm.return_value = True

        result = display_cost_warning(estimated_cost=15.0, threshold=10.0)

        assert result is True
        assert mock_console.print.call_count >= 2  # Warning message + cost
        mock_confirm.assert_called_once_with("Do you want to continue?")

    @patch("typer.confirm")
    @patch("ml_agents.cli.display.console")
    def test_display_cost_warning_over_threshold_declined(
        self, mock_console, mock_confirm
    ):
        """Test cost warning shown and declined for high costs."""
        mock_confirm.return_value = False

        result = display_cost_warning(estimated_cost=25.0, threshold=10.0)

        assert result is False
        assert mock_console.print.call_count >= 3  # Warning + cost + cancelled message
        mock_confirm.assert_called_once_with("Do you want to continue?")


class TestApproachNameFormatting:
    """Test reasoning approach name formatting."""

    def test_format_approach_name_known_approaches(self):
        """Test formatting known reasoning approaches."""
        assert format_approach_name("None") == "Baseline (No Reasoning)"
        assert format_approach_name("ChainOfThought") == "Chain of Thought"
        assert format_approach_name("ProgramOfThought") == "Program of Thought"
        assert format_approach_name("AsPlanning") == "Reasoning as Planning"
        assert format_approach_name("Reflection") == "Reflection"
        assert format_approach_name("ChainOfVerification") == "Chain of Verification"
        assert format_approach_name("SkeletonOfThought") == "Skeleton of Thought"
        assert format_approach_name("TreeOfThought") == "Tree of Thought"

    def test_format_approach_name_unknown_approach(self):
        """Test formatting unknown reasoning approach returns as-is."""
        unknown_approach = "CustomReasoningApproach"
        assert format_approach_name(unknown_approach) == unknown_approach

    def test_format_approach_name_empty_string(self):
        """Test formatting empty string."""
        assert format_approach_name("") == ""

    def test_format_approach_name_case_sensitivity(self):
        """Test that approach name formatting is case-sensitive."""
        # Should not match due to case differences
        assert format_approach_name("chainofthought") == "chainofthought"
        assert format_approach_name("CHAINOFTHOUGHT") == "CHAINOFTHOUGHT"

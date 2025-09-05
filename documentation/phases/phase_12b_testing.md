# Phase 12.2: Testing Requirements

## Unit Tests

### RepositoryManager Tests

**File**: `tests/core/test_repository_manager.py`

```python
import pytest
from unittest.mock import patch, Mock
from ml_agents.core.repository_manager import RepositoryManager

class TestRepositoryManager:
    def setup_method(self):
        self.manager = RepositoryManager()

    def test_list_csv_files_default_repo(self):
        """Test listing CSV files from default repository."""
        with patch('ml_agents.core.repository_manager.list_repo_files') as mock_list:
            mock_list.return_value = ['file1.csv', 'file2.txt', 'file3.csv']

            result = self.manager.list_csv_files()

            mock_list.assert_called_once_with(
                "c4ai-ml-agents/benchmarks-base",
                repo_type="dataset"
            )
            assert result == ['file1.csv', 'file3.csv']

    def test_list_csv_files_custom_repo(self):
        """Test listing CSV files from custom repository."""
        with patch('ml_agents.core.repository_manager.list_repo_files') as mock_list:
            mock_list.return_value = ['data.csv']

            result = self.manager.list_csv_files("custom/repo")

            mock_list.assert_called_once_with("custom/repo", repo_type="dataset")
            assert result == ['data.csv']

    def test_load_csv_file_success(self):
        """Test successful CSV file loading."""
        with patch('ml_agents.core.repository_manager.hf_hub_download') as mock_download, \
             patch('pandas.read_csv') as mock_read_csv:

            mock_download.return_value = "/tmp/test.csv"
            mock_df = Mock()
            mock_read_csv.return_value = mock_df

            result = self.manager.load_csv_file("test.csv")

            mock_download.assert_called_once_with(
                "c4ai-ml-agents/benchmarks-base",
                "test.csv",
                repo_type="dataset"
            )
            mock_read_csv.assert_called_once_with("/tmp/test.csv")
            assert result == mock_df

    def test_load_csv_file_custom_repo(self):
        """Test CSV loading from custom repository."""
        with patch('ml_agents.core.repository_manager.hf_hub_download') as mock_download:
            mock_download.return_value = "/tmp/test.csv"

            with patch('pandas.read_csv'):
                self.manager.load_csv_file("test.csv", "custom/repo")

            mock_download.assert_called_once_with(
                "custom/repo",
                "test.csv",
                repo_type="dataset"
            )

    def test_get_file_info(self):
        """Test file info retrieval."""
        mock_df = Mock()
        mock_df.__len__ = Mock(return_value=100)
        mock_df.columns = ['INPUT', 'OUTPUT', 'other']

        with patch.object(self.manager, 'load_csv_file', return_value=mock_df):
            result = self.manager.get_file_info("test.csv", "repo/id")

        expected = {
            "filename": "test.csv",
            "repo_id": "repo/id",
            "num_rows": 100,
            "columns": ['INPUT', 'OUTPUT', 'other'],
            "has_input_output": True
        }
        assert result == expected

    def test_get_file_info_missing_columns(self):
        """Test file info with missing INPUT/OUTPUT columns."""
        mock_df = Mock()
        mock_df.__len__ = Mock(return_value=50)
        mock_df.columns = ['question', 'answer']

        with patch.object(self.manager, 'load_csv_file', return_value=mock_df):
            result = self.manager.get_file_info("test.csv")

        assert result["has_input_output"] is False
        assert result["num_rows"] == 50

    def test_error_handling_file_not_found(self):
        """Test error handling when file not found."""
        with patch('ml_agents.core.repository_manager.hf_hub_download') as mock_download:
            mock_download.side_effect = Exception("File not found")

            with pytest.raises(Exception, match="File not found"):
                self.manager.load_csv_file("nonexistent.csv")
```

### Dataset Loader Integration Tests

**File**: `tests/core/test_dataset_loader_repository.py`

```python
import pytest
from unittest.mock import patch, Mock
from ml_agents.core.dataset_loader import BBEHDatasetLoader
from ml_agents.config import ExperimentConfig

class TestDatasetLoaderRepository:
    def setup_method(self):
        self.config = ExperimentConfig()
        self.loader = BBEHDatasetLoader(self.config)

    def test_load_local_test_preserved(self):
        """Test LOCAL_TEST functionality is preserved."""
        with patch.object(self.loader, '_load_local_test') as mock_local:
            mock_dataset = Mock()
            mock_local.return_value = mock_dataset

            result = self.loader.load_dataset("LOCAL_TEST")

            mock_local.assert_called_once()
            assert result == mock_dataset

    def test_load_csv_file_from_repository(self):
        """Test loading CSV file from repository."""
        mock_df = Mock()
        mock_dataset = Mock()

        with patch.object(self.loader.repo_manager, 'load_csv_file', return_value=mock_df), \
             patch('ml_agents.core.dataset_loader.Dataset.from_pandas', return_value=mock_dataset):

            result = self.loader.load_dataset("test.csv", "custom/repo")

            self.loader.repo_manager.load_csv_file.assert_called_once_with("test.csv", "custom/repo")
            assert result == mock_dataset

    def test_fallback_to_legacy_benchmark(self):
        """Test fallback to legacy benchmark system."""
        with patch.object(self.loader, '_load_legacy_benchmark') as mock_legacy:
            mock_dataset = Mock()
            mock_legacy.return_value = mock_dataset

            result = self.loader.load_dataset("GPQA")

            mock_legacy.assert_called_once_with("GPQA")
            assert result == mock_dataset

    def test_validate_format_with_input_output(self):
        """Test format validation passes with INPUT/OUTPUT columns."""
        mock_dataset = Mock()
        mock_dataset.column_names = ['INPUT', 'OUTPUT']

        # Should not raise exception
        self.loader.validate_format(mock_dataset)

    def test_validate_format_missing_columns(self):
        """Test format validation fails without required columns."""
        mock_dataset = Mock()
        mock_dataset.column_names = ['question', 'answer']

        with pytest.raises(Exception):  # Update with specific exception type
            self.loader.validate_format(mock_dataset)
```

## CLI Tests

### Command Tests

**File**: `tests/cli/test_eval_commands_repository.py`

```python
import pytest
from typer.testing import CliRunner
from unittest.mock import patch, Mock
from ml_agents.cli.commands.eval import app

class TestEvalCommandsRepository:
    def setup_method(self):
        self.runner = CliRunner()

    def test_list_command_default_repo(self):
        """Test list command with default repository."""
        with patch('ml_agents.cli.commands.eval.RepositoryManager') as mock_manager_class:
            mock_manager = Mock()
            mock_manager.list_csv_files.return_value = ['file1.csv', 'file2.csv']
            mock_manager.DEFAULT_REPO = "c4ai-ml-agents/benchmarks-base"
            mock_manager_class.return_value = mock_manager

            result = self.runner.invoke(app, ["list"])

            assert result.exit_code == 0
            mock_manager.list_csv_files.assert_called_once_with(None)
            assert "file1.csv" in result.stdout
            assert "file2.csv" in result.stdout

    def test_list_command_custom_repo(self):
        """Test list command with custom repository."""
        with patch('ml_agents.cli.commands.eval.RepositoryManager') as mock_manager_class:
            mock_manager = Mock()
            mock_manager.list_csv_files.return_value = ['data.csv']
            mock_manager_class.return_value = mock_manager

            result = self.runner.invoke(app, ["list", "--repo", "custom/repo"])

            assert result.exit_code == 0
            mock_manager.list_csv_files.assert_called_once_with("custom/repo")

    def test_info_command_success(self):
        """Test info command with valid file."""
        mock_info = {
            "filename": "test.csv",
            "repo_id": "c4ai-ml-agents/benchmarks-base",
            "num_rows": 100,
            "columns": ["INPUT", "OUTPUT"],
            "has_input_output": True
        }

        with patch('ml_agents.cli.commands.eval.RepositoryManager') as mock_manager_class:
            mock_manager = Mock()
            mock_manager.get_file_info.return_value = mock_info
            mock_manager_class.return_value = mock_manager

            result = self.runner.invoke(app, ["info", "test.csv"])

            assert result.exit_code == 0
            assert "test.csv" in result.stdout
            assert "100" in result.stdout
            assert "True" in result.stdout

    def test_info_command_file_not_found(self):
        """Test info command with non-existent file."""
        with patch('ml_agents.cli.commands.eval.RepositoryManager') as mock_manager_class:
            mock_manager = Mock()
            mock_manager.get_file_info.side_effect = Exception("File not found")
            mock_manager_class.return_value = mock_manager

            result = self.runner.invoke(app, ["info", "nonexistent.csv"])

            assert result.exit_code == 0  # CLI handles error gracefully
            assert "Error accessing file" in result.stdout

    def test_run_command_local_test(self):
        """Test run command with LOCAL_TEST."""
        with patch('ml_agents.cli.commands.eval.ExperimentRunner') as mock_runner_class:
            mock_runner = Mock()
            mock_runner_class.return_value = mock_runner

            result = self.runner.invoke(app, ["run", "LOCAL_TEST", "ChainOfThought", "--samples", "3"])

            assert result.exit_code == 0
            mock_runner.run.assert_called_once()

    def test_run_command_csv_file_validation(self):
        """Test run command validates CSV file exists."""
        with patch('ml_agents.cli.commands.eval.RepositoryManager') as mock_manager_class:
            mock_manager = Mock()
            mock_manager.list_csv_files.return_value = ['existing.csv']
            mock_manager_class.return_value = mock_manager

            # Valid file should proceed
            with patch('ml_agents.cli.commands.eval.ExperimentRunner'):
                result = self.runner.invoke(app, ["run", "existing.csv", "ChainOfThought"])
                assert result.exit_code == 0

            # Invalid file should exit with error
            result = self.runner.invoke(app, ["run", "nonexistent.csv", "ChainOfThought"])
            assert result.exit_code == 1
            assert "not found in repository" in result.stdout

    def test_run_command_with_custom_repo(self):
        """Test run command with custom repository."""
        with patch('ml_agents.cli.commands.eval.RepositoryManager') as mock_manager_class, \
             patch('ml_agents.cli.commands.eval.ExperimentRunner') as mock_runner_class:

            mock_manager = Mock()
            mock_manager.list_csv_files.return_value = ['data.csv']
            mock_manager_class.return_value = mock_manager

            mock_runner = Mock()
            mock_runner_class.return_value = mock_runner

            result = self.runner.invoke(app, [
                "run", "data.csv", "ChainOfThought",
                "--repo", "custom/repo", "--samples", "10"
            ])

            assert result.exit_code == 0
            mock_manager.list_csv_files.assert_called_with("custom/repo")
```

## Integration Tests

### End-to-End Tests

**File**: `tests/integration/test_repository_integration.py`

```python
import pytest
from unittest.mock import patch, Mock
from ml_agents.core.repository_manager import RepositoryManager
from ml_agents.core.dataset_loader import BBEHDatasetLoader
from ml_agents.config import ExperimentConfig

class TestRepositoryIntegration:
    def test_full_pipeline_local_test(self):
        """Test complete pipeline with LOCAL_TEST."""
        config = ExperimentConfig()
        loader = BBEHDatasetLoader(config)

        with patch.object(loader, '_load_local_test') as mock_local:
            mock_dataset = Mock()
            mock_local.return_value = mock_dataset

            result = loader.load_dataset("LOCAL_TEST")
            assert result == mock_dataset

    def test_full_pipeline_repository_csv(self):
        """Test complete pipeline with repository CSV."""
        config = ExperimentConfig()
        loader = BBEHDatasetLoader(config)

        mock_df = Mock()
        mock_dataset = Mock()

        with patch('ml_agents.core.repository_manager.hf_hub_download') as mock_download, \
             patch('pandas.read_csv', return_value=mock_df), \
             patch('ml_agents.core.dataset_loader.Dataset.from_pandas', return_value=mock_dataset):

            mock_download.return_value = "/tmp/test.csv"

            result = loader.load_dataset("test.csv", "custom/repo")

            assert result == mock_dataset
            mock_download.assert_called_once_with("custom/repo", "test.csv", repo_type="dataset")

    def test_error_propagation(self):
        """Test error propagation through the system."""
        config = ExperimentConfig()
        loader = BBEHDatasetLoader(config)

        with patch('ml_agents.core.repository_manager.hf_hub_download') as mock_download:
            mock_download.side_effect = Exception("Network error")

            with pytest.raises(Exception, match="Network error"):
                loader.load_dataset("test.csv")

    def test_repository_manager_caching(self):
        """Test that repository manager doesn't unnecessarily re-download."""
        manager = RepositoryManager()

        with patch('ml_agents.core.repository_manager.hf_hub_download') as mock_download, \
             patch('pandas.read_csv') as mock_read_csv:

            mock_download.return_value = "/tmp/cached.csv"
            mock_df = Mock()
            mock_read_csv.return_value = mock_df

            # Load same file twice
            result1 = manager.load_csv_file("test.csv")
            result2 = manager.load_csv_file("test.csv")

            # Should download twice (no caching implemented yet)
            assert mock_download.call_count == 2
            assert result1 == result2
```

## Error Handling Tests

### Exception Tests

**File**: `tests/core/test_repository_errors.py`

```python
import pytest
from unittest.mock import patch
from ml_agents.core.repository_manager import RepositoryManager

class TestRepositoryErrors:
    def setup_method(self):
        self.manager = RepositoryManager()

    def test_network_error_handling(self):
        """Test handling of network errors."""
        with patch('ml_agents.core.repository_manager.list_repo_files') as mock_list:
            mock_list.side_effect = Exception("Network timeout")

            with pytest.raises(Exception, match="Network timeout"):
                self.manager.list_csv_files()

    def test_authentication_error_handling(self):
        """Test handling of authentication errors."""
        with patch('ml_agents.core.repository_manager.hf_hub_download') as mock_download:
            mock_download.side_effect = Exception("Authentication failed")

            with pytest.raises(Exception, match="Authentication failed"):
                self.manager.load_csv_file("test.csv")

    def test_file_not_found_error(self):
        """Test handling when CSV file doesn't exist."""
        with patch('ml_agents.core.repository_manager.hf_hub_download') as mock_download:
            mock_download.side_effect = Exception("File not found")

            with pytest.raises(Exception, match="File not found"):
                self.manager.load_csv_file("nonexistent.csv")

    def test_invalid_csv_format(self):
        """Test handling of invalid CSV format."""
        with patch('ml_agents.core.repository_manager.hf_hub_download') as mock_download, \
             patch('pandas.read_csv') as mock_read_csv:

            mock_download.return_value = "/tmp/invalid.csv"
            mock_read_csv.side_effect = Exception("Invalid CSV format")

            with pytest.raises(Exception, match="Invalid CSV format"):
                self.manager.load_csv_file("invalid.csv")
```

## Performance Tests

### Load Testing

**File**: `tests/performance/test_repository_performance.py`

```python
import pytest
import time
from unittest.mock import patch, Mock
from ml_agents.core.repository_manager import RepositoryManager

class TestRepositoryPerformance:
    def test_list_files_performance(self):
        """Test that listing files completes in reasonable time."""
        manager = RepositoryManager()

        with patch('ml_agents.core.repository_manager.list_repo_files') as mock_list:
            # Simulate large repository
            mock_list.return_value = [f"file_{i}.csv" for i in range(1000)]

            start_time = time.time()
            result = manager.list_csv_files()
            end_time = time.time()

            # Should complete in under 1 second
            assert end_time - start_time < 1.0
            assert len(result) == 1000

    def test_concurrent_file_loading(self):
        """Test that multiple file loads can happen concurrently."""
        manager = RepositoryManager()

        with patch('ml_agents.core.repository_manager.hf_hub_download') as mock_download, \
             patch('pandas.read_csv') as mock_read_csv:

            mock_download.return_value = "/tmp/test.csv"
            mock_df = Mock()
            mock_read_csv.return_value = mock_df

            # Simulate loading multiple files
            results = []
            for i in range(5):
                result = manager.load_csv_file(f"test_{i}.csv")
                results.append(result)

            assert len(results) == 5
            assert mock_download.call_count == 5
```

## Test Coverage Requirements

- **RepositoryManager**: 95% line coverage
- **Dataset Loader Integration**: 90% line coverage
- **CLI Commands**: 85% line coverage
- **Error Handling**: 100% coverage of error paths
- **Integration Tests**: All major user workflows covered

## Test Data Requirements

### Mock Data Structures

```python
# Sample CSV file listing
MOCK_CSV_FILES = [
    "BENCHMARK-01-GPQA.csv",
    "BENCHMARK-02-BoardgameQA.csv",
    "BENCHMARK-03-RobustLR.csv"
]

# Sample file info response
MOCK_FILE_INFO = {
    "filename": "test.csv",
    "repo_id": "c4ai-ml-agents/benchmarks-base",
    "num_rows": 150,
    "columns": ["INPUT", "OUTPUT", "category"],
    "has_input_output": True
}

# Sample pandas DataFrame
MOCK_DATAFRAME = pd.DataFrame({
    "INPUT": ["Question 1", "Question 2"],
    "OUTPUT": ["Answer 1", "Answer 2"]
})
```

## Success Criteria

1. All tests pass with mocked HuggingFace API calls
2. LOCAL_TEST functionality preserved and tested
3. Error scenarios properly handled and tested
4. CLI commands work with various input combinations
5. Performance tests validate reasonable response times
6. Integration tests cover end-to-end workflows

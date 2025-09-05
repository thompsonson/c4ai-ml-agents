# Phase 12.2: Flexible Benchmark Repository System

## Purpose

Replace broken Phase 12 implementation with a working system that:

- Lists CSV files from HuggingFace dataset repositories
- Supports multiple repositories (default: `c4ai-ml-agents/benchmarks-base`)
- Uses actual HuggingFace API (`hf_hub_download`, `list_repo_files`)
- Preserves LOCAL_TEST functionality

## Core Problems to Fix

1. **Phase 12 is completely broken**: Uses wrong HuggingFace API, wrong filenames, breaks LOCAL_TEST
2. **Command fails**: `ml-agents eval run LOCAL_TEST ChainOfThought` crashes
3. **Wrong design**: Assumes `GPQA.csv` when files are `BENCHMARK-01-GPQA.csv`

## Implementation Plan

### 1. Create Flexible Repository Manager

**File**: `src/ml_agents/core/repository_manager.py`

```python
from huggingface_hub import hf_hub_download, list_repo_files
import pandas as pd

class RepositoryManager:
    """Manages CSV file access from HuggingFace dataset repositories."""

    DEFAULT_REPO = "c4ai-ml-agents/benchmarks-base"

    def list_csv_files(self, repo_id: str = None) -> List[str]:
        """List all CSV files in repository."""
        repo_id = repo_id or self.DEFAULT_REPO
        files = list_repo_files(repo_id, repo_type="dataset")
        return [f for f in files if f.endswith('.csv')]

    def load_csv_file(self, filename: str, repo_id: str = None) -> pd.DataFrame:
        """Download and load specific CSV file."""
        repo_id = repo_id or self.DEFAULT_REPO
        file_path = hf_hub_download(repo_id, filename, repo_type="dataset")
        return pd.read_csv(file_path)

    def get_file_info(self, filename: str, repo_id: str = None) -> dict:
        """Get basic info about CSV file."""
        df = self.load_csv_file(filename, repo_id)
        return {
            "filename": filename,
            "repo_id": repo_id or self.DEFAULT_REPO,
            "num_rows": len(df),
            "columns": list(df.columns),
            "has_input_output": "INPUT" in df.columns and "OUTPUT" in df.columns
        }
```

### 2. Update Dataset Loader Integration

**File**: `src/ml_agents/core/dataset_loader.py`

```python
class BBEHDatasetLoader:
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.repo_manager = RepositoryManager()

    def load_dataset(self, identifier: str, repo_id: str = None) -> Dataset:
        """Load dataset - handles LOCAL_TEST and repository CSV files."""

        # Handle LOCAL_TEST (preserve existing functionality)
        if identifier == "LOCAL_TEST":
            return self._load_local_test()

        # Handle repository CSV files
        if identifier.endswith('.csv'):
            df = self.repo_manager.load_csv_file(identifier, repo_id)
            # Convert pandas DataFrame to HuggingFace Dataset
            return Dataset.from_pandas(df)

        # Fallback to old benchmark registry (deprecated)
        return self._load_legacy_benchmark(identifier)
```

### 3. New CLI Commands

**File**: `src/ml_agents/cli/commands/eval.py`

```python
@app.command("list")
def list_csv_files(
    repo_id: str = typer.Option(None, "--repo", help="Repository ID (default: c4ai-ml-agents/benchmarks-base)")
):
    """List available CSV files in repository."""
    manager = RepositoryManager()
    csv_files = manager.list_csv_files(repo_id)

    # Display table
    table = Table(title=f"CSV Files in {repo_id or manager.DEFAULT_REPO}")
    table.add_column("Filename")
    table.add_column("Size")

    for filename in sorted(csv_files):
        table.add_row(filename, "CSV")

    console.print(table)

@app.command("info")
def file_info(
    filename: str = typer.Argument(..., help="CSV filename"),
    repo_id: str = typer.Option(None, "--repo", help="Repository ID")
):
    """Show information about specific CSV file."""
    manager = RepositoryManager()
    try:
        info = manager.get_file_info(filename, repo_id)
        console.print(f"File: {info['filename']}")
        console.print(f"Repository: {info['repo_id']}")
        console.print(f"Rows: {info['num_rows']}")
        console.print(f"Columns: {', '.join(info['columns'])}")
        console.print(f"Has INPUT/OUTPUT: {info['has_input_output']}")
    except Exception as e:
        display_error(f"Error accessing file '{filename}': {e}")

@app.command("run")
def run_single_experiment(
    identifier: str = typer.Argument(..., help="LOCAL_TEST or CSV filename (e.g., BENCHMARK-01-GPQA.csv)"),
    approach: str = typer.Argument(..., help="Reasoning approach"),
    repo_id: str = typer.Option(None, "--repo", help="Repository ID for CSV files"),
    samples: int = typer.Option(None, "--samples", "-n"),
    # ... other options
):
    """Run experiment with LOCAL_TEST or repository CSV file."""

    # Validate file exists (except LOCAL_TEST)
    if identifier != "LOCAL_TEST":
        manager = RepositoryManager()
        csv_files = manager.list_csv_files(repo_id)
        if identifier not in csv_files:
            display_error(f"File '{identifier}' not found in repository")
            raise typer.Exit(1)

    # Create config and run experiment
    config = ExperimentConfig(
        dataset_identifier=identifier,
        repository_id=repo_id,
        approaches=[approach],
        sample_count=samples
    )

    runner = ExperimentRunner(config)
    runner.run()
```

## Usage Examples

```bash
# List CSV files in default repository
ml-agents eval list

# List CSV files in specific repository
ml-agents eval list --repo c4ai-ml-agents/aqua_rat

# Get info about specific file
ml-agents eval info BENCHMARK-01-GPQA.csv
ml-agents eval info data.csv --repo c4ai-ml-agents/aqua_rat

# Run experiments
ml-agents eval run LOCAL_TEST ChainOfThought --samples 3
ml-agents eval run BENCHMARK-01-GPQA.csv ChainOfThought --samples 50
ml-agents eval run data.csv TreeOfThought --repo c4ai-ml-agents/aqua_rat
```

## Migration Steps

### Step 1: Implement Core Classes (1 hour)

- Create `RepositoryManager` class
- Add basic CSV listing and loading functionality
- Add error handling

### Step 2: Update Dataset Loader (30 minutes)

- Integrate `RepositoryManager` with `BBEHDatasetLoader`
- Ensure LOCAL_TEST still works
- Add CSV file loading path

### Step 3: Update CLI Commands (1 hour)

- Replace broken `run` command validation
- Add `list` and `info` commands
- Update help text and examples

### Step 4: Fix Existing Issues (30 minutes)

- Test that `ml-agents eval run LOCAL_TEST ChainOfThought` works
- Verify repository CSV loading works
- Update configuration classes if needed

### Step 5: Remove Broken Code (15 minutes)

- Remove or deprecate `BenchmarkRegistry` class
- Clean up Phase 12 broken implementation
- Update documentation

## Testing Requirements

```python
def test_repository_manager():
    """Test CSV file listing and loading."""
    manager = RepositoryManager()

    # Test listing files
    files = manager.list_csv_files()
    assert isinstance(files, list)
    assert all(f.endswith('.csv') for f in files)

    # Test loading file (if available)
    if files:
        df = manager.load_csv_file(files[0])
        assert isinstance(df, pd.DataFrame)

def test_local_test_preserved():
    """Ensure LOCAL_TEST still works."""
    loader = BBEHDatasetLoader(config)
    dataset = loader.load_dataset("LOCAL_TEST")
    assert dataset is not None

def test_cli_commands():
    """Test new CLI commands work."""
    # Test list command doesn't crash
    # Test info command with valid file
    # Test run command with LOCAL_TEST
```

## Success Criteria

1. **LOCAL_TEST works**: `ml-agents eval run LOCAL_TEST ChainOfThought` succeeds
2. **Repository listing works**: `ml-agents eval list` shows CSV files
3. **File loading works**: Can load and run experiments with repository CSV files
4. **Flexible repositories**: Can specify different HuggingFace dataset repositories
5. **Backward compatibility**: Existing LOCAL_TEST functionality preserved

## Files to Modify

- `src/ml_agents/core/repository_manager.py` (create)
- `src/ml_agents/core/dataset_loader.py` (update integration)
- `src/ml_agents/cli/commands/eval.py` (fix validation, add commands)
- `src/ml_agents/config.py` (add repository_id field if needed)

## Key Differences from Phase 12

- Uses working HuggingFace API (`hf_hub_download`, `list_repo_files`)
- No assumptions about filenames
- Preserves LOCAL_TEST functionality
- Supports multiple repositories
- Actually works with the repository structure you have

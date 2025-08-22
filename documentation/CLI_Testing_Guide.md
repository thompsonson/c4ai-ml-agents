# CLI Testing Guide

This document provides examples and guidance for testing the ML Agents CLI interface.

## Overview

The ML Agents CLI has comprehensive test coverage across all components:

- **CLI Component Tests**: `tests/cli/test_*.py`
- **Integration Tests**: Command execution with mocked ExperimentRunner
- **Configuration Tests**: YAML/JSON loading and validation
- **Error Handling Tests**: Invalid parameters and edge cases

## Running CLI Tests

### All CLI Tests
```bash
# Run all CLI-specific tests
pytest tests/cli/ -v

# Run with coverage
pytest tests/cli/ --cov=ml_agents.cli --cov-report=term-missing
```

### Specific Test Categories
```bash
# Configuration loading and validation
pytest tests/cli/test_config_loader.py -v

# Parameter validation and error handling
pytest tests/cli/test_validators.py -v

# Rich display formatting and output
pytest tests/cli/test_display.py -v

# Command integration with ExperimentRunner
pytest tests/cli/test_commands.py -v

# Basic CLI functionality
pytest tests/cli/test_main.py -v
```

## Test Structure

### 1. Component Tests (`test_config_loader.py`)
Tests for configuration loading and validation:

```python
def test_load_yaml_config():
    """Test loading YAML configuration file."""

def test_flatten_nested_config():
    """Test flattening nested configuration structures."""

def test_merge_config_sources():
    """Test CLI args override config file values."""
```

**Coverage**:
- YAML/JSON file loading
- Nested configuration flattening
- Configuration hierarchy (CLI → file → defaults)
- Pydantic validation with error messages
- Template generation

### 2. Validation Tests (`test_validators.py`)
Tests for parameter validation and error handling:

```python
def test_validate_reasoning_approach():
    """Test reasoning approach validation with available approaches."""

def test_validate_temperature_invalid():
    """Test temperature validation fails for values outside 0.0-2.0."""

def test_validate_output_directory():
    """Test output directory creation and write permissions."""
```

**Coverage**:
- All numeric parameter validation (temperature, top_p, max_tokens, etc.)
- File path validation (configs, checkpoints, output directories)
- API key availability checking
- Provider/model combination validation
- User confirmation flows for high-cost operations

### 3. Display Tests (`test_display.py`)
Tests for Rich output formatting and display:

```python
def test_create_experiment_table():
    """Test Rich table creation with experiment results."""

def test_display_cost_warning():
    """Test cost warning display and user confirmation."""

def test_format_approach_name():
    """Test reasoning approach name formatting for display."""
```

**Coverage**:
- Rich table formatting for experiment results
- Error, warning, success message display
- Cost summary tables with provider breakdowns
- Progress display creation
- Approach name formatting and display

### 4. Integration Tests (`test_commands.py`)
Tests for full command execution with mocked dependencies:

```python
def test_run_command_with_basic_args():
    """Test run command execution with ExperimentRunner integration."""

def test_compare_command_parallel():
    """Test compare command with parallel execution."""

def test_resume_command_valid_checkpoint():
    """Test resume command with checkpoint loading."""
```

**Coverage**:
- All CLI commands (run, compare, resume, list-checkpoints)
- Configuration precedence testing
- Error scenarios and exit codes
- Keyboard interrupt handling
- Checkpoint file validation and loading

## Manual Testing Examples

### Basic Commands
```bash
# Test help system
ml-agents --help
ml-agents run --help
ml-agents compare --help

# Test version information
ml-agents --version
ml-agents version

# Test environment validation
ml-agents validate-env

# List available approaches
ml-agents list-approaches
```

### Configuration Testing
```bash
# Create example config
cat > test_config.yaml << EOF
experiment:
  name: "test_experiment"
  sample_count: 10

model:
  provider: "openrouter"
  name: "gpt-3.5-turbo"
  temperature: 0.5

reasoning:
  approaches: ["ChainOfThought", "AsPlanning"]
EOF

# Test config loading
ml-agents run --config test_config.yaml --samples 5  # CLI override
```

### Error Scenario Testing
```bash
# Invalid reasoning approach
ml-agents run --approach "InvalidApproach"

# Invalid parameter values
ml-agents run --temperature 3.0  # Should fail: > 2.0
ml-agents run --samples 0        # Should fail: must be >= 1
ml-agents run --top-p 1.5        # Should fail: must be <= 1.0

# Missing config file
ml-agents run --config "nonexistent.yaml"

# Invalid checkpoint
ml-agents resume "nonexistent.json"
```

### Parallel Execution Testing
```bash
# Test parallel comparison
ml-agents compare \
  --approaches "ChainOfThought,AsPlanning,TreeOfThought" \
  --samples 5 \
  --parallel \
  --max-workers 3
```

## Test Coverage Metrics

Current CLI test coverage:

- **config_loader.py**: ~95% coverage (18 test methods)
- **validators.py**: ~95% coverage (25 test methods)
- **display.py**: ~90% coverage (15 test methods)
- **commands.py**: ~85% coverage (20 test methods)
- **main.py**: ~85% coverage (8 test methods)

**Total CLI Tests**: 86 test methods across 5 test files

## Common Test Patterns

### Mocking ExperimentRunner
```python
@patch('ml_agents.cli.commands.ExperimentRunner')
def test_run_command(self, mock_experiment_runner):
    mock_runner_instance = Mock()
    mock_result = Mock()
    mock_result.total_samples = 10
    mock_runner_instance.run_single_experiment.return_value = mock_result
    mock_experiment_runner.return_value = mock_runner_instance

    result = self.runner.invoke(app, ["run", "--samples", "10"])
    assert result.exit_code == 0
```

### Testing Configuration Files
```python
def test_config_file_loading(self):
    config_data = {"sample_count": 25, "provider": "anthropic"}

    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(config_data, f)
        config_path = f.name

    try:
        result = load_config_file(config_path)
        assert result["sample_count"] == 25
    finally:
        Path(config_path).unlink()
```

### Testing Rich Output
```python
@patch('ml_agents.cli.display.console')
def test_display_error(self, mock_console):
    display_error("Test error message")

    mock_console.print.assert_called_once()
    args, kwargs = mock_console.print.call_args
    assert "❌" in args[0]
    assert "Error:" in args[0]
```

## Continuous Integration

The CLI tests are integrated into the project's CI pipeline:

```bash
# In GitHub Actions / CI
pytest tests/cli/ --cov=ml_agents.cli --cov-fail-under=85
```

## Debugging Failed Tests

### Common Issues
1. **Missing mocks**: Ensure ExperimentRunner and environment checks are mocked
2. **File cleanup**: Use temporary files with proper cleanup in finally blocks
3. **Console output**: Mock Rich console for display tests
4. **Configuration precedence**: Verify CLI args properly override config files

### Debug Commands
```bash
# Run single test with verbose output
pytest tests/cli/test_commands.py::TestRunCommand::test_run_command_with_basic_args -v -s

# Run tests with pdb on failure
pytest tests/cli/ --pdb

# Show test coverage gaps
pytest tests/cli/ --cov=ml_agents.cli --cov-report=html
open htmlcov/index.html
```

## Integration with Development Workflow

The CLI tests ensure:

- ✅ All CLI commands work correctly with ExperimentRunner
- ✅ Configuration loading handles all supported formats
- ✅ Parameter validation prevents invalid experiments
- ✅ Error messages are clear and actionable
- ✅ Rich formatting displays correctly across terminals
- ✅ Checkpoint functionality works for experiment resumption

This comprehensive test suite ensures the CLI interface is production-ready for research use.

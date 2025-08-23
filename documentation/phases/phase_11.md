# Phase 11: CLI Command Stabilization & Production Readiness

## Strategic Context

**Purpose**: Finalize CLI command refactoring and establish clear boundaries between stable production-ready commands and experimental pre-alpha features.

**Integration Point**: Complete the CLI restructuring begun in previous phases, ensuring reliable user experience for stable features while clearly marking experimental functionality.

**Timeline**: 6-8 hours total implementation

## Phase 11 Strategic Decisions

### **Primary Implementation: Command Maturity Classification**

**Decision**: Classify commands into **Stable** (production-ready) and **Pre-Alpha** (experimental) categories with clear user warnings.

**Rationale**:

- Provides reliable experience for researchers using stable features
- Manages expectations for experimental functionality
- Enables iterative development of complex features
- Maintains professional CLI standards for production use

### **Command Classification**

**Stable Commands** (Production Ready):

- `setup` - Environment validation and configuration
- `db` - Database management and maintenance
- `preprocess` - Dataset preprocessing and transformation

**Pre-Alpha Commands** (Experimental):

- `eval` - Experiment execution and reasoning evaluation
- `results` - Results analysis and export

## Technical Implementation

### **Phase 11.1: Pre-Alpha Warning System (1.5 hours)**

**Implementation Strategy**: Add prominent warnings to experimental command groups.

**Warning Message Standards**:

```python
# src/ml_agents/cli/commands/eval.py
def display_pre_alpha_warning():
    """Display pre-alpha warning for evaluation commands."""
    from ml_agents.cli.display import display_warning

    console.print("\n⚠️  [bold yellow]PRE-ALPHA WARNING[/bold yellow]")
    console.print("[yellow]The 'eval' command group is in pre-alpha development.[/yellow]")
    console.print("[yellow]Features may be incomplete, unstable, or subject to breaking changes.[/yellow]")
    console.print("[yellow]For production use, consider using the stable preprocessing and database commands.[/yellow]")
    console.print("[dim]Use --skip-warnings to suppress this message.[/dim]\n")
```

**Integration Points**:

- Add warning display to all eval command entry points
- Add warning display to all results command entry points
- Implement `--skip-warnings` flag for automated scripts
- Include warnings in help text for experimental commands

**Example Integration**:

```python
def run_single_experiment(
    # ... existing parameters ...
    skip_warnings: bool = typer.Option(
        False, "--skip-warnings", help="Skip pre-alpha warnings"
    ),
) -> None:
    """Run a single reasoning experiment with the specified approach.

    ⚠️  PRE-ALPHA: This command is experimental and may be unstable.
    """
    if not skip_warnings:
        display_pre_alpha_warning()

    # ... rest of implementation
```

### **Phase 11.2: Stable Command Test Coverage Enhancement (2 hours)**

**Target Coverage Goals**:

- `setup.py`: From 23% to 80%+ coverage
- `db.py`: From 11% to 80%+ coverage
- `preprocess.py`: From 7% to 80%+ coverage

**Test Categories to Add**:

**Setup Command Tests**:

```python
# tests/cli/test_setup_commands.py
class TestSetupCommands:
    def test_validate_env_success(self):
        """Test successful environment validation."""

    def test_validate_env_missing_api_keys(self):
        """Test validation with missing API keys."""

    def test_list_approaches_complete(self):
        """Test that all approaches are listed."""

    def test_version_command_format(self):
        """Test version command output format."""
```

**Database Command Tests**:

```python
# tests/cli/test_db_commands.py
class TestDatabaseCommands:
    def test_db_init_new_database(self):
        """Test initializing a new database."""

    def test_db_init_existing_database(self):
        """Test initializing when database exists."""

    def test_db_backup_success(self):
        """Test successful database backup."""

    def test_db_stats_display(self):
        """Test database statistics display."""

    def test_db_migrate_schema_update(self):
        """Test schema migration process."""
```

**Preprocessing Command Tests**:

```python
# tests/cli/test_preprocess_commands.py
class TestPreprocessCommands:
    def test_preprocess_list_unprocessed(self):
        """Test listing unprocessed datasets."""

    def test_preprocess_inspect_dataset(self):
        """Test dataset schema inspection."""

    def test_preprocess_generate_rules(self):
        """Test transformation rule generation."""

    def test_preprocess_transform_apply(self):
        """Test applying transformations."""

    def test_preprocess_batch_processing(self):
        """Test batch processing workflow."""
```

### **Phase 11.3: CLI Argument Handling Consistency (1 hour)**

**Standardization Areas**:

**Error Message Formats**:

```python
# Standard error message patterns
def validate_sample_count(value: int) -> int:
    if value < 1:
        raise typer.BadParameter("Sample count must be at least 1")
    if value > 10000:
        raise typer.BadParameter("Sample count cannot exceed 10,000")
    return value

def validate_confidence_threshold(value: float) -> float:
    if not 0.0 <= value <= 1.0:
        raise typer.BadParameter("Confidence threshold must be between 0.0 and 1.0")
    return value
```

**Option Naming Conventions**:

- Use consistent short flags: `-n` for samples, `-o` for output, `-c` for config
- Use consistent long flags: `--sample-count` not `--samples`, `--output-dir` not `--output`
- Use consistent boolean flags: `--verbose/--quiet`, `--enable/--disable`

**Help Text Standards**:

```python
sample_count: int = typer.Option(
    50,
    "--samples", "-n",
    help="Number of samples to process (1-10,000)",
    callback=validate_sample_count
)
```

### **Phase 11.4: Integration Smoke Tests (1 hour)**

**Smoke Test Categories**:

**Environment Validation Smoke Test**:

```python
def test_setup_validate_env_integration():
    """Integration test for environment validation."""
    # Test with minimal valid environment
    # Verify error handling for missing dependencies
    # Test API key validation (mocked)
```

**Database Lifecycle Smoke Test**:

```python
def test_db_lifecycle_integration():
    """Integration test for database operations."""
    # Create temporary database
    # Initialize schema
    # Insert test data
    # Create backup
    # Verify integrity
    # Cleanup
```

**Preprocessing Pipeline Smoke Test**:

```python
def test_preprocess_pipeline_integration():
    """Integration test for preprocessing workflow."""
    # Use small test dataset
    # Inspect schema
    # Generate rules
    # Apply transformation
    # Validate output format
```

### **Phase 11.5: Help Text Formatting Consistency (45 minutes)**

**Help Text Standards**:

**Command Descriptions**:

```python
# Format: Brief description (action + object)
"Initialize database for storing experiment results."  # Good
"Creates and initializes the database."              # Avoid passive voice

# Include status indicators for command maturity
"⚠️  PRE-ALPHA: Run reasoning evaluation experiments."  # Experimental
"Initialize database for storing experiment results."    # Stable
```

**Option Descriptions**:

```python
# Include value ranges and examples
help="Sample count (1-10,000, default: 50)"
help="Output directory (default: ./outputs)"
help="Temperature value (0.0-2.0, default: 1.0)"
```

**Example Sections**:

```python
# Add usage examples to command docstrings
"""Initialize database for storing experiment results.

Examples:
    ml-agents db init
    ml-agents db init --db-path ./custom.db --force
    ml-agents db init --help
"""
```

### **Phase 11.6: Error Handling Audit (1 hour)**

**Exit Code Standards**:

```python
# Standard exit codes across all commands
SUCCESS = 0           # Operation completed successfully
GENERAL_ERROR = 1     # General command failure
INVALID_USAGE = 2     # Invalid arguments/options
USER_INTERRUPT = 130  # Ctrl+C interruption
```

**Exception Handling Patterns**:

```python
def command_wrapper(func):
    """Standard exception handling wrapper for all commands."""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except KeyboardInterrupt:
            display_warning("Operation interrupted by user")
            raise typer.Exit(130)
        except typer.BadParameter as e:
            display_error(f"Invalid parameter: {e}")
            raise typer.Exit(2)
        except FileNotFoundError as e:
            display_error(f"File not found: {e}")
            raise typer.Exit(1)
        except Exception as e:
            display_error(f"Unexpected error: {e}")
            if verbose:
                console.print_exception()
            raise typer.Exit(1)
    return wrapper
```

**Logging Standards**:

```python
# Consistent logging across commands
import logging
logger = logging.getLogger(__name__)

def db_init():
    logger.info("Starting database initialization")
    try:
        # ... operation
        logger.info("Database initialization completed successfully")
        display_success("Database initialized successfully")
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        display_error(f"Failed to initialize database: {e}")
        raise typer.Exit(1)
```

### **Phase 11.7: Documentation Updates (1 hour)**

**User Documentation Updates**:

**README.md CLI Section**:

```markdown
## Command Line Interface

### Stable Commands (Production Ready)

#### Setup & Environment
```bash
# Validate environment configuration
ml-agents setup validate-env

# List available reasoning approaches
ml-agents setup list-approaches

# Show version information
ml-agents setup version
```

#### Database Management

```bash
# Initialize results database
ml-agents db init --db-path ./results.db

# Create database backup
ml-agents db backup --source ./results.db

# Show database statistics
ml-agents db stats

# Migrate database schema
ml-agents db migrate
```

#### Dataset Preprocessing

```bash
# List unprocessed datasets
ml-agents preprocess list

# Inspect dataset schema
ml-agents preprocess inspect MilaWang/SpatialEval

# Generate transformation rules
ml-agents preprocess generate-rules MilaWang/SpatialEval

# Apply transformations
ml-agents preprocess transform MilaWang/SpatialEval rules.json

# Batch process multiple datasets
ml-agents preprocess batch --max 5

# Upload to HuggingFace Hub
ml-agents preprocess upload dataset.json --source-dataset MilaWang/SpatialEval
```

### Experimental Commands (Pre-Alpha)

⚠️  **Warning**: The following commands are in pre-alpha development and may be unstable:

#### Evaluation (Experimental)

```bash
# Single reasoning experiment (PRE-ALPHA)
ml-agents eval run --approach ChainOfThought --samples 50

# Compare multiple approaches (PRE-ALPHA)
ml-agents eval compare --approaches "CoT,Reflection" --samples 100

# Resume from checkpoint (PRE-ALPHA)
ml-agents eval resume checkpoint_001.json

# List available checkpoints (PRE-ALPHA)
ml-agents eval checkpoints
```

#### Results Analysis (Experimental)

```bash
# Export experiment results (PRE-ALPHA)
ml-agents results export EXP_001 --format csv

# Analyze experiment patterns (PRE-ALPHA)
ml-agents results analyze EXP_001 --type accuracy

# Compare experiments (PRE-ALPHA)
ml-agents results compare "EXP_001,EXP_002,EXP_003"

# List experiments (PRE-ALPHA)
ml-agents results list --status completed
```

```

**Developer Documentation Updates**:

**CONTRIBUTING.md CLI Section**:
```markdown
## CLI Development Guidelines

### Command Classification

Commands are classified into two maturity levels:

- **Stable**: Production-ready, full test coverage, stable API
- **Pre-Alpha**: Experimental, may have breaking changes

### Adding New Commands

1. Determine command maturity level
2. Add to appropriate command module (`setup.py`, `db.py`, `preprocess.py` for stable)
3. Follow error handling patterns
4. Add comprehensive tests (80%+ coverage for stable commands)
5. Update documentation

### Testing Requirements

- Stable commands: 80%+ test coverage required
- Pre-alpha commands: Basic functionality tests acceptable
- All commands: Integration smoke tests

### Error Handling Standards

- Use consistent exit codes (0, 1, 2, 130)
- Implement standard exception handling patterns
- Provide helpful error messages with suggested fixes
- Log errors appropriately
```

## Success Criteria

**Warning System Success**:

- Pre-alpha commands display clear warnings
- Users can skip warnings for automation (`--skip-warnings`)
- Help text clearly indicates command maturity
- No confusion between stable and experimental features

**Test Coverage Success**:

- Setup commands: 80%+ coverage
- Database commands: 80%+ coverage
- Preprocessing commands: 80%+ coverage
- Integration smoke tests pass consistently
- All stable commands have comprehensive error case testing

**Consistency Success**:

- Uniform error message formats across all commands
- Consistent option naming and help text
- Standardized exit codes and exception handling
- Professional CLI experience for stable commands

**Documentation Success**:

- Clear separation of stable vs experimental commands in docs
- Complete usage examples for all stable commands
- Developer guidelines for CLI contributions
- User expectations properly set for feature maturity

**User Experience Success**:

- Stable commands work reliably for production use
- Clear warnings prevent accidental use of experimental features
- Helpful error messages guide users to solutions
- Consistent behavior across all command interactions
- Professional CLI standards maintained throughout

This phase establishes ML Agents as a professional research tool with clearly defined stable features while maintaining space for experimental development.

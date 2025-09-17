# CLI Interface Design

**Version:** 1.0
**Date:** 2025-09-17
**Purpose:** Define command structure, options, and user interaction patterns

## Command Hierarchy

```
ml-agents
├── evaluate
│   ├── create    # Create new evaluation
│   ├── run       # Execute evaluation
│   └── list      # List evaluations
├── benchmark
│   ├── list      # List available benchmarks
│   └── show      # Show benchmark details
└── health        # System health check
```

## Core Commands

### ml-agents evaluate create

Create new evaluation configuration.

```bash
ml-agents evaluate create --agent <type> --model <model> --benchmark <name> [OPTIONS]
```

**Required Arguments:**

- `--agent {none,cot}` - Reasoning approach
- `--model <model_id>` - Model identifier (e.g., anthropic/claude-3-sonnet)
- `--benchmark <name>` - Benchmark short name (e.g., GPQA, FOLIO)

**Optional Arguments:**

- `--temp <float>` - Temperature 0.0-2.0 (default: 1.0)
- `--max-tokens <int>` - Maximum output tokens (default: 1000)
- `--help` - Show command help

**Output:**

```
✓ Created evaluation eval_456 (pending)
  Agent: cot
  Model: anthropic/claude-3-sonnet
  Benchmark: GPQA (150 questions)
  Run with: ml-agents evaluate run eval_456
```

**Errors:**

```
✗ Error: Invalid agent type 'xyz'. Available: none, cot
✗ Error: Benchmark 'UNKNOWN' not found. Use 'ml-agents benchmark list'
✗ Error: Model 'invalid-model' not available
```

### ml-agents evaluate run

Execute evaluation with real-time progress.

```bash
ml-agents evaluate run <evaluation_id>
```

**Required Arguments:**

- `<evaluation_id>` - Evaluation identifier

**Output (Real-time):**

```
Running evaluation eval_456...
Progress: 42/150 (28%) - Processing question 42...
Warning: Question 67 failed (parsing_error) - continuing...
Progress: 148/150 (98%) - Processing question 148...
✓ Completed: 148/150 correct (98.7%)
  Total time: 12m 34s
  Avg time per question: 5.0s
  Token usage: 245,680 tokens
```

**Interruption (Ctrl+C):**

```
^C
⚠ Evaluation interrupted by user
  Partial results: 89/150 questions completed
  Status: interrupted
  Resume with: ml-agents evaluate run eval_456
```

**Errors:**

```
✗ Error: Evaluation eval_456 not found
✗ Error: Evaluation already completed
✗ Error: OpenRouter API unavailable. Check connection and API key
```

### ml-agents evaluate list

List evaluations with filtering.

```bash
ml-agents evaluate list [OPTIONS]
```

**Optional Arguments:**

- `--status {pending,running,completed,failed,interrupted}` - Filter by status
- `--benchmark <name>` - Filter by benchmark
- `--help` - Show command help

**Output:**

```
ID        Status      Agent  Model             Benchmark  Accuracy  Created
eval_456  completed   cot    claude-3-sonnet   GPQA       98.7%     2 hours ago
eval_455  failed      none   gpt-4             FOLIO      -         1 day ago
eval_454  running     cot    claude-3-sonnet   BBEH       42.3%     5 minutes ago
```

### ml-agents benchmark list

List available benchmarks.

```bash
ml-agents benchmark list
```

**Output:**

```
Name                Questions  Description
GPQA                150        Graduate-level scientific reasoning
FOLIO               204        First-order logic inference
BBEH                2847       BIG-Bench Hard tasks
MATH3               617        Mathematical problem solving (Level 3)
LeetCode_Python_Easy 735       Programming problems in Python
```

### ml-agents benchmark show

Show benchmark details.

```bash
ml-agents benchmark show <name>
```

**Required Arguments:**

- `<name>` - Benchmark short name

**Output:**

```
Benchmark: GPQA
Description: Graduate-level scientific reasoning questions
Questions: 150
Categories: Physics (45), Chemistry (52), Biology (53)
Avg question length: 245 characters
Created: 15 days ago
```

### ml-agents health

System health and connectivity check.

```bash
ml-agents health
```

**Output:**

```
System Health: ✓ Healthy
┌─────────────┬────────┬──────────────┐
│ Component   │ Status │ Details      │
├─────────────┼────────┼──────────────┤
│ Database    │ ✓      │ Connected    │
│ OpenRouter  │ ✓      │ API key valid│
│ Benchmarks  │ ✓      │ 62 available │
└─────────────┴────────┴──────────────┘
```

## Global Options

Available for all commands:

- `--help, -h` - Show help
- `--version, -v` - Show version
- `--config <path>` - Custom config file path
- `--verbose` - Verbose output
- `--quiet` - Suppress non-essential output

## Output Formatting

### Progress Indicators

- **Spinner**: For quick operations (< 2s)
- **Progress bar**: For evaluation execution with percentage and ETA
- **Status updates**: Real-time question processing feedback

### Color Coding

- **Green (✓)**: Success, completed operations
- **Yellow (⚠)**: Warnings, non-fatal issues
- **Red (✗)**: Errors, failed operations
- **Blue (ℹ)**: Informational messages
- **Gray**: Secondary information

### Tables

- **Aligned columns**: Consistent spacing
- **Headers**: Bold or underlined
- **Sorting**: Most recent first for lists
- **Truncation**: Long values abbreviated with "..."

## Error Handling

### Error Categories

1. **Configuration Errors**

   - Invalid agent types, models, or parameters
   - Missing required arguments
   - Malformed configuration values

2. **Resource Errors**

   - Benchmark not found
   - Evaluation not found
   - API connectivity issues

3. **Runtime Errors**
   - Authentication failures
   - Network timeouts
   - System interruption

### Error Message Format

```
✗ Error: <concise description>
  <detailed explanation if helpful>
  <suggested action>
```

Examples:

```
✗ Error: Invalid temperature value '5.0'
  Temperature must be between 0.0 and 2.0
  Use: --temp 1.0

✗ Error: OpenRouter API key not configured
  Set OPENROUTER_API_KEY environment variable
  Get your key from: https://openrouter.ai/keys
```

### Exit Codes

- `0` - Success
- `1` - General error
- `2` - Configuration error
- `3` - Resource not found
- `4` - Authentication error
- `130` - User interruption (Ctrl+C)

## User Experience Patterns

### Command Discovery

- `ml-agents --help` shows all top-level commands
- `ml-agents <command> --help` shows command-specific help
- Error messages suggest correct usage

### Feedback Loops

- Immediate validation of required arguments
- Clear progress indication for long operations
- Helpful suggestions in error messages

### Workflow Optimization

- Short evaluation IDs (eval_456) for easy typing
- Tab completion support for commands and options
- Recently used values suggested in help text

### Configuration Management

- Environment variables take precedence
- Config file support for common settings
- Validation on startup with clear error messages

## Implementation Framework

### Technology Stack

- **Click**: Command-line framework with decorators
- **Rich**: Terminal formatting, progress bars, tables
- **Typer**: Type annotations and validation (alternative to Click)

### Command Structure

```python
import click
from rich.console import Console

@click.group()
@click.version_option()
def cli():
    """ML Agents v2 - Reasoning research platform"""
    pass

@cli.group()
def evaluate():
    """Evaluation management commands"""
    pass

@evaluate.command()
@click.option('--agent', type=click.Choice(['none', 'cot']), required=True)
@click.option('--model', required=True)
@click.option('--benchmark', required=True)
@click.option('--temp', type=float, default=1.0)
@click.option('--max-tokens', type=int, default=1000)
def create(agent, model, benchmark, temp, max_tokens):
    """Create new evaluation"""
    # Implementation
```

### Error Handling Pattern

```python
class CLIError(Exception):
    def __init__(self, message: str, suggestion: str = None, exit_code: int = 1):
        self.message = message
        self.suggestion = suggestion
        self.exit_code = exit_code

def handle_error(error: CLIError):
    console.print(f"✗ Error: {error.message}", style="red")
    if error.suggestion:
        console.print(f"  {error.suggestion}", style="dim")
    sys.exit(error.exit_code)
```

## Testing Considerations

### CLI Testing Framework

- **click.testing.CliRunner**: Isolated command testing
- **pytest fixtures**: Mock application services
- **Output validation**: Assert on exit codes and formatted output

### Test Categories

1. **Command parsing**: Argument validation and option handling
2. **Output formatting**: Progress bars, tables, error messages
3. **Error scenarios**: Invalid inputs, API failures, interruptions
4. **Integration**: Full workflow testing with mocked services

---

## See Also

- **[Core Behaviors](v2-core-behaviour-definition.md)** - User workflows implemented by these CLI commands
- **[Application Services Architecture](v2-application-services-architecture.md)** - Service layer called by CLI commands
- **[Project Structure](v2-project-structure.md)** - CLI implementation organization and patterns
- **[Infrastructure Requirements](v2-infrastructure-requirements.md)** - CLI framework dependencies and setup

---

This CLI design provides a consistent, user-friendly interface aligned with the core behaviors and implementation constraints.

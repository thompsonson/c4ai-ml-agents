# Cohere Labs Open Science Research into ML Agents and Reasoning

## Community Resources

- **[ML Agents Community Program](https://sites.google.com/cohere.com/coherelabs-community/community-programs/ml-agents)** - Main hub for Cohere Labs' community-driven initiative on open-source agent research, focusing on agentic frameworks, applications, evaluations, and benchmarks

- **[Project Documentation](https://docs.google.com/document/d/1fLnwUzTvO3XuvViBwLz-QuSe_y87a1p4j8Uw2R4eBiI/edit?pli=1&tab=t.0#heading=h.d0279byf6lhr)** - Detailed specifications and roadmap for the ZeroHPO (Zero-shot Hyperparameter Optimization) project for agentic tasks

- **[Project Tracker](https://docs.google.com/spreadsheets/d/1-TBlPSIiBymQfCdF_LCYJznLwaxcKtYTRJME0NT17kU/edit?usp=sharing)** - Community project tracking, task assignments, and progress monitoring

- **[Discord Community](https://discord.gg/ckaQnUakYx)** - Join the #ml-agents channel for discussions, meetings, and collaboration with the community

## Overview

This project investigates how different reasoning approaches impact AI model performance across various tasks. It provides a comprehensive framework for comparing multiple reasoning techniques with various language models.

**🎉 Phase 5 Complete**: The platform now includes a powerful CLI interface for easy experimentation alongside the original Jupyter notebook!

## Research Questions

1. **Universal Benefit**: Do all tasks benefit from reasoning?
2. **Model Variability**: Do different models show varying benefits from reasoning?
3. **Approach Comparison**: How do different reasoning approaches (CoT, PoT, etc.) compare?
4. **Task-Approach Fit**: Do certain tasks benefit more from specific reasoning methods?
5. **Cost-Benefit Analysis**: What is the tradeoff for each approach and task?
6. **Predictive Reasoning**: Can we predict the need for reasoning based on the input prompt alone?

## Reasoning Approaches Available

The platform currently supports **8 production-ready reasoning approaches**:

1. **None** - Baseline direct prompting without reasoning
2. **Chain-of-Thought (CoT)** - Step-by-step reasoning process
3. **Program-of-Thought (PoT)** - Code-based problem solving
4. **Reasoning-as-Planning** - Strategic planning with goal decomposition
5. **Reflection** - Self-evaluation and iterative improvement
6. **Chain-of-Verification** - Systematic verification with follow-up questions
7. **Skeleton-of-Thought** - Hierarchical outline-first reasoning
8. **Tree-of-Thought** - Multiple reasoning path exploration and synthesis

**Additional approaches planned**: Graph-of-Thought, ReWOO, Buffer-of-Thoughts (Phase 6)

## Quick Start

### Prerequisites

- Python 3.9+
- uv (for virtual environment management)
- API keys for at least one provider (Anthropic, Cohere, OpenRouter, or Hugging Face)

### Installation

#### Option 1: pip Install (Recommended)

Install the latest stable version from PyPI:

```bash
# Install globally
pip install ml-agents-reasoning

# Or install with development dependencies
pip install ml-agents-reasoning[dev]

# Verify installation
ml-agents --version
ml-agents --help
```

#### Option 2: Modern Python (uv/uvx)

With [uv](https://github.com/astral-sh/uv) (fastest):

```bash
# Install with uv
uv tool install ml-agents-reasoning

# Run without installing (recommended for trying out)
uvx ml-agents-reasoning run --approach CoT --samples 10

# Add to project dependencies
uv add ml-agents-reasoning
```

#### Option 3: Development Installation

For contributors or advanced users:

```bash
# Clone and install in development mode
git clone https://github.com/thompsonson/c4ai-ml-agents
cd c4ai-ml-agents
pip install -e .[dev]

# Or with uv (recommended)
uv sync --all-extras
```

### Configure API Keys

After installation, configure your API keys:

```bash
# Create configuration file
cp .env.example .env
# Edit .env with your actual API keys

# Or set environment variables directly
export ANTHROPIC_API_KEY="your-key-here"
export OPENROUTER_API_KEY="your-key-here"
```

### CLI Quick Start

Once installed, you can use the ML Agents CLI:

```bash
# Validate your environment
ml-agents validate-env

# List available reasoning approaches
ml-agents list-approaches

# Run a simple experiment
ml-agents run --approach ChainOfThought --samples 10

# Compare multiple approaches
ml-agents compare --approaches "ChainOfThought,AsPlanning,TreeOfThought" --samples 50 --parallel
```

### Jupyter Notebook (Original Interface)

To use the original Jupyter notebook interface:

```bash
jupyter notebook Reasoning_LLM.ipynb
```

## Configuration

### Supported Providers and Models

- **Hugging Face**: Gemma-2, Mistral-7B, Llama-3-8B
- **Anthropic**: Claude Opus 4, Claude Sonnet 4, Claude 3.5 Haiku
- **Cohere**: Command R+, Command R, Command Light
- **OpenRouter**: GPT-5, GPT-5 Mini, GPT OSS-120B, Gemini 2.5 Flash Lite

### Hyperparameters

- **Temperature**: 0.0 - 2.0 (controls randomness)
- **Max Tokens**: 64 - 4096 (output length limit)
- **Top P**: 0.0 - 1.0 (nucleus sampling parameter)

## MCP Integration (Phase 7)

The platform includes **SQLite database persistence** for all experiment results and supports **Claude Code MCP server integration** for direct database access during conversations.

### Database Features

- **Real-time persistence**: All experiment results are automatically saved to `ml_agents_results.db`
- **Read-only MCP access**: Query the database directly from Claude Code conversations
- **Rich export formats**: CSV, JSON, and Excel with advanced formatting
- **Advanced analytics**: Approach comparisons, failure analysis, and cost tracking

### Database CLI Commands

```bash
# Database management
ml-agents db-init --db-path ./results.db          # Initialize database
ml-agents db-backup --source ./results.db         # Create backup
ml-agents db-stats --db-path ./results.db         # Show statistics

# Export and analysis
ml-agents export EXPERIMENT_ID --format excel     # Export to Excel
ml-agents compare-experiments "exp1,exp2,exp3"    # Compare experiments
ml-agents analyze EXPERIMENT_ID --type accuracy   # Generate reports
ml-agents list-experiments --status completed     # List experiments
```

## CLI Usage Guide

### Basic Commands

#### Single Experiment

Run one reasoning approach on a dataset:

```bash
# Basic usage
ml-agents run --approach ChainOfThought --samples 50

# With specific model
ml-agents run --approach TreeOfThought --samples 100 --provider openrouter --model "openai/gpt-oss-120b"

# With advanced settings
ml-agents run --approach ChainOfVerification --multi-step-verification --max-reasoning-calls 5
```

#### Comparison Experiments

Compare multiple approaches side-by-side:

```bash
# Basic comparison
ml-agents compare --approaches "ChainOfThought,AsPlanning,TreeOfThought" --samples 100

# Parallel execution for faster results
ml-agents compare --approaches "None,ChainOfThought,Reflection" --samples 200 --parallel --max-workers 4

# Advanced reasoning comparison
ml-agents compare --approaches "ChainOfVerification,Reflection,SkeletonOfThought" --multi-step-verification --parallel
```

### Configuration Files

For complex experiments, use YAML configuration files:

```bash
# Run with configuration file
ml-agents run --config examples/configs/single_experiment.yaml

# Override specific parameters
ml-agents run --config examples/configs/comparison_study.yaml --samples 200 --parallel
```

**Example configuration** (`config.yaml`):

```yaml
experiment:
  name: "reasoning_comparison_study"
  sample_count: 100
  output_dir: "./results"

model:
  provider: "openrouter"
  name: "openai/gpt-oss-120b"
  temperature: 0.3
  max_tokens: 512

reasoning:
  approaches:
    - ChainOfThought
    - AsPlanning
    - TreeOfThought
  multi_step_verification: true
  max_reasoning_calls: 5

execution:
  parallel: true
  max_workers: 4
  save_checkpoints: true
```

### Checkpoint Management

Resume interrupted experiments:

```bash
# List available checkpoints
ml-agents list-checkpoints

# Resume from specific checkpoint
ml-agents resume checkpoint_exp_20250818_123456.json
```

### Advanced Features

#### Cost Control

```bash
# Set reasoning limits to control costs
ml-agents run --approach ChainOfVerification --max-reasoning-calls 3 --samples 50

# Monitor costs with verbose output
ml-agents compare --approaches "ChainOfThought,TreeOfThought" --samples 100 --verbose
```

#### Multi-step Reasoning

```bash
# Enable multi-step reflection
ml-agents run --approach Reflection --multi-step-reflection --max-reflection-iterations 3

# Enable multi-step verification
ml-agents run --approach ChainOfVerification --multi-step-verification --max-reasoning-calls 5
```

#### Parallel Processing

```bash
# Parallel execution with custom worker count
ml-agents compare --approaches "ChainOfThought,AsPlanning,TreeOfThought,Reflection" --parallel --max-workers 2

# Balance speed vs rate limits
ml-agents compare --approaches "None,ChainOfThought" --samples 500 --parallel --max-workers 8
```

### Output and Results

Results are automatically saved with timestamps:

```
./outputs/
├── exp_20250818_143256/
│   ├── experiment_summary.json
│   ├── results_ChainOfThought.csv
│   ├── results_AsPlanning.csv
│   └── checkpoint_exp_20250818_143256.json
```

Each result file contains:

- Input prompts and model responses
- Complete reasoning traces
- Performance metrics (accuracy, time, cost)
- Configuration details
- Error information

### Example Workflows

#### 1. Quick Testing

```bash
# Test with small sample size
ml-agents run --approach ChainOfThought --samples 5 --verbose
```

#### 2. Research Study

```bash
# Comprehensive comparison study
ml-agents compare \
  --approaches "None,ChainOfThought,AsPlanning,TreeOfThought,Reflection" \
  --samples 200 \
  --parallel \
  --max-workers 4 \
  --multi-step-verification \
  --output "./studies/comprehensive_study"
```

#### 3. Batch Processing

```bash
# Use the provided batch script
./examples/scripts/batch_experiments.sh
```

### Command Reference

| Command | Description |
|---------|-------------|
| `ml-agents run` | Single reasoning experiment |
| `ml-agents compare` | Multi-approach comparison |
| `ml-agents resume` | Resume from checkpoint |
| `ml-agents list-checkpoints` | Show available checkpoints |
| `ml-agents list-approaches` | Show available reasoning methods |
| `ml-agents validate-env` | Check environment setup |
| `ml-agents version` | Show version information |

For detailed help on any command:

```bash
ml-agents run --help
ml-agents compare --help
```

## Jupyter Notebook Usage (Original Interface)

For users who prefer the notebook interface:

1. **Setup**: Ensure dependencies are installed via `./setup.sh`
2. **Configuration**: Use interactive widgets to select models and approaches
3. **Data**: Default uses "bbeh-eval" dataset, customizable
4. **Execute**: Run experiment cells to process your dataset
5. **Results**: Tables and CSV files with format `{model}_{approach}_{timestamp}.csv`

## Dataset Requirements

Your dataset should include:

- **input** column: The question/problem to solve
- **answer** column (optional): Expected output for evaluation
- **task** column (optional): Task category for analysis

## Output Files

The notebook generates CSV files containing:

- Input prompts
- Model outputs
- Full reasoning traces
- Execution time
- Cost estimates
- Configuration details

## Project Structure

```
ml-agents/
├── src/                           # Main source code
│   ├── cli/                      # CLI interface (Phase 5)
│   │   ├── main.py              # CLI entry point
│   │   ├── commands.py          # Run/compare commands
│   │   ├── config_loader.py     # Configuration management
│   │   ├── display.py           # Rich output formatting
│   │   └── validators.py        # Input validation
│   ├── core/                    # Core experiment logic
│   │   ├── experiment_runner.py # Experiment orchestration
│   │   ├── dataset_loader.py    # Dataset loading
│   │   └── reasoning_inference.py # Inference engine
│   ├── reasoning/               # Reasoning approaches
│   │   ├── base.py             # Base reasoning class
│   │   ├── chain_of_thought.py # CoT implementation
│   │   ├── tree_of_thought.py  # ToT implementation
│   │   └── ...                 # Other approaches
│   └── utils/                   # Utilities
│       ├── api_clients.py      # API wrappers
│       ├── rate_limiter.py     # Rate limiting
│       └── logging_config.py   # Logging setup
├── examples/                    # Usage examples
│   ├── configs/                # Configuration templates
│   ├── scripts/                # Batch processing scripts
│   └── README.md               # Examples documentation
├── tests/                      # Test suite
├── outputs/                    # Experiment results
├── Reasoning_LLM.ipynb        # Original Jupyter notebook
├── config.py                  # Environment configuration
├── requirements.txt           # Python dependencies
├── setup.sh                   # Automated setup script
├── Makefile                   # Development commands
└── README.md                  # This file
```

## Best Practices

### For Researchers

1. **Start Small**: Begin with `--samples 10` to test approaches quickly
2. **Use Baselines**: Always include `None` approach for comparison
3. **Cost Control**: Monitor costs with `--verbose` and set `--max-reasoning-calls`
4. **Parallel Processing**: Use `--parallel` for faster comparison studies
5. **Reproducibility**: Save configuration files and use checkpoints

### For Cost Management

1. **Temperature Settings**: Lower values (0.1-0.3) for consistent, cost-effective results
2. **Token Limits**: Set appropriate `--max-tokens` based on your task complexity
3. **Sample Sizing**: Use smaller samples for initial exploration
4. **Provider Selection**: Compare costs across different providers
5. **Multi-step Limits**: Control `--max-reasoning-calls` for approaches like Chain-of-Verification

### For Performance

1. **Parallel Execution**: Use `--parallel --max-workers N` for comparison studies
2. **Checkpoint Usage**: Enable checkpoints for long-running experiments
3. **Rate Limiting**: Adjust `--max-workers` based on provider rate limits
4. **Batch Processing**: Use configuration files and scripts for multiple experiments

## Troubleshooting

### Common Issues

#### Environment Setup

```bash
# Check environment
ml-agents validate-env

# Fix dependency issues
make clean && make install-dev

# Verify imports
make debug-imports
```

#### API Key Problems

```bash
# Check .env file exists and has keys
cat .env

# Validate specific provider
ml-agents validate-env
```

Error messages will guide you to set missing keys:

```bash
export OPENROUTER_API_KEY="your_key_here"
export ANTHROPIC_API_KEY="your_key_here"
```

#### Rate Limiting

If you encounter rate limits:

```bash
# Reduce parallel workers
ml-agents compare --approaches "ChainOfThought,AsPlanning" --max-workers 1

# Add delays between requests
ml-agents run --approach ChainOfThought --samples 50 --parallel false
```

#### Memory Issues

For large experiments:

```bash
# Reduce sample size
ml-agents compare --approaches "ChainOfThought,TreeOfThought" --samples 50

# Disable parallel processing
ml-agents compare --approaches "..." --parallel false
```

#### NumPy Compatibility Warning

The warning about NumPy 1.x vs 2.x is cosmetic and doesn't affect functionality:

```
A module that was compiled using NumPy 1.x cannot be run in NumPy 2.3.2...
```

This is a known PyTorch compatibility issue and can be ignored.

### CLI Issues

#### Command Not Found

```bash
# Reinstall the package
make install-dev

# Check entry point
which ml-agents
```

#### Import Errors

```bash
# Activate virtual environment
source .venv/bin/activate

# Test imports
make debug-imports
```

#### Configuration Validation

For configuration errors, check:

1. YAML/JSON syntax is valid
2. All required fields are present
3. Approach names match available options (`ml-agents list-approaches`)
4. Provider/model combinations are supported

### Getting Help

1. **Command Help**: Use `--help` with any command

   ```bash
   ml-agents --help
   ml-agents run --help
   ml-agents compare --help
   ```

2. **Verbose Output**: Add `--verbose` to see detailed execution logs

   ```bash
   ml-agents run --approach ChainOfThought --samples 5 --verbose
   ```

3. **Check Status**: Validate your setup

   ```bash
   ml-agents validate-env
   ml-agents list-approaches
   make validate-env
   ```

4. **Community Support**: Join the Discord #ml-agents channel for help

### Development Issues

For developers working on the codebase:

```bash
# Run test suite
make test

# Check code quality
make lint

# Type checking
make type-check

# Full development check
make check
```

## Development Tools

### Claude Code MCP Server Setup

For developers using Claude Code, enable direct database queries in conversations:

```bash
# Configure MCP server (one-time setup)
make configure-mcp

# Or run the script directly
./scripts/install-sqlite-mcp-server.sh
```

**Available MCP Tools**:

- `read_query`: Execute validated SELECT queries
- `list_tables`: Show all database tables
- `describe_table`: Show table schemas

**⚠️ Note**: Project-scoped MCP servers don't appear in `claude mcp list` due to a [known bug](https://github.com/anthropics/claude-code/issues/5963). Use `claude mcp get sqlite-read-only` to verify installation.

## Contributing

Feel free to extend the notebook with:

- Additional reasoning approaches
- New evaluation metrics
- Support for more models/providers
- Performance optimizations

## License

### Recommended: CC BY 4.0 (Creative Commons Attribution 4.0 International)

This project is licensed under the Creative Commons Attribution 4.0 International License. This means:

- ✅ **Share** - Copy and redistribute the material in any medium or format
- ✅ **Adapt** - Remix, transform, and build upon the material for any purpose, even commercially
- ✅ **Attribution** - You must give appropriate credit, provide a link to the license, and indicate if changes were made

This license is chosen because:

1. **Open Science**: Aligns with Cohere Labs' open science mission
2. **Maximum Impact**: Allows both academic and commercial use, accelerating AI research
3. **Community Growth**: Enables derivatives while ensuring original work is credited
4. **Simplicity**: Easy to understand and implement

**Note**: For the code components specifically, you may want to consider dual-licensing with MIT or Apache 2.0 for better software compatibility.

<a rel="license" href="http://creativecommons.org/licenses/by/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by/4.0/88x31.png" /></a>

### Alternative Options Considered

- **CC BY-SA 4.0**: Adds "ShareAlike" requirement - derivatives must use same license (more restrictive but ensures openness)
- **CC BY-NC 4.0**: Adds "NonCommercial" restriction - prevents commercial use (limits industry collaboration)
- **CC0**: Public domain dedication - no attribution required (maximum freedom but no credit requirement)

# ML Agents Beginner Guide

**Quick Start Guide for AI Reasoning Research**

ML Agents is a research platform that compares different reasoning approaches (like Chain-of-Thought, Tree-of-Thought) across AI models to understand when and how reasoning improves performance. This guide gets you running experiments in under 10 minutes.

## Prerequisites

- **Python 3.9+** with pip or [uv](https://github.com/astral-sh/uv)
- **API keys** for at least one provider:
  - [OpenRouter](https://openrouter.ai/keys) (recommended - has free models)
  - [Anthropic](https://console.anthropic.com/) or [Cohere](https://dashboard.cohere.ai/)

## Installation

### Option 1: Quick Install (Recommended)
```bash
# Install from PyPI
pip install ml-agents-reasoning

# Verify installation
ml-agents --version
```

### Option 2: Modern Python (uv)
```bash
# Install with uv
uv tool install ml-agents-reasoning

# Or try without installing
uvx ml-agents-reasoning setup validate-env
```

### Option 3: Development Install
```bash
# Clone repository
git clone https://github.com/thompsonson/c4ai-ml-agents
cd c4ai-ml-agents
pip install -e .[dev]
```

## Setup

### 1. Configure API Keys
```bash
# Create environment file
cat > .env << EOF
OPENROUTER_API_KEY=sk-or-v1-your-key-here
ANTHROPIC_API_KEY=sk-ant-api-your-key-here
EOF

# Or export directly
export OPENROUTER_API_KEY="your-key-here"
```

### 2. Validate Setup
```bash
# Check environment (✅ Stable)
ml-agents setup validate-env

# List available reasoning approaches (✅ Stable)
ml-agents setup list-approaches

# Show version info (✅ Stable)
ml-agents setup version
```

## First Steps

### Initialize Database
```bash
# Create experiment database (✅ Stable)
ml-agents db init

# View database stats (✅ Stable)
ml-agents db stats
```

### Quick Test
```bash
# Run a small experiment (⚠️ Experimental)
ml-agents eval run --approach ChainOfThought --samples 5 --verbose

# This tests: API connection, reasoning pipeline, result storage
```

## Common Usage Patterns

### 1. Single Reasoning Experiment
```bash
# Basic experiment (⚠️ Experimental)
ml-agents eval run --approach ChainOfThought --samples 50

# With specific model (⚠️ Experimental)
ml-agents eval run --approach TreeOfThought --samples 100 --provider openrouter --model "openai/gpt-oss-120b"

# Cost-controlled experiment (⚠️ Experimental)
ml-agents eval run --approach ChainOfVerification --samples 50 --max-reasoning-calls 3
```

### 2. Compare Multiple Approaches
```bash
# Basic comparison (⚠️ Experimental)
ml-agents eval compare --approaches "None,ChainOfThought,TreeOfThought" --samples 100

# Parallel execution for speed (⚠️ Experimental)
ml-agents eval compare --approaches "ChainOfThought,AsPlanning,Reflection" --samples 200 --parallel --max-workers 4
```

### 3. Dataset Preprocessing
```bash
# List available datasets for preprocessing (✅ Stable)
ml-agents preprocess list

# Inspect a dataset structure (✅ Stable)
ml-agents preprocess inspect MilaWang/SpatialEval --samples 100

# Process multiple datasets (✅ Stable)
ml-agents preprocess batch --max 5
```

### 4. Database Management
```bash
# View experiment results (⚠️ Experimental)
ml-agents results list --status completed

# Export results to Excel (⚠️ Experimental)
ml-agents results export EXPERIMENT_ID --format excel

# Backup database (✅ Stable)
ml-agents db backup --source ./ml_agents_results.db
```

## Understanding Results

### Output Files
Results are saved in timestamped directories:
```
./outputs/
├── exp_20250823_143256/
│   ├── experiment_summary.json     # Experiment configuration
│   ├── results_ChainOfThought.csv  # Detailed results
│   └── errors.json                 # Any processing errors
```

### Result Columns
Each CSV contains:
- **input**: Original question/prompt
- **output**: Model's final answer
- **reasoning_trace**: Full reasoning process
- **execution_time**: Time taken (seconds)
- **total_cost**: Estimated API cost
- **approach**: Reasoning method used

### Quick Analysis
```bash
# View experiment summary (⚠️ Experimental)
ml-agents results analyze EXPERIMENT_ID --type accuracy

# Compare multiple experiments (⚠️ Experimental)
ml-agents results compare "exp1,exp2,exp3"
```

## Example Workflow

```bash
# 1. Validate setup
ml-agents setup validate-env

# 2. Initialize database
ml-agents db init

# 3. Quick test with 5 samples
ml-agents eval run --approach ChainOfThought --samples 5 --verbose

# 4. Full comparison study
ml-agents eval compare --approaches "None,ChainOfThought,TreeOfThought" --samples 100 --parallel

# 5. View results
ml-agents results list
```

## Troubleshooting

### API Key Issues
```bash
# Error: "API key not found"
ml-agents setup validate-env  # Shows which keys are missing

# Fix: Add to .env or export
export OPENROUTER_API_KEY="your-key-here"
```

### Rate Limiting
```bash
# Error: "Rate limit exceeded"
# Solution: Reduce parallel workers or use free models
ml-agents eval run --approach ChainOfThought --samples 50 --parallel false
```

### Memory/Performance Issues
```bash
# Error: High memory usage
# Solution: Reduce samples or disable parallel processing
ml-agents eval compare --approaches "ChainOfThought,TreeOfThought" --samples 50 --max-workers 1
```

### Command Not Found
```bash
# Error: "ml-agents: command not found"
# Solution: Reinstall or check PATH
pip install --force-reinstall ml-agents-reasoning
which ml-agents  # Should show installation path
```

## Understanding Command Types

**✅ Stable Commands** (Production Ready):
- `setup`, `db`, `preprocess` - Well-tested, stable API

**⚠️ Experimental Commands** (Pre-Alpha):
- `eval`, `results` - May have breaking changes, use `--skip-warnings` to suppress warnings

## Cost Management

### Free Models (No Cost)
```bash
# Use OpenRouter's free models
ml-agents eval run --approach ChainOfThought --provider openrouter --model "openai/gpt-3.5-turbo"
```

### Cost Estimation
- **5 samples**: ~$0.01-0.05 (testing)
- **100 samples**: ~$0.50-2.00 (typical study)
- **1000 samples**: ~$5-20 (full research study)

### Cost Control Tips
- Start with `--samples 5-10` for testing
- Use `--max-reasoning-calls 3` to limit multi-step approaches
- Set `--max-tokens 256` for shorter responses
- Monitor with `--verbose` flag

## Next Steps

- **Advanced Usage**: See [README.md](./README.md) for full CLI reference
- **Configuration Files**: Check [examples/](./examples/) for YAML configs
- **Research Examples**: Follow [EXAMPLE_EXPERIMENT.md](./EXAMPLE_EXPERIMENT.md)
- **Community**: Join [Discord #ml-agents](https://discord.gg/ckaQnUakYx) channel
- **Development**: See [CLAUDE.md](./CLAUDE.md) for contributor guide

## Available Reasoning Approaches

1. **None** - Baseline (direct prompting)
2. **ChainOfThought** - Step-by-step reasoning
3. **ProgramOfThought** - Code-based problem solving
4. **AsPlanning** - Strategic planning approach
5. **Reflection** - Self-evaluation and improvement
6. **ChainOfVerification** - Systematic verification
7. **SkeletonOfThought** - Outline-first reasoning
8. **TreeOfThought** - Multiple reasoning paths

Start with **ChainOfThought** (most common) and **None** (baseline) for your first comparison.

## Getting Help

```bash
# Command-specific help
ml-agents --help
ml-agents eval run --help
ml-agents setup --help

# Community support
# Discord: https://discord.gg/ckaQnUakYx (#ml-agents channel)
```

# ML Agents Beginner Guide

**Quick Start Guide for AI Reasoning Research**

ML Agents is a research platform that compares different reasoning approaches (like Chain-of-Thought, Tree-of-Thought) across AI models to understand when and how reasoning improves performance. **Phase 15** includes structured answer extraction that cleanly separates reasoning text from final answers. This guide gets you running experiments in under 10 minutes.

## Prerequisites

- **Python 3.9+** with pip or [uv](https://github.com/astral-sh/uv) (uv is recommended)
- **API keys** for at least one provider:
  - [OpenRouter](https://openrouter.ai/keys) (recommended - has free models)
  - [Anthropic](https://console.anthropic.com/)
  - [Cohere](https://dashboard.cohere.ai/)

## Installation

### Option 1: Modern Python (uv) Recommended

```bash
# Install with uv
uv tool install ml-agents-reasoning

# Verify installation
ml-agents --version

# Or try without installing
uvx --from ml-agents-reasoning ml-agents --version
```

### Option 2: Old Skool Pip Install

```bash
# Install from PyPI
pip install ml-agents-reasoning

# Verify installation
ml-agents --version
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
# export the api keys and tokens
# add random keys even if you will not use the service
export ANTHROPIC_API_KEY=sk-ant-api03-xxxxxxxxxxxxxxxxxxxx
export COHERE_API_KEY=xxxxxxxxxxxxxxxxxxxx
export OPENROUTER_API_KEY="your-key-here"
# needed to access and upload the datasets
export HF_TOKEN="your-token-here"
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

### 1. Dataset Preprocessing

Transform any HuggingFace dataset to standardized `{INPUT, OUTPUT}` format for reasoning evaluation.

**New Structure**: Preprocessing files are automatically organized in dataset-specific folders: `./outputs/{dataset_name}/preprocessing/{timestamp}/` with analysis, rules, processed data, and metadata files.

```bash
# List available datasets for preprocessing (✅ Stable)
ml-agents preprocess list

# Step 1: Inspect dataset schema and confirm INPUT/OUTPUT fields (✅ Stable)
ml-agents preprocess inspect MilaWang/SpatialEval --samples 1 --config tqa

# Step 2: Generate transformation rules file (✅ Stable)
ml-agents preprocess generate-rules MilaWang/SpatialEval --config tqa

# Step 3: Apply transformation to create standardized dataset (✅ Stable)
ml-agents preprocess transform MilaWang/SpatialEval rules.json --config tqa

# Step 4: Upload to HuggingFace Hub (✅ Stable)
ml-agents preprocess upload ./outputs/MilaWang_SpatialEval_tqa/preprocessing/latest/processed.json

# Process multiple datasets automatically (✅ Stable)
ml-agents preprocess batch --max 5
```

#### Understanding the Rules File

The `generate-rules` command creates a JSON file that defines how to transform your dataset:

```json
{
  "dataset_name": "MilaWang/SpatialEval",
  "transformation_type": "multi_field",
  "confidence": 1.0,
  "input_format": "multi_field",
  "input_fields": ["story", "question", "candidate_answers"],
  "output_field": "answer",
  "field_separator": "\n\n",
  "field_labels": {
    "story": "STORY:",
    "question": "QUESTION:",
    "candidate_answers": "OPTIONS:"
  },
  "preprocessing_steps": ["resolve_answer_index"],
  "answer_options_field": "candidate_answers"  // Optional: explicitly specify the field containing answer options
}
```

- **input_fields**: Dataset columns combined into INPUT field
- **output_field**: Dataset column used for OUTPUT field
- **preprocessing_steps**: Special transformations (e.g., convert answer indices to text)
- **field_labels**: Headers added to INPUT sections
- **answer_options_field** (optional): Explicitly specify which field contains the answer options for index resolution
  - If not specified, auto-detects from common field names: "options", "candidate_answers", "choices", "alternatives"
  - Example: GPQA dataset uses "options" field, spartqa uses "candidate_answers"

#### Manual Rules Correction (✅ Stable)

When automatic pattern detection gets the field mappings wrong, use the `fix-rules` command to correct them:

```bash
# Interactive mode (default) - prompts you for correct fields
ml-agents preprocess fix-rules ./outputs/dataset/preprocessing/latest/rules.json

# Non-interactive mode - specify corrections directly
ml-agents preprocess fix-rules rules.json \
  --input-fields story,question,candidate_answers \
  --output-field answer \
  --preprocessing-steps resolve_answer_index

# Preview changes without saving
ml-agents preprocess fix-rules rules.json --dry-run \
  --input-fields story,question,candidate_answers \
  --output-field answer
```

**Example workflow for problematic datasets**:
1. `ml-agents preprocess inspect` - see available fields
2. `ml-agents preprocess generate-rules` - creates initial (possibly wrong) rules
3. `ml-agents preprocess fix-rules` - correct the field mappings
4. `ml-agents preprocess transform` - apply the corrected rules

The command validates field names against the actual dataset schema and adds correction metadata to track manual changes.

#### Get Help for Preprocessing Commands

```bash
ml-agents preprocess --help
ml-agents preprocess generate-rules --help
ml-agents preprocess fix-rules --help
ml-agents preprocess transform --help
ml-agents preprocess upload --help
```

### 2. Single Reasoning Experiment

```bash
# Basic experiment (⚠️ Experimental)
ml-agents eval run --approach ChainOfThought --samples 50

# With specific model (⚠️ Experimental)
ml-agents eval run --approach TreeOfThought --samples 100 --provider openrouter --model "openai/gpt-oss-120b"

# Cost-controlled experiment (⚠️ Experimental)
ml-agents eval run --approach ChainOfVerification --samples 50 --max-reasoning-calls 3
```

#### Evaluation with Preprocessing Integration

Use preprocessed datasets with your evaluation experiments:

```bash
# Auto-detect latest preprocessing for a dataset (⚠️ Experimental)
ml-agents eval run --dataset dataset-name --approach ChainOfThought

# Use specific preprocessing run by ID (⚠️ Experimental)
ml-agents eval run --preprocessing-id prep_20240824_143256 --approach ChainOfThought

# Use custom preprocessed data file (⚠️ Experimental)
ml-agents eval run --preprocessing-path ./custom/processed.json --approach ChainOfThought
```

### 3. Compare Multiple Approaches

```bash
# Basic comparison (⚠️ Experimental)
ml-agents eval compare --approaches "None,ChainOfThought,TreeOfThought" --samples 100

# Parallel execution for speed (⚠️ Experimental)
ml-agents eval compare --approaches "ChainOfThought,ReasoningAsPlanning,Reflection" --samples 200 --parallel --max-workers 4
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

Results are organized by dataset with full preprocessing-evaluation traceability:

```
./outputs/
├── {dataset_name}/
│   ├── preprocessing/
│   │   ├── {timestamp}/
│   │   │   ├── analysis.json           # Dataset schema analysis
│   │   │   ├── rules.json              # Transformation rules
│   │   │   ├── processed.json          # Standardized dataset
│   │   │   └── metadata.json           # Preprocessing metadata
│   │   └── latest -> {most_recent}/    # Symlink to latest preprocessing
│   └── eval/
│       ├── {exp_timestamp}/
│       │   ├── experiment_config.json  # Experiment configuration
│       │   ├── experiment_results.csv  # Detailed results per approach
│       │   ├── experiment_summary.json # Performance summary
│       │   └── experiment_errors.json  # Any processing errors
│       └── latest -> {most_recent}/    # Symlink to latest experiment
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

### Complete Dataset → Evaluation Pipeline

```bash
# 1. Validate setup
ml-agents setup validate-env

# 2. Initialize database
ml-agents db init

# 3. Preprocess a custom dataset (creates organized folder structure)
ml-agents preprocess inspect MilaWang/SpatialEval --config tqa
ml-agents preprocess generate-rules MilaWang/SpatialEval --config tqa
ml-agents preprocess transform MilaWang/SpatialEval rules.json --config tqa

# 4. Run evaluation with preprocessed data (auto-detects latest preprocessing)
ml-agents eval run --dataset MilaWang/SpatialEval --approach ChainOfThought --samples 50

# 5. Compare approaches on same preprocessed dataset
ml-agents eval run --dataset MilaWang/SpatialEval --approach TreeOfThought --samples 50

# 6. View organized results
ml-agents results list
```

### Fixing Problematic Datasets

Some datasets need manual field mapping corrections. Here's an example with a story-question-choice dataset:

```bash
# 1. Inspect the dataset first
ml-agents preprocess inspect tasksource/spartqa-mchoice --samples 5
# Shows: story, question, candidate_answers, answer (where answer is an index)

# 2. Generate initial rules (might get it wrong)
ml-agents preprocess generate-rules tasksource/spartqa-mchoice

# 3. Check if the generated rules are wrong
# If it incorrectly maps: input_fields=["story","question"], output_field="candidate_answers"
# But should be: input_fields=["story","question","candidate_answers"], output_field="answer"

# 4. Fix the rules manually
ml-agents preprocess fix-rules ./outputs/tasksource_spartqa-mchoice/preprocessing/latest/rules.json \
  --input-fields story,question,candidate_answers \
  --output-field answer \
  --preprocessing-steps resolve_answer_index

# 5. Apply the corrected transformation
ml-agents preprocess transform tasksource/spartqa-mchoice ./outputs/tasksource_spartqa-mchoice/preprocessing/latest/rules.json

# 6. Now use for evaluation
ml-agents eval run --dataset tasksource/spartqa-mchoice --approach ChainOfThought --samples 10
```

#### Example: Different Datasets with Various Answer Option Fields

```bash
# GPQA dataset has "options" field
ml-agents preprocess inspect jeggers/gpqa_formatted --config main
# Shows: Question, options, answer

# SpartQA has "candidate_answers" field
ml-agents preprocess inspect tasksource/spartqa-mchoice
# Shows: story, question, candidate_answers, answer

# The system auto-detects these fields, but you can be explicit:
# For GPQA:
{
  "input_fields": ["Question", "options"],
  "answer_options_field": "options",  // Explicitly specify
  "preprocessing_steps": ["resolve_answer_index"]
}

# For SpartQA:
{
  "input_fields": ["story", "question", "candidate_answers"],
  "answer_options_field": "candidate_answers",  // Explicitly specify
  "preprocessing_steps": ["resolve_answer_index"]
}
```

### Quick Test Workflow

```bash
# 1. Validate setup
ml-agents setup validate-env

# 2. Initialize database
ml-agents db init

# 3. Quick test with default dataset
ml-agents eval run --approach ChainOfThought --samples 5 --verbose

# 4. View results
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
4. **ReasoningAsPlanning** - Strategic planning approach
5. **Reflection** - Self-evaluation and improvement
6. **ChainOfVerification** - Systematic verification
7. **SkeletonOfThought** - Outline-first reasoning
8. **TreeOfThought** - Multiple reasoning paths

Start with **ChainOfThought** (most common) and **None** (baseline) for your first comparison.

### 🎯 Phase 15: Clean Answer Extraction

All reasoning approaches now automatically extract clean answers:
- **Before**: "The final answer is 42"
- **After**: "42" (reasoning preserved separately)
- **Provider-optimized**: Uses best extraction mode for each provider
- **Reliable**: Built-in fallback ensures compatibility across all models

## Getting Help

```bash
# Command-specific help
ml-agents --help
ml-agents eval run --help
ml-agents setup --help

# Community support
# Discord: https://discord.gg/ckaQnUakYx (#ml-agents channel)
# Author's Discord ID: _mattthompson
```

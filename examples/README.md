# ML Agents Examples

This directory contains example configurations and scripts for running ML Agents experiments.

## Quick Start Examples

### 1. Discover Available Datasets

```bash
# List all available benchmarks and test datasets
ml-agents eval list

# Get detailed information about a specific dataset
ml-agents eval info LOCAL_TEST
ml-agents eval info BENCHMARK-01-GPQA.csv
```

### 2. Simple Single Experiment

```bash
# Run Chain-of-Thought on LOCAL_TEST with 3 samples
ml-agents eval run LOCAL_TEST ChainOfThought --samples 3

# Run on a repository benchmark with 50 samples
ml-agents eval run BENCHMARK-01-GPQA.csv ChainOfThought --samples 50

# Run on LOCAL_TEST_LARGE for scale testing
ml-agents eval run LOCAL_TEST_LARGE TreeOfThought --samples 100
```

### 3. Repository and Model Options

```bash
# Use different repository
ml-agents eval list --repo custom-org/my-benchmarks
ml-agents eval run data.csv ChainOfThought --repo custom-org/my-benchmarks

# Use different model provider and model
ml-agents eval run LOCAL_TEST Reflection --provider anthropic --model claude-3-5-haiku-20241022
ml-agents eval run BENCHMARK-02-BoardgameQA.csv ProgramOfThought --provider cohere --model command-r-plus
```

### 4. Advanced Multi-step Reasoning

```bash
# Enable multi-step verification
ml-agents eval run LOCAL_TEST ChainOfVerification --multi-step-verification --max-reasoning-calls 5

# Enable multi-step reflection
ml-agents eval run BENCHMARK-01-GPQA.csv Reflection --multi-step-reflection --samples 20
```

### 5. Compare Multiple Approaches

```bash
# Compare approaches (uses configuration-based approach selection)
ml-agents eval compare --config examples/configs/comparison_study.yaml
```

## Configuration Files

The `configs/` directory contains example YAML configuration files:

### `single_experiment.yaml`

Basic single-approach experiment configuration:

```bash
# Must specify dataset identifier and approach as arguments
ml-agents eval run LOCAL_TEST ChainOfThought --config examples/configs/single_experiment.yaml
```

### `comparison_study.yaml`

Multi-approach comparison experiment:

```bash
ml-agents eval compare --config examples/configs/comparison_study.yaml
```

### `advanced_reasoning.yaml`

Advanced multi-step reasoning configuration:

```bash
ml-agents eval run BENCHMARK-01-GPQA.csv ChainOfVerification --config examples/configs/advanced_reasoning.yaml
```

## Scripts

### `batch_experiments.sh`

Automated batch processing script that runs multiple experiments:

```bash
./examples/scripts/batch_experiments.sh
```

## Configuration Schema

All configuration files follow this structure:

```yaml
experiment:
  name: "experiment_name"
  sample_count: 100
  output_dir: "./results"

dataset:
  name: "MrLight/bbeh-eval"

model:
  provider: "openrouter"  # openrouter, anthropic, cohere
  name: "openai/gpt-oss-120b"
  temperature: 0.3
  max_tokens: 512
  top_p: 0.9

reasoning:
  approaches:
    - ChainOfThought
    - ReasoningAsPlanning
  multi_step_reflection: false
  multi_step_verification: true
  max_reasoning_calls: 3

execution:
  parallel: true
  max_workers: 4
  save_checkpoints: true

output:
  formats:
    - csv
    - json
```

## Available Reasoning Approaches

- `None` - Baseline (no reasoning)
- `ChainOfThought` - Step-by-step reasoning
- `ProgramOfThought` - Code-based problem solving
- `ReasoningAsPlanning` - Strategic planning approach
- `Reflection` - Self-evaluation and correction
- `ChainOfVerification` - Systematic verification
- `SkeletonOfThought` - Outline-first reasoning
- `TreeOfThought` - Multiple reasoning paths
- `GraphOfThought` - Graph-based reasoning connections
- `ReWOO` - Reasoning without Observation
- `BufferOfThoughts` - Buffer-based thought management

## Available Datasets

### Local Test Datasets
- `LOCAL_TEST` - 20 sample test dataset for quick testing
- `LOCAL_TEST_LARGE` - 500 sample test dataset for scale testing

### Repository Benchmarks
Use `ml-agents eval list` to see all available repository CSV files. Examples include:
- `BENCHMARK-01-GPQA.csv` - Graduate-level physics questions (448 samples)
- `BENCHMARK-02-BoardgameQA.csv` - Board game reasoning questions (1000 samples)
- `BENCHMARK-21-AquaRat.csv` - Math word problems (254 samples)
- `BENCHMARK-41-MBPP.csv` - Python programming problems (500 samples)

## Discovery Commands

### List Available Datasets

```bash
# List all datasets from default repository
ml-agents eval list

# List datasets from custom repository
ml-agents eval list --repo your-org/your-benchmarks
```

### Get Dataset Information

```bash
# Get info about LOCAL_TEST
ml-agents eval info LOCAL_TEST

# Get info about repository CSV file
ml-agents eval info BENCHMARK-01-GPQA.csv

# Get info from custom repository
ml-agents eval info my-dataset.csv --repo your-org/your-benchmarks
```

## Cost Management

- Use `--samples 10` for quick testing
- Start with smaller datasets before scaling up
- Monitor costs with `--verbose` flag
- Use `--max-reasoning-calls` to control multi-step approaches

## Troubleshooting

### Common Issues

1. **Dataset Not Found**:
   ```bash
   # Use list command to see available datasets
   ml-agents eval list

   # Check exact dataset name and spelling
   ml-agents eval info LOCAL_TEST
   ```

2. **API Key Issues**:
   ```bash
   # Validate environment setup
   ml-agents setup validate-env

   # Check your .env file has the correct keys
   ```

3. **Model Access**: Verify you have access to the specified model provider and model name

4. **Memory Issues**: Reduce `--samples` for large datasets or reduce `--max-workers` for parallel experiments

5. **Rate Limits**: Experiments automatically handle rate limits, or use `--parallel false` to disable parallel processing

### Command Reference

```bash
# Environment and setup
ml-agents setup validate-env           # Check API keys and environment
ml-agents setup list-approaches        # List available reasoning approaches
ml-agents setup version               # Show version information

# Dataset discovery
ml-agents eval list                    # List available datasets
ml-agents eval info DATASET_ID        # Get dataset information

# Run experiments
ml-agents eval run DATASET_ID APPROACH --samples N  # Single experiment
ml-agents eval compare --config FILE                # Compare approaches
ml-agents eval resume CHECKPOINT_FILE               # Resume interrupted experiment
ml-agents eval checkpoints                         # List available checkpoints

# Results and analysis
ml-agents results list                 # List completed experiments
ml-agents results export EXP_ID        # Export results to different formats
ml-agents results analyze EXP_ID       # Generate analysis reports
ml-agents results compare "exp1,exp2"  # Compare multiple experiments

# Database management
ml-agents db init                      # Initialize results database
ml-agents db stats                     # Show database statistics
ml-agents db backup                    # Create database backup
```

For more information, see the main project documentation.

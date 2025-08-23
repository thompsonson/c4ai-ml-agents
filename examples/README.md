# ML Agents Examples

This directory contains example configurations and scripts for running ML Agents experiments.

## Configuration Files

The `configs/` directory contains example YAML configuration files:

### `single_experiment.yaml`
Basic single-approach experiment configuration:
```bash
ml-agents run --config examples/configs/single_experiment.yaml
```

### `comparison_study.yaml`
Multi-approach comparison experiment:
```bash
ml-agents compare --config examples/configs/comparison_study.yaml
```

### `advanced_reasoning.yaml`
Advanced multi-step reasoning configuration:
```bash
ml-agents compare --config examples/configs/advanced_reasoning.yaml
```

## Scripts

### `batch_experiments.sh`
Automated batch processing script that runs multiple experiments:
```bash
./examples/scripts/batch_experiments.sh
```

## Quick Start Examples

### 1. Simple Single Experiment
```bash
# Run Chain-of-Thought on 50 samples
ml-agents run --approach ChainOfThought --samples 50
```

### 2. Compare Multiple Approaches
```bash
# Compare 3 approaches with parallel execution
ml-agents compare --approaches "ChainOfThought,AsPlanning,TreeOfThought" --samples 100 --parallel
```

### 3. Using Configuration Files
```bash
# Run using a configuration file
ml-agents run --config examples/configs/single_experiment.yaml

# Override specific parameters
ml-agents run --config examples/configs/single_experiment.yaml --samples 200 --parallel
```

### 4. Advanced Multi-step Reasoning
```bash
# Enable multi-step verification
ml-agents run --approach ChainOfVerification --multi-step-verification --max-reasoning-calls 5
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
    - AsPlanning
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
- `AsPlanning` - Strategic planning approach
- `Reflection` - Self-evaluation and correction
- `ChainOfVerification` - Systematic verification
- `SkeletonOfThought` - Outline-first reasoning
- `TreeOfThought` - Multiple reasoning paths

## Cost Management

- Use `--samples 10` for quick testing
- Start with smaller datasets before scaling up
- Monitor costs with `--verbose` flag
- Use `--max-reasoning-calls` to control multi-step approaches

## Troubleshooting

1. **API Key Issues**: Check your `.env` file has the correct keys
2. **Model Access**: Verify you have access to the specified model
3. **Memory Issues**: Reduce `--max-workers` or `--samples`
4. **Rate Limits**: Use `--parallel false` or reduce `--max-workers`

For more information, see the main project documentation.

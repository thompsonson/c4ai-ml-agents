# Phase 14: Reasoning Approaches Comparison Study

This directory contains the complete implementation of Phase 14, which evaluates and compares multiple reasoning approaches against a local API to measure their effectiveness.

## Overview

Phase 14 builds on Phase 13's baseline (None approach) to test three primary reasoning approaches:
- **ChainOfThought (CoT)**: Step-by-step reasoning
- **TreeOfThought (ToT)**: Multi-path exploration
- **ReasoningAsPlanning (RAP)**: Plan-based reasoning

## Quick Start

### 1. Individual Approach Testing

Test a single reasoning approach:

```bash
# Test ChainOfThought
./test_chain_of_thought.py --samples 25 --run-baseline

# Test TreeOfThought
./test_tree_of_thought.py --samples 25 --run-baseline

# Test ReasoningAsPlanning
./test_reasoning_as_planning.py --samples 25 --run-baseline
```

### 2. Comprehensive Comparison

Run all approaches and compare results:

```bash
# Compare all three approaches
./run_full_comparison.py --samples 25 --generate-viz

# Custom approach selection
./run_full_comparison.py --approaches ChainOfThought TreeOfThought --samples 50
```

### 3. Scale Testing

Test approach scalability with different dataset sizes:

```bash
# Test ChainOfThought scaling
./scale_test.py ChainOfThought --sizes 25 50 100 200

# Test with custom sizes
./scale_test.py TreeOfThought --sizes 10 25 50 100 250 500
```

### 4. Generate Reports

Create comprehensive analysis reports:

```bash
# Generate all report formats
./generate_report.py --format markdown html csv json

# Generate specific formats
./generate_report.py --format html --output-dir custom_reports/
```

## Configuration

### Environment Variables

Set these for consistent testing:

```bash
export ML_AGENTS_API_BASE_URL="http://pop-os:8000/v1"
export ML_AGENTS_MODEL="Qwen/Qwen2.5-1.5B-Instruct"
export ML_AGENTS_PROVIDER="local-openai"
```

### Command Line Options

All scripts support these common options:

- `--api-base`: API endpoint URL
- `--model`: Model identifier
- `--samples`: Number of test samples
- `--output-dir`: Output directory
- `--verbose`: Enable detailed logging

## Test Datasets

Phase 14 uses the LOCAL_TEST dataset with three question categories:

### Basic Questions (5 samples)
- Simple factual questions for baseline comparison
- Examples: "What is 2 + 2?", "What is the capital of France?"

### Reasoning Questions (5 samples)
- Logic and multi-step reasoning problems
- Examples: Mathematical word problems, logical reasoning

### Extended Reasoning (10 samples)
- Complex reasoning challenges
- Examples: Puzzle solving, advanced logic problems

### LOCAL_TEST_LARGE
- Extended dataset with 500 samples for scale testing
- Uses cycled questions from the base dataset

## Output Structure

```
outputs/phase14/
├── baseline_results.json                 # None approach baseline
├── chainofthought/
│   ├── chainofthought_results.json
│   └── comparison_report.md
├── treeoofthought/
│   ├── treeoofthought_results.json
│   └── comparison_report.md
├── full_comparison/
│   ├── baseline_results.json
│   ├── comparison_results.csv
│   ├── comparison_report.md
│   └── comparison_visualizations.png
├── scale_test/
│   └── [approach]/
│       └── scale_test_results.json
└── reports/
    ├── phase14_report.md
    ├── phase14_report.html
    ├── phase14_summary.csv
    └── analysis_data.json
```

## Expected Results

Based on Phase 14 hypotheses:

### ChainOfThought
- **Token Usage**: 2-3x baseline
- **Correctness**: +20-30% improvement
- **Response Time**: 1.5x baseline

### TreeOfThought
- **Token Usage**: 3-5x baseline
- **Correctness**: +30-40% improvement
- **Response Time**: 2-3x baseline

### ReasoningAsPlanning
- **Token Usage**: 2-4x baseline
- **Correctness**: +25-35% improvement
- **Response Time**: 2x baseline

## Success Criteria

### Phase 14.1 Success
- [ ] All three reasoning approaches tested successfully
- [ ] Metrics captured for direct comparison with baseline
- [ ] No system stability issues

### Phase 14.2 Success
- [ ] Comprehensive comparison report generated
- [ ] Clear performance/quality tradeoffs identified
- [ ] Recommendations for approach selection

### Phase 14.3 Success
- [ ] Scale tests completed without failures
- [ ] Resource usage patterns documented
- [ ] Production readiness assessment complete

## Analysis Features

### Effectiveness Analysis
- Correctness rate comparison
- Statistical significance testing (t-tests)
- Efficiency metrics (correctness per token)
- Performance stability analysis

### Scaling Analysis
- Linear scaling behavior
- Memory usage patterns
- Throughput degradation
- Scalability scoring (0-1)

### Recommendations
- Best overall approach
- Use-case specific recommendations
- Resource constraint considerations
- Statistical significance warnings

## Troubleshooting

### Common Issues

1. **API Connection Errors**
   ```bash
   # Verify API is running
   curl http://pop-os:8000/v1/models

   # Check environment variables
   echo $ML_AGENTS_API_BASE_URL
   ```

2. **Memory Issues During Scale Testing**
   ```bash
   # Monitor memory usage
   ./scale_test.py ChainOfThought --sizes 25 50 --verbose
   ```

3. **Missing Dependencies**
   ```bash
   # Install visualization dependencies
   pip install matplotlib seaborn

   # Install analysis dependencies
   pip install scipy numpy pandas
   ```

### Performance Optimization

1. **Reduce Sample Size for Quick Tests**
   ```bash
   ./run_full_comparison.py --samples 10
   ```

2. **Parallel Processing** (if supported by API)
   ```bash
   # Use with caution on local API
   ./run_full_comparison.py --parallel-requests 2
   ```

3. **Memory Monitoring**
   ```bash
   # Use system monitoring
   /usr/bin/time -v ./scale_test.py ChainOfThought
   ```

## Integration with Main CLI

Phase 14 results integrate with the main ml-agents CLI:

```bash
# View results in database
ml-agents results list --status completed

# Export specific experiments
ml-agents results export EXPERIMENT_ID --format excel

# Compare experiments
ml-agents results compare "exp1,exp2,exp3"
```

## Next Steps

After Phase 14 completion:
- **Phase 15**: Custom reasoning approaches optimized for local models
- **Phase 16**: Automated approach selection based on prompt analysis
- **Phase 17**: Reasoning approach ensemble methods

## Support

For issues or questions:
1. Check logs in `outputs/phase14/` directories
2. Verify API connectivity and model availability
3. Ensure all dependencies are installed
4. Review Phase 13 baseline results for comparison

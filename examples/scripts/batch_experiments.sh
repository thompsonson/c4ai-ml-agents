#!/bin/bash

# Batch Experiments Script for ML Agents
# Run multiple experiments with different configurations

set -e

echo "🧠 ML Agents Batch Experiments"
echo "=============================="

# Check if ml-agents CLI is available
if ! command -v ml-agents &> /dev/null; then
    echo "❌ ml-agents CLI not found. Please install the package first."
    exit 1
fi

# Validate environment
echo "🔍 Validating environment..."
ml-agents validate-env

# Create results directory
mkdir -p ./batch_results

# Array of experiments to run
declare -a experiments=(
    "single_experiment.yaml"
    "comparison_study.yaml"
    "advanced_reasoning.yaml"
)

echo ""
echo "🚀 Starting batch experiments..."

# Run each experiment
for config in "${experiments[@]}"; do
    echo ""
    echo "📋 Running experiment: $config"
    echo "----------------------------"

    config_path="examples/configs/$config"

    if [ ! -f "$config_path" ]; then
        echo "⚠️  Config file not found: $config_path"
        continue
    fi

    # Extract experiment name for output directory
    exp_name=$(basename "$config" .yaml)
    timestamp=$(date +"%Y%m%d_%H%M%S")
    output_dir="./batch_results/${exp_name}_${timestamp}"

    # Run the experiment
    if [[ "$config" == *"comparison"* ]]; then
        ml-agents compare --config "$config_path" --output "$output_dir" --verbose
    else
        ml-agents run --config "$config_path" --output "$output_dir" --verbose
    fi

    echo "✅ Completed: $config"
done

echo ""
echo "🎉 All batch experiments completed!"
echo "📂 Results saved in: ./batch_results/"
echo ""

# List all results
echo "📊 Experiment Results Summary:"
echo "-----------------------------"
find ./batch_results -name "*.csv" -type f | sort

#!/bin/bash

# Parallel benchmark script - runs multiple benchmarks simultaneously
# Usage: ./run_parallel_benchmarks.sh [APPROACH] [MAX_CONCURRENT_JOBS]
# Example: ./run_parallel_benchmarks.sh None 4

set -e

# Configuration
PROVIDER="local-openai"
MODEL="RedHatAI/Qwen2-1.5B-Instruct-quantized.w4a16"
API_BASE="http://pop-os:8000/v1"
MAX_TOKENS="8192"
APPROACH="${1:-None}"
MAX_CONCURRENT="${2:-4}"  # Number of parallel benchmark runs

# Benchmarks to run (add/remove as needed)
BENCHMARKS=(
    "BENCHMARK-51-PubMedQA.csv"
    "BENCHMARK-52-MedConceptsQA_Easy.csv"
    "BENCHMARK-53-MedConceptsQA_Medium.csv"
    "BENCHMARK-54-MedConceptsQA_Hard.csv"
    "BENCHMARK-55-ClinicBench_Treatment.csv"
    "BENCHMARK-56-ClinicBench_Hospitalization.csv"
    "BENCHMARK-57-PIID.csv"
    "BENCHMARK-58-FMD.csv"
    "BENCHMARK-59-FinBen.csv"
    "BENCHMARK-60-IFEval.csv"
    "BENCHMARK-61-CivilComments.csv"
    "BENCHMARK-62-ProSocial.csv"
)

echo "🚀 Running ${#BENCHMARKS[@]} benchmarks in parallel (max $MAX_CONCURRENT concurrent)"
echo "📊 Model: $MODEL"
echo "🔧 Provider: $PROVIDER"
echo "⚙️ Approach: $APPROACH"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Function to run a single benchmark
run_benchmark() {
    local benchmark="$1"
    local index="$2"
    local total="$3"

    echo "[$index/$total] 🔄 Starting: $benchmark (PID: $$)"

    # Run the experiment and capture output
    local log_file="./parallel_logs/${benchmark%.csv}_${approach}_$(date +%Y%m%d_%H%M%S).log"
    mkdir -p ./parallel_logs

    if uv run ml-agents eval run "$benchmark" "$APPROACH" \
        --provider "$PROVIDER" \
        --model "$MODEL" \
        --api-base "$API_BASE" \
        --skip-warnings \
        --max-tokens "$MAX_TOKENS" > "$log_file" 2>&1; then
        echo "✅ [$index/$total] SUCCESS: $benchmark"
        echo "   📄 Log: $log_file"
        return 0
    else
        echo "❌ [$index/$total] FAILED: $benchmark"
        echo "   📄 Error log: $log_file"
        return 1
    fi
}

# Export function for parallel execution
export -f run_benchmark
export PROVIDER MODEL API_BASE MAX_TOKENS APPROACH

# Counters
SUCCESSFUL=0
FAILED=0
TOTAL=${#BENCHMARKS[@]}

# Run benchmarks in parallel using GNU parallel or xargs
if command -v parallel &> /dev/null; then
    # Use GNU parallel if available (better control)
    echo "Using GNU parallel for job control..."
    printf '%s\n' "${BENCHMARKS[@]}" | \
    parallel -j "$MAX_CONCURRENT" --line-buffer \
        'run_benchmark {} {#} '"$TOTAL"
else
    # Fallback to manual job control
    echo "Using manual job control (install gnu-parallel for better performance)..."

    index=0
    for benchmark in "${BENCHMARKS[@]}"; do
        index=$((index + 1))

        # Run in background
        run_benchmark "$benchmark" "$index" "$TOTAL" &

        # Limit concurrent jobs
        while (( $(jobs -r | wc -l) >= MAX_CONCURRENT )); do
            wait -n  # Wait for any job to complete
        done
    done

    # Wait for all remaining jobs
    wait
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🏁 PARALLEL EXECUTION COMPLETE"
echo ""
echo "📊 Results saved in: ./outputs/{benchmark_name}/eval/"
echo "📄 Individual logs: ./parallel_logs/"
echo "🔍 Use 'uv run ml-agents results list' to view experiments"

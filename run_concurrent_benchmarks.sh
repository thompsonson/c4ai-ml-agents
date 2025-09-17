#!/bin/bash

# Enhanced Concurrent Benchmark Script - leverages Phase 16 concurrency
# This script combines process-level parallelism with API-level concurrency for maximum throughput
# Usage: ./run_concurrent_benchmarks.sh [APPROACH] [MAX_PROCESSES] [CONCURRENCY_LIMIT] [SAMPLES]
# Example: ./run_concurrent_benchmarks.sh None 4 10 50

set -e

# Configuration
PROVIDER="local-openai"
MODEL="RedHatAI/Qwen2-1.5B-Instruct-quantized.w4a16"
API_BASE="http://pop-os:8000/v1"
MAX_TOKENS="8192"
APPROACH="${1:-None}"
MAX_PROCESSES="${2:-4}"      # Number of parallel processes
CONCURRENCY_LIMIT="${3:-10}" # API concurrency within each process (Phase 16)
SAMPLES="${4:-}"             # Number of samples per benchmark (optional)

# Create unique run directory
RUN_TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="./concurrent_logs/$RUN_TIMESTAMP"

echo "🚀 Enhanced Concurrent Benchmark Execution (Phase 16)"
echo "📊 Process-level parallelism: $MAX_PROCESSES concurrent processes"
echo "🔀 API-level concurrency: $CONCURRENCY_LIMIT concurrent API calls per process"
echo "⚙️ Combined throughput potential: $(($MAX_PROCESSES * $CONCURRENCY_LIMIT)) concurrent API calls"
echo ""

# Benchmarks to run (focused subset for testing)
BENCHMARKS=(
    "BENCHMARK-01-GPQA.csv"
    "BENCHMARK-02-BoardgameQA.csv"
    "BENCHMARK-03-RobustLR.csv"
    "BENCHMARK-04-SciNLI.csv"
    "BENCHMARK-05-FOLIO.csv"
    "BENCHMARK-21-AquaRat.csv"
    "BENCHMARK-22-MATH3.csv"
    "BENCHMARK-23-MATH5.csv"
    "BENCHMARK-34-Text2SQL.csv"
    "BENCHMARK-35-LeetCode_Javascript_Easy.csv"
)

echo "🎯 Running ${#BENCHMARKS[@]} benchmarks with enhanced concurrency"
echo "📊 Model: $MODEL"
echo "🔧 Provider: $PROVIDER"
echo "⚙️ Approach: $APPROACH"
if [[ -n "$SAMPLES" ]]; then
    echo "🎲 Samples: $SAMPLES per benchmark"
else
    echo "🎲 Samples: Default (benchmark-specific)"
fi
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Function to update benchmark status
update_status() {
    local benchmark="$1"
    local status="$2"
    local pid="$3"
    local timestamp=$(date +%Y%m%d_%H%M%S)
    echo "$benchmark:$status:$pid:$timestamp" >> "$LOG_DIR/status.txt"
}

# Function to get current status of a benchmark
get_status() {
    local benchmark="$1"
    local status=$(grep "^$benchmark:" "$LOG_DIR/status.txt" 2>/dev/null | tail -1 | cut -d: -f2)
    case "$status" in
        "RUNNING") echo "🔄" ;;
        "SUCCESS") echo "✅" ;;
        "FAILED") echo "❌" ;;
        *) echo "⏳" ;;  # Queued/Unknown
    esac
}

# Function to get start time for a benchmark
get_benchmark_start_time() {
    local benchmark="$1"
    grep "^$benchmark:RUNNING:" "$LOG_DIR/status.txt" 2>/dev/null | tail -1 | cut -d: -f4
}

# Function to calculate and format elapsed time
get_elapsed_time() {
    local start_timestamp="$1"
    if [[ -z "$start_timestamp" ]]; then
        echo ""
        return
    fi

    local current_timestamp=$(date "+%s")
    local start_seconds=$(date -j -f "%Y%m%d_%H%M%S" "$start_timestamp" "+%s" 2>/dev/null)

    if [[ -n "$start_seconds" ]]; then
        local elapsed=$((current_timestamp - start_seconds))
        local hours=$((elapsed / 3600))
        local minutes=$(((elapsed % 3600) / 60))
        local seconds=$((elapsed % 60))

        if [[ $hours -gt 0 ]]; then
            printf "%dh%02dm" "$hours" "$minutes"
        elif [[ $minutes -gt 0 ]]; then
            printf "%dm%02ds" "$minutes" "$seconds"
        else
            printf "%ds" "$seconds"
        fi
    fi
}

# Function to extract concurrent performance metrics from log
get_concurrent_metrics() {
    local benchmark="$1"
    local log_file="$LOG_DIR/${benchmark%.csv}_${APPROACH}.log"

    if [[ -f "$log_file" ]]; then
        # Extract concurrent processing info
        local concurrent_info=$(grep "Concurrent.*processing completed" "$log_file" 2>/dev/null | tail -1)
        if [[ -n "$concurrent_info" ]]; then
            echo "$concurrent_info" | sed 's/.*Concurrent.*processing completed: //' | sed 's/, total time:.*//'
        fi
    fi
}

# Enhanced dashboard with Phase 16 metrics
show_concurrent_dashboard() {
    tput civis  # Hide cursor
    tput clear  # Initial clear

    trap 'tput cnorm; exit' INT TERM

    while true; do
        tput cup 0 0

        echo "🚀 Enhanced Concurrent Benchmark Dashboard - $(date '+%H:%M:%S')"
        echo "📊 Phase 16: Process-level (${MAX_PROCESSES}x) + API-level (${CONCURRENCY_LIMIT}x) = $(($MAX_PROCESSES * $CONCURRENCY_LIMIT))x potential"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

        local success=0 failed=0 running=0 queued=0

        for benchmark in "${BENCHMARKS[@]}"; do
            local status_icon=$(get_status "$benchmark")
            local benchmark_short=$(echo "$benchmark" | sed 's/BENCHMARK-//' | sed 's/.csv$//')
            local concurrent_metrics=$(get_concurrent_metrics "$benchmark")

            if [[ "$status_icon" == "🔄" ]]; then
                local start_time=$(get_benchmark_start_time "$benchmark")
                local runtime=$(get_elapsed_time "$start_time")
                printf "%-25s [%s] %s                \n" "$benchmark_short" "$runtime" "$status_icon"
            elif [[ "$status_icon" == "✅" && -n "$concurrent_metrics" ]]; then
                printf "%-25s (%s) %s              \n" "$benchmark_short" "$concurrent_metrics" "$status_icon"
            else
                printf "%-25s %s                                \n" "$benchmark_short" "$status_icon"
            fi

            case "$status_icon" in
                "✅") ((success++)) ;;
                "❌") ((failed++)) ;;
                "🔄") ((running++)) ;;
                "⏳") ((queued++)) ;;
            esac
        done

        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        printf "📈 Summary: ✅ %2d Success | ❌ %2d Failed | 🔄 %2d Running | ⏳ %2d Queued\n" "$success" "$failed" "$running" "$queued"
        printf "🔀 Concurrency: %d processes × %d API calls = %dx theoretical max\n" "$MAX_PROCESSES" "$CONCURRENCY_LIMIT" "$(($MAX_PROCESSES * $CONCURRENCY_LIMIT))"
        printf "📄 Logs: %s\n" "$LOG_DIR"
        printf "🔍 Press Ctrl+C to stop dashboard\n"

        # Exit if all benchmarks are done
        if [[ $((success + failed)) -eq ${#BENCHMARKS[@]} ]]; then
            printf "\n🏁 All benchmarks completed!\n"
            tput cnorm
            break
        fi

        sleep 2
    done
}

# Function to run a single benchmark with Phase 16 concurrency
run_concurrent_benchmark() {
    local benchmark="$1"
    local index="$2"
    local total="$3"

    update_status "$benchmark" "RUNNING" "$$"

    local log_file="$LOG_DIR/${benchmark%.csv}_${APPROACH}.log"
    mkdir -p "$LOG_DIR"

    # Build command with Phase 16 concurrency flags
    local cmd="uv run ml-agents eval run \"$benchmark\" \"$APPROACH\""
    cmd="$cmd --provider \"$PROVIDER\" --model \"$MODEL\" --api-base \"$API_BASE\""
    cmd="$cmd --concurrent --concurrency-limit $CONCURRENCY_LIMIT"  # Phase 16!
    cmd="$cmd --skip-warnings --max-tokens \"$MAX_TOKENS\" --verbose"

    if [[ -n "$SAMPLES" ]]; then
        cmd="$cmd --samples $SAMPLES"
    fi

    echo "🚀 Starting: $benchmark with ${CONCURRENCY_LIMIT}x API concurrency" >> "$log_file"
    echo "📝 Command: $cmd" >> "$log_file"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" >> "$log_file"

    if eval "$cmd" >> "$log_file" 2>&1; then
        update_status "$benchmark" "SUCCESS" "$$"
        echo "✅ Completed: $benchmark" >> "$log_file"
        return 0
    else
        update_status "$benchmark" "FAILED" "$$"
        echo "❌ Failed: $benchmark" >> "$log_file"
        return 1
    fi
}

# Export functions for parallel execution
export -f run_concurrent_benchmark update_status get_status get_concurrent_metrics get_benchmark_start_time get_elapsed_time
export PROVIDER MODEL API_BASE MAX_TOKENS APPROACH SAMPLES LOG_DIR CONCURRENCY_LIMIT

# Initialize
mkdir -p "$LOG_DIR"
> "$LOG_DIR/status.txt"

# Initialize all benchmarks as queued
for benchmark in "${BENCHMARKS[@]}"; do
    echo "$benchmark:QUEUED:0:$(date +%Y%m%d_%H%M%S)" >> "$LOG_DIR/status.txt"
done

TOTAL=${#BENCHMARKS[@]}

# Start enhanced dashboard in background
show_concurrent_dashboard > /dev/tty &
DASHBOARD_PID=$!
trap 'kill $DASHBOARD_PID 2>/dev/null' EXIT

echo "🚀 Starting $TOTAL benchmarks with enhanced concurrency..."
echo "📺 Dashboard running in background (PID: $DASHBOARD_PID)"
echo "🔀 Each process will use $CONCURRENCY_LIMIT concurrent API calls (Phase 16)"
echo ""

# Run benchmarks in parallel processes, each using internal API concurrency
if command -v parallel &> /dev/null; then
    printf '%s\n' "${BENCHMARKS[@]}" | \
    parallel -j "$MAX_PROCESSES" --line-buffer \
        'run_concurrent_benchmark {} {#} '"$TOTAL" 2>/dev/null
else
    # Fallback to manual job control
    index=0
    for benchmark in "${BENCHMARKS[@]}"; do
        index=$((index + 1))

        run_concurrent_benchmark "$benchmark" "$index" "$TOTAL" &

        # Limit concurrent processes
        while (( $(jobs -r | wc -l) >= MAX_PROCESSES )); do
            wait -n
        done
    done

    wait
fi

# Wait for dashboard to finish
wait $DASHBOARD_PID 2>/dev/null

# Enhanced final summary
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🏁 ENHANCED CONCURRENT EXECUTION COMPLETE (Phase 16)"
echo ""

SUCCESS=$(grep ":SUCCESS:" "$LOG_DIR/status.txt" | wc -l)
FAILED=$(grep ":FAILED:" "$LOG_DIR/status.txt" | wc -l)

echo "📈 Final Results: ✅ $SUCCESS Success | ❌ $FAILED Failed | 📊 Total: $TOTAL"
echo "🔀 Concurrency Used: $MAX_PROCESSES processes × $CONCURRENCY_LIMIT API calls = $(($MAX_PROCESSES * $CONCURRENCY_LIMIT))x"
echo "📊 Results saved in: ./outputs/{benchmark_name}/eval/"
echo "📄 Individual logs: $LOG_DIR"
echo "📄 Status log: $LOG_DIR/status.txt"
echo "🔍 Use 'uv run ml-agents results list' to view experiments"
echo ""
echo "🎉 Phase 16 concurrent processing successfully integrated!"
echo "💡 For vLLM endpoints, this should provide 2-5x performance improvement"

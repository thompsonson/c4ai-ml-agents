#!/bin/bash

# Parallel benchmark script - runs multiple benchmarks simultaneously
# Usage: ./run_parallel_benchmarks.sh [APPROACH] [MAX_CONCURRENT_JOBS] [SAMPLES]
# Example: ./run_parallel_benchmarks.sh None 4 10

set -e

# Configuration
PROVIDER="local-openai"
MODEL="RedHatAI/Qwen2-1.5B-Instruct-quantized.w4a16"
API_BASE="http://pop-os:8000/v1"
MAX_TOKENS="8192"
APPROACH="${1:-None}"
MAX_CONCURRENT="${2:-4}"  # Number of parallel benchmark runs
SAMPLES="${3:-}"  # Number of samples per benchmark (optional)

# Create unique run directory
RUN_TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="./parallel_logs/$RUN_TIMESTAMP"

# Benchmarks to run (add/remove as needed)
BENCHMARKS=(
    "BENCHMARK-01-GPQA.csv"
    "BENCHMARK-02-BoardgameQA.csv"
    "BENCHMARK-03-RobustLR.csv"
    "BENCHMARK-04-SciNLI.csv"
    "BENCHMARK-05-FOLIO.csv"
    "BENCHMARK-06-BBEH.csv"
    "BENCHMARK-07-BBH_Spatial.csv"
    "BENCHMARK-08-SpatialEval.csv"
    "BENCHMARK-09-SpartQA.csv"
    "BENCHMARK-10-BBH_Temporal.csv"
    "BENCHMARK-11-LTLbench.csv"
    "BENCHMARK-12-BBH_Temporal2.csv"
    "BENCHMARK-13-MuSR.csv"
    "BENCHMARK-14-NarrativeQA.csv"
    "BENCHMARK-15-Quac.csv"
    "BENCHMARK-16-StrategyQA.csv"
    "BENCHMARK-17-KnowLogic.csv"
    "BENCHMARK-18-ProcBench.csv"
    "BENCHMARK-19-SIQA.csv"
    "BENCHMARK-20-MuTual.csv"
    "BENCHMARK-21-AquaRat.csv"
    "BENCHMARK-22-MATH3.csv"
    "BENCHMARK-23-MATH5.csv"
    "BENCHMARK-24-AIME.csv"
    "BENCHMARK-25-MMLUPRO_LEGAL.csv"
    "BENCHMARK-26-CaseHOLD.csv"
    "BENCHMARK-27-CUAD.csv"
    "BENCHMARK-28-JEE_Chemistry.csv"
    "BENCHMARK-29-MMLUPRO_CHEMISTRY.csv"
    "BENCHMARK-30-ChemBench.csv"
    "BENCHMARK-31-JEE_Physics.csv"
    "BENCHMARK-32-MMLUPRO_PHYSICS.csv"
    "BENCHMARK-33-SciBench.csv"
    "BENCHMARK-34-Text2SQL.csv"
    "BENCHMARK-35-LeetCode_Javascript_Easy.csv"
    "BENCHMARK-36-LeetCode_Javascript_Medium.csv"
    "BENCHMARK-37-LeetCode_Javascript_Hard.csv"
    "BENCHMARK-38-LeetCode_Java_Easy.csv"
    "BENCHMARK-39-LeetCode_Java_Medium.csv"
    "BENCHMARK-40-LeetCode_Java_Hard.csv"
    "BENCHMARK-41-MBPP.csv"
    "BENCHMARK-42-LeetCode_Python_Easy.csv"
    "BENCHMARK-43-LeetCode_Python_Medium.csv"
    "BENCHMARK-44-LeetCode_Python_Hard.csv"
    "BENCHMARK-45-LeetCode_CPP_Easy.csv"
    "BENCHMARK-46-LeetCode_CPP_Medium.csv"
    "BENCHMARK-47-LeetCode_CPP_Hard.csv"
    "BENCHMARK-48-CupCASE.csv"
    "BENCHMARK-49-USMLE_MedQA.csv"
    "BENCHMARK-50-MedMCQA.csv"
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

# Function to get overall run start time
get_run_start_time() {
    grep ":QUEUED:" "$LOG_DIR/status.txt" 2>/dev/null | head -1 | cut -d: -f4
}

# Function to get sample progress from log file
get_sample_progress() {
    local benchmark="$1"
    local log_file="$LOG_DIR/${benchmark%.csv}_${APPROACH}.log"

    # Only parse for running benchmarks
    local status=$(get_status "$benchmark")
    if [[ "$status" != "🔄" ]]; then
        echo ""
        return
    fi

    # Check if log file exists
    if [[ ! -f "$log_file" ]]; then
        echo ""
        return
    fi

    # Extract total samples from log
    local total_samples=$(grep "Loaded .* samples" "$log_file" 2>/dev/null | sed -n 's/.*Loaded \([0-9]*\) samples.*/\1/p')

    # If no total samples found, return empty
    if [[ -z "$total_samples" ]]; then
        echo ""
        return
    fi

    # Get start time of this benchmark
    local start_time=$(grep "^$benchmark:RUNNING:" "$LOG_DIR/status.txt" 2>/dev/null | tail -1 | cut -d: -f4)
    if [[ -z "$start_time" ]]; then
        echo ""
        return
    fi

    # Calculate elapsed time (rough estimation)
    local start_timestamp=$(date -j -f "%Y%m%d_%H%M%S" "$start_time" "+%s" 2>/dev/null)
    local current_timestamp=$(date "+%s")

    if [[ -n "$start_timestamp" ]]; then
        local elapsed=$((current_timestamp - start_timestamp))

        # Estimate progress based on time (very rough - assume 30s per sample average)
        local estimated_samples=$((elapsed / 30))
        if [[ $estimated_samples -gt $total_samples ]]; then
            estimated_samples=$total_samples
        fi

        echo "~$estimated_samples/$total_samples"
    else
        echo "0/$total_samples"
    fi
}

# Function to show live dashboard
show_dashboard() {
    # Initialize terminal for smooth updates
    tput civis  # Hide cursor
    tput clear  # Initial clear

    # Cleanup on exit
    trap 'tput cnorm; exit' INT TERM

    while true; do
        # Position cursor at top instead of clearing (prevents flicker)
        tput cup 0 0

        echo "🚀 Benchmark Status Dashboard - $(date '+%H:%M:%S')"

        # Add run timing information
        local run_start=$(get_run_start_time)
        if [[ -n "$run_start" ]]; then
            local start_display=$(date -j -f "%Y%m%d_%H%M%S" "$run_start" "+%H:%M:%S" 2>/dev/null)
            local total_runtime=$(get_elapsed_time "$run_start")
            echo "⏱️  Run started: $start_display | Runtime: $total_runtime"
        fi

        local samples_text=""
        if [[ -n "$SAMPLES" ]]; then
            samples_text=" | 🎯 Samples: $SAMPLES"
        else
            samples_text=" | 🎯 Samples: Default (normally 50)"
        fi
        echo "📊 Model: $MODEL | 🔧 Provider: $PROVIDER | ⚙️ Approach: $APPROACH$samples_text"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

        local success=0 failed=0 running=0 queued=0

        for benchmark in "${BENCHMARKS[@]}"; do
            local status_icon=$(get_status "$benchmark")
            local benchmark_short=$(echo "$benchmark" | sed 's/BENCHMARK-//' | sed 's/.csv$//')
            local progress=$(get_sample_progress "$benchmark")

            # Add timing for running benchmarks
            if [[ "$status_icon" == "🔄" ]]; then
                local start_time=$(get_benchmark_start_time "$benchmark")
                local runtime=$(get_elapsed_time "$start_time")

                if [[ -n "$progress" ]]; then
                    printf "%-25s (%s) [%s] %s                \n" "$benchmark_short" "$progress" "$runtime" "$status_icon"
                else
                    printf "%-25s [%s] %s                        \n" "$benchmark_short" "$runtime" "$status_icon"
                fi
            else
                # Format completed/failed/queued benchmarks without timing
                if [[ -n "$progress" ]]; then
                    printf "%-25s (%s) %s                        \n" "$benchmark_short" "$progress" "$status_icon"
                else
                    printf "%-25s %s                                \n" "$benchmark_short" "$status_icon"
                fi
            fi

            case "$status_icon" in
                "✅") ((success++)) ;;
                "❌") ((failed++)) ;;
                "🔄") ((running++)) ;;
                "⏳") ((queued++)) ;;
            esac
        done

        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

        # Calculate average running time for running benchmarks
        local avg_runtime=""
        if [[ $running -gt 0 ]]; then
            local total_seconds=0
            local count=0
            for benchmark in "${BENCHMARKS[@]}"; do
                if [[ "$(get_status "$benchmark")" == "🔄" ]]; then
                    local start_time=$(get_benchmark_start_time "$benchmark")
                    if [[ -n "$start_time" ]]; then
                        local start_seconds=$(date -j -f "%Y%m%d_%H%M%S" "$start_time" "+%s" 2>/dev/null)
                        if [[ -n "$start_seconds" ]]; then
                            local elapsed=$(($(date "+%s") - start_seconds))
                            total_seconds=$((total_seconds + elapsed))
                            count=$((count + 1))
                        fi
                    fi
                fi
            done
            if [[ $count -gt 0 ]]; then
                local avg_seconds=$((total_seconds / count))
                local avg_minutes=$((avg_seconds / 60))
                local avg_secs=$((avg_seconds % 60))
                if [[ $avg_minutes -gt 0 ]]; then
                    avg_runtime=" [avg: ${avg_minutes}m${avg_secs}s]"
                else
                    avg_runtime=" [avg: ${avg_secs}s]"
                fi
            fi
        fi

        # Fixed width formatting for summary to prevent jumping
        printf "📈 Summary: ✅ %2d Success | ❌ %2d Failed | 🔄 %2d Running%s | ⏳ %2d Queued        \n" "$success" "$failed" "$running" "$avg_runtime" "$queued"
        printf "📄 Logs: %s                                    \n" "$LOG_DIR"
        printf "🔍 Press Ctrl+C to stop dashboard                                             \n"

        # Exit if all benchmarks are done
        if [[ $((success + failed)) -eq ${#BENCHMARKS[@]} ]]; then
            printf "\n🏁 All benchmarks completed!                                                   \n"
            # Restore cursor and exit cleanly
            tput cnorm
            break
        fi

        sleep 1
    done
}

# Function to run a single benchmark
run_benchmark() {
    local benchmark="$1"
    local index="$2"
    local total="$3"

    # Update status to running
    update_status "$benchmark" "RUNNING" "$$"

    # Run the experiment and capture output
    local log_file="$LOG_DIR/${benchmark%.csv}_${APPROACH}.log"
    mkdir -p "$LOG_DIR"

    # Build command with optional samples parameter
    local cmd="uv run ml-agents eval run \"$benchmark\" \"$APPROACH\" --provider \"$PROVIDER\" --model \"$MODEL\" --api-base \"$API_BASE\" --skip-warnings --max-tokens \"$MAX_TOKENS\" --verbose"
    if [[ -n "$SAMPLES" ]]; then
        cmd="$cmd --samples $SAMPLES"
    fi

    if eval "$cmd" > "$log_file" 2>&1; then
        update_status "$benchmark" "SUCCESS" "$$"
        return 0
    else
        update_status "$benchmark" "FAILED" "$$"
        return 1
    fi
}

# Export functions for parallel execution
export -f run_benchmark update_status get_status get_sample_progress get_benchmark_start_time get_elapsed_time get_run_start_time
export PROVIDER MODEL API_BASE MAX_TOKENS APPROACH SAMPLES LOG_DIR

# Initialize status file
mkdir -p "$LOG_DIR"
> "$LOG_DIR/status.txt"  # Clear status file

# Initialize all benchmarks as queued (only once)
for benchmark in "${BENCHMARKS[@]}"; do
    echo "$benchmark:QUEUED:0:$(date +%Y%m%d_%H%M%S)" >> "$LOG_DIR/status.txt"
done

TOTAL=${#BENCHMARKS[@]}

# Start dashboard in background (redirect to avoid interfering with parallel output)
show_dashboard > /dev/tty &
DASHBOARD_PID=$!

# Trap to kill dashboard on script exit
trap 'kill $DASHBOARD_PID 2>/dev/null' EXIT

echo "🚀 Starting $TOTAL benchmarks with dashboard..."
echo "📺 Dashboard running in background (PID: $DASHBOARD_PID)"
echo ""

# Run benchmarks in parallel using GNU parallel or xargs
if command -v parallel &> /dev/null; then
    # Use GNU parallel if available (better control)
    printf '%s\n' "${BENCHMARKS[@]}" | \
    parallel -j "$MAX_CONCURRENT" --line-buffer \
        'run_benchmark {} {#} '"$TOTAL" 2>/dev/null
else
    # Fallback to manual job control
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

# Wait for dashboard to finish (it will exit when all benchmarks are done)
wait $DASHBOARD_PID 2>/dev/null

# Final summary
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🏁 PARALLEL EXECUTION COMPLETE"
echo ""

# Calculate final stats
SUCCESS=$(grep ":SUCCESS:" "$LOG_DIR/status.txt" | wc -l)
FAILED=$(grep ":FAILED:" "$LOG_DIR/status.txt" | wc -l)
echo "📈 Final Results: ✅ $SUCCESS Success | ❌ $FAILED Failed | 📊 Total: $TOTAL"
echo "📊 Results saved in: ./outputs/{benchmark_name}/eval/"
echo "📄 Individual logs: $LOG_DIR"
echo "📄 Status log: $LOG_DIR/status.txt"
echo "🔍 Use 'uv run ml-agents results list' to view experiments"

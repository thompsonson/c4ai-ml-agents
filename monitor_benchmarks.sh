#!/bin/bash

# monitor_benchmarks.sh - Standalone benchmark dashboard monitor
# Usage: ./monitor_benchmarks.sh [LOG_DIR]
# Example: ./monitor_benchmarks.sh ./parallel_logs/20250908_153000

set -e

# Auto-detect latest run if no LOG_DIR specified
if [[ -z "$1" ]]; then
    LOG_DIR=$(ls -td ./parallel_logs/20*/ 2>/dev/null | head -1)
    if [[ -z "$LOG_DIR" ]]; then
        echo "❌ No benchmark runs found in ./parallel_logs/"
        echo "Usage: $0 [LOG_DIR]"
        echo "Example: $0 ./parallel_logs/20250908_153000"
        exit 1
    fi
    # Remove trailing slash for consistency
    LOG_DIR="${LOG_DIR%/}"
else
    LOG_DIR="$1"
fi

# Validate LOG_DIR exists and contains benchmark data
if [[ ! -d "$LOG_DIR" ]]; then
    echo "❌ Directory not found: $LOG_DIR"
    exit 1
fi

if [[ ! -f "$LOG_DIR/status.txt" ]]; then
    echo "❌ No status.txt found in: $LOG_DIR"
    echo "This doesn't appear to be a valid benchmark run directory."
    exit 1
fi

echo "📺 Monitoring benchmark run: $LOG_DIR"
echo "🔍 Parsing run configuration..."

# Parse configuration from existing run
parse_run_config() {
    # Extract benchmark list from status.txt
    BENCHMARKS=($(grep ":" "$LOG_DIR/status.txt" | cut -d: -f1 | sort -u))

    if [[ ${#BENCHMARKS[@]} -eq 0 ]]; then
        echo "❌ No benchmarks found in status.txt"
        exit 1
    fi

    # Find first available log file to extract configuration
    local first_log=""
    for benchmark in "${BENCHMARKS[@]}"; do
        local log_pattern="$LOG_DIR/${benchmark%.csv}_*.log"
        first_log=$(ls $log_pattern 2>/dev/null | head -1)
        if [[ -n "$first_log" ]]; then
            break
        fi
    done

    if [[ -n "$first_log" ]]; then
        # Extract MODEL, PROVIDER, APPROACH from log file
        MODEL=$(grep "Model:" "$first_log" 2>/dev/null | sed 's/.*Model:[[:space:]]*//' | sed 's/[[:space:]]*$//')
        PROVIDER=$(echo "$MODEL" | cut -d/ -f1)
        MODEL=$(echo "$MODEL" | cut -d/ -f2-)

        # Extract APPROACH from log file name or content
        APPROACH=$(basename "$first_log" | sed 's/.*_\([^_]*\)\.log$/\1/')

        # Extract SAMPLES from log content if available
        SAMPLES=$(grep "Samples:" "$first_log" 2>/dev/null | sed 's/.*Samples:[[:space:]]*//' | sed 's/[[:space:]]*$//')
    else
        # Fallback defaults if no logs available yet
        MODEL="Unknown"
        PROVIDER="Unknown"
        APPROACH="Unknown"
        SAMPLES=""
    fi

    echo "✅ Configuration loaded:"
    echo "   📊 Model: $MODEL"
    echo "   🔧 Provider: $PROVIDER"
    echo "   ⚙️ Approach: $APPROACH"
    echo "   🎯 Samples: ${SAMPLES:-Default}"
    echo "   📁 Benchmarks: ${#BENCHMARKS[@]} total"
    echo ""
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

        # Compact header - single line with all info
        local run_start=$(get_run_start_time)
        local runtime_info=""
        if [[ -n "$run_start" ]]; then
            local total_runtime=$(get_elapsed_time "$run_start")
            runtime_info=" | ⏱️ ${total_runtime}"
        fi

        # Calculate summary for header
        local success=0 failed=0 running=0 queued=0
        for benchmark in "${BENCHMARKS[@]}"; do
            local status_icon=$(get_status "$benchmark")
            case "$status_icon" in
                "✅") ((success++)) ;;
                "❌") ((failed++)) ;;
                "🔄") ((running++)) ;;
                "⏳") ((queued++)) ;;
            esac
        done

        # Calculate average running time for header
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
                local avg_hours=$((avg_seconds / 3600))
                local avg_minutes=$(((avg_seconds % 3600) / 60))
                local avg_secs=$((avg_seconds % 60))

                if [[ $avg_hours -gt 0 ]]; then
                    avg_runtime=" [avg: ${avg_hours}h${avg_minutes}m]"
                elif [[ $avg_minutes -gt 59 ]]; then
                    avg_runtime=" [avg: ${avg_hours}h${avg_minutes}m]"
                elif [[ $avg_minutes -gt 0 ]]; then
                    avg_runtime=" [avg: ${avg_minutes}m${avg_secs}s]"
                else
                    avg_runtime=" [avg: ${avg_secs}s]"
                fi
            fi
        fi

        echo "🚀 $(date '+%H:%M:%S') | $MODEL | $APPROACH | 🎯 ${SAMPLES:-Def}$runtime_info | ⏳ $queued ✅ $success ❌ $failed 🔄 $running$avg_runtime"
        echo "────────────────────────────────────────────────────────────────"

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
        done

        printf "🔍 Ctrl+C to exit | 📄 %s                                               \n" "$(basename "$LOG_DIR")"

        # Exit if all benchmarks are done
        if [[ $((success + failed)) -eq ${#BENCHMARKS[@]} ]]; then
            printf "\n🏁 All benchmarks completed!                                                   \n"
            # Restore cursor and exit cleanly
            tput cnorm
            break
        fi

        sleep 5
    done
}

# Main execution
parse_run_config
show_dashboard

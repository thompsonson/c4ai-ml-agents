#!/bin/bash

# Simple script to run all benchmark CSV files with a specific model configuration
# Usage: ./run_all_benchmarks.sh [APPROACH] [ADDITIONAL_ARGS...]
# Example: ./run_all_benchmarks.sh None --samples 10
# Example: ./run_all_benchmarks.sh ChainOfThought --samples 50 --verbose

set -e  # Exit on any error

# Default configuration - modify these as needed
PROVIDER="local-openai"
#MODEL="HuggingFaceTB/SmolLM2-1.7B-Instruct"
MODEL="RedHatAI/Qwen2-1.5B-Instruct-quantized.w4a16"
API_BASE="http://pop-os:8000/v1"
MAX_TOKENS="8192"
APPROACH="${1:-None}"  # Default to None approach if not specified

# Remove the approach from args if it was provided
if [ $# -gt 0 ]; then
    shift
fi
ADDITIONAL_ARGS="$@"

# List of all benchmarks (extracted from your list)

# already done benchmarks:
#
#
BENCHMARKS=(
#    "BENCHMARK-01-GPQA.csv"
#    "BENCHMARK-02-BoardgameQA.csv"
#    "BENCHMARK-03-RobustLR.csv"
#    "BENCHMARK-04-SciNLI.csv"
#    "BENCHMARK-05-FOLIO.csv"
#    "BENCHMARK-06-BBEH.csv"
#    "BENCHMARK-07-BBH_Spatial.csv"
#    "BENCHMARK-08-SpatialEval.csv"
#    "BENCHMARK-09-SpartQA.csv"
#    "BENCHMARK-10-BBH_Temporal.csv"
#    "BENCHMARK-11-LTLbench.csv"
#    "BENCHMARK-12-BBH_Temporal2.csv"
#    "BENCHMARK-13-MuSR.csv"
#    "BENCHMARK-14-NarrativeQA.csv"
#    "BENCHMARK-15-Quac.csv"
#    "BENCHMARK-16-StrategyQA.csv"
#    "BENCHMARK-17-KnowLogic.csv"
#    "BENCHMARK-18-ProcBench.csv"
#    "BENCHMARK-19-SIQA.csv"
#    "BENCHMARK-20-MuTual.csv"
#    "BENCHMARK-21-AquaRat.csv"
#    "BENCHMARK-22-MATH3.csv"
#    "BENCHMARK-23-MATH5.csv"
#    "BENCHMARK-24-AIME.csv"
#    "BENCHMARK-25-MMLUPRO_LEGAL.csv"
#    "BENCHMARK-26-CaseHOLD.csv"
#    "BENCHMARK-27-CUAD.csv"
#    "BENCHMARK-28-JEE_Chemistry.csv"
#    "BENCHMARK-29-MMLUPRO_CHEMISTRY.csv"
#    "BENCHMARK-30-ChemBench.csv"
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

echo "🚀 Running all ${#BENCHMARKS[@]} benchmarks with approach: $APPROACH"
echo "📊 Model: $MODEL"
echo "🔧 Provider: $PROVIDER"
echo "⚙️  Additional args: $ADDITIONAL_ARGS"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Counter for progress tracking
CURRENT=0
TOTAL=${#BENCHMARKS[@]}
SUCCESSFUL=0
FAILED=0

# Loop through each benchmark
for benchmark in "${BENCHMARKS[@]}"; do
    CURRENT=$((CURRENT + 1))
    echo ""
    echo "[$CURRENT/$TOTAL] 🔄 Running: $benchmark"
    echo "Command: uv run ml-agents eval run \"$benchmark\" \"$APPROACH\" --provider \"$PROVIDER\" --model \"$MODEL\" --api-base \"$API_BASE\" --skip-warnings --max-tokens \"$MAX_TOKENS\" $ADDITIONAL_ARGS"
    echo "────────────────────────────────────────────────────────────────────────"

    # Run the command and capture the result
    if uv run ml-agents eval run "$benchmark" "$APPROACH" \
        --provider "$PROVIDER" \
        --model "$MODEL" \
        --api-base "$API_BASE" \
        --skip-warnings \
        --max-tokens "$MAX_TOKENS" \
        $ADDITIONAL_ARGS; then
        echo "✅ SUCCESS: $benchmark"
        SUCCESSFUL=$((SUCCESSFUL + 1))
    else
        echo "❌ FAILED: $benchmark"
        FAILED=$((FAILED + 1))

        # Ask user if they want to continue on failure
        echo "❓ Continue with remaining benchmarks? [Y/n]"
        read -t 10 -n 1 response || response="y"  # Default to 'y' after 10 seconds
        echo ""
        if [[ "$response" =~ ^[Nn]$ ]]; then
            echo "🛑 Stopping execution as requested"
            break
        fi
    fi
done

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🏁 SUMMARY:"
echo "   ✅ Successful: $SUCCESSFUL/$TOTAL"
echo "   ❌ Failed: $FAILED/$TOTAL"
echo "   📊 Success rate: $(( SUCCESSFUL * 100 / (SUCCESSFUL + FAILED) ))%"
echo ""
echo "💾 Results are saved in: ./outputs/{benchmark_name}/eval/"
echo "🔍 Use 'uv run ml-agents results list' to view completed experiments"

#!/bin/bash
# =============================================================================
# lm_eval Benchmark Evaluation for Debug Set Models
# =============================================================================
# Evaluates all 10 debug set LoRA models + base model on:
#   1. MMLU (5-shot, accuracy) — Priority 1
#   2. GSM8K CoT (8-shot, exact match) — Priority 2
#   3. IFEval (0-shot, instruction following) — Priority 3
#
# Parallelization: up to 8 models evaluated concurrently (1 per GPU)
#
# Usage:
#   bash experiments/paper_scale/run_lm_eval.sh [benchmark]
#   benchmark: "mmlu" | "gsm8k" | "ifeval" | "all" (default: all)
# =============================================================================

set -e

BENCHMARK=${1:-all}
CONDA_PREFIX="/jizhicfs/karonhe/miniconda_karonhe/envs/spec_gcs"
LM_EVAL="$CONDA_PREFIX/bin/lm_eval"
export PATH="$CONDA_PREFIX/bin:$PATH"
export http_proxy="http://hy-proxy.woa.com:3128"
export https_proxy="http://hy-proxy.woa.com:3128"

BASE_MODEL="/jizhicfs/karonhe/models/LLM-Research/Meta-Llama-3___1-8B"
ADAPTER_DIR="/jizhicfs/karonhe/dataflex_saves/debug_set"
RESULT_DIR="/jizhicfs/karonhe/dataflex_saves/eval_results"
LOG_DIR="$RESULT_DIR/logs"

mkdir -p "$RESULT_DIR" "$LOG_DIR"

echo "============================================="
echo "lm_eval Benchmark Evaluation"
echo "============================================="
echo "Benchmark: $BENCHMARK"
echo "Base model: $BASE_MODEL"
echo "Adapter dir: $ADAPTER_DIR"
echo "Results: $RESULT_DIR"
echo ""

# Methods to evaluate (10 + base)
METHODS=(
    "base"
    "random"
    "loss"
    "less"
    "fisher_sft"
    "opt_gcs_logdet"
    "opt_gcs_score"
    "opt_gcs_unwhitened"
    "opt_gcs_rank50"
    "random_subspace_logdet"
    "grad_norm_topk"
)

# Benchmark configs: TASK:NSHOT
declare -A BENCH_TASK
declare -A BENCH_NSHOT
BENCH_TASK[mmlu]="mmlu"
BENCH_NSHOT[mmlu]=5
BENCH_TASK[gsm8k]="gsm8k_cot"
BENCH_NSHOT[gsm8k]=8
BENCH_TASK[ifeval]="ifeval"
BENCH_NSHOT[ifeval]=0

# =============================================================================
# Run evaluation for one method on one benchmark on one GPU
# =============================================================================
run_eval() {
    local method=$1
    local bench=$2
    local gpu=$3
    local task=${BENCH_TASK[$bench]}
    local nshot=${BENCH_NSHOT[$bench]}
    local output_dir="$RESULT_DIR/${method}/${bench}"
    local log="$LOG_DIR/${method}_${bench}.log"

    mkdir -p "$output_dir"

    # Skip if results already exist
    if ls "$output_dir"/results_*.json 1>/dev/null 2>&1; then
        echo "  [SKIP] $method/$bench (results exist)"
        return 0
    fi

    # Build model_args
    local model_args="pretrained=$BASE_MODEL,dtype=bfloat16,trust_remote_code=True"
    if [ "$method" != "base" ]; then
        local adapter_path="$ADAPTER_DIR/$method"
        if [ ! -f "$adapter_path/adapter_model.safetensors" ]; then
            echo "  [SKIP] $method/$bench (no adapter found)"
            return 1
        fi
        model_args="${model_args},peft=$adapter_path"
    fi

    echo "  [GPU $gpu] $method/$bench → $log"

    CUDA_VISIBLE_DEVICES=$gpu $LM_EVAL \
        --model hf \
        --model_args "$model_args" \
        --tasks "$task" \
        --num_fewshot "$nshot" \
        --batch_size auto \
        --output_path "$output_dir" \
        --log_samples \
        > "$log" 2>&1

    return $?
}

# =============================================================================
# Run all methods for one benchmark in parallel (8 GPUs)
# =============================================================================
run_benchmark() {
    local bench=$1
    echo ""
    echo "============================================="
    echo "Benchmark: $bench (task=${BENCH_TASK[$bench]}, nshot=${BENCH_NSHOT[$bench]})"
    echo "============================================="
    echo "Started: $(date '+%Y-%m-%d %H:%M:%S')"

    local pids=()
    local methods_running=()
    local gpu=0
    local total=${#METHODS[@]}
    local idx=0

    for method in "${METHODS[@]}"; do
        idx=$((idx + 1))

        # Wait if all 8 GPUs are busy
        if [ ${#pids[@]} -ge 8 ]; then
            # Wait for any one to finish
            wait -n "${pids[@]}" 2>/dev/null || true
            # Clean up finished PIDs
            local new_pids=()
            for pid in "${pids[@]}"; do
                if kill -0 "$pid" 2>/dev/null; then
                    new_pids+=("$pid")
                fi
            done
            pids=("${new_pids[@]}")
        fi

        # Find next free GPU
        gpu=$((idx % 8))

        run_eval "$method" "$bench" "$gpu" &
        pids+=($!)
        methods_running+=("$method")
    done

    # Wait for all remaining
    for pid in "${pids[@]}"; do
        wait "$pid" 2>/dev/null || true
    done

    echo "Finished: $(date '+%Y-%m-%d %H:%M:%S')"
    echo ""

    # Print results summary for this benchmark
    echo "--- $bench Results ---"
    for method in "${METHODS[@]}"; do
        local result_file=$(ls "$RESULT_DIR/${method}/${bench}"/results_*.json 2>/dev/null | head -1)
        if [ -f "$result_file" ]; then
            local acc=$($CONDA_PREFIX/bin/python -c "
import json, sys
with open('$result_file') as f:
    data = json.load(f)
results = data.get('results', {})
# Try different metric keys
for key in results:
    metrics = results[key]
    for mk in ['acc,none', 'exact_match,strict-match', 'prompt_level_strict_acc,none', 'acc_norm,none']:
        if mk in metrics:
            print(f'{metrics[mk]:.4f}')
            sys.exit(0)
print('N/A')
" 2>/dev/null || echo "N/A")
            printf "  %-30s %s\n" "$method" "$acc"
        else
            printf "  %-30s %s\n" "$method" "MISSING"
        fi
    done
    echo ""
}

# =============================================================================
# Main
# =============================================================================
if [[ "$BENCHMARK" == "all" || "$BENCHMARK" == "mmlu" ]]; then
    run_benchmark "mmlu"
fi

if [[ "$BENCHMARK" == "all" || "$BENCHMARK" == "gsm8k" ]]; then
    run_benchmark "gsm8k"
fi

if [[ "$BENCHMARK" == "all" || "$BENCHMARK" == "ifeval" ]]; then
    run_benchmark "ifeval"
fi

echo "============================================="
echo "All evaluations complete!"
echo "Results: $RESULT_DIR"
echo "============================================="

# =============================================================================
# Final summary table
# =============================================================================
echo ""
echo "=== FINAL SUMMARY ==="
printf "%-30s %10s %10s %10s\n" "Method" "MMLU" "GSM8K" "IFEval"
printf "%-30s %10s %10s %10s\n" "-----" "----" "-----" "------"

for method in "${METHODS[@]}"; do
    mmlu_acc="—"
    gsm_acc="—"
    ifeval_acc="—"

    for bench in mmlu gsm8k ifeval; do
        result_file=$(ls "$RESULT_DIR/${method}/${bench}"/results_*.json 2>/dev/null | head -1)
        if [ -f "$result_file" ]; then
            acc=$($CONDA_PREFIX/bin/python -c "
import json, sys
with open('$result_file') as f:
    data = json.load(f)
results = data.get('results', {})
for key in results:
    metrics = results[key]
    for mk in ['acc,none', 'exact_match,strict-match', 'prompt_level_strict_acc,none']:
        if mk in metrics:
            print(f'{metrics[mk]:.4f}')
            sys.exit(0)
print('N/A')
" 2>/dev/null || echo "N/A")
            case $bench in
                mmlu) mmlu_acc=$acc ;;
                gsm8k) gsm_acc=$acc ;;
                ifeval) ifeval_acc=$acc ;;
            esac
        fi
    done

    printf "%-30s %10s %10s %10s\n" "$method" "$mmlu_acc" "$gsm_acc" "$ifeval_acc"
done

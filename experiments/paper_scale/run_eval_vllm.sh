#!/bin/bash
# =============================================================================
# lm_eval with vLLM backend — Multi-benchmark evaluation
# =============================================================================
# Benchmarks: mmlu, gsm8k_cot, ifeval, bbh, humaneval, arc_challenge, arc_easy
#
# Usage:
#   bash run_eval_vllm.sh <model_path> [output_dir] [gpu_id]
#   bash run_eval_vllm.sh all_base_models
#
# Examples:
#   # Single model
#   bash run_eval_vllm.sh /path/to/model /path/to/output 0
#
#   # Base models comparison
#   bash run_eval_vllm.sh all_base_models
#
#   # LoRA model
#   bash run_eval_vllm.sh /path/to/base,peft=/path/to/adapter /path/to/output 0
# =============================================================================

set -e

CONDA_PREFIX="/jizhicfs/karonhe/miniconda_karonhe/envs/opencompass_hrm"
export PATH="$CONDA_PREFIX/bin:$PATH"
export http_proxy="http://hy-proxy.woa.com:3128"
export https_proxy="http://hy-proxy.woa.com:3128"
export no_proxy=".woa.com,mirrors.cloud.tencent.com,localhost,127.0.0.1"
export HF_ALLOW_CODE_EVAL="1"

LM_EVAL="$CONDA_PREFIX/bin/lm_eval"

# Benchmark configs
# Task name : few-shot
declare -A BENCHMARKS
BENCHMARKS[mmlu]="5"
BENCHMARKS[gsm8k_cot]="8"
BENCHMARKS[ifeval]="0"
BENCHMARKS[bbh]="3"
BENCHMARKS[humaneval]="0"
BENCHMARKS[arc_challenge]="25"
BENCHMARKS[arc_easy]="25"

# =============================================================================
# Run evaluation for a single model on all benchmarks
# =============================================================================
run_all_benchmarks() {
    local model_args="$1"
    local output_dir="$2"
    local gpu="$3"
    local model_name="$4"

    mkdir -p "$output_dir"
    echo "=== Evaluating: $model_name on GPU $gpu ==="
    echo "  Output: $output_dir"

    for task in "${!BENCHMARKS[@]}"; do
        local nshot=${BENCHMARKS[$task]}
        local task_dir="$output_dir/$task"
        local log="$output_dir/${task}.log"

        # Skip if already done
        if ls "$task_dir"/results_*.json 1>/dev/null 2>&1; then
            echo "  [SKIP] $task (results exist)"
            continue
        fi

        mkdir -p "$task_dir"
        echo "  [RUN] $task (${nshot}-shot) → $log"

        CUDA_VISIBLE_DEVICES=$gpu $LM_EVAL \
            --model vllm \
            --model_args "$model_args" \
            --tasks "$task" \
            --num_fewshot "$nshot" \
            --batch_size auto \
            --output_path "$task_dir" \
            --log_samples \
            > "$log" 2>&1 || echo "  [WARN] $task failed, check $log"
    done

    echo "  Done: $model_name"
    echo ""
}

# =============================================================================
# Extract results from output directory
# =============================================================================
print_results() {
    local output_dir="$1"
    local model_name="$2"

    $CONDA_PREFIX/bin/python - "$output_dir" "$model_name" << 'PYEOF'
import json, glob, sys
output_dir = sys.argv[1]
model_name = sys.argv[2]

metrics = {}
# MMLU
f = glob.glob(f"{output_dir}/mmlu/results_*.json")
if f:
    metrics['MMLU'] = json.load(open(f[-1]))['results']['mmlu']['acc,none']

# GSM8K
f = glob.glob(f"{output_dir}/gsm8k_cot/results_*.json")
if f:
    r = json.load(open(f[-1]))['results']
    for k in r:
        if 'exact_match,strict-match' in r[k]:
            metrics['GSM8K'] = r[k]['exact_match,strict-match']; break

# IFEval
f = glob.glob(f"{output_dir}/ifeval/results_*.json")
if f:
    r = json.load(open(f[-1]))['results']
    for k in r:
        if 'prompt_level_strict_acc,none' in r[k]:
            metrics['IFEval'] = r[k]['prompt_level_strict_acc,none']; break

# BBH
f = glob.glob(f"{output_dir}/bbh/results_*.json")
if f:
    r = json.load(open(f[-1]))['results']
    # BBH is a group task, look for aggregate
    for k in r:
        if 'exact_match,none' in r[k]:
            metrics['BBH'] = r[k]['exact_match,none']; break
        if 'acc_norm,none' in r[k]:
            metrics['BBH'] = r[k]['acc_norm,none']; break

# HumanEval
f = glob.glob(f"{output_dir}/humaneval/results_*.json")
if f:
    r = json.load(open(f[-1]))['results']
    for k in r:
        if 'pass@1,none' in r[k]:
            metrics['HumanEval'] = r[k]['pass@1,none']; break

# ARC-Challenge
f = glob.glob(f"{output_dir}/arc_challenge/results_*.json")
if f:
    r = json.load(open(f[-1]))['results']
    for k in r:
        if 'acc_norm,none' in r[k]:
            metrics['ARC-C'] = r[k]['acc_norm,none']; break
        if 'acc,none' in r[k]:
            metrics['ARC-C'] = r[k]['acc,none']; break

# ARC-Easy
f = glob.glob(f"{output_dir}/arc_easy/results_*.json")
if f:
    r = json.load(open(f[-1]))['results']
    for k in r:
        if 'acc_norm,none' in r[k]:
            metrics['ARC-E'] = r[k]['acc_norm,none']; break
        if 'acc,none' in r[k]:
            metrics['ARC-E'] = r[k]['acc,none']; break

print(f"\n{'='*60}")
print(f"  {model_name}")
print(f"{'='*60}")
header = "  ".join(f"{k:>9}" for k in ['MMLU','GSM8K','IFEval','BBH','HumanEval','ARC-C','ARC-E'])
print(f"  {header}")
vals = "  ".join(f"{metrics.get(k, 'N/A'):>9.4f}" if isinstance(metrics.get(k), float) else f"{'N/A':>9}" for k in ['MMLU','GSM8K','IFEval','BBH','HumanEval','ARC-C','ARC-E'])
print(f"  {vals}")
PYEOF
}

# =============================================================================
# Main: Run base models comparison
# =============================================================================
if [ "$1" == "all_base_models" ]; then
    RESULT_BASE="/jizhicfs/karonhe/dataflex_saves/eval_results_base_models"
    mkdir -p "$RESULT_BASE"

    echo "============================================="
    echo "Base Models Evaluation (vLLM backend)"
    echo "Benchmarks: mmlu, gsm8k_cot, ifeval, bbh, humaneval, arc_challenge, arc_easy"
    echo "============================================="

    # Llama-3.1-8B on GPU 0
    run_all_benchmarks \
        "pretrained=/jizhicfs/karonhe/models/LLM-Research/Meta-Llama-3___1-8B,dtype=bfloat16,tensor_parallel_size=1,gpu_memory_utilization=0.9,max_model_len=4096" \
        "$RESULT_BASE/llama3.1-8b" \
        "0" \
        "Llama-3.1-8B"

    # Llama-2-7B on GPU 1
    run_all_benchmarks \
        "pretrained=/jizhicfs/karonhe/models/Llama-2-7b-hf,dtype=bfloat16,tensor_parallel_size=1,gpu_memory_utilization=0.9,max_model_len=4096" \
        "$RESULT_BASE/llama2-7b" \
        "1" \
        "Llama-2-7B"

    # Print results
    print_results "$RESULT_BASE/llama3.1-8b" "Llama-3.1-8B"
    print_results "$RESULT_BASE/llama2-7b" "Llama-2-7B"

    echo ""
    echo "============================================="
    echo "Done! Results in: $RESULT_BASE"
    echo "============================================="

else
    # Single model mode
    MODEL_ARGS="$1"
    OUTPUT_DIR="${2:-/tmp/lm_eval_output}"
    GPU="${3:-0}"
    MODEL_NAME="${4:-custom_model}"

    if [ -z "$MODEL_ARGS" ]; then
        echo "Usage: $0 <model_args|all_base_models> [output_dir] [gpu_id] [model_name]"
        echo ""
        echo "Examples:"
        echo "  $0 all_base_models"
        echo "  $0 'pretrained=/path/to/model,dtype=bfloat16,tensor_parallel_size=1' /output 0 my_model"
    else
        run_all_benchmarks "$MODEL_ARGS" "$OUTPUT_DIR" "$GPU" "$MODEL_NAME"
        print_results "$OUTPUT_DIR" "$MODEL_NAME"
    fi
fi

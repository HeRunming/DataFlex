#!/bin/bash
# =============================================================================
# Full lm_eval for Final Table: 16 models × 7 benchmarks (vLLM backend)
# =============================================================================
# Uses opencompass_hrm env (vLLM 0.12.0 + PyTorch 2.9 + CUDA 12.8)
# Benchmarks: mmlu, gsm8k_cot, ifeval, bbh, humaneval, arc_challenge, arc_easy
# =============================================================================

set -e

CONDA_PREFIX="/jizhicfs/karonhe/miniconda_karonhe/envs/opencompass_hrm"
LM_EVAL="$CONDA_PREFIX/bin/lm_eval"
export PATH="$CONDA_PREFIX/bin:$PATH"
export http_proxy="http://hy-proxy.woa.com:3128"
export https_proxy="http://hy-proxy.woa.com:3128"
export HF_ALLOW_CODE_EVAL="1"

BASE_MODEL="/jizhicfs/karonhe/models/LLM-Research/Meta-Llama-3___1-8B"
ADAPTER_DIR="/jizhicfs/karonhe/dataflex_saves/final_table"
RESULT_DIR="/jizhicfs/karonhe/dataflex_saves/eval_results_final_full"
LOG_DIR="$RESULT_DIR/logs"

mkdir -p "$RESULT_DIR" "$LOG_DIR"

METHODS=(
    random_s42 loss_s42 less_s42 fisher_sft_s42
    grad_norm_topk_s42
    rsub_own_seed1 rsub_own_seed2 rsub_own_seed3
    hybrid_add_l025_s42 hybrid_mul_g025_s42 hybrid_mul_g05_s42 logdet_nopref_s42
    hybrid_add_l025_s1 hybrid_add_l025_s2
    hybrid_mul_g025_s1 hybrid_mul_g025_s2
)

declare -A BENCH_TASK BENCH_NSHOT
BENCH_TASK[mmlu]="mmlu"
BENCH_NSHOT[mmlu]=5
BENCH_TASK[gsm8k]="gsm8k_cot"
BENCH_NSHOT[gsm8k]=8
BENCH_TASK[ifeval]="ifeval"
BENCH_NSHOT[ifeval]=0
BENCH_TASK[bbh]="bbh"
BENCH_NSHOT[bbh]=3
BENCH_TASK[humaneval]="humaneval"
BENCH_NSHOT[humaneval]=0
BENCH_TASK[arc_c]="arc_challenge"
BENCH_NSHOT[arc_c]=25
BENCH_TASK[arc_e]="arc_easy"
BENCH_NSHOT[arc_e]=25

run_eval() {
    local method=$1 bench=$2 gpu=$3
    local task=${BENCH_TASK[$bench]} nshot=${BENCH_NSHOT[$bench]}
    local output_dir="$RESULT_DIR/${method}/${bench}"
    local log="$LOG_DIR/${method}_${bench}.log"
    mkdir -p "$output_dir"

    # Skip if results already exist
    if ls "$output_dir"/results_*.json 1>/dev/null 2>&1; then
        return 0
    fi

    local adapter_path="$ADAPTER_DIR/$method"
    # vLLM 0.12.0 + LoRA + loglikelihood crashes (CUDA illegal mem access in
    # _get_prompt_logprobs_dict) unless we disable chunked prefill / prefix
    # caching / CUDA graphs. Slower but stable.
    local model_args="pretrained=$BASE_MODEL,lora_local_path=$adapter_path,max_lora_rank=16,dtype=bfloat16,trust_remote_code=True,gpu_memory_utilization=0.85,max_model_len=4096,enable_chunked_prefill=False,enable_prefix_caching=False,enforce_eager=True"

    local extra_args=""
    if [ "$bench" == "humaneval" ]; then
        extra_args="--confirm_run_unsafe_code"
    fi

    CUDA_VISIBLE_DEVICES=$gpu $LM_EVAL \
        --model vllm \
        --model_args "$model_args" \
        --tasks "$task" \
        --num_fewshot "$nshot" \
        --batch_size auto \
        --output_path "$output_dir" \
        --log_samples \
        $extra_args \
        > "$log" 2>&1
}

run_benchmark() {
    local bench=$1
    echo ""
    echo "=== Benchmark: $bench ($(date '+%H:%M:%S')) ==="

    local pids=() gpu=0
    for method in "${METHODS[@]}"; do
        # Wait if 8 GPUs busy
        while [ ${#pids[@]} -ge 8 ]; do
            local new_pids=()
            for pid in "${pids[@]}"; do
                if kill -0 "$pid" 2>/dev/null; then
                    new_pids+=("$pid")
                fi
            done
            pids=("${new_pids[@]}")
            [ ${#pids[@]} -ge 8 ] && sleep 10
        done

        gpu=$((${#pids[@]} % 8))
        echo "  [GPU $gpu] $method/$bench"
        run_eval "$method" "$bench" "$gpu" &
        pids+=($!)
    done

    for pid in "${pids[@]}"; do wait "$pid" 2>/dev/null || true; done
    echo "  Done: $(date '+%H:%M:%S')"
}

echo "============================================="
echo "Final Table Full Evaluation: 16 models × 7 benchmarks"
echo "Started: $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================="

# Run benchmarks in order: fast ones first
run_benchmark "mmlu"
run_benchmark "arc_c"
run_benchmark "arc_e"
run_benchmark "ifeval"
run_benchmark "bbh"
run_benchmark "gsm8k"
run_benchmark "humaneval"

echo ""
echo "============================================="
echo "All evaluations complete! $(date '+%H:%M:%S')"
echo "============================================="

# Summary
$CONDA_PREFIX/bin/python << 'PYEOF'
import json, glob

RD = "/jizhicfs/karonhe/dataflex_saves/eval_results_final_full"
methods = ["random_s42","loss_s42","less_s42","fisher_sft_s42",
           "grad_norm_topk_s42",
           "rsub_own_seed1","rsub_own_seed2","rsub_own_seed3",
           "hybrid_add_l025_s42","hybrid_mul_g025_s42","hybrid_mul_g05_s42","logdet_nopref_s42",
           "hybrid_add_l025_s1","hybrid_add_l025_s2",
           "hybrid_mul_g025_s1","hybrid_mul_g025_s2"]

def get_metric(results_dict, task_name):
    for key, metrics in results_dict.items():
        if not isinstance(metrics, dict):
            continue
        if key == task_name:
            for mk in ['acc,none', 'exact_match,strict-match', 'prompt_level_strict_acc,none',
                       'pass@1,none', 'pass@1,create_test', 'acc_norm,none', 'exact_match,get-answer']:
                if mk in metrics:
                    return metrics[mk]
    return None

bench_map = {'mmlu': 'mmlu', 'gsm8k': 'gsm8k_cot', 'ifeval': 'ifeval',
             'bbh': 'bbh', 'humaneval': 'humaneval', 'arc_c': 'arc_challenge', 'arc_e': 'arc_easy'}

header = f"{'Method':<22}" + "".join(f"{b:>9}" for b in bench_map.keys())
print(header)
print("-" * len(header))

for m in methods:
    row = f"{m:<22}"
    for bench_key, task_name in bench_map.items():
        f = glob.glob(f"{RD}/{m}/{bench_key}/*/results_*.json")
        if f:
            results = json.load(open(f[-1]))['results']
            val = get_metric(results, task_name)
            row += f"{val:>9.4f}" if val is not None else f"{'???':>9}"
        else:
            row += f"{'---':>9}"
    print(row)
PYEOF

#!/bin/bash
# =============================================================================
# lm_eval for Round 2 Debug Set: 16 models × 3 benchmarks (MMLU, GSM8K, IFEval)
# =============================================================================
set -e

CONDA_PREFIX="/jizhicfs/karonhe/miniconda_karonhe/envs/spec_gcs"
LM_EVAL="$CONDA_PREFIX/bin/lm_eval"
export PATH="$CONDA_PREFIX/bin:$PATH"
export http_proxy="http://hy-proxy.woa.com:3128"
export https_proxy="http://hy-proxy.woa.com:3128"

BASE_MODEL="/jizhicfs/karonhe/models/LLM-Research/Meta-Llama-3___1-8B"
ADAPTER_DIR="/jizhicfs/karonhe/dataflex_saves/debug_set_round2"
RESULT_DIR="/jizhicfs/karonhe/dataflex_saves/eval_results_round2"
LOG_DIR="$RESULT_DIR/logs"

mkdir -p "$RESULT_DIR" "$LOG_DIR"

METHODS=(
    rsub_seed1 rsub_seed2 rsub_seed3 rsub_seed4 rsub_seed5
    hybrid_add_l025 hybrid_add_l05 hybrid_add_l10 hybrid_add_l20
    hybrid_mul_g025 hybrid_mul_g05 hybrid_mul_g10
    score_beta0 score_beta025
    logdet_pref20 logdet_nopref
)

declare -A BENCH_TASK BENCH_NSHOT
BENCH_TASK[mmlu]="mmlu"
BENCH_NSHOT[mmlu]=5
BENCH_TASK[gsm8k]="gsm8k_cot"
BENCH_NSHOT[gsm8k]=8
BENCH_TASK[ifeval]="ifeval"
BENCH_NSHOT[ifeval]=0

run_eval() {
    local method=$1 bench=$2 gpu=$3
    local task=${BENCH_TASK[$bench]} nshot=${BENCH_NSHOT[$bench]}
    local output_dir="$RESULT_DIR/${method}/${bench}"
    local log="$LOG_DIR/${method}_${bench}.log"
    mkdir -p "$output_dir"

    if ls "$output_dir"/results_*.json 1>/dev/null 2>&1; then
        return 0
    fi

    local adapter_path="$ADAPTER_DIR/$method"
    local model_args="pretrained=$BASE_MODEL,peft=$adapter_path,dtype=bfloat16,trust_remote_code=True"

    CUDA_VISIBLE_DEVICES=$gpu $LM_EVAL \
        --model hf \
        --model_args "$model_args" \
        --tasks "$task" \
        --num_fewshot "$nshot" \
        --batch_size auto \
        --output_path "$output_dir" \
        --log_samples \
        > "$log" 2>&1
}

run_benchmark() {
    local bench=$1
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
echo "Round 2 Evaluation: 16 models × 3 benchmarks"
echo "Started: $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================="

run_benchmark "mmlu"
run_benchmark "gsm8k"
run_benchmark "ifeval"

echo ""
echo "============================================="
echo "All evaluations complete! $(date '+%H:%M:%S')"
echo "============================================="

# Summary
echo ""
echo "=== Results Summary ==="
$CONDA_PREFIX/bin/python - "$RESULT_DIR" << 'PYEOF'
import json, glob, sys
RD = sys.argv[1]
methods = sorted([d.split('/')[-1] for d in glob.glob(f"{RD}/*/mmlu")])
if not methods:
    methods = ["rsub_seed1","rsub_seed2","rsub_seed3","rsub_seed4","rsub_seed5",
               "hybrid_add_l025","hybrid_add_l05","hybrid_add_l10","hybrid_add_l20",
               "hybrid_mul_g025","hybrid_mul_g05","hybrid_mul_g10",
               "score_beta0","score_beta025","logdet_pref20","logdet_nopref"]

print(f"{'Method':<22} {'MMLU':>7} {'GSM8K':>7} {'IFEval':>7}")
print("-" * 45)
for m in methods:
    mmlu = gsm = ifev = "---"
    f = glob.glob(f"{RD}/{m}/mmlu/*/results_*.json")
    if f:
        r = json.load(open(f[0]))['results']
        mmlu = f"{r['mmlu']['acc,none']:.4f}"
    f = glob.glob(f"{RD}/{m}/gsm8k/*/results_*.json")
    if f:
        r = json.load(open(f[0]))['results']
        for k in r:
            if 'exact_match,strict-match' in r[k]:
                gsm = f"{r[k]['exact_match,strict-match']:.4f}"; break
    f = glob.glob(f"{RD}/{m}/ifeval/*/results_*.json")
    if f:
        r = json.load(open(f[0]))['results']
        for k in r:
            if 'prompt_level_strict_acc,none' in r[k]:
                ifev = f"{r[k]['prompt_level_strict_acc,none']:.4f}"; break
    print(f"{m:<22} {mmlu:>7} {gsm:>7} {ifev:>7}")
PYEOF

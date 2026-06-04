#!/bin/bash
# Re-run only failed (method, benchmark) pairs from final-table eval.
# Tracks GPU assignment correctly (one job per GPU; reuse the GPU that just freed).
set -u
CONDA_PREFIX="/jizhicfs/karonhe/miniconda_karonhe/envs/opencompass_hrm"
LM_EVAL="$CONDA_PREFIX/bin/lm_eval"
export PATH="$CONDA_PREFIX/bin:$PATH"
export http_proxy="http://hy-proxy.woa.com:3128"
export https_proxy="http://hy-proxy.woa.com:3128"
export HF_ALLOW_CODE_EVAL="1"

BASE_MODEL="/jizhicfs/karonhe/models/LLM-Research/Meta-Llama-3___1-8B"
ADAPTER_DIR="/jizhicfs/karonhe/dataflex_saves/final_table"
RESULT_DIR="/jizhicfs/karonhe/dataflex_saves/eval_results_final_full"
LOG_DIR="$RESULT_DIR/logs_retry"
mkdir -p "$LOG_DIR"

declare -A BENCH_TASK BENCH_NSHOT
BENCH_TASK[mmlu]="mmlu";              BENCH_NSHOT[mmlu]=5
BENCH_TASK[gsm8k]="gsm8k_cot";        BENCH_NSHOT[gsm8k]=8
BENCH_TASK[ifeval]="ifeval";          BENCH_NSHOT[ifeval]=0
BENCH_TASK[bbh]="bbh";                BENCH_NSHOT[bbh]=3
BENCH_TASK[humaneval]="humaneval";    BENCH_NSHOT[humaneval]=0
BENCH_TASK[arc_c]="arc_challenge";    BENCH_NSHOT[arc_c]=25
BENCH_TASK[arc_e]="arc_easy";         BENCH_NSHOT[arc_e]=25

run_one() {
    local method=$1 bench=$2 gpu=$3
    local task=${BENCH_TASK[$bench]} nshot=${BENCH_NSHOT[$bench]}
    local out="$RESULT_DIR/${method}/${bench}"
    local log="$LOG_DIR/${method}_${bench}.log"
    mkdir -p "$out"
    # Skip if already done (e.g. another run filled it in)
    if ls "$out"/results_*.json 1>/dev/null 2>&1; then
        echo "  [GPU $gpu] SKIP $method/$bench (already done)"
        return 0
    fi
    # Lower gpu_memory_utilization (0.80) for safety margin
    local model_args="pretrained=$BASE_MODEL,lora_local_path=$ADAPTER_DIR/$method,max_lora_rank=16,dtype=bfloat16,trust_remote_code=True,gpu_memory_utilization=0.80,max_model_len=4096,enable_chunked_prefill=False,enable_prefix_caching=False,enforce_eager=True"
    local extra=""
    [ "$bench" == "humaneval" ] && extra="--confirm_run_unsafe_code"
    CUDA_VISIBLE_DEVICES=$gpu $LM_EVAL \
        --model vllm --model_args "$model_args" \
        --tasks "$task" --num_fewshot "$nshot" \
        --batch_size auto --output_path "$out" --log_samples \
        $extra > "$log" 2>&1
}

# Build failure list dynamically
METHODS=(random_s42 loss_s42 less_s42 fisher_sft_s42 grad_norm_topk_s42 rsub_own_seed1 rsub_own_seed2 rsub_own_seed3 hybrid_add_l025_s42 hybrid_mul_g025_s42 hybrid_mul_g05_s42 logdet_nopref_s42 hybrid_add_l025_s1 hybrid_add_l025_s2 hybrid_mul_g025_s1 hybrid_mul_g025_s2)
BENCHES=(mmlu arc_c arc_e ifeval bbh gsm8k humaneval)
JOBS=()
for m in "${METHODS[@]}"; do
    for b in "${BENCHES[@]}"; do
        if ! ls $RESULT_DIR/$m/$b/*/results_*.json >/dev/null 2>&1; then
            JOBS+=("$m:$b")
        fi
    done
done

echo "============================================="
echo "Retry $(echo ${#JOBS[@]}) failed jobs"
echo "Started: $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================="
printf '  %s\n' "${JOBS[@]}"
echo ""

# GPU pool: 8 slots, track which GPU each pid uses so we can reclaim it
declare -A GPU_PID  # gpu_index -> pid
free_gpus=(0 1 2 3 4 5 6 7)

for spec in "${JOBS[@]}"; do
    IFS=':' read -r m b <<< "$spec"
    # Wait for a free GPU
    while [ ${#free_gpus[@]} -eq 0 ]; do
        for g in 0 1 2 3 4 5 6 7; do
            pid=${GPU_PID[$g]:-0}
            if [ "$pid" != "0" ] && ! kill -0 "$pid" 2>/dev/null; then
                wait "$pid" 2>/dev/null || true
                unset GPU_PID[$g]
                free_gpus+=("$g")
            fi
        done
        [ ${#free_gpus[@]} -eq 0 ] && sleep 5
    done
    # Pop a free GPU and dispatch
    gpu=${free_gpus[0]}
    free_gpus=("${free_gpus[@]:1}")
    echo "  [$(date '+%H:%M:%S')][GPU $gpu] $m / $b"
    run_one "$m" "$b" "$gpu" &
    GPU_PID[$gpu]=$!
done

# Drain
for g in "${!GPU_PID[@]}"; do
    pid=${GPU_PID[$g]}
    wait "$pid" 2>/dev/null || true
done

echo ""
echo "============================================="
echo "Retry complete: $(date '+%H:%M:%S')"
echo "============================================="

# Re-print summary
$CONDA_PREFIX/bin/python << 'PYEOF'
import json, glob
RD = "/jizhicfs/karonhe/dataflex_saves/eval_results_final_full"
methods = ["random_s42","loss_s42","less_s42","fisher_sft_s42","grad_norm_topk_s42",
           "rsub_own_seed1","rsub_own_seed2","rsub_own_seed3",
           "hybrid_add_l025_s42","hybrid_mul_g025_s42","hybrid_mul_g05_s42","logdet_nopref_s42",
           "hybrid_add_l025_s1","hybrid_add_l025_s2","hybrid_mul_g025_s1","hybrid_mul_g025_s2"]
bench_map = {'mmlu':'mmlu','gsm8k':'gsm8k_cot','ifeval':'ifeval','bbh':'bbh',
             'humaneval':'humaneval','arc_c':'arc_challenge','arc_e':'arc_easy'}
def get_metric(d, t):
    for k, v in d.items():
        if isinstance(v, dict) and k == t:
            for mk in ['acc,none','exact_match,strict-match','prompt_level_strict_acc,none',
                       'pass@1,none','pass@1,create_test','acc_norm,none','exact_match,get-answer']:
                if mk in v: return v[mk]
    return None
hdr = f"{'Method':<22}" + "".join(f"{b:>9}" for b in bench_map)
print(hdr); print("-"*len(hdr))
for m in methods:
    row = f"{m:<22}"
    for bk, tn in bench_map.items():
        f = glob.glob(f"{RD}/{m}/{bk}/*/results_*.json")
        if f:
            v = get_metric(json.load(open(f[-1]))['results'], tn)
            row += f"{v:>9.4f}" if v is not None else f"{'???':>9}"
        else:
            row += f"{'---':>9}"
    print(row)
PYEOF

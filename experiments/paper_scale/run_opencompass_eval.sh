#!/bin/bash
# =============================================================================
# OpenCompass Evaluation for Round 2: vLLM backend with LoRA
# =============================================================================
# Evaluates all Round 2 models on GSM8K + IFEval using opencompass + vLLM
# Much faster than lm_eval HF backend for generation tasks.
#
# Usage:
#   bash experiments/paper_scale/run_opencompass_eval.sh
# =============================================================================

set -e

CONDA_PREFIX="/jizhicfs/karonhe/miniconda_karonhe/envs/opencompass_hrm"
export PATH="$CONDA_PREFIX/bin:$PATH"
export http_proxy="http://hy-proxy.woa.com:3128"
export https_proxy="http://hy-proxy.woa.com:3128"

OPENCOMPASS_DIR="/jizhicfs/karonhe/dcai_eval/math/opencompass"
BASE_MODEL="/jizhicfs/karonhe/models/LLM-Research/Meta-Llama-3___1-8B"
RESULT_DIR="/jizhicfs/karonhe/dataflex_saves/eval_results_round2/opencompass"

# Round 2 models
ADAPTER_DIR="/jizhicfs/karonhe/dataflex_saves/debug_set_round2"
# Round 1 models (for comparison)
ADAPTER_DIR_R1="/jizhicfs/karonhe/dataflex_saves/debug_set"

mkdir -p "$RESULT_DIR"

# All methods to evaluate (Round 2 + key Round 1 comparisons)
ROUND2_METHODS=(
    rsub_seed1 rsub_seed2 rsub_seed3 rsub_seed4 rsub_seed5
    hybrid_add_l025 hybrid_add_l05 hybrid_add_l10 hybrid_add_l20
    hybrid_mul_g025 hybrid_mul_g05 hybrid_mul_g10
    score_beta0 score_beta025
    logdet_pref20 logdet_nopref
)

ROUND1_METHODS=(
    random loss less fisher_sft
    opt_gcs_logdet opt_gcs_score opt_gcs_unwhitened
    random_subspace_logdet grad_norm_topk
)

echo "============================================="
echo "OpenCompass Eval: vLLM + LoRA"
echo "============================================="
echo "Base model: $BASE_MODEL"
echo "Round 2 methods: ${#ROUND2_METHODS[@]}"
echo "Round 1 methods: ${#ROUND1_METHODS[@]}"
echo ""

# Generate opencompass config for each model
generate_config() {
    local method=$1
    local adapter_path=$2
    local config_path="$RESULT_DIR/configs/${method}.py"
    mkdir -p "$(dirname $config_path)"

    cat > "$config_path" << PYEOF
from mmengine.config import read_base
from opencompass.models import VLLM

with read_base():
    from opencompass.configs.datasets.gsm8k.gsm8k_gen_1d7fe4 import gsm8k_datasets
    from opencompass.configs.datasets.IFEval.IFEval_gen_3321a3 import ifeval_datasets

datasets = gsm8k_datasets + ifeval_datasets

models = [
    dict(
        type=VLLM,
        abbr='${method}',
        path='${BASE_MODEL}',
        model_kwargs=dict(
            dtype='bfloat16',
            tensor_parallel_size=1,
            enable_lora=True,
            max_lora_rank=16,
            gpu_memory_utilization=0.85,
        ),
        lora_path='${adapter_path}',
        max_out_len=512,
        max_seq_len=4096,
        batch_size=32,
        generation_kwargs=dict(temperature=0, top_p=1.0),
        run_cfg=dict(num_gpus=1),
    )
]
PYEOF

    # Replace shell variables in the Python file
    sed -i "s|\${method}|${method}|g" "$config_path"
    sed -i "s|\${BASE_MODEL}|${BASE_MODEL}|g" "$config_path"
    sed -i "s|\${adapter_path}|${adapter_path}|g" "$config_path"

    echo "$config_path"
}

# Also generate a base model config (no LoRA)
generate_base_config() {
    local config_path="$RESULT_DIR/configs/base.py"
    mkdir -p "$(dirname $config_path)"

    cat > "$config_path" << PYEOF
from mmengine.config import read_base
from opencompass.models import VLLM

with read_base():
    from opencompass.configs.datasets.gsm8k.gsm8k_gen_1d7fe4 import gsm8k_datasets
    from opencompass.configs.datasets.IFEval.IFEval_gen_3321a3 import ifeval_datasets

datasets = gsm8k_datasets + ifeval_datasets

models = [
    dict(
        type=VLLM,
        abbr='base',
        path='${BASE_MODEL}',
        model_kwargs=dict(
            dtype='bfloat16',
            tensor_parallel_size=1,
            gpu_memory_utilization=0.85,
        ),
        max_out_len=512,
        max_seq_len=4096,
        batch_size=32,
        generation_kwargs=dict(temperature=0, top_p=1.0),
        run_cfg=dict(num_gpus=1),
    )
]
PYEOF
    sed -i "s|\${BASE_MODEL}|${BASE_MODEL}|g" "$config_path"
    echo "$config_path"
}

# =============================================================================
# Generate all configs
# =============================================================================
echo "[1/3] Generating configs..."

generate_base_config
for method in "${ROUND2_METHODS[@]}"; do
    generate_config "$method" "$ADAPTER_DIR/$method"
done
for method in "${ROUND1_METHODS[@]}"; do
    generate_config "r1_$method" "$ADAPTER_DIR_R1/$method"
done

echo "Generated $(ls $RESULT_DIR/configs/*.py | wc -l) configs"

# =============================================================================
# Run evaluations: 8 GPUs, 1 model per GPU in parallel
# =============================================================================
echo ""
echo "[2/3] Running evaluations..."

ALL_METHODS=("base")
for m in "${ROUND2_METHODS[@]}"; do ALL_METHODS+=("$m"); done
for m in "${ROUND1_METHODS[@]}"; do ALL_METHODS+=("r1_$m"); done

TOTAL=${#ALL_METHODS[@]}
PIDS=()
GPU=0

for i in "${!ALL_METHODS[@]}"; do
    method="${ALL_METHODS[$i]}"
    config="$RESULT_DIR/configs/${method}.py"
    work_dir="$RESULT_DIR/outputs/${method}"
    log="$RESULT_DIR/logs/${method}.log"
    mkdir -p "$RESULT_DIR/logs"

    if [ -d "$work_dir" ] && find "$work_dir" -name "summary_*.csv" 2>/dev/null | grep -q .; then
        echo "  [SKIP] $method (already evaluated)"
        continue
    fi

    # Wait if 8 GPUs busy
    while [ ${#PIDS[@]} -ge 8 ]; do
        NEW_PIDS=()
        for pid in "${PIDS[@]}"; do
            if kill -0 "$pid" 2>/dev/null; then
                NEW_PIDS+=("$pid")
            fi
        done
        PIDS=("${NEW_PIDS[@]}")
        [ ${#PIDS[@]} -ge 8 ] && sleep 30
    done

    GPU=$((${#PIDS[@]} % 8))
    echo "  [GPU $GPU] $method ($(($i+1))/$TOTAL)"

    CUDA_VISIBLE_DEVICES=$GPU $CONDA_PREFIX/bin/python -m opencompass.cli.main \
        "$config" \
        -w "$work_dir" \
        --max-num-workers 1 \
        > "$log" 2>&1 &
    PIDS+=($!)
done

# Wait for all
for pid in "${PIDS[@]}"; do wait "$pid" 2>/dev/null || true; done

echo ""
echo "[3/3] Collecting results..."

# =============================================================================
# Collect results
# =============================================================================
$CONDA_PREFIX/bin/python << 'PYEOF'
import os, glob, csv

result_dir = "/jizhicfs/karonhe/dataflex_saves/eval_results_round2/opencompass/outputs"
methods = sorted(os.listdir(result_dir)) if os.path.exists(result_dir) else []

print(f"{'Method':<25} {'GSM8K':>8} {'IFEval':>8}")
print("-" * 43)

for method in methods:
    summaries = glob.glob(f"{result_dir}/{method}/**/summary_*.csv", recursive=True)
    gsm = ifev = "---"
    for s in summaries:
        with open(s) as f:
            reader = csv.DictReader(f)
            for row in reader:
                ds = row.get("dataset", "")
                acc = row.get("accuracy", row.get("score", ""))
                if "gsm8k" in ds.lower():
                    gsm = acc
                elif "ifeval" in ds.lower() or "IFEval" in ds:
                    ifev = acc
    print(f"{method:<25} {gsm:>8} {ifev:>8}")
PYEOF

echo ""
echo "============================================="
echo "Done! $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================="

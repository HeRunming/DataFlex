#!/bin/bash
# =============================================================================
# Phase E — LESS-aligned final SFT (16 methods)
# =============================================================================
# For each method: take the indices from $SELECTIONS_DIR/<method>/selected_indices.npy,
# build a per-method JSON dataset, then train Llama-2-7B + LoRA r=128 + AdamW
# **from the base model** (NOT from the warmup checkpoint) for 4 epochs.
#
# This follows the LESS paper's methodology exactly: warmup checkpoint is used
# only for gradient features (Phase D); final SFT starts fresh from the base
# model on the selected 5% subset.
# =============================================================================

set -e

CONDA_ENV="${CONDA_ENV:-/jizhicfs/karonhe/miniconda_karonhe/envs/sft_train}"
export PATH="$CONDA_ENV/bin:$PATH"
export PYTHONPATH="/jizhicfs/karonhe/DataFlex/src:${PYTHONPATH:-}"
export DISABLE_VERSION_CHECK=1
export http_proxy=http://hy-proxy.woa.com:3128
export https_proxy=http://hy-proxy.woa.com:3128

REPO=/jizhicfs/karonhe/DataFlex
MODEL=/jizhicfs/karonhe/models/Llama-2-7b-hf
SOURCE_JSON=$REPO/data/tulu2_270k.json
SAVE_ROOT=/jizhicfs/karonhe/dataflex_saves/less_aligned
SELECTIONS_DIR=$SAVE_ROOT/selections
SFT_DIR=$SAVE_ROOT/sft
DATA_DIR=$REPO/data
mkdir -p "$SFT_DIR"

NPROC=8
EPOCHS=4
LR=2.0e-5

METHODS=(
    "random_s42:random:42"
    "loss_s42:loss:42"
    "less_s42:less:42"
    "fisher_sft_s42:fisher_sft:42"
    "grad_norm_topk_s42:grad_norm_topk:42"
    "rsub_own_seed1:random_subspace_logdet_seed1:42"
    "rsub_own_seed2:random_subspace_logdet_seed2:42"
    "rsub_own_seed3:random_subspace_logdet_seed3:42"
    "hybrid_add_l025_s42:opt_gcs_hybrid_add_lambda0.25:42"
    "hybrid_mul_g025_s42:opt_gcs_hybrid_mul_gamma0.25:42"
    "hybrid_mul_g05_s42:opt_gcs_hybrid_mul_gamma0.5:42"
    "logdet_nopref_s42:opt_gcs_logdet_no_prefilter:42"
    "hybrid_add_l025_s1:opt_gcs_hybrid_add_lambda0.25:1"
    "hybrid_add_l025_s2:opt_gcs_hybrid_add_lambda0.25:2"
    "hybrid_mul_g025_s1:opt_gcs_hybrid_mul_gamma0.25:1"
    "hybrid_mul_g025_s2:opt_gcs_hybrid_mul_gamma0.25:2"
)

# -----------------------------------------------------------------------------
# Step 1: build per-method dataset JSONs from selected indices
# -----------------------------------------------------------------------------
$CONDA_ENV/bin/python <<PYEOF
import json, os, sys
import numpy as np

source = "$SOURCE_JSON"
sel_root = "$SELECTIONS_DIR"
out_root = "$DATA_DIR"
methods = """$(printf '%s\n' "${METHODS[@]}")""".strip().split("\n")

print(f"[build] loading source pool from {source}", flush=True)
with open(source) as f:
    pool = json.load(f)
print(f"[build] pool size: {len(pool)}", flush=True)

dataset_info_path = os.path.join(out_root, "dataset_info.json")
with open(dataset_info_path) as f:
    di = json.load(f)

for spec in methods:
    method = spec.split(":")[0]
    idx_path = os.path.join(sel_root, method, "selected_indices.npy")
    if not os.path.exists(idx_path):
        print(f"[skip] {method}: no selected_indices.npy", flush=True)
        continue
    indices = np.load(idx_path)
    out_json = f"tulu2_270k_sel_{method}.json"
    out_path = os.path.join(out_root, out_json)
    if os.path.exists(out_path):
        print(f"[skip] {method}: {out_json} already exists", flush=True)
    else:
        subset = [pool[int(i)] for i in indices]
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(subset, f, ensure_ascii=False)
        print(f"[build] {method}: wrote {len(subset)} examples -> {out_json}", flush=True)
    ds_name = f"tulu2_sel_{method}"
    di[ds_name] = {
        "file_name": out_json,
        "columns": {"prompt": "instruction", "query": "input", "response": "output"},
    }

with open(dataset_info_path, "w") as f:
    json.dump(di, f, indent=2, ensure_ascii=False)
print(f"[build] dataset_info.json updated", flush=True)
PYEOF

# -----------------------------------------------------------------------------
# Step 2: per-method final SFT (8-GPU dispatch, sequential since each uses 8 GPUs)
# -----------------------------------------------------------------------------
gen_yaml() {
    local method=$1 train_seed=$2
    local yaml_path="$SFT_DIR/$method/sft.yaml"
    local out_dir="$SFT_DIR/$method"
    mkdir -p "$out_dir"

    cat > "$yaml_path" <<EOF
### model
model_name_or_path: $MODEL
trust_remote_code: true

### method
stage: sft
do_train: true
finetuning_type: lora
lora_target: all
lora_rank: 128
lora_alpha: 512

### dataset
dataset: tulu2_sel_$method
template: alpaca
cutoff_len: 1024
overwrite_cache: false
preprocessing_num_workers: 16
dataloader_num_workers: 0
seed: $train_seed

### output
output_dir: $out_dir
logging_steps: 50
save_steps: 100000   # save once at end
save_strategy: epoch
plot_loss: true
save_only_model: true
overwrite_output_dir: true
report_to: none

### DataFlex (vanilla static SFT)
train_type: static

### train (LESS paper)
per_device_train_batch_size: 1
gradient_accumulation_steps: 1
learning_rate: $LR
num_train_epochs: $EPOCHS
lr_scheduler_type: linear
warmup_ratio: 0.03
bf16: true
optim: adamw_torch
adam_beta1: 0.9
adam_beta2: 0.999
weight_decay: 0.0
ddp_timeout: 180000000
EOF
    echo "$yaml_path"
}

cd $REPO

for method_spec in "${METHODS[@]}"; do
    IFS=':' read -r method component train_seed <<< "$method_spec"
    out_dir="$SFT_DIR/$method"
    if [ -f "$out_dir/adapter_model.safetensors" ]; then
        echo "[skip] $method: adapter already exists"
        continue
    fi
    if [ ! -f "$REPO/data/tulu2_270k_sel_${method}.json" ]; then
        echo "[skip] $method: per-method dataset not found"
        continue
    fi
    yaml_path=$(gen_yaml "$method" "$train_seed")
    echo ""
    echo "============================================="
    echo "[$method] training (seed=$train_seed)"
    echo "============================================="
    torchrun --nproc_per_node=$NPROC --nnodes=1 \
        src/dataflex/launcher.py "$yaml_path" \
        2>&1 | tee "$out_dir/sft.log" || echo "[fail] $method"
done

echo ""
echo "============================================="
echo "All SFT runs complete. Adapters:"
echo "============================================="
ls -la $SFT_DIR/*/adapter_model.safetensors 2>/dev/null | wc -l

#!/bin/bash
# =============================================================================
# Phase C — LESS-aligned warmup training (simplified)
# =============================================================================
# Pre-select a fixed 5% subset of 270K (13,533 examples), then run vanilla
# LLaMA-Factory SFT for 4 epochs on that subset. The standard HF Trainer
# checkpoint contains both the LoRA adapter and the AdamW optimizer state
# (exp_avg, exp_avg_sq), which is what Phase D needs.
#
# Avoids DataFlex's online select-and-train pipeline entirely; this is a
# straightforward LoRA fine-tune from base.
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
WARMUP_DATASET_NAME=tulu2_270k_warmup_5pct
WARMUP_JSON=$REPO/data/${WARMUP_DATASET_NAME}.json
SAVE_ROOT=/jizhicfs/karonhe/dataflex_saves/less_aligned
WARMUP_DIR=$SAVE_ROOT/warmup
mkdir -p "$WARMUP_DIR"

NPROC=8
EPOCHS=4
LR=2.0e-5
NUM_WARMUP_SAMPLES=13533

# -----------------------------------------------------------------------------
# Step 1: build the warmup subset (deterministic seed)
# -----------------------------------------------------------------------------
$CONDA_ENV/bin/python <<PYEOF
import json, os
import numpy as np

src = "$SOURCE_JSON"
out = "$WARMUP_JSON"
n_target = $NUM_WARMUP_SAMPLES
seed = 42

if os.path.exists(out):
    with open(out) as f:
        existing = json.load(f)
    print(f"[skip] {out} already exists with {len(existing)} examples")
else:
    print(f"[load] {src}", flush=True)
    with open(src) as f:
        pool = json.load(f)
    print(f"[load] pool size: {len(pool)}", flush=True)
    rng = np.random.RandomState(seed)
    indices = rng.permutation(len(pool))[:n_target]
    indices.sort()
    subset = [pool[int(i)] for i in indices]
    print(f"[save] {out} ({len(subset)} examples)", flush=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(subset, f, ensure_ascii=False)
    np.save(out.replace(".json", "_indices.npy"), indices)

# Register
di_path = "$REPO/data/dataset_info.json"
with open(di_path) as f:
    di = json.load(f)
ds_name = "$WARMUP_DATASET_NAME"
di[ds_name] = {
    "file_name": "${WARMUP_DATASET_NAME}.json",
    "columns": {"prompt": "instruction", "query": "input", "response": "output"},
}
with open(di_path, "w") as f:
    json.dump(di, f, indent=2, ensure_ascii=False)
print(f"[register] dataset_info.json updated: {ds_name}")
PYEOF

# -----------------------------------------------------------------------------
# Step 2: run vanilla LLaMA-Factory SFT for 4 epochs
# -----------------------------------------------------------------------------
YAML=$WARMUP_DIR/warmup.yaml
cat > "$YAML" <<EOF
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
dataset: $WARMUP_DATASET_NAME
template: alpaca
cutoff_len: 1024
overwrite_cache: false
preprocessing_num_workers: 16
dataloader_num_workers: 0
seed: 42

### output
output_dir: $WARMUP_DIR/warmup_ckpt
logging_steps: 50
save_strategy: epoch
save_total_limit: 1
save_only_model: false   # need optimizer.pt for Phase D
plot_loss: true
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

echo "============================================="
echo "Phase C — LESS-aligned warmup training"
echo "============================================="
echo "Model:    $MODEL"
echo "Dataset:  $WARMUP_DATASET_NAME (5% of 270K = $NUM_WARMUP_SAMPLES)"
echo "Epochs:   $EPOCHS, lr=$LR, LoRA r=128"
echo "Output:   $WARMUP_DIR/warmup_ckpt/"
echo ""

cd $REPO
torchrun --nproc_per_node=$NPROC --nnodes=1 \
    src/dataflex/launcher.py "$YAML" \
    2>&1 | tee "$WARMUP_DIR/warmup.log"

echo ""
echo "============================================="
echo "Verifying warmup checkpoint contents:"
echo "============================================="
# After training, the latest checkpoint dir should contain adapter_model.safetensors + optimizer.pt
LATEST_CKPT=$(ls -d $WARMUP_DIR/warmup_ckpt/checkpoint-* 2>/dev/null | sort -V | tail -1)
if [ -n "$LATEST_CKPT" ]; then
    echo "Latest checkpoint: $LATEST_CKPT"
    ls -la "$LATEST_CKPT" | head -20
    # Symlink so Phase D's --warmup_ckpt path is stable
    ln -sfn "$LATEST_CKPT" "$WARMUP_DIR/warmup_ckpt_latest"
    echo ""
    echo "Stable symlink: $WARMUP_DIR/warmup_ckpt_latest -> $LATEST_CKPT"
else
    echo "[ERROR] no checkpoint produced under $WARMUP_DIR/warmup_ckpt/"
fi

#!/bin/bash
# =============================================================================
# Opt-GCS Debug Set: 10 methods × budget=5000
# =============================================================================
# Quick validation run after round-3 bug fixes.
# One budget (5k) to verify code correctness before full paper-scale.
#
# Methods:
#   1. random (baseline)
#   2. loss (baseline)
#   3. less (baseline, needs eval_dataset)
#   4. fisher_sft (baseline)
#   5. opt_gcs_logdet (main method)
#   6. opt_gcs_score (score variant)
#   7. opt_gcs_unwhitened (ablation: β=0)
#   8. opt_gcs_rank50 (ablation: fixed rank=50)
#   9. random_subspace_logdet (negative control)
#  10. grad_norm_topk (negative control)
#
# Usage:
#   bash experiments/paper_scale/run_debug_set.sh [method_name]
#   - No argument: run all 10 methods sequentially
#   - With argument: run only that method (e.g., "opt_gcs_logdet")
# =============================================================================

set -e

FILTER=${1:-""}
CONDA_PREFIX="/jizhicfs/karonhe/miniconda_karonhe/envs/spec_gcs"
export PATH="$CONDA_PREFIX/bin:$PATH"
export PYTHONPATH="/jizhicfs/karonhe/DataFlex/src:$PYTHONPATH"
export DISABLE_VERSION_CHECK=1
export http_proxy="http://hy-proxy.woa.com:3128"
export https_proxy="http://hy-proxy.woa.com:3128"

WORK_DIR="/jizhicfs/karonhe/DataFlex"
SAVE_DIR="/jizhicfs/karonhe/dataflex_saves/debug_set"
MODEL="/jizhicfs/karonhe/models/LLM-Research/Meta-Llama-3___1-8B"
TORCHRUN="$CONDA_PREFIX/bin/torchrun"
NPROC=8
MASTER_PORT=29700
BUDGET=5000

cd "$WORK_DIR"
mkdir -p "$SAVE_DIR/logs" "$SAVE_DIR/configs"

echo "============================================="
echo "Opt-GCS Debug Set: 10 methods × budget=5000"
echo "============================================="
echo "Model: $MODEL"
echo "GPUs: $NPROC"
echo "Budget: $BUDGET"
echo "Save: $SAVE_DIR"
echo "Filter: ${FILTER:-all}"
echo ""

# Training params for budget=5000
# 5000/8 = 625 steps/epoch → ~2 epochs = 1250 steps
TRAIN_STEPS=1250
WARMUP_CALC=10
UPDATE_STEP_CALC=$((BUDGET / NPROC))  # 625
EVAL_STEPS=$((TRAIN_STEPS / 3))       # ~416

# =============================================================================
# Method list: METHOD_NAME:COMPONENT_NAME
# =============================================================================
METHODS=(
    "random:random"
    "loss:loss"
    "less:less"
    "fisher_sft:fisher_sft"
    "opt_gcs_logdet:opt_gcs_logdet"
    "opt_gcs_score:spec_gcs_score"
    "opt_gcs_unwhitened:opt_gcs_unwhitened"
    "opt_gcs_rank50:opt_gcs_rank50"
    "random_subspace_logdet:random_subspace_logdet"
    "grad_norm_topk:grad_norm_topk"
)

# =============================================================================
# Generate YAML for a method
# =============================================================================
generate_yaml() {
    local method=$1
    local component=$2
    local yaml_path="$SAVE_DIR/configs/${method}.yaml"
    local output_dir="$SAVE_DIR/${method}"

    mkdir -p "$output_dir"

    cat > "$yaml_path" << YAML
### model
model_name_or_path: $MODEL
trust_remote_code: true

### method
stage: sft
do_train: true
finetuning_type: lora
lora_target: all
lora_rank: 16
lora_alpha: 8

### dataset
dataset: openhermes_10w
template: llama3
cutoff_len: 4096
overwrite_cache: true
preprocessing_num_workers: 16
dataloader_num_workers: 0
seed: 42

### output
output_dir: $output_dir
logging_steps: 10
save_steps: $TRAIN_STEPS
plot_loss: true
save_only_model: true
overwrite_output_dir: true
report_to: none

### train
per_device_train_batch_size: 1
gradient_accumulation_steps: 1
learning_rate: 1.0e-4
num_train_epochs: 1.0
lr_scheduler_type: cosine
warmup_ratio: 0.1
bf16: true
ddp_timeout: 180000000

### DataFlex
train_type: dynamic_select
components_cfg_file: src/dataflex/configs/components.yaml
component_name: $component
warmup_step: $WARMUP_CALC
update_step: $UPDATE_STEP_CALC
update_times: 2

### eval
eval_dataset: mmlu_valid_cot
per_device_eval_batch_size: 1
metric_for_best_model: eval_loss
greater_is_better: false
load_best_model_at_end: false
eval_strategy: steps
eval_steps: $EVAL_STEPS
YAML

    echo "$yaml_path"
}

# =============================================================================
# Run all methods
# =============================================================================
TOTAL=${#METHODS[@]}
CURRENT=0
SUCCEEDED=0
FAILED=0
SKIPPED=0

for method_spec in "${METHODS[@]}"; do
    IFS=':' read -r method component <<< "$method_spec"
    CURRENT=$((CURRENT + 1))

    # Filter if specified
    if [ -n "$FILTER" ] && [ "$method" != "$FILTER" ]; then
        continue
    fi

    output_dir="$SAVE_DIR/${method}"
    log="$SAVE_DIR/logs/${method}.log"

    # Skip if already trained
    if [ -f "$output_dir/adapter_model.safetensors" ]; then
        echo "[$CURRENT/$TOTAL] SKIP $method (already trained)"
        SKIPPED=$((SKIPPED + 1))
        continue
    fi

    # Generate config
    yaml=$(generate_yaml "$method" "$component")

    echo "[$CURRENT/$TOTAL] RUNNING: $method (component=$component)"
    echo "  Config: $yaml"
    echo "  Log: $log"
    echo "  Started: $(date '+%Y-%m-%d %H:%M:%S')"

    PORT=$((MASTER_PORT + CURRENT))

    if $TORCHRUN --nproc_per_node=$NPROC --master_port=$PORT \
        "$WORK_DIR/src/dataflex/launcher.py" "$yaml" \
        > "$log" 2>&1; then
        echo "  ✓ $method SUCCEEDED ($(date '+%H:%M:%S'))"
        SUCCEEDED=$((SUCCEEDED + 1))
    else
        echo "  ✗ $method FAILED (check $log)"
        FAILED=$((FAILED + 1))
        # Print last 20 lines of log for quick diagnosis
        echo "  --- Last 20 lines of log ---"
        tail -20 "$log" | sed 's/^/  | /'
        echo "  ---"
    fi
    echo ""
done

echo "============================================="
echo "Debug Set Complete!"
echo "  Succeeded: $SUCCEEDED"
echo "  Failed:    $FAILED"
echo "  Skipped:   $SKIPPED"
echo "============================================="

# =============================================================================
# Quick results summary
# =============================================================================
echo ""
echo "=== Eval Loss Summary ==="
for method_spec in "${METHODS[@]}"; do
    IFS=':' read -r method component <<< "$method_spec"
    log="$SAVE_DIR/logs/${method}.log"
    if [ -f "$log" ]; then
        eval_loss=$(grep -o "'eval_loss': [0-9.]*" "$log" | tail -1 | grep -o '[0-9.]*$' || echo "N/A")
        train_loss=$(grep -o "'loss': [0-9.]*" "$log" | tail -1 | grep -o '[0-9.]*$' || echo "N/A")
        printf "  %-30s train_loss=%-8s eval_loss=%-8s\n" "$method" "$train_loss" "$eval_loss"
    fi
done

echo ""
echo "=== Selection Cache Files ==="
find "$SAVE_DIR" -name "*.json" -path "*/step_*" | head -20
echo ""
echo "=== Trained Models ==="
find "$SAVE_DIR" -name "adapter_model.safetensors" | sort

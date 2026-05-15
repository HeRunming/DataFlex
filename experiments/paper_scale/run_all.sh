#!/bin/bash
# =============================================================================
# Opt-GCS Paper-Scale Experiment Runner
# =============================================================================
# This script runs the full experiment matrix for the Opt-GCS paper:
# - Multiple selection methods (baselines + ours + ablations + negative controls)
# - Multiple selection budgets (1%, 5%, 10%, 20%)
# - MMLU evaluation for all trained models
#
# Usage:
#   bash experiments/paper_scale/run_all.sh [stage]
#   stage: "select" | "train" | "eval" | "all" (default: all)
#
# Prerequisites:
#   - Model: /jizhicfs/karonhe/models/LLM-Research/Meta-Llama-3___1-8B
#   - Data: /jizhicfs/karonhe/DataFlex/data/{Openhermes_train,MMLU_valid_cot}.json
#   - Environment: spec_gcs conda env
# =============================================================================

set -e

STAGE=${1:-all}
CONDA_PREFIX="/jizhicfs/karonhe/miniconda_karonhe/envs/spec_gcs"
export PATH="$CONDA_PREFIX/bin:$PATH"
export PYTHONPATH="/jizhicfs/karonhe/DataFlex/src:$PYTHONPATH"
export DISABLE_VERSION_CHECK=1
export http_proxy="http://hy-proxy.woa.com:3128"
export https_proxy="http://hy-proxy.woa.com:3128"

WORK_DIR="/jizhicfs/karonhe/DataFlex"
SAVE_DIR="/jizhicfs/karonhe/dataflex_saves/paper_experiments"
MODEL="/jizhicfs/karonhe/models/LLM-Research/Meta-Llama-3___1-8B"
TORCHRUN="$CONDA_PREFIX/bin/torchrun"
NPROC=8
MASTER_PORT=29600

cd "$WORK_DIR"
mkdir -p "$SAVE_DIR/logs"

echo "============================================="
echo "Opt-GCS Paper-Scale Experiments"
echo "============================================="
echo "Stage: $STAGE"
echo "Model: $MODEL"
echo "GPUs: $NPROC"
echo "Save: $SAVE_DIR"
echo ""

# =============================================================================
# Method list and configurations
# =============================================================================
# Format: METHOD_NAME:COMPONENT_NAME:NEEDS_EVAL_DATASET
METHODS=(
    # Baselines
    "random:random:no"
    "loss:loss:no"
    "less:less:yes"
    # FisherSFT baseline
    "fisher_sft:fisher_sft:no"
    # Opt-GCS main method
    "opt_gcs_logdet:opt_gcs_logdet:no"
    # Opt-GCS ablations
    "opt_gcs_score:spec_gcs_score:no"
    "opt_gcs_unwhitened:opt_gcs_unwhitened:no"
    "opt_gcs_raw_sgd:opt_gcs_raw_sgd:no"
    # Negative controls
    "grad_norm_topk:grad_norm_topk:no"
    "random_subspace_logdet:random_subspace_logdet:no"
)

# Selection budgets (as fraction of total data)
# For 100k dataset: 1%=1000, 5%=5000, 10%=10000, 20%=20000
BUDGETS=(1000 5000 10000)

# Training steps for each budget
# Rule: ~2-3 epochs over selected data, batch_size=8 (global)
# 1000 samples / 8 = 125 steps/epoch → 250-375 steps
# 5000 samples / 8 = 625 steps/epoch → 1250-1875 steps
# 10000 samples / 8 = 1250 steps/epoch → 2500-3750 steps
declare -A TRAIN_STEPS
TRAIN_STEPS[1000]=300
TRAIN_STEPS[5000]=1500
TRAIN_STEPS[10000]=3000

# =============================================================================
# Helper function: generate training YAML for a method/budget combination
# =============================================================================
generate_yaml() {
    local method=$1
    local component=$2
    local needs_eval=$3
    local budget=$4
    local train_steps=${TRAIN_STEPS[$budget]}
    local output_dir="$SAVE_DIR/${method}/budget_${budget}"
    local yaml_path="$SAVE_DIR/configs/${method}_budget_${budget}.yaml"

    mkdir -p "$(dirname $yaml_path)"
    mkdir -p "$output_dir"

    # Calculate warmup and update steps
    # warmup: 10% of total steps
    # update_step: remaining / 2 (two selection rounds)
    local warmup_step=$((train_steps / 10))
    local remaining=$((train_steps - warmup_step))
    local update_step=$((remaining / 2))

    # num_samples per selection = budget (select budget samples each round)
    # In DataFlex: num_samples = total_batch_size * update_step
    # total_batch_size = per_device * n_gpu = 1 * 8 = 8
    # So update_step = budget / 8
    local update_step_calc=$((budget / NPROC))
    # warmup_step should be small
    local warmup_step_calc=10

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
save_steps: $train_steps
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
warmup_step: $warmup_step_calc
update_step: $update_step_calc
update_times: 2

### eval
eval_dataset: mmlu_valid_cot
per_device_eval_batch_size: 1
metric_for_best_model: eval_loss
greater_is_better: false
load_best_model_at_end: false
eval_strategy: steps
eval_steps: $((train_steps / 3))
YAML

    echo "$yaml_path"
}

# =============================================================================
# Stage: Generate all configs
# =============================================================================
if [[ "$STAGE" == "all" || "$STAGE" == "config" ]]; then
    echo "[Stage: config] Generating experiment YAML files..."
    for method_spec in "${METHODS[@]}"; do
        IFS=':' read -r method component needs_eval <<< "$method_spec"
        for budget in "${BUDGETS[@]}"; do
            yaml=$(generate_yaml "$method" "$component" "$needs_eval" "$budget")
            echo "  Generated: $yaml"
        done
    done
    echo ""
fi

# =============================================================================
# Stage: Run training
# =============================================================================
if [[ "$STAGE" == "all" || "$STAGE" == "train" ]]; then
    echo "[Stage: train] Running training experiments..."
    for method_spec in "${METHODS[@]}"; do
        IFS=':' read -r method component needs_eval <<< "$method_spec"
        for budget in "${BUDGETS[@]}"; do
            yaml="$SAVE_DIR/configs/${method}_budget_${budget}.yaml"
            log="$SAVE_DIR/logs/${method}_budget_${budget}.log"
            output_dir="$SAVE_DIR/${method}/budget_${budget}"

            if [ -f "$output_dir/adapter_model.safetensors" ]; then
                echo "  [SKIP] $method budget=$budget (already trained)"
                continue
            fi

            if [ ! -f "$yaml" ]; then
                yaml=$(generate_yaml "$method" "$component" "$needs_eval" "$budget")
            fi

            echo "  [RUN] $method budget=$budget → $log"
            $TORCHRUN --nproc_per_node=$NPROC --master_port=$((MASTER_PORT++)) \
                "$WORK_DIR/src/dataflex/launcher.py" "$yaml" \
                > "$log" 2>&1 || echo "  [WARN] $method budget=$budget failed. Check $log"
        done
    done
    echo ""
fi

# =============================================================================
# Stage: Evaluation
# =============================================================================
if [[ "$STAGE" == "all" || "$STAGE" == "eval" ]]; then
    echo "[Stage: eval] Running MMLU evaluation..."
    echo "  TODO: Implement lm_eval / opencompass evaluation"
    echo "  Models to evaluate:"
    for method_spec in "${METHODS[@]}"; do
        IFS=':' read -r method component needs_eval <<< "$method_spec"
        for budget in "${BUDGETS[@]}"; do
            output_dir="$SAVE_DIR/${method}/budget_${budget}"
            if [ -d "$output_dir" ]; then
                echo "    $output_dir"
            fi
        done
    done
    echo ""
fi

echo "============================================="
echo "Experiment pipeline complete!"
echo "============================================="

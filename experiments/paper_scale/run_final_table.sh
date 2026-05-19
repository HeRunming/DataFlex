#!/bin/bash
# =============================================================================
# Clean Final Table: Unified experiment for paper submission
# =============================================================================
# Re-runs ALL core methods under the same code version, cache isolation,
# and training config. No R1/R2 mixing.
#
# Methods (12):
#   Baselines: random, loss, less, fisher_sft
#   Controls: grad_norm_topk, random_subspace (3 seeds, own gradients)
#   Ours: hybrid_add_l025, hybrid_mul_g025, hybrid_mul_g05, logdet_nopref
#
# Each method computes its own gradients. No cross-method cache sharing.
# Hybrid methods have 3 training seeds for variance estimation.
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
SAVE_DIR="/jizhicfs/karonhe/dataflex_saves/final_table"
MODEL="/jizhicfs/karonhe/models/LLM-Research/Meta-Llama-3___1-8B"
TORCHRUN="$CONDA_PREFIX/bin/torchrun"
NPROC=8
MASTER_PORT=29900
BUDGET=5000

cd "$WORK_DIR"
mkdir -p "$SAVE_DIR/logs" "$SAVE_DIR/configs"

# Training params
WARMUP_CALC=10
UPDATE_STEP_CALC=$((BUDGET / NPROC))  # 625
UPDATE_TIMES=2
TRAIN_STEPS=$((WARMUP_CALC + UPDATE_STEP_CALC * UPDATE_TIMES))  # 1260
EVAL_STEPS=$((TRAIN_STEPS / 3))

echo "============================================="
echo "Clean Final Table Experiments"
echo "============================================="
echo "Model: $MODEL"
echo "Budget: $BUDGET, Steps: $TRAIN_STEPS"
echo "Save: $SAVE_DIR"
echo ""

# Generate isolated components file
FINAL_COMPONENTS="$SAVE_DIR/configs/components_final.yaml"
$CONDA_PREFIX/bin/python - "$SAVE_DIR" << 'PYEOF'
import sys, yaml
save_dir = sys.argv[1]
with open("src/dataflex/configs/components.yaml") as f:
    cfg = yaml.safe_load(f)
# Override all cache_dirs to be fully isolated
for sname, sconf in cfg.get("selectors", {}).items():
    if "params" in sconf and "cache_dir" in sconf["params"]:
        sconf["params"]["cache_dir"] = f"{save_dir}/cache/{sname}"
# Enable compute_own_grads for random_subspace and grad_norm_topk controls
for sname, sconf in cfg.get("selectors", {}).items():
    if sname.startswith("random_subspace_logdet"):
        sconf["params"]["compute_own_grads"] = True
        sconf["params"]["gradient_type"] = "adam_diag"
        sconf["params"]["save_interval"] = 16
        sconf["params"]["projector_seed"] = 42  # fixed across subspace seeds
        sconf["params"]["clipping_method"] = "adaptive"
        sconf["params"].pop("source_grad_dirs", None)
    if sname == "grad_norm_topk":
        sconf["params"]["compute_own_grads"] = True
        sconf["params"]["gradient_type"] = "adam_diag"
        sconf["params"]["proj_dim"] = 4096
        sconf["params"]["save_interval"] = 16
out_path = f"{save_dir}/configs/components_final.yaml"
with open(out_path, "w") as f:
    yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)
print(f"Generated: {out_path}")
PYEOF

# =============================================================================
# Method definitions: METHOD_NAME:COMPONENT_NAME:TRAINING_SEED
# =============================================================================
METHODS=(
    # Baselines (single seed for now)
    "random_s42:random:42"
    "loss_s42:loss:42"
    "less_s42:less:42"
    "fisher_sft_s42:fisher_sft:42"
    # Negative controls (own gradient computation)
    "grad_norm_topk_s42:grad_norm_topk:42"
    "rsub_own_seed1:random_subspace_logdet_seed1:42"
    "rsub_own_seed2:random_subspace_logdet_seed2:42"
    "rsub_own_seed3:random_subspace_logdet_seed3:42"
    # Our methods (main seed)
    "hybrid_add_l025_s42:opt_gcs_hybrid_add_lambda0.25:42"
    "hybrid_mul_g025_s42:opt_gcs_hybrid_mul_gamma0.25:42"
    "hybrid_mul_g05_s42:opt_gcs_hybrid_mul_gamma0.5:42"
    "logdet_nopref_s42:opt_gcs_logdet_no_prefilter:42"
    # Our methods (extra seeds for variance)
    "hybrid_add_l025_s1:opt_gcs_hybrid_add_lambda0.25:1"
    "hybrid_add_l025_s2:opt_gcs_hybrid_add_lambda0.25:2"
    "hybrid_mul_g025_s1:opt_gcs_hybrid_mul_gamma0.25:1"
    "hybrid_mul_g025_s2:opt_gcs_hybrid_mul_gamma0.25:2"
)

generate_yaml() {
    local method=$1
    local component=$2
    local train_seed=$3
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
seed: $train_seed

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
components_cfg_file: $FINAL_COMPONENTS
component_name: $component
warmup_step: $WARMUP_CALC
update_step: $UPDATE_STEP_CALC
update_times: $UPDATE_TIMES

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
# Run
# =============================================================================
TOTAL=${#METHODS[@]}
CURRENT=0
SUCCEEDED=0
FAILED=0
SKIPPED=0

for method_spec in "${METHODS[@]}"; do
    IFS=':' read -r method component train_seed <<< "$method_spec"
    CURRENT=$((CURRENT + 1))

    # Filter support
    if [ -n "$FILTER" ] && [ "$method" != "$FILTER" ]; then
        continue
    fi

    output_dir="$SAVE_DIR/${method}"
    log="$SAVE_DIR/logs/${method}.log"

    if [ -f "$output_dir/adapter_model.safetensors" ]; then
        echo "[$CURRENT/$TOTAL] SKIP $method (already trained)"
        SKIPPED=$((SKIPPED + 1))
        continue
    fi

    yaml=$(generate_yaml "$method" "$component" "$train_seed")

    echo "[$CURRENT/$TOTAL] RUNNING: $method (component=$component, seed=$train_seed)"
    echo "  Log: $log"
    echo "  Started: $(date '+%Y-%m-%d %H:%M:%S')"

    PORT=$((MASTER_PORT + CURRENT))

    if $TORCHRUN --nproc_per_node=$NPROC --master_port=$PORT \
        "$WORK_DIR/src/dataflex/launcher.py" "$yaml" \
        > "$log" 2>&1; then
        echo "  OK $method SUCCEEDED ($(date '+%H:%M:%S'))"
        SUCCEEDED=$((SUCCEEDED + 1))
    else
        echo "  FAIL $method FAILED (check $log)"
        FAILED=$((FAILED + 1))
        tail -20 "$log" | sed 's/^/  | /'
    fi
    echo ""
done

echo "============================================="
echo "Clean Final Table Complete!"
echo "  Succeeded: $SUCCEEDED"
echo "  Failed:    $FAILED"
echo "  Skipped:   $SKIPPED"
echo "============================================="

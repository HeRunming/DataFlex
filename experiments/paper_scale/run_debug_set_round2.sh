#!/bin/bash
# =============================================================================
# Opt-GCS Debug Set Round 2: Hybrid methods + multi-seed random_subspace
# =============================================================================
# Priority experiments from GPT analysis:
# 1. random_subspace_logdet multi-seed (verify if seed=42 was lucky)
# 2. Hybrid additive lambda sweep (score + coverage)
# 3. Hybrid multiplicative gamma sweep
# 4. Score/LogDet ablations
#
# All at budget=5000, same setup as round 1.
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
SAVE_DIR="/jizhicfs/karonhe/dataflex_saves/debug_set_round2"
MODEL="/jizhicfs/karonhe/models/LLM-Research/Meta-Llama-3___1-8B"
TORCHRUN="$CONDA_PREFIX/bin/torchrun"
NPROC=8
MASTER_PORT=29800
BUDGET=5000

cd "$WORK_DIR"
mkdir -p "$SAVE_DIR/logs" "$SAVE_DIR/configs"

# Generate round2-specific components.yaml with cache_dir overrides
# This ensures gradient/selection caches are isolated per round
ROUND2_COMPONENTS="$SAVE_DIR/configs/components_round2.yaml"
/jizhicfs/karonhe/miniconda_karonhe/envs/spec_gcs/bin/python - "$SAVE_DIR" << 'PYEOF'
import sys, yaml
save_dir = sys.argv[1]
with open("src/dataflex/configs/components.yaml") as f:
    cfg = yaml.safe_load(f)
# Override cache_dir for all selectors to be under the round2 save directory
for sname, sconf in cfg.get("selectors", {}).items():
    if "params" in sconf and "cache_dir" in sconf["params"]:
        sconf["params"]["cache_dir"] = f"{save_dir}/cache/{sname}"
# Inject source_grad_dirs for random_subspace seeds and grad_norm_topk
# They should look for gradients in the score_beta0 cache (which runs first)
grad_source = f"{save_dir}/cache/opt_gcs_score_beta0"
for sname, sconf in cfg.get("selectors", {}).items():
    if sname.startswith("random_subspace_logdet_seed") or sname == "grad_norm_topk":
        sconf["params"]["source_grad_dirs"] = [grad_source]
out_path = f"{save_dir}/configs/components_round2.yaml"
with open(out_path, "w") as f:
    yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)
print(f"Generated round2 components: {out_path}")
PYEOF

echo "============================================="
echo "Opt-GCS Debug Set Round 2"
echo "============================================="
echo "Model: $MODEL"
echo "GPUs: $NPROC"
echo "Budget: $BUDGET"
echo "Save: $SAVE_DIR"
echo ""

WARMUP_CALC=10
UPDATE_STEP_CALC=$((BUDGET / NPROC))  # 5000/8=625
UPDATE_TIMES=2
TRAIN_STEPS=$((WARMUP_CALC + UPDATE_STEP_CALC * UPDATE_TIMES))  # 10+625*2=1260
EVAL_STEPS=$((TRAIN_STEPS / 3))

# =============================================================================
# Method list: METHOD_NAME:COMPONENT_NAME
# Priority order: random_subspace seeds first, then hybrids
# =============================================================================
METHODS=(
    # First: generate gradient cache that random_subspace seeds will reuse
    "score_beta0:opt_gcs_score_beta0"
    # Random subspace multi-seed (highest priority experiment)
    # These reuse gradients from score_beta0 via sibling scan
    "rsub_seed1:random_subspace_logdet_seed1"
    "rsub_seed2:random_subspace_logdet_seed2"
    "rsub_seed3:random_subspace_logdet_seed3"
    "rsub_seed4:random_subspace_logdet_seed4"
    "rsub_seed5:random_subspace_logdet_seed5"
    # Score ablation (reuses score_beta0 gradients where step_id matches)
    "score_beta025:opt_gcs_score_beta0.25"
    # Hybrid additive lambda sweep
    "hybrid_add_l025:opt_gcs_hybrid_add_lambda0.25"
    "hybrid_add_l05:opt_gcs_hybrid_add_lambda0.5"
    "hybrid_add_l10:opt_gcs_hybrid_add_lambda1.0"
    "hybrid_add_l20:opt_gcs_hybrid_add_lambda2.0"
    # Hybrid multiplicative gamma sweep
    "hybrid_mul_g025:opt_gcs_hybrid_mul_gamma0.25"
    "hybrid_mul_g05:opt_gcs_hybrid_mul_gamma0.5"
    "hybrid_mul_g10:opt_gcs_hybrid_mul_gamma1.0"
    # LogDet ablations
    "logdet_pref20:opt_gcs_logdet_pref20"
    "logdet_nopref:opt_gcs_logdet_no_prefilter"
)

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
components_cfg_file: $ROUND2_COMPONENTS
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
    IFS=':' read -r method component <<< "$method_spec"
    CURRENT=$((CURRENT + 1))

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

    yaml=$(generate_yaml "$method" "$component")

    echo "[$CURRENT/$TOTAL] RUNNING: $method (component=$component)"
    echo "  Config: $yaml"
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
        echo "  ---"
    fi
    echo ""
done

echo "============================================="
echo "Debug Set Round 2 Complete!"
echo "  Succeeded: $SUCCEEDED"
echo "  Failed:    $FAILED"
echo "  Skipped:   $SKIPPED"
echo "============================================="

echo ""
echo "=== Eval Loss Summary ==="
for method_spec in "${METHODS[@]}"; do
    IFS=':' read -r method component <<< "$method_spec"
    log="$SAVE_DIR/logs/${method}.log"
    if [ -f "$log" ]; then
        eval_loss=$(grep -o "'eval_loss': [0-9.]*" "$log" | tail -1 | grep -o '[0-9.]*$' || echo "N/A")
        train_loss=$(grep "{'loss'" "$log" | tail -1 | grep -o "'loss': [0-9.]*" | grep -o '[0-9.]*$' || echo "N/A")
        printf "  %-30s train_loss=%-8s eval_loss=%-8s\n" "$method" "$train_loss" "$eval_loss"
    fi
done

#!/bin/bash
# Spec-GCS Full Experiment Pipeline
# Runs all selector methods on Open-Hermes-2.5 and evaluates on MMLU
#
# Usage: bash scripts/run_experiment.sh
#
# Prerequisites:
#   - Model at /jizhicfs/karonhe/models/LLM-Research/Meta-Llama-3___1-8B
#   - Data at /jizhicfs/karonhe/DataFlex/data/{Openhermes_train,MMLU_valid_cot,MMLUSubset_test}.json
#   - Environment: conda activate spec_gcs

set -e

export CONDA_PREFIX="/jizhicfs/karonhe/miniconda_karonhe/envs/spec_gcs"
export PATH="$CONDA_PREFIX/bin:$PATH"
export PYTHONPATH="/jizhicfs/karonhe/DataFlex/src:$PYTHONPATH"

# Proxy for any HuggingFace downloads
export http_proxy="http://hy-proxy.woa.com:3128"
export https_proxy="http://hy-proxy.woa.com:3128"

SAVE_DIR="/jizhicfs/karonhe/dataflex_saves"
WORK_DIR="/jizhicfs/karonhe/DataFlex"
cd "$WORK_DIR"

echo "============================================="
echo "Spec-GCS Experiment Pipeline"
echo "============================================="
echo "Working directory: $WORK_DIR"
echo "Save directory: $SAVE_DIR"
echo ""

# =============================================
# Step 1: Run Diagnostic (optional, skip if already done)
# =============================================
if [ ! -f "$SAVE_DIR/diagnostic/spectral_results.json" ]; then
    echo "[Step 1] Running spectral diagnostic..."
    python scripts/diagnostic_spectral.py \
        --model_path /jizhicfs/karonhe/models/LLM-Research/Meta-Llama-3___1-8B \
        --data_path /jizhicfs/karonhe/DataFlex/data/Openhermes_train.json \
        --num_samples 5000 \
        --proj_dim 4096 \
        --output_dir "$SAVE_DIR/diagnostic"
    echo "[Step 1] Diagnostic complete."
else
    echo "[Step 1] Diagnostic already exists, skipping."
fi
echo ""

# =============================================
# Step 2: Train with Spec-GCS LogDet (main method)
# =============================================
echo "[Step 2] Training with Spec-GCS LogDet..."
FORCE_TORCHRUN=1 DISABLE_VERSION_CHECK=1 \
    dataflex-cli train examples/train_lora/selectors/spec_gcs_logdet.yaml
echo "[Step 2] Spec-GCS LogDet training complete."
echo ""

# =============================================
# Step 3: Train with Spec-GCS Score (ablation)
# =============================================
echo "[Step 3] Training with Spec-GCS Score..."
FORCE_TORCHRUN=1 DISABLE_VERSION_CHECK=1 \
    dataflex-cli train examples/train_lora/selectors/spec_gcs_score.yaml
echo "[Step 3] Spec-GCS Score training complete."
echo ""

# =============================================
# Step 4: Train with Random baseline
# =============================================
echo "[Step 4] Training with Random baseline..."
FORCE_TORCHRUN=1 DISABLE_VERSION_CHECK=1 \
    dataflex-cli train examples/train_lora/selectors/random.yaml
echo "[Step 4] Random baseline training complete."
echo ""

# =============================================
# Step 5: Evaluation (MMLU)
# =============================================
echo "[Step 5] TODO: Run MMLU evaluation on all checkpoints"
echo "  Use lm_eval or opencompass to evaluate:"
echo "    - $SAVE_DIR/Llama-3.1-8B/spec_gcs_logdet"
echo "    - $SAVE_DIR/Llama-3.1-8B/spec_gcs_score"
echo "    - $SAVE_DIR/Llama-3.1-8B/random"
echo ""

echo "============================================="
echo "Pipeline Complete!"
echo "============================================="

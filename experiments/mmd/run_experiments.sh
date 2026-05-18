#!/usr/bin/env bash
set -euo pipefail

###############################################################################
# MMD Evaluation Experiment Runner
# Usage: bash experiments/mmd/run_experiments.sh
# Run from the DataFlex repository root.
###############################################################################

# ─── Configuration ───────────────────────────────────────────────────────────
NUM_GPUS="${NUM_GPUS:-1}"
SEEDS="${SEEDS:-42 123 456}"
MODEL="${MODEL:-Qwen/Qwen2.5-0.5B}"
EXPERIMENT_DIR="experiments/mmd"
CONFIGS_DIR="${EXPERIMENT_DIR}/configs"
EMBEDDINGS_DIR="${EXPERIMENT_DIR}/embeddings"
RESULTS_DIR="${EXPERIMENT_DIR}/results"
OUTPUT_BASE="${EXPERIMENT_DIR}/outputs"

# Training env
export FORCE_TORCHRUN="${FORCE_TORCHRUN:-1}"
export DISABLE_VERSION_CHECK="${DISABLE_VERSION_CHECK:-1}"

mkdir -p "${RESULTS_DIR}" "${OUTPUT_BASE}" "${EMBEDDINGS_DIR}"

echo "============================================================"
echo " MMD Evaluation Experiments"
echo "============================================================"
echo " NUM_GPUS:       ${NUM_GPUS}"
echo " SEEDS:          ${SEEDS}"
echo " MODEL:          ${MODEL}"
echo " EXPERIMENT_DIR: ${EXPERIMENT_DIR}"
echo "============================================================"

###############################################################################
# Step 0: Offline embedding computation
###############################################################################
echo ""
echo ">>> Step 0: Computing offline embeddings..."
echo "------------------------------------------------------------"

if [ ! -f "${EMBEDDINGS_DIR}/candidate_embeddings.npy" ]; then
    python src/dataflex/offline_selector/offline_mmd_selector.py \
        --model_name_or_path "${MODEL}" \
        --candidate_dataset alpaca_en_demo \
        --target_dataset alpaca_zh_demo \
        --output_dir "${EMBEDDINGS_DIR}" \
        --batch_size 32 \
        --max_length 4096
    echo "    Embeddings saved to ${EMBEDDINGS_DIR}/"
else
    echo "    Embeddings already exist, skipping computation."
fi

###############################################################################
# Step 1: Baseline experiments (random, less)
###############################################################################
echo ""
echo ">>> Step 1: Running baseline experiments..."
echo "------------------------------------------------------------"

for SEED in ${SEEDS}; do
    echo "  [random] seed=${SEED}"
    dataflex-cli train "${CONFIGS_DIR}/random_baseline.yaml" \
        --seed "${SEED}" \
        --output_dir "${OUTPUT_BASE}/random/seed_${SEED}" \
        --overwrite_output_dir true

    echo "  [less] seed=${SEED}"
    dataflex-cli train "${CONFIGS_DIR}/less_baseline.yaml" \
        --seed "${SEED}" \
        --output_dir "${OUTPUT_BASE}/less/seed_${SEED}" \
        --overwrite_output_dir true
done

###############################################################################
# Step 2: MMD variant experiments (emb_rbf, grad_rbf, grad_cov)
###############################################################################
echo ""
echo ">>> Step 2: Running MMD variant experiments..."
echo "------------------------------------------------------------"

MMD_VARIANTS="mmd_emb_rbf mmd_grad_rbf mmd_grad_cov"

for VARIANT in ${MMD_VARIANTS}; do
    for SEED in ${SEEDS}; do
        echo "  [${VARIANT}] seed=${SEED}"
        dataflex-cli train "${CONFIGS_DIR}/${VARIANT}.yaml" \
            --seed "${SEED}" \
            --output_dir "${OUTPUT_BASE}/${VARIANT}/seed_${SEED}" \
            --overwrite_output_dir true
    done
done

###############################################################################
# Step 3: Lambda ablation for emb_rbf
###############################################################################
echo ""
echo ">>> Step 3: Running lambda ablation for emb_rbf..."
echo "------------------------------------------------------------"

LAMBDAS="0.01 0.1 0.5 1.0 2.0 5.0 10.0"
ABLATION_SEED="${ABLATION_SEED:-42}"

for LAMBDA in ${LAMBDAS}; do
    echo "  [emb_rbf] lambda=${LAMBDA}, seed=${ABLATION_SEED}"
    dataflex-cli train "${CONFIGS_DIR}/mmd_emb_rbf.yaml" \
        --seed "${ABLATION_SEED}" \
        --sigma "${LAMBDA}" \
        --output_dir "${OUTPUT_BASE}/ablation_lambda/lambda_${LAMBDA}" \
        --overwrite_output_dir true
done

###############################################################################
# Step 4: Evaluation
###############################################################################
echo ""
echo ">>> Step 4: Running evaluation..."
echo "------------------------------------------------------------"

python "${EXPERIMENT_DIR}/evaluate.py" \
    --results_dir "${OUTPUT_BASE}" \
    --embeddings_dir "${EMBEDDINGS_DIR}" \
    --output "${RESULTS_DIR}/mmd_evaluation_results.json"

echo ""
echo "============================================================"
echo " All experiments completed!"
echo " Results saved to: ${RESULTS_DIR}/mmd_evaluation_results.json"
echo "============================================================"

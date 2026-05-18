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

CANDIDATE_DATA="data/alpaca_en_demo.json"
TARGET_DATA="data/alpaca_zh_demo.json"
EMBED_MODEL="${MODEL}"

if [ ! -f "${EMBEDDINGS_DIR}/candidate_embeddings.npy" ]; then
    python src/dataflex/offline_selector/offline_mmd_selector.py \
        --candidate_path "${CANDIDATE_DATA}" \
        --query_path "${TARGET_DATA}" \
        --embed_model "${EMBED_MODEL}" \
        --save_dir "${EMBEDDINGS_DIR}" \
        --mode embed \
        --batch_size 32
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
# Step 3: Lambda ablation for emb_rbf (offline selection)
###############################################################################
echo ""
echo ">>> Step 3: Running lambda ablation for emb_rbf..."
echo "------------------------------------------------------------"

for LAMBDA in 0.0 0.1 0.3 0.5 0.7 1.0 2.0; do
    echo "  [emb_rbf] lambda_redundancy=${LAMBDA}"
    python src/dataflex/offline_selector/offline_mmd_selector.py \
        --candidate_path "${CANDIDATE_DATA}" \
        --query_path "${TARGET_DATA}" \
        --embed_model "${EMBED_MODEL}" \
        --save_dir "${OUTPUT_BASE}/lambda_ablation/lambda_${LAMBDA}" \
        --mode select \
        --num_select 5000 \
        --lambda_redundancy ${LAMBDA} \
        --sigma auto
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

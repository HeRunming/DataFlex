#!/bin/bash
set -euo pipefail

###############################################################################
# Embedding-Based Static Selection Experiments
#
# Methods: random, mean_target_sim, max_target_sim, mmd_emb_rbf
# These do NOT require model forward passes or GPU for selection.
###############################################################################

# === CONFIGURATION ===
CANDIDATE_DATA="${CANDIDATE_DATA:-data/alpaca_en_demo.json}"
TARGET_DATA_GSM8K="${TARGET_DATA_GSM8K:-data/alpaca_zh_demo.json}"
EMBED_MODEL="${EMBED_MODEL:-/jizhicfs/karonhe/models/sentence-transformers/all-MiniLM-L6-v2}"
OUTPUT_BASE="${OUTPUT_BASE:-experiments/less_aligned/results}"
SEEDS=(42 123 456)
RATIOS=(0.01 0.05 0.10)

echo "================================================================"
echo " Embedding-Based Static Selection"
echo " Candidate: ${CANDIDATE_DATA}"
echo " Embed model: ${EMBED_MODEL}"
echo " Seeds: ${SEEDS[*]}"
echo " Ratios: ${RATIOS[*]}"
echo "================================================================"

###############################################################################
# Helper function
###############################################################################
run_select() {
    local method=$1
    local target=$2
    local ratio=$3
    local seed=$4
    local task_name=$5
    local outdir="${OUTPUT_BASE}/${method}/${task_name}/ratio_${ratio}/seed_${seed}"

    echo "  [SELECT] ${method} | ${task_name} | ratio=${ratio} | seed=${seed}"
    python scripts/static_select_and_train.py select \
        --method "${method}" \
        --candidate_data "${CANDIDATE_DATA}" \
        --target_data "${target}" \
        --embed_model "${EMBED_MODEL}" \
        --selection_ratio "${ratio}" \
        --output_dir "${outdir}" \
        --seed "${seed}" \
        2>&1 | tail -3
}

###############################################################################
# Run selections
###############################################################################
echo ""
echo ">>> Running embedding-based data selection..."

for seed in "${SEEDS[@]}"; do
    for ratio in "${RATIOS[@]}"; do
        echo ""
        echo "--- Seed: ${seed}, Ratio: ${ratio} ---"

        # Random baseline
        run_select "random" "${TARGET_DATA_GSM8K}" "${ratio}" "${seed}" "gsm8k"

        # Mean target similarity (cosine, target relevance only)
        run_select "mean_target_sim" "${TARGET_DATA_GSM8K}" "${ratio}" "${seed}" "gsm8k"

        # Max target similarity (strict nearest neighbor)
        run_select "max_target_sim" "${TARGET_DATA_GSM8K}" "${ratio}" "${seed}" "gsm8k"

        # Mean target RBF (same kernel as MMD, but no redundancy - exact ablation)
        run_select "mean_target_rbf" "${TARGET_DATA_GSM8K}" "${ratio}" "${seed}" "gsm8k"

        # MMD-Emb-RBF (exact marginal greedy, with redundancy penalty)
        run_select "mmd_emb_rbf" "${TARGET_DATA_GSM8K}" "${ratio}" "${seed}" "gsm8k"
    done
done

echo ""
echo "================================================================"
echo " Embedding selection complete. Results in: ${OUTPUT_BASE}"
echo " Next: train on selected subsets via:"
echo "   python scripts/static_select_and_train.py train \\"
echo "     --base_config experiments/less_aligned/configs/train_llama7b_lora.yaml \\"
echo "     --selected_indices <output_dir>/selected_indices.json \\"
echo "     --candidate_data ${CANDIDATE_DATA} \\"
echo "     --output_dir <model_output>"
echo "================================================================"

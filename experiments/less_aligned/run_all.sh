#!/bin/bash
set -euo pipefail

###############################################################################
# LESS-Aligned Experiment Runner
# Runs all selection methods + training + evaluation for paper comparison
###############################################################################

# === CONFIGURATION ===
CANDIDATE_DATA="${CANDIDATE_DATA:-data/flan_v2_100k.json}"
TARGET_DATA_GSM8K="${TARGET_DATA_GSM8K:-data/gsm8k_train_64.json}"
TARGET_DATA_MMLU="${TARGET_DATA_MMLU:-data/mmlu_dev_64.json}"
MODEL="${MODEL:-meta-llama/Llama-2-7b-hf}"
EMBED_MODEL="${EMBED_MODEL:-sentence-transformers/all-MiniLM-L6-v2}"
BASE_CONFIG="${BASE_CONFIG:-experiments/less_aligned/configs/train_llama7b_lora.yaml}"
OUTPUT_BASE="${OUTPUT_BASE:-experiments/less_aligned/results}"
SEEDS=(42 123 456)
RATIOS=(0.01 0.05 0.10)

echo "================================================================"
echo " LESS-Aligned Experiments"
echo " Candidate: ${CANDIDATE_DATA}"
echo " Model: ${MODEL}"
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
        --model_name_or_path "${MODEL}" \
        --embed_model "${EMBED_MODEL}" \
        --selection_ratio "${ratio}" \
        --output_dir "${outdir}" \
        --seed "${seed}" \
        2>&1 | tail -3
}

###############################################################################
# Step 1: Run selections
###############################################################################
echo ""
echo ">>> Step 1: Running data selection..."

for seed in "${SEEDS[@]}"; do
    for ratio in "${RATIOS[@]}"; do
        echo ""
        echo "--- Seed: ${seed}, Ratio: ${ratio} ---"

        # GSM8K target
        run_select "random" "${TARGET_DATA_GSM8K}" "${ratio}" "${seed}" "gsm8k"
        run_select "mmd_emb_rbf" "${TARGET_DATA_GSM8K}" "${ratio}" "${seed}" "gsm8k"
        run_select "embedding_nn" "${TARGET_DATA_GSM8K}" "${ratio}" "${seed}" "gsm8k"

        # MMLU target
        run_select "random" "${TARGET_DATA_MMLU}" "${ratio}" "${seed}" "mmlu"
        run_select "mmd_emb_rbf" "${TARGET_DATA_MMLU}" "${ratio}" "${seed}" "mmlu"
        run_select "embedding_nn" "${TARGET_DATA_MMLU}" "${ratio}" "${seed}" "mmlu"
    done
done

###############################################################################
# Step 2: Training (requires GPU)
###############################################################################
echo ""
echo ">>> Step 2: Training on selected subsets..."
echo "    NOTE: Run training separately with appropriate GPU resources."
echo "    Example:"
echo "      dataflex-cli train ${BASE_CONFIG} \\"
echo "        dataset_dir=<selected_subset.json> \\"
echo "        output_dir=<output_dir>"

###############################################################################
# Step 3: Evaluation
###############################################################################
echo ""
echo ">>> Step 3: Evaluation"
echo "    Run lm-evaluation-harness on trained checkpoints:"
echo "      lm_eval --model hf --model_args pretrained=<checkpoint> \\"
echo "        --tasks gsm8k,mmlu --batch_size 8"

echo ""
echo "================================================================"
echo " Selection complete. See results in: ${OUTPUT_BASE}"
echo "================================================================"

#!/bin/bash
###############################################################################
# Parallel BBH CoT Evaluation - one model per GPU
###############################################################################
set -e

BASE_MODEL="/jizhicfs/karonhe/models/shakechen/Llama-2-7b-hf"
SFT_DIR="/jizhicfs/karonhe/dataflex_saves/sft_results"
OUTPUT_DIR="/jizhicfs/karonhe/dataflex_saves/eval_results/bbh"
mkdir -p "$OUTPUT_DIR"

TASK="bbh_cot_fewshot"
BATCH_SIZE=8

run_eval() {
    local GPU=$1
    local METHOD=$2
    local PEFT_ARG=$3
    local RESULT_DIR="${OUTPUT_DIR}/${METHOD}"

    echo "[GPU ${GPU}] Starting: ${METHOD}"

    CUDA_VISIBLE_DEVICES=${GPU} lm_eval --model hf \
        --model_args "pretrained=${BASE_MODEL}${PEFT_ARG},dtype=bfloat16,trust_remote_code=True" \
        --tasks "${TASK}" \
        --batch_size "${BATCH_SIZE}" \
        --output_path "${RESULT_DIR}" \
        2>&1 > "${OUTPUT_DIR}/${METHOD}.log"

    echo "[GPU ${GPU}] Done: ${METHOD}"
}

# Launch in parallel
run_eval 0 "base_model" "" &
run_eval 1 "random_selected" ",peft=${SFT_DIR}/random_selected" &
run_eval 2 "less_sgd_selected" ",peft=${SFT_DIR}/less_sgd_selected" &
run_eval 3 "mmd_grad_rbf_sgd_selected" ",peft=${SFT_DIR}/mmd_grad_rbf_sgd_selected" &
run_eval 4 "mmd_grad_cov_sgd_selected" ",peft=${SFT_DIR}/mmd_grad_cov_sgd_selected" &

echo "All evaluations launched in parallel on GPUs 0-4"
echo "Waiting for all to complete..."
wait

echo ""
echo "============================================"
echo "All evaluations complete!"
echo "============================================"

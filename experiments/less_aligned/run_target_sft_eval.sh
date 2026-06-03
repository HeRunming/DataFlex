#!/bin/bash
###############################################################################
# Batch SFT + eval for all (target × method) combos.
# 16 SFT runs total (8 methods × 2 targets), 8 GPUs → 2 waves.
# After each SFT: MMLU → lm-eval mmlu (5-shot); TydiQA → custom F1 harness.
###############################################################################
set -uo pipefail
cd /jizhicfs/karonhe/DataFlex
DF=/jizhicfs/karonhe/DataFlex
SAVES=/jizhicfs/karonhe/dataflex_saves
BASE=/jizhicfs/karonhe/models/shakechen/Llama-2-7b-hf
SFT_DIR=$SAVES/sft_results
LOGD=/tmp/target_sft
mkdir -p $LOGD

train_eval() {
    local comp=$1 target=$2 gpu=$3
    local sft_out=$SFT_DIR/${comp}_selected
    # ---- SFT ----
    if [ ! -f "$sft_out/adapter_model.safetensors" ]; then
        echo "[GPU $gpu] SFT $comp"
        CUDA_VISIBLE_DEVICES=$gpu dataflex-cli train $DF/experiments/less_aligned/configs/train_llama7b_lora.yaml \
            dataset=${comp}_selected dataset_dir=$DF/data \
            output_dir=$sft_out lora_alpha=512 num_train_epochs=4 \
            > $LOGD/${comp}_sft.log 2>&1
    fi
    if [ ! -f "$sft_out/adapter_model.safetensors" ]; then
        echo "[FAIL] $comp SFT (see $LOGD/${comp}_sft.log)"; return 1
    fi
    # ---- eval ----
    if [ "$target" = "mmlu" ]; then
        local eval_out=$SAVES/eval_results/mmlu/${comp}_selected
        if ! find "$eval_out" -name "results_*.json" 2>/dev/null | grep -q .; then
            echo "[GPU $gpu] MMLU eval $comp"
            CUDA_VISIBLE_DEVICES=$gpu lm_eval --model hf \
                --model_args "pretrained=$BASE,peft=$sft_out,dtype=bfloat16,trust_remote_code=True" \
                --tasks mmlu --num_fewshot 5 --batch_size 8 \
                --output_path "$eval_out" > $LOGD/${comp}_eval.log 2>&1
        fi
    else  # tydiqa
        local eval_out=$SAVES/eval_results/tydiqa/${comp}_selected.json
        if [ ! -f "$eval_out" ]; then
            echo "[GPU $gpu] TydiQA eval $comp"
            CUDA_VISIBLE_DEVICES=$gpu python scripts/eval_tydiqa.py \
                --adapter $sft_out --output "$eval_out" --batch_size 16 \
                > $LOGD/${comp}_eval.log 2>&1
        fi
    fi
    echo "[done] $comp"
}

METHODS=(less_adam mmd_grad_rbf_adam mmd_grad_cov_adam less_sgd mmd_grad_rbf_sgd mmd_grad_cov_sgd mmd_emb_rbf mmd_emb_rbf_stochastic)

# Wave 1: MMLU (8 methods on GPUs 0-7)
echo "=== WAVE 1: MMLU ==="
gpu=0
for m in "${METHODS[@]}"; do
    train_eval "${m}_mmlu" mmlu $gpu &
    gpu=$((gpu+1))
done
wait
echo "=== WAVE 1 done ==="

# Wave 2: TydiQA (8 methods on GPUs 0-7)
echo "=== WAVE 2: TydiQA ==="
gpu=0
for m in "${METHODS[@]}"; do
    train_eval "${m}_tydiqa" tydiqa $gpu &
    gpu=$((gpu+1))
done
wait
echo "=== WAVE 2 done — all target experiments complete ==="

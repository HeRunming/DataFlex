#!/bin/bash
###############################################################################
# TSDS baseline: SFT + eval for the 3 target-specific TSDS selections.
# Unified pipeline (same as all other methods): from-base Llama-2-7B LoRA
# r=128 a=512, 4 epochs, batch 128; eval BBH/MMLU/TyDiQA(Answer: trigger).
# Idempotent.
###############################################################################
set -uo pipefail
cd /jizhicfs/karonhe/DataFlex
export PATH=/jizhicfs/karonhe/envs/dataflex/bin:$PATH
export http_proxy="http://hy-proxy.woa.com:3128"
export https_proxy="http://hy-proxy.woa.com:3128"
export no_proxy=".woa.com,localhost,127.0.0.1,mirrors.tencent.com"
export HF_ALLOW_CODE_EVAL=1
BASE=/jizhicfs/karonhe/models/shakechen/Llama-2-7b-hf
SFT=/jizhicfs/karonhe/dataflex_saves/sft_results
EVAL=/jizhicfs/karonhe/dataflex_saves/eval_results
LOGD=/tmp/dataflex_logs
mkdir -p "$EVAL/bbh" "$EVAL/mmlu" "$EVAL/tydiqa" "$LOGD"

# ---- SFT (8-GPU, sequential) ----
for tgt in bbh mmlu tydiqa; do
  ds=tsds_${tgt}
  out="$SFT/${ds}_selected"
  if [ -f "$out/adapter_model.safetensors" ]; then echo "[skip sft] $ds"; continue; fi
  echo "=== [$(date +%H:%M:%S)] SFT $ds (8-GPU) ==="
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 FORCE_TORCHRUN=1 dataflex-cli train \
    experiments/less_aligned/configs/train_llama7b_lora.yaml \
    dataset="${ds}_selected" dataset_dir=data output_dir="$out" \
    per_device_train_batch_size=16 gradient_accumulation_steps=1 \
    lora_alpha=512 num_train_epochs=4 > "$LOGD/sft_${ds}.log" 2>&1
  [ -f "$out/adapter_model.safetensors" ] && echo "[done sft] $ds" || { echo "[FAIL sft] $ds"; }
done

# ---- Eval (one per GPU) ----
echo "=== EVAL ==="
ad=$SFT/tsds_bbh_selected
CUDA_VISIBLE_DEVICES=0 lm_eval --model hf \
  --model_args "pretrained=$BASE,peft=$ad,dtype=bfloat16,trust_remote_code=True" \
  --tasks bbh_cot_fewshot --batch_size 8 --output_path $EVAL/bbh/tsds_bbh > $LOGD/eval_bbh_tsds_bbh.log 2>&1 &
ad=$SFT/tsds_mmlu_selected
CUDA_VISIBLE_DEVICES=1 lm_eval --model hf \
  --model_args "pretrained=$BASE,peft=$ad,dtype=bfloat16,trust_remote_code=True" \
  --tasks mmlu --num_fewshot 5 --batch_size 8 --output_path $EVAL/mmlu/tsds_mmlu > $LOGD/eval_mmlu_tsds_mmlu.log 2>&1 &
ad=$SFT/tsds_tydiqa_selected
CUDA_VISIBLE_DEVICES=2 python scripts/eval_tydiqa.py --adapter $ad --output $EVAL/tydiqa/tsds_tydiqa.json --batch_size 16 > $LOGD/eval_tydiqa_tsds_tydiqa.log 2>&1 &
wait
echo "=== TSDS SFT+EVAL COMPLETE [$(date +%H:%M:%S)] ==="

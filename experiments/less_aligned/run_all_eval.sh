#!/bin/bash
###############################################################################
# Evaluate all 9 Adam SFT adapters on their target benchmark.
#   BBH    (3 adapters): lm_eval bbh_cot_fewshot (CoT 3-shot)
#   MMLU   (3 adapters): lm_eval mmlu --num_fewshot 5
#   TyDiQA (3 adapters): scripts/eval_tydiqa.py (1-shot GoldP F1/EM)
# Runs one model per GPU (8 GPUs); 9 jobs => last one waits a slot.
# Idempotent: skips evals whose output already exists.
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

run_bbh() {
  local m=$1 gpu=$2; local ad=$SFT/${m}_selected; local out=$EVAL/bbh/${m}
  if find "$out" -name "results_*.json" 2>/dev/null | grep -q .; then echo "[skip] bbh $m"; return; fi
  echo "[GPU $gpu] BBH $m"
  CUDA_VISIBLE_DEVICES=$gpu lm_eval --model hf \
    --model_args "pretrained=$BASE,peft=$ad,dtype=bfloat16,trust_remote_code=True" \
    --tasks bbh_cot_fewshot --batch_size 8 --output_path "$out" \
    > "$LOGD/eval_bbh_${m}.log" 2>&1
}
run_mmlu() {
  local m=$1 gpu=$2; local ad=$SFT/${m}_selected; local out=$EVAL/mmlu/${m}
  if find "$out" -name "results_*.json" 2>/dev/null | grep -q .; then echo "[skip] mmlu $m"; return; fi
  echo "[GPU $gpu] MMLU $m"
  CUDA_VISIBLE_DEVICES=$gpu lm_eval --model hf \
    --model_args "pretrained=$BASE,peft=$ad,dtype=bfloat16,trust_remote_code=True" \
    --tasks mmlu --num_fewshot 5 --batch_size 8 --output_path "$out" \
    > "$LOGD/eval_mmlu_${m}.log" 2>&1
}
run_tydiqa() {
  local m=$1 gpu=$2; local ad=$SFT/${m}_selected; local out=$EVAL/tydiqa/${m}.json
  if [ -f "$out" ]; then echo "[skip] tydiqa $m"; return; fi
  echo "[GPU $gpu] TyDiQA $m"
  CUDA_VISIBLE_DEVICES=$gpu python scripts/eval_tydiqa.py \
    --adapter "$ad" --output "$out" --batch_size 16 \
    > "$LOGD/eval_tydiqa_${m}.log" 2>&1
}

# Launch all 9 across GPUs 0-7 (+ one extra on GPU0 after a slot frees).
run_bbh   less_adam_bbh            0 &
run_bbh   mmd_grad_rbf_adam_bbh    1 &
run_bbh   mmd_grad_cov_adam_bbh    2 &
run_mmlu  less_adam_mmlu           3 &
run_mmlu  mmd_grad_rbf_adam_mmlu   4 &
run_mmlu  mmd_grad_cov_adam_mmlu   5 &
run_tydiqa less_adam_tydiqa        6 &
run_tydiqa mmd_grad_rbf_adam_tydiqa 7 &
wait
# 9th (cov tydiqa) on GPU0 now free
run_tydiqa mmd_grad_cov_adam_tydiqa 0 &
wait
echo "=== ALL EVAL COMPLETE [$(date +%H:%M:%S)] ==="

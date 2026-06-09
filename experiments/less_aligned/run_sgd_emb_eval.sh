#!/bin/bash
###############################################################################
# Evaluate the 15 SGD+emb adapters on their target benchmark (unified pipeline,
# same eval as the Adam run, incl. the TyDiQA 'Answer:' trigger fix).
#   BBH    -> lm_eval bbh_cot_fewshot
#   MMLU   -> lm_eval mmlu --num_fewshot 5
#   TyDiQA -> scripts/eval_tydiqa.py
# 8 GPUs, idempotent. Two waves (8 + 7).
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

ev() {  # method gpu target
  local m=$1 gpu=$2 tgt=$3; local ad=$SFT/${m}_selected
  if [ "$tgt" = bbh ]; then
    local out=$EVAL/bbh/$m
    find "$out" -name "results_*.json" 2>/dev/null | grep -q . && { echo "[skip] $m"; return; }
    CUDA_VISIBLE_DEVICES=$gpu lm_eval --model hf \
      --model_args "pretrained=$BASE,peft=$ad,dtype=bfloat16,trust_remote_code=True" \
      --tasks bbh_cot_fewshot --batch_size 8 --output_path "$out" > $LOGD/eval_bbh_$m.log 2>&1
  elif [ "$tgt" = mmlu ]; then
    local out=$EVAL/mmlu/$m
    find "$out" -name "results_*.json" 2>/dev/null | grep -q . && { echo "[skip] $m"; return; }
    CUDA_VISIBLE_DEVICES=$gpu lm_eval --model hf \
      --model_args "pretrained=$BASE,peft=$ad,dtype=bfloat16,trust_remote_code=True" \
      --tasks mmlu --num_fewshot 5 --batch_size 8 --output_path "$out" > $LOGD/eval_mmlu_$m.log 2>&1
  else
    local out=$EVAL/tydiqa/$m.json
    [ -f "$out" ] && { echo "[skip] $m"; return; }
    CUDA_VISIBLE_DEVICES=$gpu python scripts/eval_tydiqa.py --adapter "$ad" --output "$out" --batch_size 16 > $LOGD/eval_tydiqa_$m.log 2>&1
  fi
  echo "[done] $m"
}

# Wave 1 (8): all BBH + all MMLU + 2 tydiqa
ev less_sgd_bbh 0 bbh &
ev mmd_grad_rbf_sgd_bbh 1 bbh &
ev mmd_grad_cov_sgd_bbh 2 bbh &
ev mmd_emb_rbf_bbh 3 bbh &
ev mmd_emb_rbf_stochastic_bbh 4 bbh &
ev less_sgd_mmlu 5 mmlu &
ev mmd_grad_rbf_sgd_mmlu 6 mmlu &
ev mmd_grad_cov_sgd_mmlu 7 mmlu &
wait
# Wave 2 (7): rest of MMLU + all tydiqa
ev mmd_emb_rbf_mmlu 0 mmlu &
ev mmd_emb_rbf_stochastic_mmlu 1 mmlu &
ev less_sgd_tydiqa 2 tydiqa &
ev mmd_grad_rbf_sgd_tydiqa 3 tydiqa &
ev mmd_grad_cov_sgd_tydiqa 4 tydiqa &
ev mmd_emb_rbf_tydiqa 5 tydiqa &
ev mmd_emb_rbf_stochastic_tydiqa 6 tydiqa &
wait
echo "=== ALL SGD+emb EVAL COMPLETE [$(date +%H:%M:%S)] ==="

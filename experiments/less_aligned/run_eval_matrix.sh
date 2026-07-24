#!/bin/bash
###############################################################################
# Multi-seed eval matrix (idempotent). Evaluates every SFT adapter
# sft_results/{method}_{target}_seed{s} on its matching benchmark.
#   BBH  : lm_eval bbh_cot_fewshot   (data-parallel across all 8 GPUs)
#   MMLU : lm_eval mmlu --num_fewshot 5  (data-parallel)
#   TyDiQA: scripts/eval_tydiqa.py    (single GPU)
# Proxy REQUIRED (datasets fetched from HF Hub).
# Skips any eval whose result already exists.
###############################################################################
set -uo pipefail
ROOT=/jizhicfs/karonhe/DataFlex_fa
ENVBIN=/jizhicfs/karonhe/envs/dataflex-fa/bin
SAVES=/jizhicfs/karonhe/dataflex_saves
SFT=$SAVES/sft_results
EVAL=$SAVES/eval_results
LOGD=$SAVES/logs
BASE=/jizhicfs/karonhe/models/shakechen/Llama-2-7b-hf
export PATH=$ENVBIN:$PATH
export http_proxy="http://hy-proxy.woa.com:3128"
export https_proxy="http://hy-proxy.woa.com:3128"
export no_proxy=".woa.com,localhost,127.0.0.1,mirrors.tencent.com"
mkdir -p $EVAL/bbh $EVAL/mmlu $EVAL/tydiqa $LOGD
cd $ROOT
SEEDS=(${SEEDS:-42 1})
METHODS=(less_sgd less_adam mmd_grad_rbf_sgd mmd_grad_rbf_adam mmd_grad_cov_sgd mmd_grad_cov_adam mmd_emb_rbf mmd_emb_rbf_stochastic tsds nice)
log(){ echo "[$(date +%m-%d_%H:%M:%S)] $*"; }

# BBH + MMLU: data-parallel across 8 GPUs, one adapter at a time (fast per adapter)
for s in "${SEEDS[@]}"; do
  for m in "${METHODS[@]}"; do
    ad=$SFT/${m}_bbh_seed${s}; out=$EVAL/bbh/${m}_bbh_seed${s}
    [ -f "$ad/adapter_model.safetensors" ] || { log "[miss adapter] $ad"; continue; }
    if find "$out" -name "results_*.json" 2>/dev/null | grep -q .; then log "[skip bbh] ${m}_seed${s}"; else
      log "BBH ${m}_seed${s} (8-GPU dp)"
      accelerate launch --num_processes 8 --main_process_port 29561 -m lm_eval --model hf \
        --model_args "pretrained=$BASE,peft=$ad,dtype=bfloat16,trust_remote_code=True" \
        --tasks bbh_cot_fewshot --batch_size 16 --output_path "$out" \
        > $LOGD/eval_bbh_${m}_seed${s}.log 2>&1
    fi
  done
done
for s in "${SEEDS[@]}"; do
  for m in "${METHODS[@]}"; do
    ad=$SFT/${m}_mmlu_seed${s}; out=$EVAL/mmlu/${m}_mmlu_seed${s}
    [ -f "$ad/adapter_model.safetensors" ] || { log "[miss adapter] $ad"; continue; }
    if find "$out" -name "results_*.json" 2>/dev/null | grep -q .; then log "[skip mmlu] ${m}_seed${s}"; else
      log "MMLU ${m}_seed${s} (8-GPU dp)"
      accelerate launch --num_processes 8 --main_process_port 29562 -m lm_eval --model hf \
        --model_args "pretrained=$BASE,peft=$ad,dtype=bfloat16,trust_remote_code=True" \
        --tasks mmlu --num_fewshot 5 --batch_size 16 --output_path "$out" \
        > $LOGD/eval_mmlu_${m}_seed${s}.log 2>&1
    fi
  done
done
# TyDiQA: single-GPU each, run 8 in parallel across GPUs
gi=0; declare -a P=()
for s in "${SEEDS[@]}"; do
  for m in "${METHODS[@]}"; do
    ad=$SFT/${m}_tydiqa_seed${s}; out=$EVAL/tydiqa/${m}_tydiqa_seed${s}.json
    [ -f "$ad/adapter_model.safetensors" ] || { log "[miss adapter] $ad"; continue; }
    [ -f "$out" ] && { log "[skip tydiqa] ${m}_seed${s}"; continue; }
    log "TyDiQA ${m}_seed${s} (GPU $gi)"
    CUDA_VISIBLE_DEVICES=$gi $ENVBIN/python scripts/eval_tydiqa.py \
      --adapter "$ad" --output "$out" --batch_size 16 > $LOGD/eval_tydiqa_${m}_seed${s}.log 2>&1 &
    P+=($!); gi=$(( (gi+1) % 8 ))
    [ ${#P[@]} -ge 8 ] && { for p in "${P[@]}"; do wait "$p"; done; P=(); }
  done
done
for p in "${P[@]}"; do wait "$p"; done
log "=== EVAL MATRIX COMPLETE ==="

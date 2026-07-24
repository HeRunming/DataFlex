#!/bin/bash
###############################################################################
# Skew experiment: SFT 10 skew adapters (seed 42) then eval by MMLU category
# (mmlu_stem + mmlu_humanities). Idempotent. 8-GPU SFT; data-parallel category eval.
###############################################################################
set -uo pipefail
ROOT=/jizhicfs/karonhe/DataFlex_fa
ENVBIN=/jizhicfs/karonhe/envs/dataflex-fa/bin
SAVES=/jizhicfs/karonhe/dataflex_saves
SFT=$SAVES/sft_results; EVAL=$SAVES/eval_results; LOGD=$SAVES/logs
BASE=/jizhicfs/karonhe/models/shakechen/Llama-2-7b-hf
export PATH=$ENVBIN:$PATH
mkdir -p $EVAL/skew $LOGD; cd $ROOT
# NOTE: do NOT export http_proxy globally — it hangs the torchrun/NCCL rendezvous
# during 8-GPU SFT. Proxy is set inline ONLY for the eval phase (HF downloads).
METHODS=(less_adam nice tsds mmd_grad_cov_sgd mmd_grad_cov_adam)
TS=(stem80 hum80)
log(){ echo "[$(date +%m-%d_%H:%M:%S)] $*"; }

# --- SFT (8-GPU, seed 42) ---
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
for t in "${TS[@]}"; do for m in "${METHODS[@]}"; do
  out=$SFT/skew_${m}_${t}
  [ -f $out/adapter_model.safetensors ] && { log "[skip sft] ${m}_${t}"; continue; }
  log "SFT skew_${m}_${t}"
  dataflex-cli train experiments/less_aligned/configs/train_llama7b_lora.yaml \
    dataset=skew_${m}_${t}_sel output_dir=$out seed=42 \
    per_device_train_batch_size=4 gradient_accumulation_steps=4 lora_alpha=512 num_train_epochs=4 \
    > $LOGD/sft_skew_${m}_${t}.log 2>&1
  [ -f $out/adapter_model.safetensors ] && log "[done] ${m}_${t}" || log "[FAIL] ${m}_${t}"
done; done

# --- eval by category (mmlu_stem + mmlu_humanities), 8-GPU data-parallel ---
for t in "${TS[@]}"; do for m in "${METHODS[@]}"; do
  ad=$SFT/skew_${m}_${t}; out=$EVAL/skew/skew_${m}_${t}
  [ -f $ad/adapter_model.safetensors ] || { log "[miss] ${m}_${t}"; continue; }
  if find "$out" -name "results_*.json" 2>/dev/null | grep -q .; then log "[skip eval] ${m}_${t}"; continue; fi
  log "EVAL skew_${m}_${t} (mmlu_stem+humanities)"
  NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 \
  http_proxy="http://hy-proxy.woa.com:3128" https_proxy="http://hy-proxy.woa.com:3128" \
  no_proxy=".woa.com,localhost,127.0.0.1,mirrors.tencent.com" \
  accelerate launch --num_processes 8 --main_process_port 29650 -m lm_eval --model hf \
    --model_args "pretrained=$BASE,peft=$ad,dtype=bfloat16,trust_remote_code=True" \
    --tasks mmlu_stem,mmlu_humanities --num_fewshot 5 --batch_size 16 --output_path "$out" \
    > $LOGD/eval_skew_${m}_${t}.log 2>&1
done; done
log "=== SKEW EXPERIMENT COMPLETE ==="

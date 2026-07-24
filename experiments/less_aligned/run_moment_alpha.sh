#!/bin/bash
set -uo pipefail
ENVBIN=/jizhicfs/karonhe/envs/dataflex-fa/bin; SAVES=/jizhicfs/karonhe/dataflex_saves
SFT=$SAVES/sft_results; EVAL=$SAVES/eval_results; LOGD=$SAVES/logs
BASE=/jizhicfs/karonhe/models/shakechen/Llama-2-7b-hf
export PATH=$ENVBIN:$PATH; cd /jizhicfs/karonhe/DataFlex_fa; mkdir -p $EVAL/skew $LOGD
log(){ echo "[$(date +%m-%d_%H:%M:%S)] $*"; }
ALPHAS=(0.0 0.25 0.5 0.75 1.0)
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
for a in "${ALPHAS[@]}"; do
  out=$SFT/moment_a${a}_stem80
  [ -f $out/adapter_model.safetensors ] && { log "[skip sft] a=$a"; continue; }
  log "SFT moment a=$a"
  dataflex-cli train experiments/less_aligned/configs/train_llama7b_lora.yaml \
    dataset=moment_a${a}_stem80_sel output_dir=$out seed=42 \
    per_device_train_batch_size=4 gradient_accumulation_steps=4 lora_alpha=512 num_train_epochs=4 \
    > $LOGD/sft_moment_a${a}_stem80.log 2>&1
  [ -f $out/adapter_model.safetensors ] && log "[done] a=$a" || log "[FAIL] a=$a"
done
for a in "${ALPHAS[@]}"; do
  ad=$SFT/moment_a${a}_stem80; out=$EVAL/skew/moment_a${a}_stem80
  [ -f $ad/adapter_model.safetensors ] || continue
  find "$out" -name "results_*.json" 2>/dev/null|grep -q . && { log "[skip eval] a=$a"; continue; }
  log "EVAL moment a=$a"
  NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 http_proxy="http://hy-proxy.woa.com:3128" https_proxy="http://hy-proxy.woa.com:3128" \
  no_proxy=".woa.com,localhost,127.0.0.1,mirrors.tencent.com" \
  accelerate launch --num_processes 8 --main_process_port 29680 -m lm_eval --model hf \
    --model_args "pretrained=$BASE,peft=$ad,dtype=bfloat16,trust_remote_code=True" \
    --tasks mmlu_stem,mmlu_humanities --num_fewshot 5 --batch_size 16 --output_path "$out" \
    > $LOGD/eval_moment_a${a}_stem80.log 2>&1
done
log "=== MOMENT ALPHA SWEEP COMPLETE ==="

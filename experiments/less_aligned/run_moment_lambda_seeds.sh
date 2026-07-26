#!/bin/bash
# choice_0726.md option 2: paired-seed confirmation. Fixed selected subsets (both from
# seed-42 gradient cache); vary only the SFT seed in {1,2} to form paired diffs with the
# existing seed-42 run. Compare ONLY GradCov(λ=0) vs best joint(λ=0.02). Not λ=0.07.
#   pair diff Δ_s = BalAcc(λ=0.02,s) - BalAcc(λ=0,s),  s∈{42,1,2}
set -uo pipefail
ENVBIN=/jizhicfs/karonhe/envs/dataflex-fa/bin
SAVES=/jizhicfs/karonhe/dataflex_saves
SFT=$SAVES/sft_results; EVAL=$SAVES/eval_results; LOGD=$SAVES/logs
BASE=/jizhicfs/karonhe/models/shakechen/Llama-2-7b-hf
export PATH=$ENVBIN:$PATH; cd /jizhicfs/karonhe/DataFlex_fa
mkdir -p $EVAL/skew $LOGD
log(){ echo "[$(date +%m-%d_%H:%M:%S)] $*"; }
# name -> registered dataset key
declare -A DS=( [gradcov]=moment_a0.0_stem80_sel [joint002]=lmoment_l0.02_stem80_sel )
SEEDS=(1 2)
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# ---- SFT ----
for s in "${SEEDS[@]}"; do
  for name in gradcov joint002; do
    out=$SFT/${name}_stem80_seed${s}
    [ -f $out/adapter_model.safetensors ] && { log "[skip sft] $name s=$s"; continue; }
    log "SFT $name seed=$s"
    dataflex-cli train experiments/less_aligned/configs/train_llama7b_lora.yaml \
      dataset=${DS[$name]} output_dir=$out seed=$s \
      per_device_train_batch_size=4 gradient_accumulation_steps=4 lora_alpha=512 num_train_epochs=4 \
      > $LOGD/sft_${name}_stem80_seed${s}.log 2>&1
    [ -f $out/adapter_model.safetensors ] && log "[done sft] $name s=$s" || log "[FAIL sft] $name s=$s"
  done
done

# ---- eval ----
port=29682
for s in "${SEEDS[@]}"; do
  for name in gradcov joint002; do
    ad=$SFT/${name}_stem80_seed${s}; out=$EVAL/skew/${name}_stem80_seed${s}
    [ -f $ad/adapter_model.safetensors ] || { log "[no adapter] $name s=$s"; continue; }
    find "$out" -name "results_*.json" 2>/dev/null|grep -q . && { log "[skip eval] $name s=$s"; continue; }
    log "EVAL $name seed=$s"
    NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 http_proxy="http://hy-proxy.woa.com:3128" https_proxy="http://hy-proxy.woa.com:3128" \
    no_proxy=".woa.com,localhost,127.0.0.1,mirrors.tencent.com" \
    accelerate launch --num_processes 8 --main_process_port $port -m lm_eval --model hf \
      --model_args "pretrained=$BASE,peft=$ad,dtype=bfloat16,trust_remote_code=True" \
      --tasks mmlu_stem,mmlu_humanities --num_fewshot 5 --batch_size 16 --output_path "$out" \
      > $LOGD/eval_${name}_stem80_seed${s}.log 2>&1
    port=$((port+1))
    find "$out" -name "results_*.json" 2>/dev/null|grep -q . && log "[done eval] $name s=$s" || log "[FAIL eval] $name s=$s"
  done
done
log "=== PAIRED SEED PHASE COMPLETE ==="

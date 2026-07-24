#!/bin/bash
###############################################################################
# Extend skew experiment to more seeds. For each NEW seed s:
#   - 6 gradient skew selections (reuse seed-s caches via symlink)
#   - 2 NICE skew selections (seed-s warmup ckpt + seed-s adam cache)
#   - TSDS/emb selection is SEED-INDEPENDENT (bge) -> reuse seed-42's selection,
#     only re-SFT per seed with that seed's training seed.
#   - SFT 10 methods x 2 T (seed=s), eval by mmlu_stem+mmlu_humanities.
# Idempotent. NO short timeouts (8-GPU init is slow ~13min on loaded FS).
###############################################################################
set -uo pipefail
ROOT=/jizhicfs/karonhe/DataFlex_fa; ENVBIN=/jizhicfs/karonhe/envs/dataflex-fa/bin
SAVES=/jizhicfs/karonhe/dataflex_saves; SFT=$SAVES/sft_results; EVAL=$SAVES/eval_results
SUB=$SAVES/sft_subsets; LOGD=$SAVES/logs; CAND=$ROOT/data/less_train_all.jsonl
BASE=/jizhicfs/karonhe/models/shakechen/Llama-2-7b-hf; PY=$ENVBIN/python
export PATH=$ENVBIN:$PATH
mkdir -p $EVAL/skew $SUB $LOGD; cd $ROOT
NEW_SEEDS=(${NEW_SEEDS:-1 2})
GRAD=(less_adam mmd_grad_cov_adam mmd_grad_cov_sgd)
ALL=(less_adam nice tsds mmd_grad_cov_sgd mmd_grad_cov_adam)
TS=(stem80 hum80)
log(){ echo "[$(date +%m-%d_%H:%M:%S)] $*"; }

for s in "${NEW_SEEDS[@]}"; do
  ckpt=$SAVES/sft_results/warmup_seed${s}/checkpoint-1692
  # --- gradient selections (6) ---
  for cfg in experiments/less_aligned/configs/skew/select_*_seed${s}.yaml; do
    cn=$(grep -oE "component_name: .*" "$cfg" | awk '{print $2}')
    [ -f $SAVES/${cn}_output/step_1.json ] && { log "[skip sel] $cn"; continue; }
    log "SEL $cn"; CUDA_VISIBLE_DEVICES=0 dataflex-cli train "$cfg" > $LOGD/sel_${cn}.log 2>&1 || log "[sel FAIL] $cn"
  done
  # --- NICE selections (2) ---
  for t in "${TS[@]}"; do
    out=$SAVES/nice_${t}_seed${s}_output
    [ -f $out/step_1.json ] && { log "[skip nice] ${t}_seed${s}"; continue; }
    log "NICE ${t}_seed${s}"
    CUDA_VISIBLE_DEVICES=0 http_proxy="" https_proxy="" $PY scripts/nice_select.py \
      --candidate_grads $SAVES/less_output_seed${s}/train/1/all_projected_grads.pt \
      --base_model $BASE --adapter $ckpt --target_data data/mmlu_target_${t}.jsonl --target_name mmlu \
      --out_cache_dir $out --proj_dim 8192 --seed 123 --mc 16 --temperature 1.0 --max_new_tokens 16 \
      --selection_ratio 0.05 > $LOGD/nice_${t}_seed${s}.log 2>&1 || log "[nice FAIL] ${t}_seed${s}"
  done

  # --- register subsets (gradient+nice per-seed; tsds reuse seed-42 selection) ---
  $PY - "$s" <<'PYEOF'
import json,sys,subprocess,os
s=sys.argv[1]; SAVES="/jizhicfs/karonhe/dataflex_saves"; SUB=f"{SAVES}/sft_subsets"; CAND="data/less_train_all.jsonl"
info=json.load(open("data/dataset_info.json"))
def reg(cn, cache, key, jsonl):
    if os.path.exists(jsonl):
        info[key]={"file_name":jsonl,"formatting":"sharegpt","columns":{"messages":"messages"},"tags":{"role_tag":"role","content_tag":"content","user_tag":"user","assistant_tag":"assistant"}}; return
    subprocess.run(["/jizhicfs/karonhe/envs/dataflex-fa/bin/python","scripts/export_gradient_selection.py","--candidate_data",CAND,
      "--cache_dir",cache,"--output_dir",f"{SUB}/{key}_export","--method",key,"--target_data","data/mmlu_target.jsonl",
      "--selection_ratio","0.05","--seed","42"],stdout=open(f"{SAVES}/logs/export_{key}.log","w"),stderr=subprocess.STDOUT)
    d=json.load(open(f"{SUB}/{key}_export/selected_subset.json"))
    open(jsonl,"w").write("\n".join(json.dumps(r,ensure_ascii=False) for r in d)+"\n")
    info[key]={"file_name":jsonl,"formatting":"sharegpt","columns":{"messages":"messages"},"tags":{"role_tag":"role","content_tag":"content","user_tag":"user","assistant_tag":"assistant"}}
for t in ["stem80","hum80"]:
    for m in ["less_adam","mmd_grad_cov_adam","mmd_grad_cov_sgd"]:
        reg(f"{m}_{t}_seed{s}", f"{SAVES}/{m}_{t}_seed{s}_output", f"skew_{m}_{t}_seed{s}_sel", f"{SUB}/skew_{m}_{t}_seed{s}_selected.jsonl")
    reg(f"nice_{t}_seed{s}", f"{SAVES}/nice_{t}_seed{s}_output", f"skew_nice_{t}_seed{s}_sel", f"{SUB}/skew_nice_{t}_seed{s}_selected.jsonl")
    # tsds: reuse seed-42 selection (seed-independent), but distinct SFT dataset key per seed points to same jsonl
    reg(f"tsds_{t}", f"{SAVES}/tsds_{t}_output", f"skew_tsds_{t}_seed{s}_sel", f"{SUB}/skew_tsds_{t}_selected.jsonl")
json.dump(info,open("data/dataset_info.json","w"),indent=2,ensure_ascii=False)
print(f"seed{s}: subsets registered")
PYEOF

  # --- SFT (8-GPU, seed=s) ---
  export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
  for t in "${TS[@]}"; do for m in "${ALL[@]}"; do
    out=$SFT/skew_${m}_${t}_seed${s}
    [ -f $out/adapter_model.safetensors ] && { log "[skip sft] ${m}_${t}_seed${s}"; continue; }
    log "SFT skew_${m}_${t}_seed${s}"
    dataflex-cli train experiments/less_aligned/configs/train_llama7b_lora.yaml \
      dataset=skew_${m}_${t}_seed${s}_sel output_dir=$out seed=$s \
      per_device_train_batch_size=4 gradient_accumulation_steps=4 lora_alpha=512 num_train_epochs=4 \
      > $LOGD/sft_skew_${m}_${t}_seed${s}.log 2>&1
    [ -f $out/adapter_model.safetensors ] && log "[done] ${m}_${t}_seed${s}" || log "[FAIL] ${m}_${t}_seed${s}"
  done; done

  # --- eval by category ---
  for t in "${TS[@]}"; do for m in "${ALL[@]}"; do
    ad=$SFT/skew_${m}_${t}_seed${s}; out=$EVAL/skew/skew_${m}_${t}_seed${s}
    [ -f $ad/adapter_model.safetensors ] || continue
    find "$out" -name "results_*.json" 2>/dev/null | grep -q . && { log "[skip eval] ${m}_${t}_seed${s}"; continue; }
    log "EVAL skew_${m}_${t}_seed${s}"
    NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 http_proxy="http://hy-proxy.woa.com:3128" https_proxy="http://hy-proxy.woa.com:3128" \
    no_proxy=".woa.com,localhost,127.0.0.1,mirrors.tencent.com" \
    accelerate launch --num_processes 8 --main_process_port 29660 -m lm_eval --model hf \
      --model_args "pretrained=$BASE,peft=$ad,dtype=bfloat16,trust_remote_code=True" \
      --tasks mmlu_stem,mmlu_humanities --num_fewshot 5 --batch_size 16 --output_path "$out" \
      > $LOGD/eval_skew_${m}_${t}_seed${s}.log 2>&1
  done; done
  log "=== SKEW SEED $s DONE ==="
done
log "=== SKEW SEED EXTENSION COMPLETE ==="

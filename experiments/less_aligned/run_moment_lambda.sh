#!/bin/bash
# Phase-2: train the two Pareto lambda-joint candidates (choice_0725.md) on T_stem80,
# single seed 42, then eval mmlu_stem+mmlu_humanities. NOT beta=0.25/0.5/0.75 (degenerate).
#   A = lambda 0.07 (marginal-balanced) ; B = lambda 0.02 (GradCov-preserving)
# Reuses: existing lmoment_l{lam}_stem80_output selections, export_gradient_selection.py,
# train_llama7b_lora.yaml config, GradCov(=lambda0) as the paired reference.
set -uo pipefail
ENVBIN=/jizhicfs/karonhe/envs/dataflex-fa/bin; PY=$ENVBIN/python
SAVES=/jizhicfs/karonhe/dataflex_saves
SFT=$SAVES/sft_results; EVAL=$SAVES/eval_results; LOGD=$SAVES/logs; SUB=$SAVES/sft_subsets
BASE=/jizhicfs/karonhe/models/shakechen/Llama-2-7b-hf
CAND=/jizhicfs/karonhe/DataFlex_fa/data/less_train_all.jsonl
export PATH=$ENVBIN:$PATH; cd /jizhicfs/karonhe/DataFlex_fa
mkdir -p $EVAL/skew $LOGD $SUB
log(){ echo "[$(date +%m-%d_%H:%M:%S)] $*"; }
LAMS=(0.02 0.07)

# ---- export + register subsets ----
$PY - "${LAMS[*]}" <<'PYEOF'
import json,sys,subprocess,os
LAMS=sys.argv[1].split()
SAVES="/jizhicfs/karonhe/dataflex_saves"; SUB=f"{SAVES}/sft_subsets"
CAND="data/less_train_all.jsonl"
info=json.load(open("data/dataset_info.json"))
def reg(cache,key,jsonl):
    if not os.path.exists(jsonl):
        subprocess.run([f"/jizhicfs/karonhe/envs/dataflex-fa/bin/python","scripts/export_gradient_selection.py",
          "--candidate_data",CAND,"--cache_dir",cache,"--output_dir",f"{SUB}/{key}_export","--method",key,
          "--target_data","data/mmlu_target_stem80.jsonl","--selection_ratio","0.05","--seed","42"],
          stdout=open(f"{SAVES}/logs/export_{key}.log","w"),stderr=subprocess.STDOUT)
        d=json.load(open(f"{SUB}/{key}_export/selected_subset.json"))
        open(jsonl,"w").write("\n".join(json.dumps(r,ensure_ascii=False) for r in d)+"\n")
    info[key]={"file_name":jsonl,"formatting":"sharegpt","columns":{"messages":"messages"},
               "tags":{"role_tag":"role","content_tag":"content","user_tag":"user","assistant_tag":"assistant"}}
for lam in LAMS:
    reg(f"{SAVES}/lmoment_l{lam}_stem80_output", f"lmoment_l{lam}_stem80_sel",
        f"{SUB}/lmoment_l{lam}_stem80_sel.jsonl")
json.dump(info,open("data/dataset_info.json","w"),indent=2,ensure_ascii=False)
print("subsets registered:", ["lmoment_l%s_stem80_sel"%l for l in LAMS])
PYEOF
log "registration done"

# ---- SFT ----
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
for lam in "${LAMS[@]}"; do
  out=$SFT/lmoment_l${lam}_stem80
  [ -f $out/adapter_model.safetensors ] && { log "[skip sft] lam=$lam"; continue; }
  log "SFT lam=$lam"
  dataflex-cli train experiments/less_aligned/configs/train_llama7b_lora.yaml \
    dataset=lmoment_l${lam}_stem80_sel output_dir=$out seed=42 \
    per_device_train_batch_size=4 gradient_accumulation_steps=4 lora_alpha=512 num_train_epochs=4 \
    > $LOGD/sft_lmoment_l${lam}_stem80.log 2>&1
  [ -f $out/adapter_model.safetensors ] && log "[done sft] lam=$lam" || log "[FAIL sft] lam=$lam"
done

# ---- eval mmlu_stem + mmlu_humanities ----
for lam in "${LAMS[@]}"; do
  ad=$SFT/lmoment_l${lam}_stem80; out=$EVAL/skew/lmoment_l${lam}_stem80
  [ -f $ad/adapter_model.safetensors ] || { log "[no adapter] lam=$lam"; continue; }
  find "$out" -name "results_*.json" 2>/dev/null|grep -q . && { log "[skip eval] lam=$lam"; continue; }
  log "EVAL lam=$lam"
  NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 http_proxy="http://hy-proxy.woa.com:3128" https_proxy="http://hy-proxy.woa.com:3128" \
  no_proxy=".woa.com,localhost,127.0.0.1,mirrors.tencent.com" \
  accelerate launch --num_processes 8 --main_process_port 29681 -m lm_eval --model hf \
    --model_args "pretrained=$BASE,peft=$ad,dtype=bfloat16,trust_remote_code=True" \
    --tasks mmlu_stem,mmlu_humanities --num_fewshot 5 --batch_size 16 --output_path "$out" \
    > $LOGD/eval_lmoment_l${lam}_stem80.log 2>&1
  find "$out" -name "results_*.json" 2>/dev/null|grep -q . && log "[done eval] lam=$lam" || log "[FAIL eval] lam=$lam"
done
log "=== LAMBDA PARETO PHASE-2 COMPLETE ==="

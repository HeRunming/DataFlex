#!/bin/bash
# GIST rank-rule development gate (code_review_0730_2): choose ONE global rank rule for GIST-SharedProj
# between k=M (target_dim=150 capped) and EVR95 (k<M spectral filtering), on the OLD STEM80/HUM80
# development targets — then freeze it (do NOT tune rank on the new target draws).
#   4 SFT: {full=k=M, evr95} x {stem80, hum80}, seed 42. Selections already generated
#   (gistsp_full_*, gistsp_evr95_*). Pick the rule with higher mean balanced acc across directions.
set -Eeuo pipefail
ENVBIN=/jizhicfs/karonhe/envs/dataflex-fa/bin; PY=$ENVBIN/python
SAVES=/jizhicfs/karonhe/dataflex_saves
SFT=$SAVES/sft_results; EVAL=$SAVES/eval_results; LOGD=$SAVES/logs; SUB=$SAVES/sft_subsets
BASE=/jizhicfs/karonhe/models/shakechen/Llama-2-7b-hf
CAND=/jizhicfs/karonhe/DataFlex_fa/data/less_train_all.jsonl
K=13533
export PATH=$ENVBIN:$PATH; cd /jizhicfs/karonhe/DataFlex_fa
mkdir -p $EVAL/skew $LOGD $SUB
log(){ echo "[$(date +%m-%d_%H:%M:%S)] $*"; }
trap 'log "[FATAL] line=$LINENO command=$BASH_COMMAND"' ERR
declare -A TDATA=( [stem80]=data/mmlu_target_stem80.jsonl [hum80]=data/mmlu_target_hum80.jsonl )

# ---- export + register the 4 GIST-SharedProj subsets ----
$PY - "$K" <<'PYEOF'
import json,subprocess,os,sys
K=int(sys.argv[1]); SAVES="/jizhicfs/karonhe/dataflex_saves"; SUB=f"{SAVES}/sft_subsets"; CAND="data/less_train_all.jsonl"
TDATA={"stem80":"data/mmlu_target_stem80.jsonl","hum80":"data/mmlu_target_hum80.jsonl"}
info=json.load(open("data/dataset_info.json"))
def reg(cache,key,jsonl,tdata):
    if not os.path.exists(jsonl):
        with open(f"{SAVES}/logs/export_{key}.log","w") as lf:
            subprocess.run([f"/jizhicfs/karonhe/envs/dataflex-fa/bin/python","scripts/export_gradient_selection.py",
              "--candidate_data",CAND,"--cache_dir",cache,"--output_dir",f"{SUB}/{key}_export","--method",key,
              "--target_data",tdata,"--selection_ratio","0.05","--seed","42"],stdout=lf,stderr=subprocess.STDOUT,check=True)
        d=json.load(open(f"{SUB}/{key}_export/selected_subset.json")); assert len(d)==K,f"{key} {len(d)}!={K}"
        open(jsonl,"w").write("\n".join(json.dumps(r,ensure_ascii=False) for r in d)+"\n")
    n=sum(1 for _ in open(jsonl)); assert n==K,f"{key} {n}!={K}"
    info[key]={"file_name":jsonl,"formatting":"sharegpt","columns":{"messages":"messages"},
               "tags":{"role_tag":"role","content_tag":"content","user_tag":"user","assistant_tag":"assistant"}}
for t in ["stem80","hum80"]:
    for rule in ["full","evr95"]:
        reg(f"{SAVES}/gistsp_{rule}_{t}", f"gistsp_{rule}_{t}_sel", f"{SUB}/gistsp_{rule}_{t}_sel.jsonl", TDATA[t])
json.dump(info,open("data/dataset_info.json","w"),indent=2,ensure_ascii=False); print("registered 4 gistsp subsets")
PYEOF
log "registration done"

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
for t in stem80 hum80; do for rule in full evr95; do
  out=$SFT/gistsp_${rule}_${t}_seed42
  [[ -f $out/adapter_model.safetensors ]] && { log "[skip sft] $rule $t"; continue; }
  log "SFT gistsp $rule $t seed=42"
  if ! dataflex-cli train experiments/less_aligned/configs/train_llama7b_lora.yaml \
      dataset=gistsp_${rule}_${t}_sel output_dir=$out seed=42 \
      per_device_train_batch_size=4 gradient_accumulation_steps=4 lora_alpha=512 num_train_epochs=4 \
      > $LOGD/sft_gistsp_${rule}_${t}.log 2>&1; then log "[FAIL sft] $rule $t"; exit 1; fi
  [[ -f $out/adapter_model.safetensors ]] || { log "[FAIL sft] $rule $t (no adapter)"; exit 1; }
  log "[done sft] $rule $t"
done; done

port=29710
for t in stem80 hum80; do for rule in full evr95; do
  ad=$SFT/gistsp_${rule}_${t}_seed42; out=$EVAL/skew/gistsp_${rule}_${t}_seed42
  find "$out" -name "results_*.json" 2>/dev/null|grep -q . && { log "[skip eval] $rule $t"; port=$((port+1)); continue; }
  log "EVAL gistsp $rule $t"
  if ! NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 http_proxy="http://hy-proxy.woa.com:3128" https_proxy="http://hy-proxy.woa.com:3128" \
    no_proxy=".woa.com,localhost,127.0.0.1,mirrors.tencent.com" \
    accelerate launch --num_processes 8 --main_process_port $port -m lm_eval --model hf \
      --model_args "pretrained=$BASE,peft=$ad,dtype=bfloat16,trust_remote_code=True" \
      --tasks mmlu_stem,mmlu_humanities --num_fewshot 5 --batch_size 16 --output_path "$out" \
      > $LOGD/eval_gistsp_${rule}_${t}.log 2>&1; then log "[FAIL eval] $rule $t"; exit 1; fi
  port=$((port+1)); log "[done eval] $rule $t"
done; done

$PY - <<'PYS'
import json,glob
def bal(name):
    fs=glob.glob(f"/jizhicfs/karonhe/dataflex_saves/eval_results/skew/{name}/**/results_*.json",recursive=True)
    if not fs: return None
    r=json.load(open(sorted(fs)[-1]))["results"]
    def acc(k):
        for kk in r:
            if kk==k or kk.endswith(k): return r[kk].get("acc,none",r[kk].get("acc"))
    return (acc("mmlu_stem")+acc("mmlu_humanities"))/2
rows=[]
for rule in ["full","evr95"]:
    b=[bal(f"gistsp_{rule}_{t}_seed42") for t in ["stem80","hum80"]]
    m=sum(b)/2
    rows.append((rule,b[0],b[1],m)); print(f"GIST-{rule}: stem80={b[0]:.4f} hum80={b[1]:.4f} MEAN={m:.4f}")
win=max(rows,key=lambda r:r[3])[0]
print(f"WINNER rank rule = {win} (freeze this globally for the pilot)")
open("/jizhicfs/karonhe/dataflex_saves/eval_results/skew/gist_rankrule_gate.json","w").write(
    json.dumps({"rows":rows,"winner":win},indent=2))
PYS
log "=== GIST RANK-RULE GATE COMPLETE ==="

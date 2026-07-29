#!/bin/bash
# review_0729 attribution gate: representation × selector 2x2 on existing STEM80 + HUM80, seed 42.
#   rows = representation {first-order u , second-order (u^Tv)^2}
#   cols = selector {relevance top-k (no repulsion) , MMD coreset}
# Existing cells (reuse, do NOT retrain):
#   1st × MMD  = Linear-MMD : stem80 moment_a1.0_stem80 ; hum80 linear_hum80_seed42
#   2nd × MMD  = DSMC       : stem80 moment_a0.0_stem80 ; hum80 gradcov_hum80_seed42
# NEW cells (this script) = the relevance top-k column (NOT round-robin; code_review_0729):
#   1st × TopK = First-TopK   : select_relevance_topk --order first
#   2nd × TopK = Second-TopK  : select_relevance_topk --order second
# => 4 selections + 4 SFT + 4 eval, seed 42. Decides attribution (representation vs diversity)
#    BEFORE any target-draw matrix. Same offline caches / warmup provenance as the mirror.
#    (cache dir / dataset keys keep firstrr/secondrr names; only the CSV labels say TopK.)
#    A true greedy round-robin selector is deferred to the external-validity phase.
set -Eeuo pipefail
ENVBIN=/jizhicfs/karonhe/envs/dataflex-fa/bin; PY=$ENVBIN/python
SAVES=/jizhicfs/karonhe/dataflex_saves
SFT=$SAVES/sft_results; EVAL=$SAVES/eval_results; LOGD=$SAVES/logs; SUB=$SAVES/sft_subsets
BASE=/jizhicfs/karonhe/models/shakechen/Llama-2-7b-hf
CAND=/jizhicfs/karonhe/DataFlex_fa/data/less_train_all.jsonl
CANDGRAD=$SAVES/less_output/train/1/all_projected_grads.pt
K=13533
export PATH=$ENVBIN:$PATH; cd /jizhicfs/karonhe/DataFlex_fa
mkdir -p $EVAL/skew $LOGD $SUB
log(){ echo "[$(date +%m-%d_%H:%M:%S)] $*"; }
trap 'log "[FATAL] line=$LINENO command=$BASH_COMMAND"' ERR
# target grad cache per direction (SGD target, LESS-aligned; same as prior skew runs)
declare -A TGT=( [stem80]=$SAVES/mmd_grad_cov_adam_stem80_output/target/1/all_projected_grads.pt
                 [hum80]=$SAVES/mmd_grad_cov_adam_hum80_output/target/1/all_projected_grads.pt )
declare -A TDATA=( [stem80]=data/mmlu_target_stem80.jsonl [hum80]=data/mmlu_target_hum80.jsonl )

validate_selection(){ $PY - "$1" "$K" <<'PYV'
import json,sys
d=json.load(open(sys.argv[1])); K=int(sys.argv[2]); idx=d["indices"]
assert len(idx)==K and len(set(idx))==K and min(idx)>=0 and max(idx)<270679
PYV
}

# ---- 1. selection: First-RR / Second-RR on both directions ----
for t in stem80 hum80; do
  for ord in first second; do
    tag=$([ $ord = first ] && echo firstrr || echo secondrr)
    out=$SAVES/${tag}_${t}_output
    if [[ -s $out/step_1.json ]]; then validate_selection $out/step_1.json; log "[skip select, validated] $tag $t"; continue; fi
    log "SELECT $tag $t"
    if ! CUDA_VISIBLE_DEVICES=0 $PY scripts/select_relevance_topk.py \
        --train_grads $CANDGRAD --target_grads ${TGT[$t]} --out_cache_dir $out \
        --num_select $K --order $ord > $LOGD/sel_${tag}_${t}.log 2>&1; then log "[FAIL select] $tag $t"; exit 1; fi
    validate_selection $out/step_1.json; log "[done select] $tag $t"
  done
done

# ---- 2. export + register (fail-fast, validate 13533 unique rows) ----
$PY - "$K" <<'PYEOF'
import json,subprocess,os,sys
K=int(sys.argv[1])
SAVES="/jizhicfs/karonhe/dataflex_saves"; SUB=f"{SAVES}/sft_subsets"; CAND="data/less_train_all.jsonl"
TDATA={"stem80":"data/mmlu_target_stem80.jsonl","hum80":"data/mmlu_target_hum80.jsonl"}
info=json.load(open("data/dataset_info.json"))
def reg(cache,key,jsonl,tdata):
    if not os.path.exists(jsonl):
        with open(f"{SAVES}/logs/export_{key}.log","w") as lf:
            subprocess.run(["/jizhicfs/karonhe/envs/dataflex-fa/bin/python","scripts/export_gradient_selection.py",
              "--candidate_data",CAND,"--cache_dir",cache,"--output_dir",f"{SUB}/{key}_export","--method",key,
              "--target_data",tdata,"--selection_ratio","0.05","--seed","42"],
              stdout=lf,stderr=subprocess.STDOUT,check=True)
        d=json.load(open(f"{SUB}/{key}_export/selected_subset.json"))
        assert len(d)==K, f"{key}: exported {len(d)} != {K}"
        open(jsonl,"w").write("\n".join(json.dumps(r,ensure_ascii=False) for r in d)+"\n")
    n=sum(1 for _ in open(jsonl)); assert n==K, f"{key}: {n} lines != {K}"
    info[key]={"file_name":jsonl,"formatting":"sharegpt","columns":{"messages":"messages"},
               "tags":{"role_tag":"role","content_tag":"content","user_tag":"user","assistant_tag":"assistant"}}
for t in ["stem80","hum80"]:
    for tag in ["firstrr","secondrr"]:
        reg(f"{SAVES}/{tag}_{t}_output", f"{tag}_{t}_sel", f"{SUB}/{tag}_{t}_sel.jsonl", TDATA[t])
json.dump(info,open("data/dataset_info.json","w"),indent=2,ensure_ascii=False)
print("registered+validated: firstrr/secondrr x stem80/hum80")
PYEOF
log "registration done"

# ---- 3. SFT (4 new cells, seed 42) ----
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
for t in stem80 hum80; do for tag in firstrr secondrr; do
  out=$SFT/${tag}_${t}_seed42
  [[ -f $out/adapter_model.safetensors ]] && { log "[skip sft] $tag $t"; continue; }
  log "SFT $tag $t seed=42"
  if ! dataflex-cli train experiments/less_aligned/configs/train_llama7b_lora.yaml \
      dataset=${tag}_${t}_sel output_dir=$out seed=42 \
      per_device_train_batch_size=4 gradient_accumulation_steps=4 lora_alpha=512 num_train_epochs=4 \
      > $LOGD/sft_${tag}_${t}_seed42.log 2>&1; then log "[FAIL sft] $tag $t"; exit 1; fi
  [[ -f $out/adapter_model.safetensors ]] || { log "[FAIL sft] $tag $t (no adapter)"; exit 1; }
  log "[done sft] $tag $t"
done; done

# ---- 4. eval ----
port=29700
for t in stem80 hum80; do for tag in firstrr secondrr; do
  ad=$SFT/${tag}_${t}_seed42; out=$EVAL/skew/${tag}_${t}_seed42
  [[ -f $ad/adapter_model.safetensors ]] || { log "[no adapter] $tag $t"; exit 1; }
  find "$out" -name "results_*.json" 2>/dev/null|grep -q . && { log "[skip eval] $tag $t"; port=$((port+1)); continue; }
  log "EVAL $tag $t"
  if ! NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 http_proxy="http://hy-proxy.woa.com:3128" https_proxy="http://hy-proxy.woa.com:3128" \
    no_proxy=".woa.com,localhost,127.0.0.1,mirrors.tencent.com" \
    accelerate launch --num_processes 8 --main_process_port $port -m lm_eval --model hf \
      --model_args "pretrained=$BASE,peft=$ad,dtype=bfloat16,trust_remote_code=True" \
      --tasks mmlu_stem,mmlu_humanities --num_fewshot 5 --batch_size 16 --output_path "$out" \
      > $LOGD/eval_${tag}_${t}_seed42.log 2>&1; then log "[FAIL eval] $tag $t"; exit 1; fi
  find "$out" -name "results_*.json" 2>/dev/null|grep -q . || { log "[FAIL eval] $tag $t (no results)"; exit 1; }
  port=$((port+1)); log "[done eval] $tag $t"
done; done

# ---- 5. per-run completeness check + 2x2 CSV (reuses existing MMD cells) ----
for spec in firstrr:stem80 secondrr:stem80 firstrr:hum80 secondrr:hum80; do
  tag=${spec%%:*}; t=${spec##*:}; od="$EVAL/skew/${tag}_${t}_seed42"
  [[ $(find "$od" -name "results_*.json" 2>/dev/null | wc -l) -ge 1 ]] || { log "[FATAL] no eval $tag $t"; exit 1; }
done
$PY - <<'PYS'
import json,glob
def acc(r,k):
    for kk in r:
        if kk==k or kk.endswith(k): return r[kk].get("acc,none",r[kk].get("acc"))
def bal(name):
    fs=glob.glob(f"/jizhicfs/karonhe/dataflex_saves/eval_results/skew/{name}/**/results_*.json",recursive=True)
    if not fs: return None
    r=json.load(open(sorted(fs)[-1]))["results"]; s=acc(r,"mmlu_stem"); h=acc(r,"mmlu_humanities")
    return s,h,(s+h)/2
# 2x2 cells per direction: (representation, selector) -> eval dir name.
# NB cache dirs keep the firstrr/secondrr filenames (already written), but the cells are
# relevance TOP-K not round-robin (code_review_0729) -> labelled First-TopK / Second-TopK.
cells={
 "stem80":{"1st-TopK":"firstrr_stem80_seed42","1st-MMD(Linear)":"moment_a1.0_stem80",
           "2nd-TopK":"secondrr_stem80_seed42","2nd-MMD(DSMC)":"moment_a0.0_stem80"},
 "hum80":{"1st-TopK":"firstrr_hum80_seed42","1st-MMD(Linear)":"linear_hum80_seed42",
          "2nd-TopK":"secondrr_hum80_seed42","2nd-MMD(DSMC)":"gradcov_hum80_seed42"},
}
out="/jizhicfs/karonhe/dataflex_saves/eval_results/skew/attribution_2x2_results.csv"
lines=["target,cell,representation,selector,stem,hum,balanced"]
for t,d in cells.items():
    for cell,name in d.items():
        v=bal(name)
        rep="2nd" if cell.startswith("2nd") else "1st"; selr="MMD" if "MMD" in cell else "TopK"
        if not v: lines.append(f"{t},{cell},{rep},{selr},NA,NA,NA"); continue
        lines.append(f"{t},{cell},{rep},{selr},{v[0]:.4f},{v[1]:.4f},{v[2]:.4f}")
open(out,"w").write("\n".join(lines)+"\n"); print("wrote",out); print("\n".join(lines))
PYS
log "=== 2x2 ATTRIBUTION GATE COMPLETE ==="

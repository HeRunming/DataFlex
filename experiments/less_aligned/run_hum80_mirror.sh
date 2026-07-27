#!/bin/bash
# review_0727.md + code_review_0727.md: slim T_hum80 mirror for skew-direction invariance.
# Three points, SAME offline greedy code paths as the stem80 experiment (so GradCov-vs-joint is
# not confounded by selector):
#   gradcov = select_moment_mmd.py    --alpha 0.0   (pure 2nd-order; == select_moment_lambda --lam 0)
#   joint   = select_moment_lambda.py --lam   0.02  (best interior joint)
#   linear  = select_moment_mmd.py    --alpha 1.0   (1st-order endpoint)
# Seeds: GradCov & joint get {42,1,2}; linear gets {42} only (endpoint explainer, not the main
#   paired comparison -> 7 SFT not 9). If linear@42 lands near GradCov, add seeds 1,2 after.
# Provenance (verified before commit): candidate cache less_output/train/1 and hum80 target
#   cache both from warmup_seed42/checkpoint-1692 (adapter + optimizer.pt sha256 identical to
#   random_selected/checkpoint-1692). Adam-candidate / SGD-target = LESS-aligned protocol (kept
#   for stem80 comparability; symmetric Adam/Adam or SGD/SGD is a separate main-paper ablation).
# NO further lambda sweeps (frozen: lambda_default=0).
set -Eeuo pipefail
ENVBIN=/jizhicfs/karonhe/envs/dataflex-fa/bin; PY=$ENVBIN/python
SAVES=/jizhicfs/karonhe/dataflex_saves
SFT=$SAVES/sft_results; EVAL=$SAVES/eval_results; LOGD=$SAVES/logs; SUB=$SAVES/sft_subsets
BASE=/jizhicfs/karonhe/models/shakechen/Llama-2-7b-hf
CAND=/jizhicfs/karonhe/DataFlex_fa/data/less_train_all.jsonl
CANDGRAD=$SAVES/less_output/train/1/all_projected_grads.pt
TGT=$SAVES/mmd_grad_cov_adam_hum80_output/target/1/all_projected_grads.pt
K=13533
export PATH=$ENVBIN:$PATH; cd /jizhicfs/karonhe/DataFlex_fa
mkdir -p $EVAL/skew $LOGD $SUB
log(){ echo "[$(date +%m-%d_%H:%M:%S)] $*"; }
trap 'log "[FATAL] line=$LINENO command=$BASH_COMMAND"' ERR

# fail-fast selection wrapper: run, then require a well-formed step_1.json
run_selection(){
  local name=$1 out=$2; shift 2
  if [[ -s "$out/step_1.json" ]]; then log "[skip select] $name"; return; fi
  log "SELECT $name"
  if ! "$@" > "$LOGD/sel_${name}.log" 2>&1; then log "[FAIL select] $name (see sel_${name}.log)"; exit 1; fi
  $PY - "$out/step_1.json" "$K" <<'PYV'
import json,sys
d=json.load(open(sys.argv[1])); K=int(sys.argv[2]); idx=d["indices"]
assert len(idx)==K, f"{len(idx)}!={K}"
assert len(set(idx))==K, "duplicate indices"
assert min(idx)>=0 and max(idx)<270679, "index out of candidate range"
PYV
  log "[done select] $name"
}

# ---- 1. selection (3 points, offline greedy on hum80 target) ----
run_selection gradcov $SAVES/hmoment_gradcov_hum80_output \
  $PY scripts/select_moment_mmd.py --train_grads $CANDGRAD --target_grads $TGT \
  --out_cache_dir $SAVES/hmoment_gradcov_hum80_output --num_select $K --alpha 0.0
run_selection joint002 $SAVES/hmoment_l0.02_hum80_output \
  $PY scripts/select_moment_lambda.py --train_grads $CANDGRAD --target_grads $TGT \
  --out_cache_dir $SAVES/hmoment_l0.02_hum80_output --num_select $K --lam 0.02
run_selection linear $SAVES/hmoment_linear_hum80_output \
  $PY scripts/select_moment_mmd.py --train_grads $CANDGRAD --target_grads $TGT \
  --out_cache_dir $SAVES/hmoment_linear_hum80_output --num_select $K --alpha 1.0

# ---- 2. export + register subsets (fail-fast; validate 13533 unique rows) ----
$PY - "$K" <<'PYEOF'
import json,subprocess,os,sys
K=int(sys.argv[1])
SAVES="/jizhicfs/karonhe/dataflex_saves"; SUB=f"{SAVES}/sft_subsets"; CAND="data/less_train_all.jsonl"
info=json.load(open("data/dataset_info.json"))
def reg(cache,key,jsonl):
    if not os.path.exists(jsonl):
        with open(f"{SAVES}/logs/export_{key}.log","w") as lf:
            subprocess.run(["/jizhicfs/karonhe/envs/dataflex-fa/bin/python","scripts/export_gradient_selection.py",
              "--candidate_data",CAND,"--cache_dir",cache,"--output_dir",f"{SUB}/{key}_export","--method",key,
              "--target_data","data/mmlu_target_hum80.jsonl","--selection_ratio","0.05","--seed","42"],
              stdout=lf,stderr=subprocess.STDOUT,check=True)
        d=json.load(open(f"{SUB}/{key}_export/selected_subset.json"))
        assert len(d)==K, f"{key}: exported {len(d)} rows != {K}"
        open(jsonl,"w").write("\n".join(json.dumps(r,ensure_ascii=False) for r in d)+"\n")
    n=sum(1 for _ in open(jsonl))
    assert n==K, f"{key}: {jsonl} has {n} lines != {K}"
    info[key]={"file_name":jsonl,"formatting":"sharegpt","columns":{"messages":"messages"},
               "tags":{"role_tag":"role","content_tag":"content","user_tag":"user","assistant_tag":"assistant"}}
reg(f"{SAVES}/hmoment_gradcov_hum80_output","hmoment_gradcov_hum80_sel",f"{SUB}/hmoment_gradcov_hum80_sel.jsonl")
reg(f"{SAVES}/hmoment_l0.02_hum80_output",  "hmoment_l0.02_hum80_sel",  f"{SUB}/hmoment_l0.02_hum80_sel.jsonl")
reg(f"{SAVES}/hmoment_linear_hum80_output", "hmoment_linear_hum80_sel", f"{SUB}/hmoment_linear_hum80_sel.jsonl")
json.dump(info,open("data/dataset_info.json","w"),indent=2,ensure_ascii=False)
print("registered+validated: gradcov / l0.02 / linear hum80")
PYEOF
log "registration done"

# ---- 3. SFT (gradcov,joint x {42,1,2} ; linear x {42}) = 7 runs ----
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
declare -A DS=( [gradcov]=hmoment_gradcov_hum80_sel [joint002]=hmoment_l0.02_hum80_sel [linear]=hmoment_linear_hum80_sel )
run_sft(){
  local name=$1 s=$2 out=$SFT/${1}_hum80_seed${2}
  [[ -f $out/adapter_model.safetensors ]] && { log "[skip sft] $name s=$s"; return; }
  log "SFT $name seed=$s"
  if ! dataflex-cli train experiments/less_aligned/configs/train_llama7b_lora.yaml \
      dataset=${DS[$name]} output_dir=$out seed=$s \
      per_device_train_batch_size=4 gradient_accumulation_steps=4 lora_alpha=512 num_train_epochs=4 \
      > $LOGD/sft_${name}_hum80_seed${s}.log 2>&1; then log "[FAIL sft] $name s=$s"; exit 1; fi
  [[ -f $out/adapter_model.safetensors ]] || { log "[FAIL sft] $name s=$s (no adapter)"; exit 1; }
  log "[done sft] $name s=$s"
}
for s in 42 1 2; do run_sft gradcov $s; run_sft joint002 $s; done
run_sft linear 42

# ---- 4. eval (mmlu_stem + mmlu_humanities) ----
port=29690
run_eval(){
  local name=$1 s=$2 ad=$SFT/${1}_hum80_seed${2} out=$EVAL/skew/${1}_hum80_seed${2}
  [[ -f $ad/adapter_model.safetensors ]] || { log "[no adapter] $name s=$s"; exit 1; }
  find "$out" -name "results_*.json" 2>/dev/null|grep -q . && { log "[skip eval] $name s=$s"; return; }
  log "EVAL $name seed=$s"
  if ! NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 http_proxy="http://hy-proxy.woa.com:3128" https_proxy="http://hy-proxy.woa.com:3128" \
    no_proxy=".woa.com,localhost,127.0.0.1,mirrors.tencent.com" \
    accelerate launch --num_processes 8 --main_process_port $port -m lm_eval --model hf \
      --model_args "pretrained=$BASE,peft=$ad,dtype=bfloat16,trust_remote_code=True" \
      --tasks mmlu_stem,mmlu_humanities --num_fewshot 5 --batch_size 16 --output_path "$out" \
      > $LOGD/eval_${name}_hum80_seed${s}.log 2>&1; then log "[FAIL eval] $name s=$s"; exit 1; fi
  find "$out" -name "results_*.json" 2>/dev/null|grep -q . || { log "[FAIL eval] $name s=$s (no results)"; exit 1; }
  log "[done eval] $name s=$s"
}
for s in 42 1 2; do run_eval gradcov $s; port=$((port+1)); run_eval joint002 $s; port=$((port+1)); done
run_eval linear 42; port=$((port+1))

# ---- 5. completeness check + CSV (do not trust the COMPLETE log line) ----
expected=7
actual=$(find "$EVAL/skew" -path "*_hum80_seed*" -name "results_*.json" 2>/dev/null | \
         grep -E "/(gradcov|joint002|linear)_hum80_seed(42|1|2)/" | wc -l)
if [[ "$actual" -ne "$expected" ]]; then log "[FATAL] expected $expected eval outputs, found $actual"; exit 1; fi
$PY - <<'PYS'
import json,glob,os
rows=[("gradcov",42),("gradcov",1),("gradcov",2),("joint002",42),("joint002",1),("joint002",2),("linear",42)]
out="/jizhicfs/karonhe/dataflex_saves/eval_results/skew/hum80_mirror_results.csv"
def acc(r,k):
    for kk in r:
        if kk==k or kk.endswith(k): return r[kk].get("acc,none",r[kk].get("acc"))
lines=["method,seed,stem,hum,balanced"]
for name,s in rows:
    fs=glob.glob(f"/jizhicfs/karonhe/dataflex_saves/eval_results/skew/{name}_hum80_seed{s}/**/results_*.json",recursive=True)
    r=json.load(open(sorted(fs)[-1]))["results"]
    st=acc(r,"mmlu_stem"); h=acc(r,"mmlu_humanities")
    lines.append(f"{name},{s},{st:.4f},{h:.4f},{(st+h)/2:.4f}")
open(out,"w").write("\n".join(lines)+"\n")
print("wrote",out); print("\n".join(lines))
PYS
log "=== T_hum80 MIRROR COMPLETE (7 SFT, all eval present) ==="

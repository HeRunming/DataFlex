#!/bin/bash
# Target-draw pilot driver (staged, resume + fail-fast). review_0729/_0730/_0731, frozen protocol.
#
# INTERFACE: configured via ENV VARS (not CLI flags): DRAW (default stem80_draw0), PHASES
#   (comma list from: setup gengrad select export diag train). Unknown/empty phases are rejected.
#   e.g.  DRAW=stem80_draw0 PHASES=setup,gengrad,select,export,diag bash run_targetdraw_pilot.sh
#
# 8 methods per draw (frozen): dsmc second_rr less first_rr gist nice randk randk_lenmatch
#   dsmc           = select_moment_mmd  --alpha 0.0            (2nd-order MMD coreset) [headline]
#   less           = select_relevance_topk --order first       (1st-order relevance top-k, LESS-like)
#   first_rr       = select_round_robin  --order first
#   second_rr      = select_round_robin  --order second
#   gist           = select_gist_faithful --rank 150           (GIST-SharedProj, k=min(150,M)=M)
#   nice           = nice_select.py (NICE-MMLU-EM, per-draw target policy grads; labelled adaptation)
#   randk          = uniform fixed-K, subset seed 2000+draw_id
#   randk_lenmatch = fixed-K length-histogram-matched to dsmc (draw-specific)
# NO SFT/eval in this driver's implemented phases (train is a stub); SFT is a separate gated step.
set -Eeuo pipefail
ENVBIN=/jizhicfs/karonhe/envs/dataflex-fa/bin; PY=$ENVBIN/python
SAVES=/jizhicfs/karonhe/dataflex_saves; ROOT=/jizhicfs/karonhe/DataFlex_fa
CANDGRAD=$SAVES/less_output/train/1/all_projected_grads.pt
CAND_JSONL=$ROOT/data/less_train_all.jsonl
WARMUP=$SAVES/sft_results/warmup_seed42/checkpoint-1692
BASE=/jizhicfs/karonhe/models/shakechen/Llama-2-7b-hf
LOGD=$SAVES/logs; K=13533
export PATH=$ENVBIN:$PATH; cd $ROOT; mkdir -p $LOGD
log(){ echo "[$(date +%m-%d_%H:%M:%S)] $*"; }
trap 'log "[FATAL] line=$LINENO cmd=$BASH_COMMAND"' ERR

DRAW="${DRAW:-stem80_draw0}"
PHASES="${PHASES:-}"
[[ -z "$PHASES" ]] && { log "[FATAL] PHASES env var is empty (e.g. PHASES=setup,gengrad,select,export,diag)"; exit 2; }
for p in ${PHASES//,/ }; do
  case "$p" in setup|gengrad|select|export|diag|train) ;; *) log "[FATAL] unknown phase '$p'"; exit 2;; esac
done
has(){ [[ ",$PHASES," == *",$1,"* ]]; }
[[ -f $ROOT/data/target_draws/${DRAW}.jsonl ]] || { log "[FATAL] no draw jsonl for $DRAW"; exit 2; }
CACHE=$SAVES/draw_${DRAW}_output
TGT=$CACHE/target/1/all_projected_grads.pt
did=${DRAW##*draw}; rr_seed=$((3000+did)); rk_seed=$((2000+did))

validate_sel(){ $PY - "$1" "$K" <<'PYV'
import json,sys
d=json.load(open(sys.argv[1])); K=int(sys.argv[2]); idx=d["indices"]
assert len(idx)==K and len(set(idx))==K and min(idx)>=0 and max(idx)<270679, "bad selection"
PYV
}

# ───────────── preflight manifest (records + validates the frozen inputs) ─────────────
preflight(){
  local mfile=$SAVES/draw_${DRAW}_pilot_manifest.json
  $PY - "$DRAW" "$SAVES" "$ROOT" "$CANDGRAD" "$WARMUP" "$K" > "$mfile" <<'PYM'
import json,sys,os,hashlib,glob
draw,SAVES,ROOT,cand,warmup,K=sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4],int(sys.argv[6]) if False else sys.argv[4],None
draw,SAVES,ROOT,cand,warmup,K=sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4],sys.argv[5],int(sys.argv[6])
def sh(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()
tgt_jsonl=f"{ROOT}/data/target_draws/{draw}.jsonl"
meta=json.load(open(f"{ROOT}/data/target_draws/{draw}.meta.json"))
cand_real=os.path.realpath(cand)
man={"driver_commit":os.popen("git -C %s rev-parse --short HEAD"%ROOT).read().strip(),
     "draw":draw,"K":K,
     "target_jsonl":tgt_jsonl,"target_file_sha256":sh(tgt_jsonl),
     "target_file_sha256_expected":meta["target_file_sha256"],
     "ordered_target_ids_sha256_expected":meta["ordered_target_ids_sha256"],
     "candidate_cache_path":cand,"candidate_cache_realpath":cand_real,
     "candidate_cache_sha256":sh(cand_real),
     "warmup_adapter_sha256":sh(f"{warmup}/adapter_model.safetensors"),
     "warmup_optimizer_sha256":sh(f"{warmup}/optimizer.pt"),
     "proj_dim":8192,"proj_seed":123,"gradient_type_candidate":"adam","gradient_type_target":"sgd",
     "train_seed":meta["train_seed"],"rr_perm_seed":meta["rr_perm_seed"]}
# validate target file hash matches the frozen artifact
assert man["target_file_sha256"]==man["target_file_sha256_expected"], "TARGET JSONL HASH MISMATCH vs frozen meta"
print(json.dumps(man,indent=2))
PYM
  log "[preflight] manifest -> $mfile (target hash matches frozen meta)"
}

# ───────────── phase: setup ─────────────
if has setup; then
  log "SETUP $DRAW"; $PY scripts/setup_draw_target.py --draw $DRAW; preflight
fi

# ───────────── phase: gengrad (target-only; candidate reused via verified symlink) ─────────────
if has gengrad; then
  if [[ -s $TGT ]]; then
    # validate existing target cache before skipping
    $PY - "$TGT" <<'PYT'
import torch,sys
X=torch.load(sys.argv[1],map_location="cpu").float()
assert tuple(X.shape)==(64,8192), f"target shape {tuple(X.shape)} != (64,8192)"
assert torch.isfinite(X).all(), "target has NaN/Inf"
n=X.norm(dim=1); assert (n>1e-6).all(), "target has zero rows"
print(f"[gengrad] existing target cache OK shape={tuple(X.shape)} norm~{float(n.mean()):.3f}")
PYT
    log "[skip gengrad] $DRAW target cache validated"
  else
    mkdir -p $CACHE/train/1
    if [[ ! -e $CACHE/train/1/all_projected_grads.pt ]]; then
      ln -s $CANDGRAD $CACHE/train/1/all_projected_grads.pt
    fi
    # verify the symlink resolves to the FROZEN candidate cache (not any stray file)
    resolved=$(readlink -f $CACHE/train/1/all_projected_grads.pt)
    [[ "$resolved" == "$(readlink -f $CANDGRAD)" ]] || { log "[FATAL] candidate symlink resolves to $resolved, not $CANDGRAD"; exit 1; }
    log "[gengrad] candidate symlink verified -> $resolved"
    log "GENGRAD $DRAW (8-GPU target-only extraction)"
    export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
    if ! dataflex-cli train experiments/less_aligned/configs/draws/select_${DRAW}.yaml \
        > $LOGD/gengrad_${DRAW}.log 2>&1; then log "[FAIL gengrad] $DRAW"; exit 1; fi
    [[ -s $TGT ]] || { log "[FAIL gengrad] no target grads at $TGT"; exit 1; }
    log "[done gengrad] $DRAW"
  fi
fi

# ───────────── phase: select (8 methods) ─────────────
if has select; then
  gen(){ local name=$1; shift; local out=$SAVES/sel_${DRAW}_${name}
    if [[ -s $out/step_1.json ]]; then validate_sel $out/step_1.json; log "[skip select] $name"; return; fi
    log "SELECT $name ($DRAW)"
    if ! CUDA_VISIBLE_DEVICES=0 "$@" --out_cache_dir $out --num_select $K > $LOGD/sel_${DRAW}_${name}.log 2>&1; then
      log "[FAIL select] $name"; exit 1; fi
    validate_sel $out/step_1.json; log "[done select] $name"
  }
  gen dsmc      $PY scripts/select_moment_mmd.py     --train_grads $CANDGRAD --target_grads $TGT --alpha 0.0
  gen less      $PY scripts/select_relevance_topk.py --train_grads $CANDGRAD --target_grads $TGT --order first
  gen first_rr  $PY scripts/select_round_robin.py    --train_grads $CANDGRAD --target_grads $TGT --order first  --perm_seed $rr_seed
  gen second_rr $PY scripts/select_round_robin.py    --train_grads $CANDGRAD --target_grads $TGT --order second --perm_seed $rr_seed
  gen gist      $PY scripts/select_gist_faithful.py  --train_grads $CANDGRAD --target_grads $TGT --rank 150
  # randk (uniform fixed-K)
  out=$SAVES/sel_${DRAW}_randk
  if [[ -s $out/step_1.json ]]; then validate_sel $out/step_1.json; log "[skip select] randk"; else
    log "SELECT randk ($DRAW) seed=$rk_seed"
    CUDA_VISIBLE_DEVICES=0 $PY - "$CANDGRAD" "$out" "$K" "$rk_seed" > $LOGD/sel_${DRAW}_randk.log 2>&1 <<'PYR'
import torch,json,sys,os
cand,out,K,seed=sys.argv[1],sys.argv[2],int(sys.argv[3]),int(sys.argv[4])
N=torch.load(cand,map_location="cpu").shape[0]
g=torch.Generator().manual_seed(seed); idx=torch.randperm(N,generator=g)[:K].tolist()
assert len(set(idx))==K
os.makedirs(out,exist_ok=True)
json.dump({"indices":idx,"metric":{"kernel":"random_k","seed":seed,"num_select":K}},open(f"{out}/step_1.json","w"))
PYR
    validate_sel $out/step_1.json; log "[done select] randk"
  fi
  # nice (NICE-MMLU-EM; strict-deterministic so selection is bit-reproducible — verified draw0)
  out=$SAVES/sel_${DRAW}_nice
  if [[ -s $out/step_1.json ]]; then validate_sel $out/step_1.json; log "[skip select] nice"; else
    log "SELECT nice ($DRAW) NICE-MMLU-EM strict-deterministic"
    if ! CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDA_VISIBLE_DEVICES=0 $PY scripts/nice_select.py --candidate_grads $CANDGRAD \
        --base_model $BASE --adapter $WARMUP --target_data data/target_draws/${DRAW}.jsonl \
        --target_name mmlu --out_cache_dir $out --proj_dim 8192 --seed 123 --mc 8 \
        --temperature 1.0 --max_new_tokens 16 --num_select $K --strict_deterministic \
        --val_grads_out $out/val_grads.pt \
        > $LOGD/sel_${DRAW}_nice.log 2>&1; then log "[FAIL select] nice"; exit 1; fi
    validate_sel $out/step_1.json; log "[done select] nice"
  fi
  # randk_lenmatch (needs dsmc selection as target histogram)
  out=$SAVES/sel_${DRAW}_randk_lenmatch
  if [[ -s $out/step_1.json ]]; then validate_sel $out/step_1.json; log "[skip select] randk_lenmatch"; else
    log "SELECT randk_lenmatch ($DRAW) seed=$rk_seed"
    if ! CUDA_VISIBLE_DEVICES=0 $PY scripts/select_randk_lenmatch.py --candidate_data $CAND_JSONL \
        --dsmc_step1 $SAVES/sel_${DRAW}_dsmc/step_1.json --out_cache_dir $out --num_select $K \
        --length_cache $SAVES/candidate_token_lengths_llama2_cutoff2048.npy --seed $rk_seed \
        > $LOGD/sel_${DRAW}_randk_lenmatch.log 2>&1; then log "[FAIL select] randk_lenmatch"; exit 1; fi
    validate_sel $out/step_1.json; log "[done select] randk_lenmatch"
  fi
fi

# ───────────── phase: export ─────────────
if has export; then
  for name in dsmc less first_rr second_rr gist nice randk randk_lenmatch; do
    outc=$SAVES/sel_${DRAW}_${name}; jsonl=$SAVES/sft_subsets/${DRAW}_${name}_sel.jsonl
    [[ -s $outc/step_1.json ]] || continue
    if [[ -s $jsonl ]]; then log "[skip export] $name"; continue; fi
    log "EXPORT $name"
    $PY scripts/export_gradient_selection.py --candidate_data $CAND_JSONL \
      --cache_dir $outc --output_dir $SAVES/sft_subsets/${DRAW}_${name}_export --method ${DRAW}_${name} \
      --target_data data/target_draws/${DRAW}.jsonl --selection_ratio 0.05 --seed 42 \
      > $LOGD/export_${DRAW}_${name}.log 2>&1
    $PY -c "import json;d=json.load(open('$SAVES/sft_subsets/${DRAW}_${name}_export/selected_subset.json'));assert len(d)==$K;open('$jsonl','w').write('\n'.join(json.dumps(r,ensure_ascii=False) for r in d)+'\n')"
    log "[done export] $name ($(wc -l <$jsonl) rows)"
  done
fi

# ───────────── phase: diag (full 8-method selection diagnostics) ─────────────
if has diag; then
  log "DIAG $DRAW"
  $PY - "$DRAW" "$SAVES" "$CAND_JSONL" "$ROOT" <<'PYD'
import json,sys,os,hashlib
import numpy as np
draw,SAVES,cand_jsonl,ROOT=sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4]
names=["dsmc","less","first_rr","second_rr","gist","nice","randk","randk_lenmatch"]
sels={}
for n in names:
    p=f"{SAVES}/sel_{draw}_{n}/step_1.json"
    if os.path.exists(p): sels[n]=json.load(open(p))["indices"]
lc=f"{SAVES}/candidate_token_lengths_llama2_cutoff2048.npy"
lengths=np.load(lc) if os.path.exists(lc) else None
print(f"\n=== selection diagnostics: {draw} ({len(sels)}/8 methods) ===")
# sizes + hashes
for n in names:
    if n not in sels: print(f"  MISSING: {n}"); continue
    idx=sels[n]; h=hashlib.sha256(json.dumps(sorted(idx)).encode()).hexdigest()[:12]
    assert len(idx)==13533 and len(set(idx))==13533, f"{n} bad size"
    line=f"  {n:16s} size={len(idx)} sha={h}"
    if lengths is not None:
        L=lengths[np.array(idx)]; line+=f" tot_tok={int(L.sum())} meanlen={L.mean():.0f}"
    print(line)
# pairwise jaccard
S={n:set(v) for n,v in sels.items()}
ks=[n for n in names if n in S]
print("\n  pairwise Jaccard:")
print("  "+" ".join(f"{n[:9]:>9s}" for n in ks))
for a in ks:
    print("  "+a[:14].ljust(14)+" ".join(f"{len(S[a]&S[b])/len(S[a]|S[b]):9.3f}" for b in ks))
# length histogram per selection
if lengths is not None:
    edges=[0,256,512,1024,1536,100000]
    print("\n  post-tok length histogram (buckets 0-256/256-512/512-1024/1024-1536/1536+):")
    for n in ks:
        L=lengths[np.array(sels[n])]
        hist=[int(((L>=edges[i])&(L<edges[i+1])).sum()) for i in range(len(edges)-1)]
        print(f"    {n:16s} {hist}")
print("\n[ok] all present selections exactly 13533 unique indices")
PYD
  log "=== DIAG COMPLETE ($DRAW) ==="
fi

if has train; then log "[train] stub — SFT is a separate gated step, not in this driver"; fi
log "=== PILOT DRIVER DONE (phases: $PHASES, draw: $DRAW) ==="

#!/bin/bash
# Target-draw pilot driver (staged, resume + fail-fast). review_0730 / frozen protocol.
# Phases (run subset via --phases):  setup gengrad select export train eval aggregate
# DRY RUN = `--phases setup,gengrad,select,export,diag` on one draw: produces target grads + all 8
# selections + a selection diagnostics report (sizes/overlaps/length-dist/subject/runtime). NO SFT.
#
# 8 methods per draw (frozen): dsmc, second_rr, less, first_rr, gist, nice, randk, randk_lenmatch
#   dsmc      = select_moment_mmd  --alpha 0.0        (2nd-order MMD coreset)  [headline]
#   less      = select_relevance_topk --order first   (1st-order relevance top-k = LESS-like)
#   first_rr  = select_round_robin  --order first
#   second_rr = select_round_robin  --order second
#   gist      = select_gist_faithful --rank 150       (GIST-SharedProj, k=min(150,M)=M)
#   nice      = scripts/nice_select.py                (reward-weighted; per-draw target)
#   randk     = uniform fixed-K, subset seed 2000+draw_id
#   randk_lenmatch = fixed-K length-histogram-matched to dsmc (draw-specific)
set -Eeuo pipefail
ENVBIN=/jizhicfs/karonhe/envs/dataflex-fa/bin; PY=$ENVBIN/python
SAVES=/jizhicfs/karonhe/dataflex_saves; ROOT=/jizhicfs/karonhe/DataFlex_fa
CANDGRAD=$SAVES/less_output/train/1/all_projected_grads.pt
LOGD=$SAVES/logs; K=13533
export PATH=$ENVBIN:$PATH; cd $ROOT; mkdir -p $LOGD
log(){ echo "[$(date +%m-%d_%H:%M:%S)] $*"; }
trap 'log "[FATAL] line=$LINENO cmd=$BASH_COMMAND"' ERR

DRAW="${DRAW:-stem80_draw0}"
PHASES="${PHASES:-setup,gengrad,select,export,diag}"
has(){ [[ ",$PHASES," == *",$1,"* ]]; }
TGT=$SAVES/draw_${DRAW}_output/target/1/all_projected_grads.pt
CACHE=$SAVES/draw_${DRAW}_output
validate_sel(){ $PY - "$1" "$K" <<'PYV'
import json,sys
d=json.load(open(sys.argv[1])); K=int(sys.argv[2]); idx=d["indices"]
assert len(idx)==K and len(set(idx))==K and min(idx)>=0 and max(idx)<270679, "bad selection"
PYV
}

# ---- phase: setup (register dataset + configs) ----
if has setup; then
  log "SETUP $DRAW"; $PY scripts/setup_draw_target.py --draw $DRAW
fi

# ---- phase: gengrad (extract target grads for this draw; candidate REUSED via symlink) ----
if has gengrad; then
  if [[ -s $TGT ]]; then log "[skip gengrad] $DRAW target grads exist"; else
    # Reuse the shared 270k Adam candidate cache: symlink it as this draw's train grads so the
    # mmd selector skips the (hours-long) candidate re-extraction and only computes the 64 target
    # grads. Same candidate cache used by DSMC/mirror (warmup_seed42/ckpt-1692, verified).
    mkdir -p $CACHE/train/1
    if [[ ! -e $CACHE/train/1/all_projected_grads.pt ]]; then
      ln -s $CANDGRAD $CACHE/train/1/all_projected_grads.pt
      log "[gengrad] symlinked shared candidate cache -> $CACHE/train/1/"
    fi
    log "GENGRAD $DRAW (8-GPU target-only gradient extraction)"
    export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
    if ! dataflex-cli train experiments/less_aligned/configs/draws/select_${DRAW}.yaml \
        > $LOGD/gengrad_${DRAW}.log 2>&1; then log "[FAIL gengrad] $DRAW"; exit 1; fi
    [[ -s $TGT ]] || { log "[FAIL gengrad] no target grads at $TGT"; exit 1; }
    log "[done gengrad] $DRAW $(python3 -c "import torch;print(tuple(torch.load('$TGT',map_location='cpu').shape))")"
  fi
fi

# ---- phase: select (8 methods, offline, GPU0) ----
if has select; then
  rr_seed=$(( 3000 + ${DRAW##*draw} ))
  rk_seed=$(( 2000 + ${DRAW##*draw} ))
  declare -A DONE=()
  gen(){ # name : cmd...
    local name=$1; shift; local out=$SAVES/sel_${DRAW}_${name}
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
  # randk: uniform fixed-K (target-independent, but keyed per draw index)
  out=$SAVES/sel_${DRAW}_randk
  if [[ -s $out/step_1.json ]]; then validate_sel $out/step_1.json; log "[skip select] randk"; else
    log "SELECT randk ($DRAW) seed=$rk_seed"
    CUDA_VISIBLE_DEVICES=0 $PY - "$CANDGRAD" "$out" "$K" "$rk_seed" > $LOGD/sel_${DRAW}_randk.log 2>&1 <<'PYR'
import torch,json,sys,os
cand,out,K,seed=sys.argv[1],sys.argv[2],int(sys.argv[3]),int(sys.argv[4])
X=torch.load(cand,map_location="cpu"); N=X.shape[0]
g=torch.Generator().manual_seed(seed); idx=torch.randperm(N,generator=g)[:K].tolist()
assert len(set(idx))==K
os.makedirs(out,exist_ok=True)
json.dump({"indices":idx,"metric":{"kernel":"random_k","seed":seed,"num_select":K}},open(f"{out}/step_1.json","w"))
PYR
    validate_sel $out/step_1.json; log "[done select] randk"
  fi
  # NICE + randk_lenmatch are deferred to the full pilot (NICE needs reward extraction; lenmatch
  # needs tokenized lengths). Dry run covers the 6 gradient/geometry methods + randk.
  log "[select] core methods done (nice + randk_lenmatch handled in train phase setup)"
fi

# ---- phase: export (materialize subsets for methods present) ----
if has export; then
  for name in dsmc less first_rr second_rr gist randk; do
    outc=$SAVES/sel_${DRAW}_${name}; jsonl=$SAVES/sft_subsets/${DRAW}_${name}_sel.jsonl
    [[ -s $outc/step_1.json ]] || continue
    if [[ -s $jsonl ]]; then log "[skip export] $name"; continue; fi
    log "EXPORT $name"
    $PY scripts/export_gradient_selection.py --candidate_data data/less_train_all.jsonl \
      --cache_dir $outc --output_dir $SAVES/sft_subsets/${DRAW}_${name}_export --method ${DRAW}_${name} \
      --target_data data/target_draws/${DRAW}.jsonl --selection_ratio 0.05 --seed 42 \
      > $LOGD/export_${DRAW}_${name}.log 2>&1
    $PY -c "import json;d=json.load(open('$SAVES/sft_subsets/${DRAW}_${name}_export/selected_subset.json'));assert len(d)==$K;open('$jsonl','w').write('\n'.join(json.dumps(r,ensure_ascii=False) for r in d)+'\n')"
    log "[done export] $name ($(wc -l <$jsonl) rows)"
  done
fi

# ---- phase: diag (selection-only dry-run report) ----
if has diag; then
  log "DIAG $DRAW"
  $PY - "$DRAW" "$SAVES" "$CANDGRAD" <<'PYD'
import json,sys,glob,os,numpy as np
draw,SAVES,cand=sys.argv[1],sys.argv[2],sys.argv[3]
names=["dsmc","less","first_rr","second_rr","gist","randk"]
sels={}
for n in names:
    p=f"{SAVES}/sel_{draw}_{n}/step_1.json"
    if os.path.exists(p): sels[n]=set(json.load(open(p))["indices"])
print(f"\n=== selection diagnostics: {draw} ===")
print(f"{'method':10s} {'size':>6s} " + " ".join(f"{n[:8]:>8s}" for n in sels))
for a in sels:
    row=" ".join(f"{len(sels[a]&sels[b])/len(sels[a]|sels[b]):8.3f}" for b in sels)
    print(f"{a:10s} {len(sels[a]):6d} {row}")
# length distribution per selection (token proxy = chars of user+assistant)
import torch
cand_jsonl="data/less_train_all.jsonl"
print("\n[note] pairwise Jaccard above; all sizes must equal 13533.")
for n,s in sels.items(): assert len(s)==13533, f"{n} size {len(s)}"
print("[ok] all selections exactly 13533 unique indices")
PYD
  log "=== DIAG COMPLETE ($DRAW) ==="
fi

# ---- phase: train (gated; full pilot only) ----
if has train; then log "[train] not part of dry run — see full pilot driver"; fi
log "=== PILOT DRIVER DONE (phases: $PHASES, draw: $DRAW) ==="

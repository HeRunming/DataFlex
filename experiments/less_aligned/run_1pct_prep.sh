#!/bin/bash
# 1% budget pre-SFT prep (decision_0806 / review_0806). Separate namespace from 5%.
# Assumes derive_1pct_from_5pct.py already wrote sel1pct_<draw>_<method>/ for the 7 prefixable
# methods (dsmc,less,first_rr,second_rr,gist,nice,randk). This script:
#   1. builds randk_lenmatch at K=2707 (matched to each draw's 1% DSMC length histogram)
#   2. exports all 80 1% subsets -> <draw>_<method>_1pct_sel.jsonl
#   3. builds the 1% run plan (tag pilot1pct, budget 2707) + validates counts
# NO SFT here. Run once; idempotent.
set -Eeuo pipefail
ENVBIN=/jizhicfs/karonhe/envs/dataflex-fa/bin; PY=$ENVBIN/python
SAVES=/jizhicfs/karonhe/dataflex_saves; ROOT=/jizhicfs/karonhe/DataFlex_fa
CAND_JSONL=$ROOT/data/less_train_all.jsonl; LOGD=$SAVES/logs; K1=2707
export PATH=$ENVBIN:$PATH; cd $ROOT; mkdir -p $LOGD
log(){ echo "[$(date +%m-%d_%H:%M:%S)] $*"; }
trap 'log "[FATAL] line=$LINENO cmd=$BASH_COMMAND"' ERR
DRAWS="stem80_draw0 stem80_draw1 stem80_draw2 stem80_draw3 stem80_draw4 hum80_draw0 hum80_draw1 hum80_draw2 hum80_draw3 hum80_draw4"
PREFIX_METHODS="dsmc less first_rr second_rr gist nice randk"

# 1. randk_lenmatch @ K=2707 (matched to 1% DSMC subset per draw)
for d in $DRAWS; do
  out=$SAVES/sel1pct_${d}_randk_lenmatch; rk_seed=$(( 2000 + ${d##*draw} ))
  [[ -s $out/step_1.json ]] && { log "[skip lenmatch] $d"; continue; }
  log "LENMATCH-1pct $d (K=$K1 seed=$rk_seed)"
  CUDA_VISIBLE_DEVICES=0 $PY scripts/select_randk_lenmatch.py --candidate_data $CAND_JSONL \
    --dsmc_step1 $SAVES/sel1pct_${d}_dsmc/step_1.json --out_cache_dir $out --num_select $K1 \
    --length_cache $SAVES/candidate_token_lengths_llama2_cutoff2048.npy --seed $rk_seed \
    > $LOGD/sel1pct_${d}_randk_lenmatch.log 2>&1
done

# 2. export all 80 1% subsets
for d in $DRAWS; do
  for m in $PREFIX_METHODS randk_lenmatch; do
    outc=$SAVES/sel1pct_${d}_${m}; jsonl=$SAVES/sft_subsets/${d}_${m}_1pct_sel.jsonl
    [[ -s $jsonl ]] && continue
    $PY scripts/export_gradient_selection.py --candidate_data $CAND_JSONL \
      --cache_dir $outc --output_dir $SAVES/sft_subsets/${d}_${m}_1pct_export --method ${d}_${m}_1pct \
      --target_data data/target_draws/${d}.jsonl --selection_ratio 0.01 --seed 42 \
      > $LOGD/export1pct_${d}_${m}.log 2>&1
    $PY -c "import json;dd=json.load(open('$SAVES/sft_subsets/${d}_${m}_1pct_export/selected_subset.json'));assert len(dd)==$K1,len(dd);open('$jsonl','w').write('\n'.join(json.dumps(r,ensure_ascii=False) for r in dd)+'\n')"
  done
  log "[exported 1pct subsets] $d"
done

# 3. build 1% run plan (tag pilot1pct, budget 2707)
$PY scripts/build_pilot_run_plan.py --tag pilot1pct --budget $K1 \
  --out $ROOT/experiments/less_aligned/pilot1pct_run_plan.json \
  --subset_tmpl "$SAVES/sft_subsets/{draw}_{m}_1pct_sel.jsonl" \
  --randk_subset_tmpl "$SAVES/sft_subsets/{draw}_randk_1pct_sel.jsonl"
log "=== 1pct PRE-SFT PREP DONE ==="

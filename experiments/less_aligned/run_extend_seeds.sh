#!/bin/bash
###############################################################################
# Master driver to extend the multi-seed table to n=5 (seeds 2,3,4).
# For each seed: selections (run_seed_selections.sh) -> SFT matrix -> eval matrix.
# Fully idempotent; safe to re-launch. Sequential per seed.
###############################################################################
set -uo pipefail
ROOT=/jizhicfs/karonhe/DataFlex_fa
D=$ROOT/experiments/less_aligned
LOGD=/jizhicfs/karonhe/dataflex_saves/logs
cd $ROOT
NEW_SEEDS=(${NEW_SEEDS:-2 3 4})
for s in "${NEW_SEEDS[@]}"; do
  echo "[$(date +%m-%d_%H:%M:%S)] === SEED $s: selections ==="
  SEED=$s bash $D/run_seed_selections.sh
  echo "[$(date +%m-%d_%H:%M:%S)] === SEED $s: SFT matrix ==="
  SEEDS="$s" bash $D/run_sft_matrix.sh > $LOGD/sft_matrix_seed${s}.log 2>&1
  echo "[$(date +%m-%d_%H:%M:%S)] === SEED $s: eval matrix ==="
  SEEDS="$s" bash $D/run_eval_matrix.sh > $LOGD/eval_matrix_seed${s}.log 2>&1
  echo "[$(date +%m-%d_%H:%M:%S)] === SEED $s DONE ==="
done
echo "[$(date +%m-%d_%H:%M:%S)] === ALL NEW SEEDS COMPLETE — run aggregate_multiseed.py --seeds 42 1 2 3 4 ==="

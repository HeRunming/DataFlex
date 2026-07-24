#!/bin/bash
###############################################################################
# Multi-seed SFT matrix driver (idempotent). For every (method, target, seed):
#   1. locate the selection step_1.json in its cache dir
#   2. export -> subset jsonl, register in dataset_info.json (once per selection)
#   3. 8-GPU DDP SFT (eff batch 128, 4ep, LoRA r128 a512, seed={s})
#        -> sft_results/{method}_{target}_seed{s}
# Skips any run whose adapter_model.safetensors already exists.
#
# Selection cache dir conventions:
#   gradient : {method}_{target}_seed{s}_output          (per seed)
#   nice     : nice_{target}_seed{s}_output              (per seed)
#   emb/tsds : {method}_{target}_output                  (seed-independent, shared)
###############################################################################
set -uo pipefail
ROOT=/jizhicfs/karonhe/DataFlex_fa
ENVBIN=/jizhicfs/karonhe/envs/dataflex-fa/bin
SAVES=/jizhicfs/karonhe/dataflex_saves
SFT=$SAVES/sft_results
SUB=$SAVES/sft_subsets
LOGD=$SAVES/logs
CAND=$ROOT/data/less_train_all.jsonl
PY=$ENVBIN/python
export PATH=$ENVBIN:$PATH
mkdir -p $SFT $SUB $LOGD
cd $ROOT

SEEDS=(${SEEDS:-42 1 2})
GRAD=(less_sgd less_adam mmd_grad_rbf_sgd mmd_grad_rbf_adam mmd_grad_cov_sgd mmd_grad_cov_adam)
EMB=(mmd_emb_rbf mmd_emb_rbf_stochastic tsds)
TARGETS=(bbh mmlu tydiqa)
declare -A TGTDS=( [bbh]=bbh_target_100 [mmlu]=mmlu_target [tydiqa]=tydiqa_target )
log(){ echo "[$(date +%m-%d_%H:%M:%S)] $*"; }

# selection cache dir for (method,target,seed)
cache_dir(){ local m=$1 t=$2 s=$3
  if [[ " ${EMB[*]} " == *" $m "* ]]; then echo "$SAVES/${m}_${t}_output"
  elif [ "$m" = nice ]; then echo "$SAVES/nice_${t}_seed${s}_output"
  else echo "$SAVES/${m}_${t}_seed${s}_output"; fi; }

# dataset key + jsonl for (method,target,seed). emb/tsds subset shared across seeds.
register_subset(){ local m=$1 t=$2 s=$3 cdir=$4
  local key jsonl
  if [[ " ${EMB[*]} " == *" $m "* ]]; then key="${m}_${t}_sel"; jsonl="$SUB/${m}_${t}_selected.jsonl"
  else key="${m}_${t}_seed${s}_sel"; jsonl="$SUB/${m}_${t}_seed${s}_selected.jsonl"; fi
  if [ ! -f "$jsonl" ]; then
    # all diagnostics to stderr so stdout carries ONLY the key
    $PY scripts/export_gradient_selection.py --candidate_data $CAND --cache_dir "$cdir" \
      --output_dir "$SUB/${key}_export" --method "$key" --target_data data/${TGTDS[$t]}.jsonl \
      --selection_ratio 0.05 --seed 42 > $LOGD/export_${key}.log 2>&1
    $PY -c "
import json,sys
d=json.load(open('$SUB/${key}_export/selected_subset.json'))
open('$jsonl','w').write('\n'.join(json.dumps(r,ensure_ascii=False) for r in d)+'\n')
info=json.load(open('data/dataset_info.json'))
info['$key']={'file_name':'$jsonl','formatting':'sharegpt','columns':{'messages':'messages'},
  'tags':{'role_tag':'role','content_tag':'content','user_tag':'user','assistant_tag':'assistant'}}
json.dump(info,open('data/dataset_info.json','w'),indent=2,ensure_ascii=False)
print('registered $key', len(d), file=sys.stderr)
" 1>&2
  fi
  printf '%s' "$key"
}

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
ALL=("${GRAD[@]}" "${EMB[@]}" nice)
for s in "${SEEDS[@]}"; do
  for t in "${TARGETS[@]}"; do
    for m in "${ALL[@]}"; do
      out=$SFT/${m}_${t}_seed${s}
      if [ -f "$out/adapter_model.safetensors" ]; then log "[skip sft] ${m}_${t}_seed${s}"; continue; fi
      cdir=$(cache_dir $m $t $s)
      if [ ! -f "$cdir/step_1.json" ]; then log "[MISSING selection] $cdir — skip ${m}_${t}_seed${s}"; continue; fi
      key=$(register_subset $m $t $s "$cdir")
      log "SFT ${m}_${t}_seed${s} (dataset=$key)"
      dataflex-cli train experiments/less_aligned/configs/train_llama7b_lora.yaml \
        dataset="$key" output_dir="$out" seed=$s \
        per_device_train_batch_size=4 gradient_accumulation_steps=4 \
        lora_alpha=512 num_train_epochs=4 \
        > $LOGD/sft_${m}_${t}_seed${s}.log 2>&1
      [ -f "$out/adapter_model.safetensors" ] && log "[done] ${m}_${t}_seed${s}" || log "[FAIL] ${m}_${t}_seed${s}"
    done
  done
done
log "=== SFT MATRIX COMPLETE ==="

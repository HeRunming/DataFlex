#!/bin/bash
# Target-draw SFT/eval/aggregate driver (staged, resume + fail-fast). advice_0731 / code_review_0801.
# Budget-agnostic: PLAN env selects the run plan (default 5%; set PLAN=...pilot1pct_run_plan.json for
# 1%). All budget checks derive K from the plan — nothing budget-specific is hardcoded here.
# Phases via PHASES env: register train eval aggregate. Optional ADAPTERS="id1 id2" restricts
# TRAIN+EVAL to specific adapter_ids (canary). NOTE: register ALWAYS processes ALL adapters in the
# plan (cheap, idempotent) regardless of ADAPTERS — datasets are shared config, not per-canary state.
#   register : hash-validate (rows + SHA256 vs plan, at the plan's budget) + register dataset keys
#   train    : SFT each unique adapter once (seed from plan, eff-batch 128, 4 epochs) + train_manifest;
#              resume skips only a HASH-VALIDATED matching adapter (not mere file existence)
#   eval     : lm_eval mmlu_stem+humanities into a UNIQUE run subdir; require exactly 1 results file;
#              write eval_manifest; resume skips only a hash-validated matching eval
#   aggregate: expand adapters -> cells; DSMC-method paired diff; 51/64,13/64 weights
set -Eeuo pipefail
ENVBIN=/jizhicfs/karonhe/envs/dataflex-fa/bin; PY=$ENVBIN/python
SAVES=/jizhicfs/karonhe/dataflex_saves; ROOT=/jizhicfs/karonhe/DataFlex_fa
BASE=/jizhicfs/karonhe/models/shakechen/Llama-2-7b-hf
LOGD=$SAVES/logs; PLAN="${PLAN:-$ROOT/experiments/less_aligned/pilot_run_plan.json}"
# MASTER = shared, budget-INDEPENDENT target geometry (draws, target grads, ckpt, cache, projection,
# env). The budget-DEPENDENT selection/subset hashes live in pilot{5,1}pct_selection_manifest.json.
MASTER=$ROOT/experiments/less_aligned/targetdraw_10draw_master_manifest.json
PROV="$PY scripts/pilot_provenance.py"
export PATH=$ENVBIN:$PATH; cd $ROOT; mkdir -p $LOGD $SAVES/eval_results/skew
log(){ echo "[$(date +%m-%d_%H:%M:%S)] $*"; }
trap 'log "[FATAL] line=$LINENO cmd=$BASH_COMMAND"' ERR
PHASES="${PHASES:-}"; ADAPTERS="${ADAPTERS:-}"
[[ -z "$PHASES" ]] && { log "[FATAL] PHASES empty (register,train,eval,aggregate)"; exit 2; }
for p in ${PHASES//,/ }; do case "$p" in register|train|eval|aggregate) ;; *) log "[FATAL] unknown phase $p"; exit 2;; esac; done
has(){ [[ ",$PHASES," == *",$1,"* ]]; }
[[ -f $PLAN ]] || { log "[FATAL] no run plan $PLAN (run build_pilot_run_plan.py)"; exit 2; }
want(){ [[ -z "$ADAPTERS" ]] || [[ " $ADAPTERS " == *" $1 "* ]]; }
mapfile -t AIDS < <($PY -c "import json;print('\n'.join(json.load(open('$PLAN'))['adapters']))")

# ───────────── register (all adapters in plan; rows+SHA hash-validated at the plan budget) ─────────────
if has register; then
  $PROV register --plan $PLAN
  log "register done"
fi

# ───────────── train ─────────────
if has train; then
  export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
  for aid in "${AIDS[@]}"; do
    want "$aid" || continue
    read -r dkey seed out <<< "$($PY -c "import json;p=json.load(open('$PLAN'));a=p['adapters']['$aid'];c=[c for c in p['cells'] if c['adapter_id']=='$aid'][0];print(a['dataset_key'],a['train_seed'],c.get('sft_out','$SAVES/sft_results/pilot_$aid'))")"
    if $PROV check_train --plan $PLAN --aid $aid --adapter_dir $out; then log "[skip train, validated] $aid"; continue; fi
    log "TRAIN $aid (dataset=$dkey seed=$seed)"
    if ! dataflex-cli train experiments/less_aligned/configs/train_llama7b_lora.yaml \
        dataset=$dkey output_dir=$out seed=$seed \
        per_device_train_batch_size=4 gradient_accumulation_steps=4 lora_alpha=512 num_train_epochs=4 \
        > $LOGD/${aid}_train.log 2>&1; then log "[FAIL train] $aid"; exit 1; fi
    [[ -f $out/adapter_model.safetensors ]] || { log "[FAIL train] $aid no adapter"; exit 1; }
    $PROV write_train_manifest --plan $PLAN --aid $aid --adapter_dir $out --master $MASTER --base_model $BASE
    log "[done train] $aid"
  done
fi

# ───────────── eval ─────────────
if has eval; then
  port=29720
  for aid in "${AIDS[@]}"; do
    want "$aid" || { port=$((port+1)); continue; }
    read -r ad base <<< "$($PY -c "import json;c=[c for c in json.load(open('$PLAN'))['cells'] if c['adapter_id']=='$aid'][0];print(c.get('sft_out','$SAVES/sft_results/pilot_$aid'),c.get('eval_out','$SAVES/eval_results/skew/pilot_$aid'))")"
    [[ -f $ad/adapter_model.safetensors ]] || { log "[no adapter] $aid"; exit 1; }
    if $PROV check_eval --eval_dir $base --adapter_dir $ad; then log "[skip eval, validated] $aid"; port=$((port+1)); continue; fi
    # fresh unique run subdir so results are never mixed across runs
    run="$base/run_$(date +%s)_$port"; mkdir -p "$run"
    log "EVAL $aid -> $run"
    # HF_*_OFFLINE: the mmlu dataset is already in the local HF cache; without this, 8 ranks each
    # hit the Hub API and can get HTTP 429 rate-limited (observed in the 1% canary). Infra-only fix.
    if ! HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
      NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 http_proxy="http://hy-proxy.woa.com:3128" https_proxy="http://hy-proxy.woa.com:3128" \
      no_proxy=".woa.com,localhost,127.0.0.1,mirrors.tencent.com" \
      accelerate launch --num_processes 8 --main_process_port $port -m lm_eval --model hf \
        --model_args "pretrained=$BASE,peft=$ad,dtype=bfloat16,trust_remote_code=True" \
        --tasks mmlu_stem,mmlu_humanities --num_fewshot 5 --batch_size 16 --output_path "$run" \
        > $LOGD/${aid}_eval.log 2>&1; then log "[FAIL eval] $aid"; exit 1; fi
    # require exactly one authoritative results file, then pin it in eval_manifest
    $PROV find_eval --eval_dir "$run" >/dev/null || { log "[FAIL eval] $aid not exactly 1 results file"; exit 1; }
    $PROV write_eval_manifest --aid $aid --adapter_dir $ad --eval_dir "$base"
    port=$((port+1)); log "[done eval] $aid"
  done
fi

# ───────────── aggregate ─────────────
if has aggregate; then
  $PY scripts/aggregate_pilot.py --plan $PLAN --saves $SAVES ${AGG_ALLOW_PARTIAL:+--allow-partial}
  log "aggregate done"
fi
log "=== PILOT SFT DRIVER DONE (phases: $PHASES) ==="

#!/bin/bash
# Target-draw pilot SFT/eval/aggregate driver (staged, resume + fail-fast). advice_0731.
# Consumes experiments/less_aligned/pilot_run_plan.json (32 cells -> 30 unique adapters).
# Phases via PHASES env: register train eval aggregate. Optional ADAPTERS="id1 id2" to restrict
# to specific adapter_ids (used for the 2-adapter canary). NO methodological knobs here.
#
#   register : register each unique adapter's subset jsonl as a llamafactory dataset key (validates
#              subset hash + 13533 rows)
#   train    : SFT each unique adapter once (seed from plan, eff-batch 128, 4 epochs). 8-GPU.
#   eval     : lm_eval mmlu_stem+mmlu_humanities per unique adapter (one authoritative results file)
#   aggregate: expand 30 adapters -> 32 cells, compute balanced + target-weighted, paired Δ vs DSMC
set -Eeuo pipefail
ENVBIN=/jizhicfs/karonhe/envs/dataflex-fa/bin; PY=$ENVBIN/python
SAVES=/jizhicfs/karonhe/dataflex_saves; ROOT=/jizhicfs/karonhe/DataFlex_fa
BASE=/jizhicfs/karonhe/models/shakechen/Llama-2-7b-hf
LOGD=$SAVES/logs; PLAN=$ROOT/experiments/less_aligned/pilot_run_plan.json; K=13533
export PATH=$ENVBIN:$PATH; cd $ROOT; mkdir -p $LOGD $SAVES/eval_results/skew
log(){ echo "[$(date +%m-%d_%H:%M:%S)] $*"; }
trap 'log "[FATAL] line=$LINENO cmd=$BASH_COMMAND"' ERR
PHASES="${PHASES:-}"; ADAPTERS="${ADAPTERS:-}"
[[ -z "$PHASES" ]] && { log "[FATAL] PHASES empty (register,train,eval,aggregate)"; exit 2; }
for p in ${PHASES//,/ }; do case "$p" in register|train|eval|aggregate) ;; *) log "[FATAL] unknown phase $p"; exit 2;; esac; done
has(){ [[ ",$PHASES," == *",$1,"* ]]; }
[[ -f $PLAN ]] || { log "[FATAL] no run plan $PLAN (run build_pilot_run_plan.py)"; exit 2; }
# adapter filter helper
want(){ [[ -z "$ADAPTERS" ]] || [[ " $ADAPTERS " == *" $1 "* ]]; }

# list unique adapter ids
mapfile -t AIDS < <($PY -c "import json;print('\n'.join(json.load(open('$PLAN'))['adapters']))")

# ───────────── register ─────────────
if has register; then
  $PY - "$PLAN" "$K" <<'PYEOF'
import json,sys,os
plan=json.load(open(sys.argv[1])); K=int(sys.argv[2])
info=json.load(open("data/dataset_info.json"))
for aid,a in plan["adapters"].items():
    j=a["subset_jsonl"]
    n=sum(1 for _ in open(j)); assert n==K, f"{aid}: {j} has {n} != {K}"
    info[a["dataset_key"]]={"file_name":j,"formatting":"sharegpt","columns":{"messages":"messages"},
        "tags":{"role_tag":"role","content_tag":"content","user_tag":"user","assistant_tag":"assistant"}}
json.dump(info,open("data/dataset_info.json","w"),indent=2,ensure_ascii=False)
print(f"registered {len(plan['adapters'])} unique adapter datasets")
PYEOF
  log "register done"
fi

# ───────────── train ─────────────
if has train; then
  export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
  for aid in "${AIDS[@]}"; do
    want "$aid" || continue
    read -r dkey seed <<< "$($PY -c "import json;a=json.load(open('$PLAN'))['adapters']['$aid'];print(a['dataset_key'],a['train_seed'])")"
    out=$SAVES/sft_results/pilot_${aid}
    [[ -f $out/adapter_model.safetensors ]] && { log "[skip train] $aid"; continue; }
    log "TRAIN $aid (dataset=$dkey seed=$seed)"
    if ! dataflex-cli train experiments/less_aligned/configs/train_llama7b_lora.yaml \
        dataset=$dkey output_dir=$out seed=$seed \
        per_device_train_batch_size=4 gradient_accumulation_steps=4 lora_alpha=512 num_train_epochs=4 \
        > $LOGD/pilot_train_${aid}.log 2>&1; then log "[FAIL train] $aid"; exit 1; fi
    [[ -f $out/adapter_model.safetensors ]] || { log "[FAIL train] $aid no adapter"; exit 1; }
    log "[done train] $aid"
  done
fi

# ───────────── eval ─────────────
if has eval; then
  port=29720
  for aid in "${AIDS[@]}"; do
    want "$aid" || { port=$((port+1)); continue; }
    ad=$SAVES/sft_results/pilot_${aid}; out=$SAVES/eval_results/skew/pilot_${aid}
    [[ -f $ad/adapter_model.safetensors ]] || { log "[no adapter] $aid"; exit 1; }
    find "$out" -name "results_*.json" 2>/dev/null|grep -q . && { log "[skip eval] $aid"; port=$((port+1)); continue; }
    log "EVAL $aid"
    if ! NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 http_proxy="http://hy-proxy.woa.com:3128" https_proxy="http://hy-proxy.woa.com:3128" \
      no_proxy=".woa.com,localhost,127.0.0.1,mirrors.tencent.com" \
      accelerate launch --num_processes 8 --main_process_port $port -m lm_eval --model hf \
        --model_args "pretrained=$BASE,peft=$ad,dtype=bfloat16,trust_remote_code=True" \
        --tasks mmlu_stem,mmlu_humanities --num_fewshot 5 --batch_size 16 --output_path "$out" \
        > $LOGD/pilot_eval_${aid}.log 2>&1; then log "[FAIL eval] $aid"; exit 1; fi
    find "$out" -name "results_*.json" 2>/dev/null|grep -q . || { log "[FAIL eval] $aid no results"; exit 1; }
    port=$((port+1)); log "[done eval] $aid"
  done
fi

# ───────────── aggregate ─────────────
if has aggregate; then
  $PY scripts/aggregate_pilot.py --plan $PLAN --saves $SAVES
  log "aggregate done"
fi
log "=== PILOT SFT DRIVER DONE (phases: $PHASES) ==="

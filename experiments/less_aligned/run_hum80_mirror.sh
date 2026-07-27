#!/bin/bash
# review_0727.md: slim T_hum80 mirror for skew-direction invariance. Three points, SAME offline
# code paths as the stem80 experiment (so GradCov-vs-joint is not confounded by selector):
#   gradcov = select_moment_mmd.py    --alpha 0.0   (pure 2nd-order, == select_moment_lambda --lam 0)
#   joint   = select_moment_lambda.py --lam   0.02  (best interior joint)
#   linear  = select_moment_mmd.py    --alpha 1.0   (1st-order endpoint)
# Paired seeds {42,1,2}: fixed selected subset per point, vary only the SFT seed.
# NO further lambda sweeps (frozen per review). Candidate pool = same 270k seed-42 grad cache.
set -uo pipefail
ENVBIN=/jizhicfs/karonhe/envs/dataflex-fa/bin; PY=$ENVBIN/python
SAVES=/jizhicfs/karonhe/dataflex_saves
SFT=$SAVES/sft_results; EVAL=$SAVES/eval_results; LOGD=$SAVES/logs; SUB=$SAVES/sft_subsets
BASE=/jizhicfs/karonhe/models/shakechen/Llama-2-7b-hf
CAND=/jizhicfs/karonhe/DataFlex_fa/data/less_train_all.jsonl
CANDGRAD=$SAVES/less_output/train/1/all_projected_grads.pt
TGT=$SAVES/mmd_grad_cov_adam_hum80_output/target/1/all_projected_grads.pt
export PATH=$ENVBIN:$PATH; cd /jizhicfs/karonhe/DataFlex_fa
mkdir -p $EVAL/skew $LOGD $SUB
log(){ echo "[$(date +%m-%d_%H:%M:%S)] $*"; }
SEEDS=(42 1 2)

# ---- 1. selection (3 points, offline greedy on hum80 target) ----
gc=$SAVES/hmoment_gradcov_hum80_output
jo=$SAVES/hmoment_l0.02_hum80_output
li=$SAVES/hmoment_linear_hum80_output
[ -f $gc/step_1.json ] || { log "SELECT gradcov(α=0)"; CUDA_VISIBLE_DEVICES=0 $PY scripts/select_moment_mmd.py \
  --train_grads $CANDGRAD --target_grads $TGT --out_cache_dir $gc --num_select 13533 --alpha 0.0 \
  > $LOGD/sel_hmoment_gradcov.log 2>&1 && log "[done] gradcov"; }
[ -f $jo/step_1.json ] || { log "SELECT joint(λ=0.02)"; CUDA_VISIBLE_DEVICES=0 $PY scripts/select_moment_lambda.py \
  --train_grads $CANDGRAD --target_grads $TGT --out_cache_dir $jo --num_select 13533 --lam 0.02 \
  > $LOGD/sel_hmoment_l0.02.log 2>&1 && log "[done] joint"; }
[ -f $li/step_1.json ] || { log "SELECT linear(α=1)"; CUDA_VISIBLE_DEVICES=0 $PY scripts/select_moment_mmd.py \
  --train_grads $CANDGRAD --target_grads $TGT --out_cache_dir $li --num_select 13533 --alpha 1.0 \
  > $LOGD/sel_hmoment_linear.log 2>&1 && log "[done] linear"; }

# ---- 2. export + register subsets ----
$PY - <<'PYEOF'
import json,subprocess,os
SAVES="/jizhicfs/karonhe/dataflex_saves"; SUB=f"{SAVES}/sft_subsets"; CAND="data/less_train_all.jsonl"
info=json.load(open("data/dataset_info.json"))
def reg(cache,key,jsonl):
    if not os.path.exists(jsonl):
        subprocess.run([f"/jizhicfs/karonhe/envs/dataflex-fa/bin/python","scripts/export_gradient_selection.py",
          "--candidate_data",CAND,"--cache_dir",cache,"--output_dir",f"{SUB}/{key}_export","--method",key,
          "--target_data","data/mmlu_target_hum80.jsonl","--selection_ratio","0.05","--seed","42"],
          stdout=open(f"{SAVES}/logs/export_{key}.log","w"),stderr=subprocess.STDOUT)
        d=json.load(open(f"{SUB}/{key}_export/selected_subset.json"))
        open(jsonl,"w").write("\n".join(json.dumps(r,ensure_ascii=False) for r in d)+"\n")
    info[key]={"file_name":jsonl,"formatting":"sharegpt","columns":{"messages":"messages"},
               "tags":{"role_tag":"role","content_tag":"content","user_tag":"user","assistant_tag":"assistant"}}
reg(f"{SAVES}/hmoment_gradcov_hum80_output", "hmoment_gradcov_hum80_sel", f"{SUB}/hmoment_gradcov_hum80_sel.jsonl")
reg(f"{SAVES}/hmoment_l0.02_hum80_output",   "hmoment_l0.02_hum80_sel",   f"{SUB}/hmoment_l0.02_hum80_sel.jsonl")
reg(f"{SAVES}/hmoment_linear_hum80_output",  "hmoment_linear_hum80_sel",  f"{SUB}/hmoment_linear_hum80_sel.jsonl")
json.dump(info,open("data/dataset_info.json","w"),indent=2,ensure_ascii=False)
print("registered: gradcov / l0.02 / linear hum80")
PYEOF
log "registration done"

# ---- 3. SFT (3 points x 3 seeds) ----
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
declare -A DS=( [gradcov]=hmoment_gradcov_hum80_sel [joint002]=hmoment_l0.02_hum80_sel [linear]=hmoment_linear_hum80_sel )
for s in "${SEEDS[@]}"; do
  for name in gradcov joint002 linear; do
    out=$SFT/${name}_hum80_seed${s}
    [ -f $out/adapter_model.safetensors ] && { log "[skip sft] $name s=$s"; continue; }
    log "SFT $name seed=$s"
    dataflex-cli train experiments/less_aligned/configs/train_llama7b_lora.yaml \
      dataset=${DS[$name]} output_dir=$out seed=$s \
      per_device_train_batch_size=4 gradient_accumulation_steps=4 lora_alpha=512 num_train_epochs=4 \
      > $LOGD/sft_${name}_hum80_seed${s}.log 2>&1
    [ -f $out/adapter_model.safetensors ] && log "[done sft] $name s=$s" || log "[FAIL sft] $name s=$s"
  done
done

# ---- 4. eval (mmlu_stem + mmlu_humanities) ----
port=29690
for s in "${SEEDS[@]}"; do
  for name in gradcov joint002 linear; do
    ad=$SFT/${name}_hum80_seed${s}; out=$EVAL/skew/${name}_hum80_seed${s}
    [ -f $ad/adapter_model.safetensors ] || { log "[no adapter] $name s=$s"; continue; }
    find "$out" -name "results_*.json" 2>/dev/null|grep -q . && { log "[skip eval] $name s=$s"; continue; }
    log "EVAL $name seed=$s"
    NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 http_proxy="http://hy-proxy.woa.com:3128" https_proxy="http://hy-proxy.woa.com:3128" \
    no_proxy=".woa.com,localhost,127.0.0.1,mirrors.tencent.com" \
    accelerate launch --num_processes 8 --main_process_port $port -m lm_eval --model hf \
      --model_args "pretrained=$BASE,peft=$ad,dtype=bfloat16,trust_remote_code=True" \
      --tasks mmlu_stem,mmlu_humanities --num_fewshot 5 --batch_size 16 --output_path "$out" \
      > $LOGD/eval_${name}_hum80_seed${s}.log 2>&1
    port=$((port+1))
    find "$out" -name "results_*.json" 2>/dev/null|grep -q . && log "[done eval] $name s=$s" || log "[FAIL eval] $name s=$s"
  done
done
log "=== T_hum80 MIRROR COMPLETE ==="

#!/bin/bash
###############################################################################
# Per-seed selection pipeline (idempotent): warmup -> adam+sgd caches ->
# 18 gradient selections -> 3 NICE selections, for ONE seed.
# Usage: SEED=2 bash run_seed_selections.sh
# After this completes for a seed, run_sft_matrix.sh (with that seed) can SFT.
###############################################################################
set -uo pipefail
ROOT=/jizhicfs/karonhe/DataFlex_fa
ENVBIN=/jizhicfs/karonhe/envs/dataflex-fa/bin
SAVES=/jizhicfs/karonhe/dataflex_saves
LOGD=$SAVES/logs
BASE=/jizhicfs/karonhe/models/shakechen/Llama-2-7b-hf
PY=$ENVBIN/python
export PATH=$ENVBIN:$PATH
export http_proxy="" https_proxy=""   # no net needed for these
mkdir -p $LOGD; cd $ROOT
s=${SEED:?set SEED}
ALLGPU=0,1,2,3,4,5,6,7
wckpt=$SAVES/sft_results/warmup_seed${s}/checkpoint-1692
log(){ echo "[$(date +%m-%d_%H:%M:%S)][seed$s] $*"; }

# 1. warmup
if [ ! -f $wckpt/optimizer.pt ]; then
  log "warmup"; CUDA_VISIBLE_DEVICES=$ALLGPU dataflex-cli train /tmp/warmup_seed${s}.yaml > $LOGD/warmup_seed${s}.log 2>&1
fi
[ -f $wckpt/optimizer.pt ] || { log "WARMUP FAILED"; exit 1; }

# 2. adam + sgd candidate caches (dedicated per-seed components + select cfg)
$PY - "$s" <<'PYEOF'
import yaml,sys
s=sys.argv[1]; S="/jizhicfs/karonhe/dataflex_saves"; ckpt=f"{S}/sft_results/warmup_seed{s}/checkpoint-1692"
M="/jizhicfs/karonhe/models/shakechen/Llama-2-7b-hf"
comp={"selectors":{
  f"less_adam_cache_seed{s}":{"name":"less","params":{"cache_dir":f"{S}/less_output_seed{s}","gradient_type":"adam","proj_dim":8192,"seed":123,"save_interval":16}},
  f"less_sgd_cache_seed{s}":{"name":"less","params":{"cache_dir":f"{S}/less_sgd_output_seed{s}","gradient_type":"sgd","proj_dim":8192,"seed":123,"save_interval":16}}}}
yaml.safe_dump(comp,open(f"src/dataflex/configs/components_cache_seed{s}.yaml","w"),sort_keys=False)
base=dict(model_name_or_path=M,adapter_name_or_path=ckpt,trust_remote_code=True,stage="sft",do_train=True,
 finetuning_type="lora",lora_rank=128,lora_alpha=512,lora_target="q_proj,k_proj,v_proj,o_proj",lora_dropout=0.1,
 dataset="less_train_all",template="llama2",cutoff_len=2048,overwrite_cache=True,preprocessing_num_workers=16,
 logging_steps=10,overwrite_output_dir=True,save_steps=99999,report_to="none",per_device_train_batch_size=1,
 gradient_accumulation_steps=1,learning_rate=2.0e-5,num_train_epochs=1.0,bf16=True,ddp_timeout=180000000,train_step=2,
 train_type="dynamic_select",components_cfg_file=f"src/dataflex/configs/components_cache_seed{s}.yaml",
 warmup_step=1,update_step=1,update_times=1,selection_ratio=0.05,optimizer_state_path=ckpt,
 target_dataset="bbh_target_100",eval_dataset="bbh_target_100",eval_strategy="no")
for gt in ("adam","sgd"):
    c=dict(base); c["component_name"]=f"less_{gt}_cache_seed{s}"; c["output_dir"]=f"{S}/less_aligned/less_{gt}_cache_seed{s}"
    yaml.safe_dump(c,open(f"/tmp/cache_{gt}_seed{s}.yaml","w"),sort_keys=False)
print("cache configs written")
PYEOF
if [ ! -f $SAVES/less_output_seed${s}/train/1/all_projected_grads.pt ]; then
  log "adam cache"; CUDA_VISIBLE_DEVICES=$ALLGPU dataflex-cli train /tmp/cache_adam_seed${s}.yaml > $LOGD/cache_adam_seed${s}.log 2>&1
fi
if [ ! -f $SAVES/less_sgd_output_seed${s}/train/1/all_projected_grads.pt ]; then
  log "sgd cache"; CUDA_VISIBLE_DEVICES=$ALLGPU dataflex-cli train /tmp/cache_sgd_seed${s}.yaml > $LOGD/cache_sgd_seed${s}.log 2>&1
fi

# 3. gradient selections (18) via gen_multiseed_configs + batched run
$PY scripts/gen_multiseed_configs.py --seed $s > $LOGD/gen_seed${s}.log 2>&1
cfgs=(experiments/less_aligned/configs/multiseed/select_*_seed${s}.yaml)
gpus=(0 1 2 3 4 5 6 7); gi=0; declare -a P=()
for cfg in "${cfgs[@]}"; do
  comp=$(grep -oE "component_name: .*" "$cfg" | awk '{print $2}')
  [ -f $SAVES/${comp}_output/step_1.json ] && { log "[skip sel] $comp"; continue; }
  CUDA_VISIBLE_DEVICES=${gpus[$gi]} dataflex-cli train "$cfg" > $LOGD/sel_${comp}.log 2>&1 &
  P+=($!); gi=$(((gi+1)%8)); [ ${#P[@]} -ge 8 ] && { for p in "${P[@]}"; do wait "$p"; done; P=(); }
done
for p in "${P[@]}"; do wait "$p"; done
log "gradient selections done: $(ls $SAVES/*_seed${s}_output/step_1.json 2>/dev/null | grep -v nice | wc -l)/18"

# 4. NICE selections (3) — policy grad uses seed's warmup ckpt + seed's adam cache
CAND=$SAVES/less_output_seed${s}/train/1/all_projected_grads.pt
run_nice(){ local t=$1 n=$2 g=$3 mnt=$4
  [ -f $SAVES/nice_${n}_seed${s}_output/step_1.json ] && { log "[skip nice] $n"; return; }
  CUDA_VISIBLE_DEVICES=$g $PY scripts/nice_select.py --candidate_grads $CAND --base_model $BASE \
    --adapter $wckpt --target_data data/${t}.jsonl --target_name $n \
    --out_cache_dir $SAVES/nice_${n}_seed${s}_output --proj_dim 8192 --seed 123 --mc 16 \
    --temperature 1.0 --max_new_tokens $mnt --selection_ratio 0.05 > $LOGD/nice_${n}_seed${s}.log 2>&1; }
run_nice bbh_target_100 bbh 0 512 &
run_nice mmlu_target mmlu 1 16 &
run_nice tydiqa_target tydiqa 2 64 &
wait
log "NICE selections done: $(ls $SAVES/nice_*_seed${s}_output/step_1.json 2>/dev/null | wc -l)/3"
log "=== SEED $s SELECTIONS COMPLETE ==="

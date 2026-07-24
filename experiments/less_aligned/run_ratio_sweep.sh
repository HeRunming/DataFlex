#!/bin/bash
###############################################################################
# Ratio sweep: STEM-majority skew at ratios stem50/stem70/stem90 (stem80 already
# done in the 3-seed skew experiment). 4 methods × 3 ratios × 3 seeds.
#   methods: less_adam, nice, tsds, mmd_grad_cov_adam
# Per (ratio, method, seed): select (reuse seed caches / shared bge for tsds) ->
#   SFT (8-GPU seed=s) -> eval mmlu_stem+mmlu_humanities.
# Idempotent, NO short timeouts (8-GPU init slow on loaded FS).
###############################################################################
set -uo pipefail
ROOT=/jizhicfs/karonhe/DataFlex_fa; ENVBIN=/jizhicfs/karonhe/envs/dataflex-fa/bin
SAVES=/jizhicfs/karonhe/dataflex_saves; SFT=$SAVES/sft_results; EVAL=$SAVES/eval_results
SUB=$SAVES/sft_subsets; LOGD=$SAVES/logs; EMB=$SAVES/embeddings; CAND=$ROOT/data/less_train_all.jsonl
BASE=/jizhicfs/karonhe/models/shakechen/Llama-2-7b-hf; PY=$ENVBIN/python; MODEL=$BASE
export PATH=$ENVBIN:$PATH
mkdir -p $EVAL/skew $SUB $LOGD; cd $ROOT
RATIOS=(${RATIOS:-stem50 stem70 stem90})
SEEDS=(${SEEDS:-42 1 2})
log(){ echo "[$(date +%m-%d_%H:%M:%S)] $*"; }
# gradient method spec: reg gradient_type shared-cache-base kernel-extra(yaml frag)
gcfg(){ # $1 method -> echoes "reg|gt|base|kernelkey|targ_gt"
  case $1 in
    less_adam) echo "less|adam|less_output||";;
    mmd_grad_cov_adam) echo "mmd|adam|less_output|grad_cov|sgd";;
  esac; }

for R in "${RATIOS[@]}"; do
 tds=mmlu_target_${R}
 for s in "${SEEDS[@]}"; do
  ckpt=$SAVES/sft_results/warmup_seed${s}/checkpoint-1692
  scur=$([ "$s" = 42 ] && echo "" || echo "_seed${s}")
  # ---- gradient selections (less_adam, grad_cov_adam) ----
  for m in less_adam mmd_grad_cov_adam; do
    IFS='|' read reg gt base kk tgt <<< "$(gcfg $m)"
    cn=${m}_${R}${scur}; cache=$SAVES/${cn}_output
    if [ ! -f $cache/step_1.json ]; then
      mkdir -p $cache/train
      [ -e $cache/train/1 ] || ln -s $SAVES/${base}${scur}/train/1 $cache/train/1
      compf=src/dataflex/configs/components_sweep_${cn}.yaml
      $PY - "$cn" "$reg" "$gt" "$kk" "$tgt" "$cache" "$compf" <<'PYEOF'
import yaml,sys
cn,reg,gt,kk,tgt,cache,compf=sys.argv[1:8]
p={"cache_dir":cache,"proj_dim":8192,"save_interval":16,"seed":123,"candidate_subsample":-1,"greedy_device":"auto","gradient_type":gt,"sigma":None}
if kk: p["kernel_type"]=kk; p["target_gradient_type"]=tgt
yaml.safe_dump({"selectors":{cn:{"name":reg,"params":p}}},open(compf,"w"),sort_keys=False)
PYEOF
      selc=experiments/less_aligned/configs/sweep_select_${cn}.yaml
      $PY - "$cn" "$ckpt" "$tds" "$compf" "$selc" "$MODEL" "$SAVES" <<'PYEOF'
import yaml,sys
cn,ckpt,tds,compf,selc,MODEL,SAVES=sys.argv[1:8]
yaml.safe_dump(dict(model_name_or_path=MODEL,adapter_name_or_path=ckpt,trust_remote_code=True,stage="sft",do_train=True,
 finetuning_type="lora",lora_rank=128,lora_alpha=512,lora_target="q_proj,k_proj,v_proj,o_proj",lora_dropout=0.1,
 dataset="less_train_all",template="llama2",cutoff_len=2048,overwrite_cache=True,preprocessing_num_workers=16,
 output_dir=f"{SAVES}/less_aligned/{cn}",logging_steps=10,overwrite_output_dir=True,save_steps=99999,report_to="none",
 per_device_train_batch_size=1,gradient_accumulation_steps=1,learning_rate=2.0e-5,num_train_epochs=1.0,bf16=True,
 ddp_timeout=180000000,train_step=2,train_type="dynamic_select",components_cfg_file=compf,component_name=cn,
 warmup_step=1,update_step=1,update_times=1,selection_ratio=0.05,optimizer_state_path=ckpt,
 target_dataset=tds,eval_dataset=tds,eval_strategy="no"),open(selc,"w"),sort_keys=False)
PYEOF
      log "SEL $cn"; CUDA_VISIBLE_DEVICES=0 dataflex-cli train $selc > $LOGD/sel_${cn}.log 2>&1 || log "[sel FAIL] $cn"
    fi
  done
  # ---- NICE selection ----
  ncn=nice_${R}${scur}
  if [ ! -f $SAVES/${ncn}_output/step_1.json ]; then
    log "NICE $ncn"
    CUDA_VISIBLE_DEVICES=0 http_proxy="" https_proxy="" $PY scripts/nice_select.py \
      --candidate_grads $SAVES/less_output${scur}/train/1/all_projected_grads.pt --base_model $BASE --adapter $ckpt \
      --target_data data/${tds}.jsonl --target_name mmlu --out_cache_dir $SAVES/${ncn}_output \
      --proj_dim 8192 --seed 123 --mc 16 --temperature 1.0 --max_new_tokens 16 --selection_ratio 0.05 \
      > $LOGD/nice_${ncn}.log 2>&1 || log "[nice FAIL] $ncn"
  fi
  # ---- TSDS selection (seed-independent; compute once at seed 42) ----
  if [ "$s" = 42 ] && [ ! -f $SAVES/tsds_${R}_output/step_1.json ]; then
    log "TSDS $R"
    OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0 $PY scripts/run_tsds_from_emb.py \
      --candidate_emb $EMB/candidate_270k.npy --target_emb $EMB/target_${tds}.npy --num_select 13533 \
      --probs_out $SAVES/tsds_${R}_output/probs.npy --indices_out $SAVES/tsds_${R}_output/indices.npy --seed 42 \
      > $LOGD/tsds_${R}.log 2>&1
    $PY -c "import numpy as np,json,os; os.makedirs('$SAVES/tsds_${R}_output',exist_ok=True); idx=np.load('$SAVES/tsds_${R}_output/indices.npy').astype(int).tolist(); json.dump({'indices':idx,'metric':{'method':'tsds'}},open('$SAVES/tsds_${R}_output/step_1.json','w'))"
  fi
 done
done

# ---- register subsets + SFT + eval ----
$PY - "${RATIOS[*]}" "${SEEDS[*]}" <<'PYEOF'
import json,sys,subprocess,os
RATIOS=sys.argv[1].split(); SEEDS=sys.argv[2].split()
SAVES="/jizhicfs/karonhe/dataflex_saves"; SUB=f"{SAVES}/sft_subsets"; CAND="data/less_train_all.jsonl"
info=json.load(open("data/dataset_info.json"))
def reg(cache,key,jsonl):
    if not os.path.exists(jsonl):
        subprocess.run(["/jizhicfs/karonhe/envs/dataflex-fa/bin/python","scripts/export_gradient_selection.py","--candidate_data",CAND,
          "--cache_dir",cache,"--output_dir",f"{SUB}/{key}_export","--method",key,"--target_data","data/mmlu_target.jsonl",
          "--selection_ratio","0.05","--seed","42"],stdout=open(f"{SAVES}/logs/export_{key}.log","w"),stderr=subprocess.STDOUT)
        d=json.load(open(f"{SUB}/{key}_export/selected_subset.json"))
        open(jsonl,"w").write("\n".join(json.dumps(r,ensure_ascii=False) for r in d)+"\n")
    info[key]={"file_name":jsonl,"formatting":"sharegpt","columns":{"messages":"messages"},"tags":{"role_tag":"role","content_tag":"content","user_tag":"user","assistant_tag":"assistant"}}
for R in RATIOS:
    for s in SEEDS:
        sc="" if s=="42" else f"_seed{s}"
        for m in ["less_adam","nice","mmd_grad_cov_adam"]:
            reg(f"{SAVES}/{m}_{R}{sc}_output", f"sw_{m}_{R}_seed{s}_sel", f"{SUB}/sw_{m}_{R}_seed{s}_selected.jsonl")
        reg(f"{SAVES}/tsds_{R}_output", f"sw_tsds_{R}_seed{s}_sel", f"{SUB}/sw_tsds_{R}_seed{s}_selected.jsonl")
json.dump(info,open("data/dataset_info.json","w"),indent=2,ensure_ascii=False)
print("subsets registered")
PYEOF

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
for R in "${RATIOS[@]}"; do for s in "${SEEDS[@]}"; do for m in less_adam nice tsds mmd_grad_cov_adam; do
  out=$SFT/sw_${m}_${R}_seed${s}
  [ -f $out/adapter_model.safetensors ] && { log "[skip sft] ${m}_${R}_s${s}"; continue; }
  log "SFT sw_${m}_${R}_seed${s}"
  dataflex-cli train experiments/less_aligned/configs/train_llama7b_lora.yaml \
    dataset=sw_${m}_${R}_seed${s}_sel output_dir=$out seed=$s \
    per_device_train_batch_size=4 gradient_accumulation_steps=4 lora_alpha=512 num_train_epochs=4 \
    > $LOGD/sft_sw_${m}_${R}_seed${s}.log 2>&1
  [ -f $out/adapter_model.safetensors ] && log "[done] ${m}_${R}_s${s}" || log "[FAIL] ${m}_${R}_s${s}"
done; done; done
for R in "${RATIOS[@]}"; do for s in "${SEEDS[@]}"; do for m in less_adam nice tsds mmd_grad_cov_adam; do
  ad=$SFT/sw_${m}_${R}_seed${s}; out=$EVAL/skew/sw_${m}_${R}_seed${s}
  [ -f $ad/adapter_model.safetensors ] || continue
  find "$out" -name "results_*.json" 2>/dev/null | grep -q . && { log "[skip eval] ${m}_${R}_s${s}"; continue; }
  log "EVAL sw_${m}_${R}_seed${s}"
  NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 http_proxy="http://hy-proxy.woa.com:3128" https_proxy="http://hy-proxy.woa.com:3128" \
  no_proxy=".woa.com,localhost,127.0.0.1,mirrors.tencent.com" \
  accelerate launch --num_processes 8 --main_process_port 29670 -m lm_eval --model hf \
    --model_args "pretrained=$BASE,peft=$ad,dtype=bfloat16,trust_remote_code=True" \
    --tasks mmlu_stem,mmlu_humanities --num_fewshot 5 --batch_size 16 --output_path "$out" \
    > $LOGD/eval_sw_${m}_${R}_seed${s}.log 2>&1
done; done; done
log "=== RATIO SWEEP COMPLETE ==="

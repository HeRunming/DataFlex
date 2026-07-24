#!/bin/bash
###############################################################################
# NICE baseline: export -> register -> 8-GPU SFT -> eval, for 3 targets.
# Sequential 8-GPU DDP SFT (per_device 4 x accum 4 x 8 = eff batch 128, 4 epochs).
# Each target's NICE selection is SFT'd then evaluated on its matching benchmark.
###############################################################################
set -uo pipefail

ROOT=/jizhicfs/karonhe/DataFlex_fa
ENVBIN=/jizhicfs/karonhe/envs/dataflex-fa/bin
SAVES=/jizhicfs/karonhe/dataflex_saves
SFT=$SAVES/sft_results
EVAL=$SAVES/eval_results
LOGD=$SAVES/logs
BASE=/jizhicfs/karonhe/models/shakechen/Llama-2-7b-hf
CAND=$ROOT/data/less_train_all.jsonl
export PATH=$ENVBIN:$PATH
mkdir -p $SFT $EVAL/bbh $EVAL/mmlu $EVAL/tydiqa $LOGD $SAVES/sft_subsets
cd $ROOT

# --- 1. export + register the 3 NICE selections (bbh already needs export) ---
for n in bbh mmlu tydiqa; do
  exp=$SAVES/sft_subsets/nice_${n}_export
  if [ ! -f $exp/selected_subset.json ]; then
    $ENVBIN/python scripts/export_gradient_selection.py \
      --candidate_data $CAND --cache_dir $SAVES/nice_${n}_output \
      --output_dir $exp --method nice_${n} \
      --target_data data/${n}_target.jsonl --selection_ratio 0.05 --seed 42 2>&1 | tail -1
  fi
done

$ENVBIN/python - <<'PY'
import json
info_path="data/dataset_info.json"; info=json.load(open(info_path))
for n in ["bbh","mmlu","tydiqa"]:
    src=f"/jizhicfs/karonhe/dataflex_saves/sft_subsets/nice_{n}_export/selected_subset.json"
    out=f"/jizhicfs/karonhe/dataflex_saves/sft_subsets/nice_{n}_selected.jsonl"
    data=json.load(open(src))
    with open(out,"w") as f:
        for r in data: f.write(json.dumps(r,ensure_ascii=False)+"\n")
    key=f"nice_{n}_selected"
    info[key]={"file_name":out,"formatting":"sharegpt","columns":{"messages":"messages"},
               "tags":{"role_tag":"role","content_tag":"content","user_tag":"user","assistant_tag":"assistant"}}
    print(f"registered {key}: {len(data)}")
json.dump(info,open(info_path,"w"),indent=2,ensure_ascii=False)
PY

# --- 2. sequential 8-GPU SFT ---
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
for n in bbh mmlu tydiqa; do
  out=$SFT/nice_${n}_selected
  if [ -f $out/adapter_model.safetensors ]; then echo "[skip sft] nice_$n"; continue; fi
  echo "=== SFT nice_$n (8-GPU DDP) ==="
  dataflex-cli train experiments/less_aligned/configs/train_llama7b_lora.yaml \
    dataset=nice_${n}_selected output_dir=$out \
    lora_alpha=512 num_train_epochs=4 \
    per_device_train_batch_size=4 gradient_accumulation_steps=4 \
    > $LOGD/sft_nice_${n}.log 2>&1
  echo "  sft nice_$n exit $?"
done

# --- 3. eval (one benchmark per target, single GPU each is fine; run parallel on 0/1/2) ---
echo "=== EVAL ==="
CUDA_VISIBLE_DEVICES=0 lm_eval --model hf \
  --model_args "pretrained=$BASE,peft=$SFT/nice_bbh_selected,dtype=bfloat16,trust_remote_code=True" \
  --tasks bbh_cot_fewshot --batch_size 8 --output_path $EVAL/bbh/nice_bbh \
  > $LOGD/eval_nice_bbh.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 lm_eval --model hf \
  --model_args "pretrained=$BASE,peft=$SFT/nice_mmlu_selected,dtype=bfloat16,trust_remote_code=True" \
  --tasks mmlu --num_fewshot 5 --batch_size 8 --output_path $EVAL/mmlu/nice_mmlu \
  > $LOGD/eval_nice_mmlu.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 $ENVBIN/python scripts/eval_tydiqa.py \
  --adapter $SFT/nice_tydiqa_selected --output $EVAL/tydiqa/nice_tydiqa.json --batch_size 16 \
  > $LOGD/eval_nice_tydiqa.log 2>&1 &
wait
echo "=== ALL DONE ==="

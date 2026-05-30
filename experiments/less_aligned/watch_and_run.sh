#!/bin/bash
# Watcher: when MMD-GradCov-Adam V2 step_1.json appears, export → SFT → BBH eval
set -euo pipefail
cd /jizhicfs/karonhe/DataFlex
DF=/jizhicfs/karonhe/DataFlex
SAVES=/jizhicfs/karonhe/dataflex_saves
METHOD=$1
CACHE=$2
GPU=$3

echo "[$(date)] Watching ${METHOD} cache=${CACHE}"
while [ ! -f "${CACHE}/step_1.json" ]; do
    sleep 600
done
echo "[$(date)] ${METHOD} ready"

# Export
out=${SAVES}/sft_subsets/${METHOD}_selected.jsonl
python scripts/export_gradient_selection.py \
    --candidate_data data/less_train_all.jsonl \
    --cache_dir "${CACHE}" \
    --output_dir ${SAVES}/sft_subsets/${METHOD}_export \
    --method ${METHOD} \
    --target_data data/bbh_target_100.jsonl \
    --selection_ratio 0.05 --seed 42 2>&1 | tail -3
python -c "
import json
data = json.load(open('${SAVES}/sft_subsets/${METHOD}_export/selected_subset.json'))
with open('${out}', 'w') as f:
    for r in data: f.write(json.dumps(r, ensure_ascii=False) + '\n')
print(f'Wrote {len(data)} samples')
"

# Register
python -c "
import json
info_path='data/dataset_info.json'
with open(info_path) as f: info=json.load(f)
key='${METHOD}_selected'
info[key]={'file_name':'${out}','formatting':'sharegpt','columns':{'messages':'messages'},'tags':{'role_tag':'role','content_tag':'content','user_tag':'user','assistant_tag':'assistant'}}
with open(info_path,'w') as f: json.dump(info,f,indent=2,ensure_ascii=False)
print('Registered',key)
"

# SFT
sft_out=${SAVES}/sft_results/${METHOD}_selected
echo "[$(date)] SFT on GPU ${GPU}"
CUDA_VISIBLE_DEVICES=${GPU} dataflex-cli train experiments/less_aligned/configs/train_llama7b_lora.yaml \
    dataset=${METHOD}_selected dataset_dir=${DF}/data \
    output_dir=${sft_out} lora_alpha=512 num_train_epochs=4 \
    > ${SAVES}/sft_results/${METHOD}_selected.log 2>&1
echo "[$(date)] SFT done"

# Eval
eval_out=${SAVES}/eval_results/bbh/${METHOD}_selected
mkdir -p ${eval_out}
echo "[$(date)] BBH eval on GPU ${GPU}"
CUDA_VISIBLE_DEVICES=${GPU} lm_eval --model hf \
    --model_args "pretrained=/jizhicfs/karonhe/models/shakechen/Llama-2-7b-hf,peft=${sft_out},dtype=bfloat16,trust_remote_code=True" \
    --tasks bbh_cot_fewshot --batch_size 8 \
    --output_path ${eval_out} \
    > ${SAVES}/eval_results/bbh/${METHOD}_selected.log 2>&1
echo "[$(date)] eval done"
python scripts/aggregate_bbh_results.py 2>&1 | tail -20

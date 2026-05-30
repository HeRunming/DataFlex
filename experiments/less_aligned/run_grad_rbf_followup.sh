#!/bin/bash
###############################################################################
# Wait for MMD-Grad-RBF-Adam to finish, then run its SFT + BBH eval.
###############################################################################
set -euo pipefail

DATAFLEX_ROOT="/jizhicfs/karonhe/DataFlex"
SAVES="/jizhicfs/karonhe/dataflex_saves"
CANDIDATE="${DATAFLEX_ROOT}/data/less_train_all.jsonl"
METHOD="mmd_grad_rbf_adam"
CACHE="${SAVES}/mmd_grad_rbf_output"
SUBSETS_DIR="${SAVES}/sft_subsets"
SFT_DIR="${SAVES}/sft_results"
EVAL_DIR="${SAVES}/eval_results/bbh"

echo "=== Waiting for ${METHOD} step_1.json ==="
while [ ! -f "${CACHE}/step_1.json" ]; do
    echo "[wait $(date +%H:%M:%S)] not ready"
    sleep 600
done
echo "[ready] ${METHOD}"

# Export
out="${SUBSETS_DIR}/${METHOD}_selected.jsonl"
if [ ! -f "${out}" ]; then
    python "${DATAFLEX_ROOT}/scripts/export_gradient_selection.py" \
        --candidate_data "${CANDIDATE}" \
        --cache_dir "${CACHE}" \
        --output_dir "${SUBSETS_DIR}/${METHOD}_export" \
        --method "${METHOD}" \
        --target_data "${DATAFLEX_ROOT}/data/bbh_target_100.jsonl" \
        --selection_ratio 0.05 \
        --seed 42
    python -c "
import json
data = json.load(open('${SUBSETS_DIR}/${METHOD}_export/selected_subset.json'))
with open('${out}', 'w') as f:
    for r in data:
        f.write(json.dumps(r, ensure_ascii=False) + '\n')
print(f'Wrote {len(data)} samples')
"
fi

# Register
python -c "
import json
info_path = '${DATAFLEX_ROOT}/data/dataset_info.json'
with open(info_path) as f:
    info = json.load(f)
key = '${METHOD}_selected'
info[key] = {
    'file_name': '${SUBSETS_DIR}/' + key + '.jsonl',
    'formatting': 'sharegpt',
    'columns': {'messages': 'messages'},
    'tags': {'role_tag': 'role', 'content_tag': 'content', 'user_tag': 'user', 'assistant_tag': 'assistant'},
}
with open(info_path, 'w') as f:
    json.dump(info, f, indent=2, ensure_ascii=False)
print('Registered', key)
"

# Find a free GPU
GPU=2  # GPU 2 was the one running grad_rbf, now free
sft_out="${SFT_DIR}/${METHOD}_selected"
echo "=== SFT on GPU ${GPU} ==="
CUDA_VISIBLE_DEVICES=${GPU} dataflex-cli train \
    ${DATAFLEX_ROOT}/experiments/less_aligned/configs/train_llama7b_lora.yaml \
    dataset="${METHOD}_selected" \
    dataset_dir="${DATAFLEX_ROOT}/data" \
    output_dir="${sft_out}" \
    lora_alpha=512 \
    num_train_epochs=4 \
    > "${SFT_DIR}/${METHOD}_selected.log" 2>&1
echo "SFT done"

# BBH eval
eval_out="${EVAL_DIR}/${METHOD}_selected"
echo "=== BBH eval on GPU ${GPU} ==="
CUDA_VISIBLE_DEVICES=${GPU} lm_eval --model hf \
    --model_args "pretrained=/jizhicfs/karonhe/models/shakechen/Llama-2-7b-hf,peft=${sft_out},dtype=bfloat16,trust_remote_code=True" \
    --tasks bbh_cot_fewshot \
    --batch_size 8 \
    --output_path "${eval_out}" \
    > "${EVAL_DIR}/${METHOD}_selected.log" 2>&1
echo "Eval done"

# Re-aggregate
python "${DATAFLEX_ROOT}/scripts/aggregate_bbh_results.py"

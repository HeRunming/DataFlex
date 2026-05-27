#!/bin/bash
###############################################################################
# Adam-aware + emb_rbf full pipeline:
#   1. Wait for all 4 selections to finish (LESS-Adam, MMD-Grad-RBF-Adam,
#      MMD-GradCov-Adam, MMD-Emb-RBF)
#   2. Export selected indices → JSONL subsets
#   3. Register in dataset_info.json
#   4. Run 4-epoch SFT on each (parallel, one per GPU)
#   5. Run BBH CoT 3-shot eval on each
###############################################################################
set -euo pipefail

DATAFLEX_ROOT="/jizhicfs/karonhe/DataFlex"
SAVES="/jizhicfs/karonhe/dataflex_saves"
CANDIDATE="${DATAFLEX_ROOT}/data/less_train_all.jsonl"

# Selection cache dirs (each method has its own per components.yaml)
declare -A CACHE_DIRS=(
    [less_adam]="${SAVES}/less_output"
    [mmd_grad_rbf_adam]="${SAVES}/mmd_grad_rbf_output"
    [mmd_grad_cov_adam]="${SAVES}/mmd_grad_cov_output"
    [mmd_emb_rbf]="${SAVES}/mmd_emb_rbf_output"
)

# 1. Wait for selections
echo "=== Waiting for selections ==="
for method in "${!CACHE_DIRS[@]}"; do
    cache="${CACHE_DIRS[$method]}"
    while [ ! -f "${cache}/step_1.json" ]; do
        echo "[wait] ${method}: ${cache}/step_1.json not ready"
        sleep 600
    done
    echo "[ready] ${method}"
done

# 2. Export to JSONL subsets
SUBSETS_DIR="${SAVES}/sft_subsets"
mkdir -p "${SUBSETS_DIR}"
echo ""
echo "=== Exporting subsets ==="
for method in "${!CACHE_DIRS[@]}"; do
    cache="${CACHE_DIRS[$method]}"
    out="${SUBSETS_DIR}/${method}_selected.jsonl"
    if [ -f "${out}" ]; then
        echo "[skip] ${out} already exists"
        continue
    fi
    python "${DATAFLEX_ROOT}/scripts/export_gradient_selection.py" \
        --candidate_data "${CANDIDATE}" \
        --cache_dir "${cache}" \
        --output_dir "${SUBSETS_DIR}/${method}_export" \
        --method "${method}" \
        --target_data "${DATAFLEX_ROOT}/data/bbh_target_100.jsonl" \
        --selection_ratio 0.05 \
        --seed 42 2>&1 | tail -5
    # The export creates selected_subset.json, copy it to jsonl
    if [ -f "${SUBSETS_DIR}/${method}_export/selected_subset.json" ]; then
        python -c "
import json
data = json.load(open('${SUBSETS_DIR}/${method}_export/selected_subset.json'))
with open('${out}', 'w') as f:
    for r in data:
        f.write(json.dumps(r, ensure_ascii=False) + '\n')
print(f'Wrote {len(data)} samples to ${out}')
"
    fi
done

# 3. Register in dataset_info.json
echo ""
echo "=== Registering datasets ==="
python -c "
import json
info_path = '${DATAFLEX_ROOT}/data/dataset_info.json'
with open(info_path) as f:
    info = json.load(f)
for method in ['less_adam', 'mmd_grad_rbf_adam', 'mmd_grad_cov_adam', 'mmd_emb_rbf']:
    key = f'{method}_selected'
    info[key] = {
        'file_name': '${SUBSETS_DIR}/' + key + '.jsonl',
        'formatting': 'sharegpt',
        'columns': {'messages': 'messages'},
        'tags': {'role_tag': 'role', 'content_tag': 'content', 'user_tag': 'user', 'assistant_tag': 'assistant'},
    }
    print(f'Registered: {key}')
with open(info_path, 'w') as f:
    json.dump(info, f, indent=2, ensure_ascii=False)
"

# 4. SFT training in parallel
echo ""
echo "=== SFT training (parallel) ==="
SFT_DIR="${SAVES}/sft_results"
mkdir -p "${SFT_DIR}"
GPU=0
declare -a SFT_PIDS=()
for method in less_adam mmd_grad_rbf_adam mmd_grad_cov_adam mmd_emb_rbf; do
    out="${SFT_DIR}/${method}_selected"
    if [ -d "${out}" ] && [ -f "${out}/adapter_model.safetensors" ]; then
        echo "[skip] ${method} SFT already done"
        GPU=$((GPU+1))
        continue
    fi

    echo "[GPU ${GPU}] Starting SFT: ${method}"
    CUDA_VISIBLE_DEVICES=${GPU} dataflex-cli train \
        ${DATAFLEX_ROOT}/experiments/less_aligned/configs/train_llama7b_lora.yaml \
        dataset="${method}_selected" \
        output_dir="${out}" \
        lora_alpha=512 \
        num_train_epochs=4 \
        > "${SFT_DIR}/${method}_selected.log" 2>&1 &
    SFT_PIDS+=($!)
    GPU=$((GPU+1))
done

# Wait for all SFTs
echo "Waiting for ${#SFT_PIDS[@]} SFT runs..."
for pid in "${SFT_PIDS[@]}"; do wait "${pid}"; done
echo "All SFTs done"

# 5. BBH eval
echo ""
echo "=== BBH eval ==="
EVAL_DIR="${SAVES}/eval_results/bbh"
mkdir -p "${EVAL_DIR}"
GPU=0
declare -a EVAL_PIDS=()
for method in less_adam mmd_grad_rbf_adam mmd_grad_cov_adam mmd_emb_rbf; do
    adapter="${SFT_DIR}/${method}_selected"
    out="${EVAL_DIR}/${method}_selected"
    if [ -f "${out}/results_*.json" ] || ls ${out}/**/results_*.json 2>/dev/null | head -1 | grep -q .; then
        echo "[skip] ${method} eval already done"
        GPU=$((GPU+1))
        continue
    fi

    echo "[GPU ${GPU}] Eval BBH: ${method}"
    CUDA_VISIBLE_DEVICES=${GPU} lm_eval --model hf \
        --model_args "pretrained=/jizhicfs/karonhe/models/shakechen/Llama-2-7b-hf,peft=${adapter},dtype=bfloat16,trust_remote_code=True" \
        --tasks bbh_cot_fewshot \
        --batch_size 8 \
        --output_path "${out}" \
        > "${EVAL_DIR}/${method}_selected.log" 2>&1 &
    EVAL_PIDS+=($!)
    GPU=$((GPU+1))
done

for pid in "${EVAL_PIDS[@]}"; do wait "${pid}"; done
echo "All BBH evals done"

# 6. Aggregate results
echo ""
echo "=== Aggregating results ==="
python "${DATAFLEX_ROOT}/scripts/aggregate_bbh_results.py"

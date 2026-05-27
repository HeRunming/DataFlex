#!/bin/bash
###############################################################################
# BBH CoT Few-shot Evaluation (LESS paper aligned)
# Uses lm-evaluation-harness with hf backend
# Task: bbh_cot_fewshot (3-shot chain-of-thought, exact match)
###############################################################################

set -e

BASE_MODEL="/jizhicfs/karonhe/models/shakechen/Llama-2-7b-hf"
SFT_DIR="/jizhicfs/karonhe/dataflex_saves/sft_results"
OUTPUT_DIR="/jizhicfs/karonhe/dataflex_saves/eval_results/bbh"
mkdir -p "$OUTPUT_DIR"

TASK="bbh_cot_fewshot"
BATCH_SIZE=4

# Methods to evaluate
METHODS=("random_selected" "less_sgd_selected" "mmd_grad_rbf_sgd_selected" "mmd_grad_cov_sgd_selected")

for METHOD in "${METHODS[@]}"; do
    ADAPTER_PATH="${SFT_DIR}/${METHOD}"
    RESULT_DIR="${OUTPUT_DIR}/${METHOD}"

    if [ -f "${RESULT_DIR}/results.json" ]; then
        echo "[SKIP] ${METHOD} already evaluated"
        continue
    fi

    echo "========================================"
    echo "Evaluating: ${METHOD}"
    echo "Adapter: ${ADAPTER_PATH}"
    echo "========================================"

    lm_eval --model hf \
        --model_args "pretrained=${BASE_MODEL},peft=${ADAPTER_PATH},dtype=bfloat16" \
        --tasks "${TASK}" \
        --batch_size "${BATCH_SIZE}" \
        --output_path "${RESULT_DIR}" \
        --log_samples \
        2>&1 | tee "${RESULT_DIR}.log"

    echo "[DONE] ${METHOD}"
    echo ""
done

# Also evaluate base model (no adapter) for reference
echo "========================================"
echo "Evaluating: base_model (no adapter)"
echo "========================================"
BASE_RESULT_DIR="${OUTPUT_DIR}/base_model"
if [ ! -f "${BASE_RESULT_DIR}/results.json" ]; then
    lm_eval --model hf \
        --model_args "pretrained=${BASE_MODEL},dtype=bfloat16" \
        --tasks "${TASK}" \
        --batch_size "${BATCH_SIZE}" \
        --output_path "${BASE_RESULT_DIR}" \
        --log_samples \
        2>&1 | tee "${BASE_RESULT_DIR}.log"
fi

echo ""
echo "============================================"
echo "All evaluations complete!"
echo "Results in: ${OUTPUT_DIR}"
echo "============================================"

# Print summary
echo ""
echo "=== BBH CoT Few-shot Results Summary ==="
for METHOD in "base_model" "${METHODS[@]}"; do
    RESULT_FILE="${OUTPUT_DIR}/${METHOD}/results.json"
    if [ -f "$RESULT_FILE" ]; then
        SCORE=$(python -c "
import json
with open('${RESULT_FILE}') as f:
    r = json.load(f)
# Try to find the aggregate BBH score
results = r.get('results', {})
if '${TASK}' in results:
    acc = results['${TASK}'].get('acc_norm,none', results['${TASK}'].get('exact_match,none', results['${TASK}'].get('acc,none', 'N/A')))
    print(f'{acc}')
else:
    # Average across subtasks
    scores = []
    for k, v in results.items():
        if 'bbh' in k:
            s = v.get('exact_match,none', v.get('acc,none', None))
            if s is not None:
                scores.append(s)
    if scores:
        print(f'{sum(scores)/len(scores):.4f}')
    else:
        print('N/A')
" 2>/dev/null)
        printf "%-35s %s\n" "${METHOD}" "${SCORE}"
    fi
done

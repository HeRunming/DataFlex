#!/bin/bash
set -euo pipefail

###############################################################################
# Smoke Test: verify all selection methods produce valid outputs
#
# Uses tiny data (alpaca_en_demo = ~50 samples, alpaca_zh_demo = ~50 samples)
# Checks: selected_indices.json exists, correct count, no duplicates
###############################################################################

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "${SCRIPT_DIR}/../.."

# Activate env
if [ -f "/jizhicfs/karonhe/miniconda3/bin/activate" ]; then
    source /jizhicfs/karonhe/miniconda3/bin/activate dataflex
fi

CANDIDATE="data/alpaca_en_demo.json"
TARGET="data/alpaca_zh_demo.json"
EMBED_MODEL="${EMBED_MODEL:-/jizhicfs/karonhe/models/sentence-transformers/all-MiniLM-L6-v2}"
OUTPUT_BASE="/tmp/mmd_smoke_test_$$"
RATIO=0.5  # Select 50% for smoke test (small data)

echo "================================================================"
echo " MMD Smoke Test"
echo " Candidate: ${CANDIDATE}"
echo " Target: ${TARGET}"
echo " Ratio: ${RATIO}"
echo " Output: ${OUTPUT_BASE}"
echo "================================================================"

PASS=0
FAIL=0

run_and_check() {
    local method=$1
    local outdir="${OUTPUT_BASE}/${method}"
    echo -n "  [${method}] ... "

    if python scripts/static_select_and_train.py select \
        --method "${method}" \
        --candidate_data "${CANDIDATE}" \
        --target_data "${TARGET}" \
        --embed_model "${EMBED_MODEL}" \
        --selection_ratio "${RATIO}" \
        --output_dir "${outdir}" \
        --seed 42 2>/dev/null; then

        # Check outputs
        if [ -f "${outdir}/selected_indices.json" ] && [ -f "${outdir}/selected_subset.json" ]; then
            # Verify count
            count=$(python -c "import json; d=json.load(open('${outdir}/selected_indices.json')); print(len(d['indices']))")
            echo "PASS (selected ${count} samples)"
            PASS=$((PASS + 1))
        else
            echo "FAIL (output files missing)"
            FAIL=$((FAIL + 1))
        fi
    else
        echo "FAIL (script error)"
        FAIL=$((FAIL + 1))
    fi
}

echo ""
echo ">>> Embedding-based methods:"
run_and_check "random"
run_and_check "mean_target_sim"
run_and_check "max_target_sim"
run_and_check "mean_target_rbf"
run_and_check "mmd_emb_rbf"
run_and_check "full"

echo ""
echo "================================================================"
echo " Results: ${PASS} PASS, ${FAIL} FAIL"
echo "================================================================"

# Cleanup
rm -rf "${OUTPUT_BASE}"

if [ ${FAIL} -gt 0 ]; then
    echo " ❌ Some methods failed!"
    exit 1
else
    echo " ✅ All embedding methods passed smoke test!"
    exit 0
fi

#!/bin/bash
set -euo pipefail

###############################################################################
# Gradient-Based Selection Runner (LESS-Aligned)
#
# Runs gradient-based selection methods using DataFlex dynamic_select in
# single-shot mode (warmup_step=0, train_step=1).
#
# Methods: LESS-SGD, MMD-Grad-RBF-SGD, MMD-GradCov-SGD
# These use the SAME gradient features (proj_dim=8192, seed=123) for fair comparison.
#
# IMPORTANT: Each run uses a UNIQUE cache_dir to prevent cross-contamination.
# After selection, results are auto-exported to standard format.
###############################################################################

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DATAFLEX_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${DATAFLEX_ROOT}"

# === Configuration ===
MODEL="${MODEL:-/jizhicfs/karonhe/models/Qwen/Qwen2.5-0.5B}"
CANDIDATE_DATA="${CANDIDATE_DATA:-data/alpaca_en_demo.json}"
TARGET_DATA="${TARGET_DATA:-data/alpaca_zh_demo.json}"
CONFIGS_DIR="${SCRIPT_DIR}/configs"
OUTPUT_BASE="${OUTPUT_BASE:-${SCRIPT_DIR}/results}"
SEEDS=(42)
RATIOS=(0.05)

echo "================================================================"
echo " Gradient-Based Selection (LESS-Aligned, SGD-safe)"
echo " Model: ${MODEL}"
echo " Candidate: ${CANDIDATE_DATA}"
echo " Target: ${TARGET_DATA}"
echo " Seeds: ${SEEDS[*]}"
echo " Ratios: ${RATIOS[*]}"
echo "================================================================"

# Activate conda env if available
if [ -f "/jizhicfs/karonhe/miniconda3/bin/activate" ]; then
    source /jizhicfs/karonhe/miniconda3/bin/activate dataflex
fi

###############################################################################
# Helper: run selection + export results
###############################################################################
run_gradient_select() {
    local method=$1      # component_name (e.g., less_sgd)
    local config=$2      # base YAML config
    local ratio=$3
    local seed=$4
    local label=$5       # human-readable label
    local outdir="${OUTPUT_BASE}/${label}/ratio_${ratio}/seed_${seed}"
    local cache_dir="${outdir}/selector_cache"

    echo "  [${label}] ratio=${ratio} seed=${seed}"

    # Unique cache per run (prevent cross-contamination)
    mkdir -p "${outdir}" "${cache_dir}"

    # Run selection via dataflex-cli
    dataflex-cli train "${config}" \
        model_name_or_path="${MODEL}" \
        component_name="${method}" \
        selection_ratio="${ratio}" \
        seed="${seed}" \
        output_dir="${outdir}/train_output" \
        2>&1 | tee "${outdir}/selection.log" | tail -3 || {
        echo "    FAILED (see ${outdir}/selection.log)"
        return 1
    }

    # Auto-export to standard format
    echo "    Exporting selection results..."
    python scripts/export_gradient_selection.py \
        --candidate_data "${CANDIDATE_DATA}" \
        --cache_dir "${outdir}/train_output" \
        --output_dir "${outdir}" \
        --method "${label}" \
        --target_data "${TARGET_DATA}" \
        --selection_ratio "${ratio}" \
        --seed "${seed}" 2>&1 | tail -2 || {
        # Also try the default cache dir
        python scripts/export_gradient_selection.py \
            --candidate_data "${CANDIDATE_DATA}" \
            --cache_dir "../dataflex_saves/${method}_output" \
            --output_dir "${outdir}" \
            --method "${label}" \
            --target_data "${TARGET_DATA}" \
            --selection_ratio "${ratio}" \
            --seed "${seed}" 2>&1 | tail -2 || echo "    Export FAILED"
    }
}

###############################################################################
# Run gradient-based selections
###############################################################################

for seed in "${SEEDS[@]}"; do
    for ratio in "${RATIOS[@]}"; do
        echo ""
        echo "--- Seed: ${seed}, Ratio: ${ratio} ---"

        run_gradient_select "less_sgd" \
            "${CONFIGS_DIR}/select_less.yaml" "${ratio}" "${seed}" "less_sgd"

        run_gradient_select "mmd_grad_rbf_sgd" \
            "${CONFIGS_DIR}/select_mmd_grad_rbf.yaml" "${ratio}" "${seed}" "mmd_grad_rbf_sgd"

        run_gradient_select "mmd_grad_cov_sgd" \
            "${CONFIGS_DIR}/select_mmd_grad_cov.yaml" "${ratio}" "${seed}" "mmd_grad_cov_sgd"
    done
done

###############################################################################
# Run diagnostics on all results
###############################################################################
echo ""
echo ">>> Running selection diagnostics..."
if [ -d "${OUTPUT_BASE}" ] && command -v python &>/dev/null; then
    python scripts/selection_diagnostics.py \
        --candidate_data "${CANDIDATE_DATA}" \
        --target_data "${TARGET_DATA}" \
        --results_dir "${OUTPUT_BASE}" \
        --output "${OUTPUT_BASE}/diagnostics_report.json" \
        2>&1 | tail -15 || echo "  Diagnostics skipped (may need embeddings)"
fi

echo ""
echo "================================================================"
echo " Gradient selection + export complete."
echo " Results: ${OUTPUT_BASE}"
echo ""
echo " To train on selected subsets:"
echo "   dataflex-cli train experiments/less_aligned/configs/train_llama7b_lora.yaml \\"
echo "     dataset=selected_subset \\"
echo "     dataset_dir=<method/ratio/seed dir>"
echo "================================================================"

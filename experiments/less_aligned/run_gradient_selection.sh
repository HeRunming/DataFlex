#!/bin/bash
set -euo pipefail

###############################################################################
# Gradient-Based Selection Runner (LESS-Aligned)
#
# Runs gradient-based selection methods using DataFlex dynamic_select in
# single-shot mode (warmup_step=0, train_step=1).
#
# Methods: LESS, MMD-Grad-RBF (SGD), MMD-GradCov (SGD)
# These use the SAME gradient features (proj_dim=8192, seed=123) for fair comparison.
#
# Pre-requisite: warmup checkpoint (if using Adam-aware gradients)
# For SGD-safe first pass, no warmup needed.
###############################################################################

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DATAFLEX_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${DATAFLEX_ROOT}"

# === Configuration ===
MODEL="${MODEL:-/jizhicfs/karonhe/models/Qwen/Qwen2.5-0.5B}"
CONFIGS_DIR="${SCRIPT_DIR}/configs"
OUTPUT_BASE="${OUTPUT_BASE:-${SCRIPT_DIR}/results}"
SEEDS=(42)
RATIOS=(0.05)

echo "================================================================"
echo " Gradient-Based Selection (LESS-Aligned)"
echo " Model: ${MODEL}"
echo " Seeds: ${SEEDS[*]}"
echo " Ratios: ${RATIOS[*]}"
echo "================================================================"

# Activate conda env if available
if [ -f "/jizhicfs/karonhe/miniconda3/bin/activate" ]; then
    source /jizhicfs/karonhe/miniconda3/bin/activate dataflex
fi

###############################################################################
# Run gradient-based selections
###############################################################################

for seed in "${SEEDS[@]}"; do
    for ratio in "${RATIOS[@]}"; do
        echo ""
        echo "--- Seed: ${seed}, Ratio: ${ratio} ---"

        # LESS (SGD gradient - no Adam state needed)
        echo "  [LESS SGD] Running selection..."
        dataflex-cli train "${CONFIGS_DIR}/select_less.yaml" \
            model_name_or_path="${MODEL}" \
            component_name=less_sgd \
            selection_ratio="${ratio}" \
            seed="${seed}" \
            output_dir="${OUTPUT_BASE}/less_sgd/ratio_${ratio}/seed_${seed}" \
            2>&1 | tail -5 || echo "  [LESS SGD] FAILED"

        # MMD-Grad-RBF (SGD)
        echo "  [MMD-Grad-RBF SGD] Running selection..."
        dataflex-cli train "${CONFIGS_DIR}/select_mmd_grad_rbf.yaml" \
            model_name_or_path="${MODEL}" \
            component_name=mmd_grad_rbf_sgd \
            selection_ratio="${ratio}" \
            seed="${seed}" \
            output_dir="${OUTPUT_BASE}/mmd_grad_rbf_sgd/ratio_${ratio}/seed_${seed}" \
            2>&1 | tail -5 || echo "  [MMD-Grad-RBF SGD] FAILED"

        # MMD-GradCov (SGD)
        echo "  [MMD-GradCov SGD] Running selection..."
        dataflex-cli train "${CONFIGS_DIR}/select_mmd_grad_cov.yaml" \
            model_name_or_path="${MODEL}" \
            component_name=mmd_grad_cov_sgd \
            selection_ratio="${ratio}" \
            seed="${seed}" \
            output_dir="${OUTPUT_BASE}/mmd_grad_cov_sgd/ratio_${ratio}/seed_${seed}" \
            2>&1 | tail -5 || echo "  [MMD-GradCov SGD] FAILED"
    done
done

echo ""
echo "================================================================"
echo " Gradient selection complete."
echo " Results in: ${OUTPUT_BASE}"
echo " Next: extract selected_indices and train on subset."
echo "================================================================"

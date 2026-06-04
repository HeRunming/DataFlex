#!/bin/bash
# =============================================================================
# Phase D — LESS-aligned offline selection (wrapper)
# =============================================================================
# Loads the Phase C warmup checkpoint, then for each of 16 methods runs
# selector.select() over the 270K Tulu pool exactly once and saves
# selected_indices.npy under $SAVE_DIR/<method>/.
# =============================================================================

set -e

CONDA_ENV="${CONDA_ENV:-/jizhicfs/karonhe/miniconda_karonhe/envs/sft_train}"
export PATH="$CONDA_ENV/bin:$PATH"
export PYTHONPATH="/jizhicfs/karonhe/DataFlex/src:${PYTHONPATH:-}"
export DISABLE_VERSION_CHECK=1
export http_proxy=http://hy-proxy.woa.com:3128
export https_proxy=http://hy-proxy.woa.com:3128

REPO=/jizhicfs/karonhe/DataFlex
MODEL=/jizhicfs/karonhe/models/Llama-2-7b-hf
SAVE_ROOT=/jizhicfs/karonhe/dataflex_saves/less_aligned
WARMUP_CKPT=$SAVE_ROOT/warmup/warmup_ckpt_latest    # symlinked by run_warmup.sh
SELECTIONS=$SAVE_ROOT/selections
DATASET_PATH=$REPO/data/tulu2_270k.json
EVAL_DATASET=$REPO/data/MMLU_valid_cot.json   # used as LESS target task (already in repo)

mkdir -p "$SELECTIONS"

NUM_SAMPLES=13533     # 5% of 270K, LESS-paper default
NPROC=8

METHODS=(
    "random_s42"
    "loss_s42"
    "less_s42"
    "fisher_sft_s42"
    "grad_norm_topk_s42"
    "rsub_own_seed1"
    "rsub_own_seed2"
    "rsub_own_seed3"
    "hybrid_add_l025_s42"
    "hybrid_mul_g025_s42"
    "hybrid_mul_g05_s42"
    "logdet_nopref_s42"
    "hybrid_add_l025_s1"
    "hybrid_add_l025_s2"
    "hybrid_mul_g025_s1"
    "hybrid_mul_g025_s2"
)

# Method names map to component names in components.yaml. The "_sN" suffix
# is a training-seed tag, not a separate selector — collapse to base names.
declare -A METHOD_TO_COMPONENT
METHOD_TO_COMPONENT[random_s42]=random
METHOD_TO_COMPONENT[loss_s42]=loss
METHOD_TO_COMPONENT[less_s42]=less
METHOD_TO_COMPONENT[fisher_sft_s42]=fisher_sft
METHOD_TO_COMPONENT[grad_norm_topk_s42]=grad_norm_topk
METHOD_TO_COMPONENT[rsub_own_seed1]=random_subspace_logdet_seed1
METHOD_TO_COMPONENT[rsub_own_seed2]=random_subspace_logdet_seed2
METHOD_TO_COMPONENT[rsub_own_seed3]=random_subspace_logdet_seed3
METHOD_TO_COMPONENT[hybrid_add_l025_s42]=opt_gcs_hybrid_add_lambda0.25
METHOD_TO_COMPONENT[hybrid_add_l025_s1]=opt_gcs_hybrid_add_lambda0.25
METHOD_TO_COMPONENT[hybrid_add_l025_s2]=opt_gcs_hybrid_add_lambda0.25
METHOD_TO_COMPONENT[hybrid_mul_g025_s42]=opt_gcs_hybrid_mul_gamma0.25
METHOD_TO_COMPONENT[hybrid_mul_g025_s1]=opt_gcs_hybrid_mul_gamma0.25
METHOD_TO_COMPONENT[hybrid_mul_g025_s2]=opt_gcs_hybrid_mul_gamma0.25
METHOD_TO_COMPONENT[hybrid_mul_g05_s42]=opt_gcs_hybrid_mul_gamma0.5
METHOD_TO_COMPONENT[logdet_nopref_s42]=opt_gcs_logdet_no_prefilter

# Build the comma-free list of unique component names to actually run
COMPONENT_LIST=()
for m in "${METHODS[@]}"; do
    COMPONENT_LIST+=("${METHOD_TO_COMPONENT[$m]}")
done

echo "============================================="
echo "Phase D — Offline selection"
echo "Warmup ckpt: $WARMUP_CKPT"
echo "Pool:        $DATASET_PATH"
echo "Save dir:    $SELECTIONS"
echo "Methods:     ${#METHODS[@]}"
echo "============================================="

cd $REPO

# Pre-tokenize pool on a single process to avoid 8 ranks racing & timing out.
# (The driver also has rank-0-first guards, but doing it cleanly here is safer.)
PRETOK_DIR="$SELECTIONS/_tokenized_pool"
if [ ! -d "$PRETOK_DIR" ] || [ ! -f "$PRETOK_DIR/dataset_info.json" ]; then
    echo "[pretok] tokenizing 270K pool to $PRETOK_DIR (single-process) ..."
    $CONDA_ENV/bin/python <<PYEOF
import sys
sys.path.insert(0, '/jizhicfs/karonhe/DataFlex/experiments/less_aligned/scripts')
import importlib.util
spec = importlib.util.spec_from_file_location('rso', '/jizhicfs/karonhe/DataFlex/experiments/less_aligned/scripts/run_select_offline.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained('$MODEL', use_fast=True)
if tok.pad_token is None: tok.pad_token = tok.eos_token
ds = m.load_alpaca_dataset_as_hf('$DATASET_PATH', tok, max_length=1024)
print(f'tokenized {len(ds)} examples')
ds.save_to_disk('$PRETOK_DIR')
print(f'saved to $PRETOK_DIR')
PYEOF
else
    echo "[pretok] $PRETOK_DIR already exists, skipping"
fi

# Same for eval set if specified
EVAL_TOK_DIR="$SELECTIONS/_tokenized_eval"
if [ -f "$EVAL_DATASET" ] && [ ! -f "$EVAL_TOK_DIR/dataset_info.json" ]; then
    echo "[pretok-eval] tokenizing $EVAL_DATASET to $EVAL_TOK_DIR ..."
    $CONDA_ENV/bin/python <<PYEOF
import sys
sys.path.insert(0, '/jizhicfs/karonhe/DataFlex/experiments/less_aligned/scripts')
import importlib.util
spec = importlib.util.spec_from_file_location('rso', '/jizhicfs/karonhe/DataFlex/experiments/less_aligned/scripts/run_select_offline.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained('$MODEL', use_fast=True)
if tok.pad_token is None: tok.pad_token = tok.eos_token
ds = m.load_alpaca_dataset_as_hf('$EVAL_DATASET', tok, max_length=1024)
print(f'tokenized eval {len(ds)} examples')
ds.save_to_disk('$EVAL_TOK_DIR')
PYEOF
fi

# Bump rendezvous + collective op timeouts for distributed
export TORCHELASTIC_RDZV_TIMEOUT=3600
export NCCL_TIMEOUT=3600

torchrun --nproc_per_node=$NPROC --nnodes=1 \
    experiments/less_aligned/scripts/run_select_offline.py \
    --warmup_ckpt "$WARMUP_CKPT" \
    --base_model "$MODEL" \
    --dataset_path "$DATASET_PATH" \
    --eval_dataset_path "$EVAL_DATASET" \
    --save_dir "$SELECTIONS" \
    --num_samples "$NUM_SAMPLES" \
    --shared_grad_method opt_gcs_logdet \
    --methods "${COMPONENT_LIST[@]}" \
    2>&1 | tee "$SAVE_ROOT/select_offline.log"

echo ""
echo "============================================="
echo "Selections per method:"
echo "============================================="
for m in "${METHODS[@]}"; do
    comp="${METHOD_TO_COMPONENT[$m]}"
    n=$(python -c "import numpy as np, os; p='$SELECTIONS/$comp/selected_indices.npy'; print(len(np.load(p)) if os.path.exists(p) else 'MISSING')")
    printf "  %-25s -> %s   (%s)\n" "$m" "$comp" "$n"
    # Symlink component-named indices into method-named dirs for downstream Phase E
    if [ -f "$SELECTIONS/$comp/selected_indices.npy" ] && [ ! -e "$SELECTIONS/$m/selected_indices.npy" ]; then
        mkdir -p "$SELECTIONS/$m"
        ln -sf "$SELECTIONS/$comp/selected_indices.npy" "$SELECTIONS/$m/selected_indices.npy"
    fi
done

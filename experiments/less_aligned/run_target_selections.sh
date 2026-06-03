#!/bin/bash
###############################################################################
# Run all (target × method) SELECTIONS (cache-reuse, fast).
# Gradient methods reuse symlinked candidate grads → only target grads + greedy.
# Emb methods reuse candidate embeddings → only greedy.
# Runs sequentially per GPU group to avoid contention on the shared grad cache.
###############################################################################
set -uo pipefail
cd /jizhicfs/karonhe/DataFlex
CFG=experiments/less_aligned/configs/targets
LOG=/tmp/target_select
mkdir -p $LOG

run_sel() {
    local comp=$1 gpu=$2
    local out=/jizhicfs/karonhe/dataflex_saves/${comp}_output
    if [ -f "${out}/step_1.json" ]; then
        echo "[skip] ${comp} already has step_1.json"
        return
    fi
    echo "[GPU ${gpu}] selecting ${comp}"
    CUDA_VISIBLE_DEVICES=${gpu} dataflex-cli train ${CFG}/select_${comp}.yaml > ${LOG}/${comp}.log 2>&1
    if [ -f "${out}/step_1.json" ]; then echo "[done] ${comp}"; else echo "[FAIL] ${comp} (see ${LOG}/${comp}.log)"; fi
}

# MMLU methods on GPUs 0-3, TydiQA on GPUs 4-7. Gradient methods need a GPU
# (target grad compute); emb methods also use GPU for greedy.
# Run gradient-adam first (need warmup ckpt load), then sgd, then emb.

# --- MMLU ---
run_sel less_adam_mmlu 0 &
run_sel mmd_grad_rbf_adam_mmlu 1 &
run_sel mmd_grad_cov_adam_mmlu 2 &
run_sel less_sgd_mmlu 3 &
# --- TydiQA ---
run_sel less_adam_tydiqa 4 &
run_sel mmd_grad_rbf_adam_tydiqa 5 &
run_sel mmd_grad_cov_adam_tydiqa 6 &
run_sel less_sgd_tydiqa 7 &
wait
echo "=== wave 1 done ==="

run_sel mmd_grad_rbf_sgd_mmlu 0 &
run_sel mmd_grad_cov_sgd_mmlu 1 &
run_sel mmd_emb_rbf_mmlu 2 &
run_sel mmd_emb_rbf_stochastic_mmlu 3 &
run_sel mmd_grad_rbf_sgd_tydiqa 4 &
run_sel mmd_grad_cov_sgd_tydiqa 5 &
run_sel mmd_emb_rbf_tydiqa 6 &
run_sel mmd_emb_rbf_stochastic_tydiqa 7 &
wait
echo "=== wave 2 done — all selections complete ==="
ls /jizhicfs/karonhe/dataflex_saves/*_{mmlu,tydiqa}_output/step_1.json 2>/dev/null | wc -l

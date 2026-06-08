#!/bin/bash
###############################################################################
# Sequential 8-GPU SFT for the 15 SGD+emb selections (unified pipeline,
# same as the Adam run: from-base Llama-2-7B LoRA r=128 a=512, 4 epochs,
# effective batch 128). Idempotent.
###############################################################################
set -uo pipefail
cd /jizhicfs/karonhe/DataFlex
export PATH=/jizhicfs/karonhe/envs/dataflex/bin:$PATH
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export FORCE_TORCHRUN=1
SFT_DIR=/jizhicfs/karonhe/dataflex_saves/sft_results
LOGD=/tmp/dataflex_logs
mkdir -p "$SFT_DIR" "$LOGD"

DATASETS=(
  less_sgd_bbh mmd_grad_rbf_sgd_bbh mmd_grad_cov_sgd_bbh
  less_sgd_mmlu mmd_grad_rbf_sgd_mmlu mmd_grad_cov_sgd_mmlu
  less_sgd_tydiqa mmd_grad_rbf_sgd_tydiqa mmd_grad_cov_sgd_tydiqa
  mmd_emb_rbf_bbh mmd_emb_rbf_stochastic_bbh
  mmd_emb_rbf_mmlu mmd_emb_rbf_stochastic_mmlu
  mmd_emb_rbf_tydiqa mmd_emb_rbf_stochastic_tydiqa
)
for ds in "${DATASETS[@]}"; do
  out="$SFT_DIR/${ds}_selected"
  if [ -f "$out/adapter_model.safetensors" ]; then echo "[skip] $ds"; continue; fi
  echo "=== [$(date +%H:%M:%S)] SFT $ds (8-GPU) ==="
  dataflex-cli train experiments/less_aligned/configs/train_llama7b_lora.yaml \
    dataset="${ds}_selected" dataset_dir=data output_dir="$out" \
    per_device_train_batch_size=16 gradient_accumulation_steps=1 \
    lora_alpha=512 num_train_epochs=4 \
    > "$LOGD/sft_${ds}.log" 2>&1
  [ -f "$out/adapter_model.safetensors" ] && echo "[done] $ds" || echo "[FAIL] $ds"
done
echo "=== ALL SGD+emb SFT COMPLETE [$(date +%H:%M:%S)] ==="

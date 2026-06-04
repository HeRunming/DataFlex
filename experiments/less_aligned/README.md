# LESS-Aligned Experiment (Llama-2-7B + 270K Tulu-V2 + LoRA r=128 + AdamW)

Re-run of the 16-method × 7-benchmark data-selection comparison, with every
setting aligned to the **LESS paper** (Xia et al., ICML 2024) so results are
directly comparable.

## Settings

| Knob | Value |
|------|-------|
| Base model | Llama-2-7B (`/jizhicfs/karonhe/models/Llama-2-7b-hf`) |
| Data pool | 270,659 examples = LESS official Tulu-V2 mix (FLAN v2 100K + CoT 100K + Dolly 15K + Oasst1 55.7K), from `princeton-nlp/less_data` |
| Budget | 5% = 13,533 examples per method |
| LoRA | rank 128, alpha 512, target=all |
| Optimizer | AdamW, lr 2e-5, betas 0.9/0.999, wd 0.0, linear schedule, 4 epochs |
| Template | alpaca |
| Gradient features | **all** gradient-based selectors use the same Adam-preconditioned, TRAK-projected (4096-d Rademacher) gradients |

## Pipeline (6 phases)

1. **Phase A** — `data/download_less_data.py`: build `data/tulu2_270k.json` from the LESS official zip.
2. **Phase B** — Adam-preconditioning unified into `Selector.adam_precondition_grads()` (base_selector.py); all selectors set `gradient_type: adam`.
3. **Phase C** — `run_warmup.sh`: 4-epoch LoRA warmup on a fixed 5% random subset; saves adapter + AdamW optimizer state (`exp_avg`/`exp_avg_sq`) for gradient features.
4. **Phase D** — `run_select.sh` → `scripts/run_select_offline.py`: load warmup ckpt, compute Adam-preconditioned gradients **once** (shared across methods), run all 16 selectors, write `selected_indices.npy`.
5. **Phase E** — `run_final_sft.sh`: train Llama-2-7B + LoRA r=128 **from base** on each method's selected 5%, 4 epochs.
6. **Phase F** — `run_eval.sh`: vLLM `lm_eval` on 7 benchmarks (`max_lora_rank=128`, `enable_chunked_prefill=False, enable_prefix_caching=False, enforce_eager=True`, `gpu_memory_utilization=0.75`).

## Key findings

See `results/results_summary.md` for the full table. Headline results:

- **LESS wins** (AVG 0.3802, AvgRk 5.00): supervised gradient-similarity selection reproduces the paper's core claim — target-aware selection beats unsupervised geometric selection in this regime.
- **Most methods barely beat random** (random AVG 0.3568, rank 6/16). At 5% budget on a weak base model, SFT gives only marginal average gains over the Llama-2-7B base (0.3759), and regresses on IFEval/ARC-E.
- **OptGCS / hybrid variants underperform random** here (ranks 9–16) — the opposite of the earlier Llama-3.1-8B + Openhermes setting where hybrid_mul topped GSM8K. This suggests the unsupervised "cover the training geometry" objective does not hold under small-budget + weak-base conditions, and motivates a **target-aware OptGCS** variant.

## Reproduce

```bash
# Phase A: data
python experiments/less_aligned/data/download_less_data.py    # needs LESS zip extracted to /jizhicfs/karonhe/less_data_zip
# Phase C-F (each waits on the previous):
bash experiments/less_aligned/run_warmup.sh
bash experiments/less_aligned/run_select.sh
bash experiments/less_aligned/run_final_sft.sh
bash experiments/less_aligned/run_eval.sh
```

Environment: `sft_train` conda env (PyTorch 2.6 + CUDA 12.4) for training/selection;
`opencompass_hrm` (vLLM 0.12.0) for eval. `traker` + `fast_jl` required in `sft_train`
for the TRAK CudaProjector (the BasicProjector OOMs at r=128).

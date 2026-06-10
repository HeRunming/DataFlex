# Unified Comparison: SGD vs Adam × LESS vs MMD (all 8 methods × 3 targets)

**Date:** 2026-06-09
**Why this table exists:** every cell below was produced by ONE pipeline —
fp32 candidate gradients (or bge-base-en-v1.5 embeddings), identical 4-epoch
LoRA SFT, and identical eval (BBH CoT 3-shot, MMLU 5-shot, TyDiQA 1-shot with
the `Answer:` trigger fix). This is the first table where SGD-vs-Adam and
MMD-vs-LESS-vs-embedding are all directly comparable. The earlier
`three_target_summary` mixed pipelines (bf16 bug, old eval) and must not be
compared against these numbers.

## Setup

- Base: Llama-2-7B; pool: 270,679 LESS Tulu-V2; budget: 5% (13,533).
- Gradient methods: TRAK proj 8192-dim, seed 123. Adam variants precondition
  candidate grads with the warmup AdamW state (checkpoint-1692), target grads
  always SGD. SGD variants use raw SGD grads.
- Embedding methods: bge-base-en-v1.5, greedy MMD (exact / stochastic ε=0.01).
- SFT: from base, LoRA r=128 α=512, 4 epochs, effective batch 128.

## Results (all 24 cells, same pipeline)

| Method | grad | BBH | MMLU | TyDiQA-F1 |
|---|---|---|---|---|
| less_sgd | sgd | 0.3847 | 0.4642 | 0.5433 |
| mmd_grad_rbf_sgd | sgd | 0.3649 | 0.4648 | 0.5387 |
| mmd_grad_cov_sgd | sgd | 0.3873 | **0.4704** | 0.5572 |
| less_adam | adam | 0.3726 | 0.4574 | 0.5418 |
| mmd_grad_rbf_adam | adam | 0.3744 | 0.4542 | 0.5610 |
| mmd_grad_cov_adam | adam | 0.3821 | 0.4527 | **0.5735** |
| mmd_emb_rbf | emb | 0.3916 | 0.4495 | 0.5468 |
| mmd_emb_rbf_stochastic | emb | **0.3930** | 0.4516 | 0.5535 |

### Per-target best
- **BBH:** mmd_emb_rbf_stochastic (0.3930)
- **MMLU:** mmd_grad_cov_sgd (0.4704)
- **TyDiQA:** mmd_grad_cov_adam (0.5735)

→ An MMD variant wins **every** target. LESS never tops a column.

## Key reading: GradCov beats its own LESS on 5/6 (target, optimizer) cells

| Target | less_sgd → cov_sgd | less_adam → cov_adam |
|---|---|---|
| BBH | +0.0026 | +0.0095 |
| MMLU | +0.0062 | −0.0047 |
| TyDiQA | +0.0138 | **+0.0317** |

MMD-GradCov ≥ LESS under both optimizers on every target except MMLU-Adam
(−0.005, within noise). The gradient-covariance kernel is the most consistent
improvement over LESS — strongest on TyDiQA (the most heterogeneous, 9-language
target), exactly where matching the target gradient *distribution* rather than
its mean should help most.

## Other observations

- **GradRBF** is inconsistent (helps under Adam on TyDiQA, hurts under SGD on
  BBH). GradCov is the more reliable kernel.
- **Embedding MMD** is surprisingly strong on BBH (best overall there) but
  weakest on MMLU — semantic matching suits BBH's reasoning-style coverage but
  not MMLU's broad factual spread.
- **Adam vs SGD:** no uniform winner. SGD is better on MMLU/BBH for most
  methods; Adam is better on TyDiQA. The fp32 Adam fix matters specifically for
  the covariance kernel on heterogeneous targets (TyDiQA cov_adam 0.5735 is the
  single best cell).

## Caveat

Single seed; gaps of 0.003–0.03 need multi-seed confirmation before strong
claims. Next: 3 seeds on the GradCov-vs-LESS cells, and the 1%/10% budget
scaling (LESS only ever reports 5%, so this is net-new).

Raw per-method outputs: `eval_results/{bbh,mmlu,tydiqa}/` (gitignored, large);
aggregated scores in `unified_24_summary.json`.

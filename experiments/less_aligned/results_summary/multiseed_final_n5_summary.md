# FINAL multi-seed comparison (n=5: seeds 42,1,2,3,4) — mean ± std

**Date:** 2026-07-14. Same env throughout (transformers 4.50, llamafactory 0.9.3).
Full-pipeline multi-seed (NICE-paper protocol, 5 seeds): each seed = distinct
warmup 5% subset → re-warmup → re-cache candidate grads → re-select → SFT → eval.
Error bars capture **selection + training variance** (embedding methods'
selection is bge-based & seed-independent, so their spread is SFT-only → smaller).
10 methods × 3 targets × 5 seeds = 150 SFT + 150 eval, all complete.

## Table (mean ± std over 5 seeds)

| Method | grad/emb | BBH | MMLU | TyDiQA-F1 |
|---|---|---|---|---|
| less_sgd | sgd | 0.3812 ± 0.0063 | 0.4462 ± 0.0050 | 0.5713 ± 0.0175 |
| mmd_grad_rbf_sgd | sgd | 0.3793 ± 0.0059 | 0.4550 ± 0.0075 | 0.5433 ± 0.0419 |
| mmd_grad_cov_sgd | sgd | 0.3879 ± 0.0081 | 0.4565 ± 0.0071 | 0.5700 ± 0.0288 |
| less_adam | adam | 0.3852 ± 0.0065 | 0.4454 ± 0.0076 | 0.5578 ± 0.0444 |
| mmd_grad_rbf_adam | adam | 0.3615 ± 0.0091 | 0.4413 ± 0.0073 | 0.5657 ± 0.0079 |
| mmd_grad_cov_adam | adam | 0.3696 ± 0.0056 | 0.4539 ± 0.0095 | **0.5794 ± 0.0073** |
| mmd_emb_rbf | emb | 0.3961 ± 0.0039 | 0.4448 ± 0.0028 | 0.5516 ± 0.0066 |
| mmd_emb_rbf_stochastic | emb | 0.3939 ± 0.0040 | 0.4450 ± 0.0015 | 0.5525 ± 0.0054 |
| tsds | emb (OT) | **0.3974 ± 0.0047** | 0.4586 ± 0.0020 | 0.5605 ± 0.0064 |
| nice | adam+policy | 0.3918 ± 0.0049 | **0.4617 ± 0.0071** | 0.5747 ± 0.0137 |

## Per-target: leader + who ties (2·SE of difference, n=5)

- **BBH** — leader **tsds 0.3974**. Statistical ties (within 2·SE): mmd_emb_rbf
  (0.3961), emb_stoch (0.3939), nice (0.3918). → **semantic/embedding methods win
  BBH**; all gradient methods (LESS, GradCov, GradRBF) sit significantly lower.
  GradCov-sgd (0.388) is the best gradient method but still below the emb cluster.
- **MMLU** — leader **nice 0.4617**. Ties: tsds (0.4586), grad_cov_sgd (0.4565),
  grad_rbf_sgd (0.4550), grad_cov_adam (0.4539). → **NICE tops MMLU**, in a 5-way
  tie with tsds + the SGD gradient MMDs. LESS and emb_rbf are significantly lower.
- **TyDiQA** — leader **mmd_grad_cov_adam 0.5794**. Ties: nice (0.5747),
  less_sgd (0.5713), grad_cov_sgd (0.5700), (less_adam 0.5578 also within its huge
  ±0.044). → **GradCov-adam tops TyDiQA**, tied with NICE and the SGD methods.

## Headline readings

1. **No single method dominates — the winner rotates by target**, and every
   "win" is a *statistical tie* with 2-4 other methods once n=5 error bars are in:
   - BBH → TSDS / embedding methods
   - MMLU → NICE (tied with TSDS, GradCov-sgd)
   - TyDiQA → MMD-GradCov-adam (tied with NICE)

2. **MMD-GradCov is competitive but NOT a clear winner.** It leads TyDiQA (tied
   with NICE) and is in the MMLU top-tie, but loses BBH to embedding methods and
   is mid-pack there. The original single-seed claim "GradCov beats LESS & TSDS on
   MMLU/TyDiQA" **weakens to "GradCov is in the top statistical tie on MMLU/TyDiQA;
   the gaps over TSDS/NICE are within noise."**

3. **NICE is the strongest all-round baseline** — outright best on MMLU, tied-2nd
   on TyDiQA, top-tie on BBH. The reward-driven policy-gradient signal is very
   competitive; it does not lose to GradCov on any target.

4. **TSDS wins BBH and is top-tie on MMLU** — embedding/OT selection remains the
   method to beat on reasoning-coverage (BBH) and broad-factual (MMLU).

5. **Adam vs SGD is target-dependent for MMD** (confirmed, not the LESS pattern):
   - BBH: SGD ≫ Adam (grad_cov_sgd 0.388 vs adam 0.370; grad_rbf even worse under Adam)
   - TyDiQA: Adam > SGD for grad_cov (0.579 vs 0.570)
   - MMLU: ~tied. There is no uniform Adam>SGD; the "Adam higher" memory was a
     LESS-only, partly bf16-bug artifact (see earlier note).

6. **Variance is large relative to gaps** — TyDiQA std reaches ±0.044 (less_adam),
   ±0.041 (grad_rbf_sgd), ±0.029 (grad_cov_sgd). This is exactly why single-seed
   rankings were unreliable, and why most top-of-column differences are ties.

## Bottom line for the paper
The honest, error-barred story: **MMD-GradCov is a competitive targeted-selection
method — best or tied-best on MMLU and TyDiQA — but it does not statistically beat
the strongest baselines (NICE, TSDS) on any single target, and it trails on BBH.**
NICE and TSDS are both very strong; the method that wins depends on the target's
nature (semantic-coverage → TSDS/emb; heterogeneous/multilingual → GradCov/NICE).

Raw: `multiseed_summary.{md,json,csv}`; adapters `sft_results/{method}_{target}_seed{42,1,2,3,4}`.

# Multi-seed same-env comparison (2 seeds: 42, 1) — mean ± std

**Date:** 2026-07-07. Same env throughout (transformers 4.50, llamafactory 0.9.3).
Full-pipeline multi-seed: each seed = different warmup 5% subset → re-warmup →
re-cache candidate grads → re-select → SFT → eval. So error bars capture
**selection + training variance** (embedding methods emb_rbf/emb_stoch/tsds use
seed-independent bge selection, so their spread is SFT-only). 10 methods × 3
targets × 2 seeds = 60 SFT + 60 eval, all complete.

## Table (mean ± std over seeds 42, 1)

| Method | grad/emb | BBH | MMLU | TyDiQA-F1 |
|---|---|---|---|---|
| less_sgd | sgd | 0.3846 ± 0.0037 | 0.4472 ± 0.0064 | 0.5573 ± 0.0138 |
| mmd_grad_rbf_sgd | sgd | 0.3795 ± 0.0041 | 0.4550 ± 0.0134 | 0.4991 ± 0.0038 |
| mmd_grad_cov_sgd | sgd | 0.3937 ± 0.0038 | 0.4559 ± 0.0096 | 0.5480 ± 0.0412 |
| less_adam | adam | 0.3866 ± 0.0024 | 0.4458 ± 0.0014 | 0.5277 ± 0.0644 |
| mmd_grad_rbf_adam | adam | 0.3664 ± 0.0043 | 0.4414 ± 0.0033 | 0.5682 ± 0.0117 |
| mmd_grad_cov_adam | adam | 0.3681 ± 0.0013 | 0.4554 ± 0.0089 | **0.5776 ± 0.0114** |
| mmd_emb_rbf | emb | 0.3942 ± 0.0021 | 0.4432 ± 0.0020 | 0.5523 ± 0.0124 |
| mmd_emb_rbf_stochastic | emb | 0.3934 ± 0.0057 | 0.4440 ± 0.0014 | 0.5532 ± 0.0105 |
| tsds | emb (OT) | **0.4020 ± 0.0019** | **0.4606 ± 0.0002** | 0.5585 ± 0.0095 |
| nice | adam+policy | 0.3889 ± 0.0056 | 0.4539 ± 0.0002 | 0.5748 ± 0.0039 |

## Per-target winner + significance (gap vs pooled ±1σ; 2 seeds → crude)

- **BBH:** tsds 0.4020 clearly leads. All others ≥0.008 below, gap > pooled 1σ
  → the tsds/semantic lead on BBH looks real. GradCov-sgd (0.394) is best gradient method.
- **MMLU:** tsds 0.4606 highest, but grad_cov_sgd (0.456), grad_cov_adam (0.455),
  grad_rbf_sgd (0.455) are all **within noise** of it. Effectively a 4-way tie at the top.
- **TyDiQA:** mmd_grad_cov_adam 0.5776 best; **nice 0.5748 within noise** (tied);
  tsds (0.5585) and LESS (≤0.557) fall > 1σ below → GradCov-adam & NICE lead the
  heterogeneous target, both clearly above LESS/TSDS.

## Key readings

1. **Single-seed rankings were partly noise.** Many std are comparable to or
   larger than the gaps they'd need to explain (e.g. less_adam TyDiQA
   ±0.064, grad_cov_sgd TyDiQA ±0.041). The 2-seed table already dissolves
   several apparent single-seed orderings into ties. This validates the whole
   multi-seed exercise.

2. **GradCov is NOT a clean overall winner** once error bars are in:
   - MMLU: tied with tsds/rbf (within noise).
   - TyDiQA: GradCov-adam best but statistically tied with NICE.
   - BBH: loses to tsds (semantic), consistent with prior finding.
   So the honest story is "GradCov is competitive at the top on MMLU/TyDiQA,
   tied with the strongest baselines," not "GradCov wins."

3. **NICE is a strong baseline**, tied with GradCov-adam on TyDiQA (0.575 vs
   0.578) and close on MMLU — its reward-driven policy gradient is genuinely
   competitive on the heterogeneous/multilingual target.

4. **TSDS is the BBH & MMLU leader** (semantic OT selection), reinforcing that
   embedding-space methods win where reasoning/broad-factual coverage matters.

5. **Adam vs SGD is task-dependent for MMD** (as in every prior era, incl.
   unified_with_tsds): BBH favors SGD (grad_cov_sgd 0.394 > adam 0.368), TyDiQA
   favors Adam (grad_cov_adam 0.578 > sgd 0.548), MMLU ~tied. There was never a
   uniform Adam>SGD for MMD; that pattern was mostly a LESS phenomenon.

## Caveat / next
2 seeds → std is a 1-df estimate (crude). A 3rd seed would tighten the intervals
and firm up the MMLU/TyDiQA ties. Seed-2 warmup is already done; its caches +
selections + SFT + eval (~another ~2 days) would give n=3. Decision pending.

Raw: `multiseed_summary.{md,json,csv}`; per-adapter outputs under
`sft_results/{method}_{target}_seed{42,1}` and `eval_results/`.

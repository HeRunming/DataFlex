# Adding TSDS: 9-method × 3-target unified comparison

**Date:** 2026-06-10
TSDS (Zhang et al. 2024, OT-based distribution matching) added as a strong
baseline under the LESS setting. Same pipeline as everything else: bge-base-en-v1.5
embeddings (the SAME ones MMD-Emb-RBF uses), 4-epoch LoRA SFT, unified eval
(TyDiQA with `Answer:` trigger). TSDS solver is a faithful port of the official
ZifanL/TSDS (faiss IVF + KDE near-dup down-weighting + OT heap solver; defaults
α=0.5, C=5.0, σ=0.75, max_K=5000, kde_K=1000); per-target probs → no-replacement
sample of 13,533 (5%).

## Results

| Method | grad/emb | BBH | MMLU | TyDiQA-F1 |
|---|---|---|---|---|
| less_sgd | sgd | 0.3847 | 0.4642 | 0.5433 |
| mmd_grad_rbf_sgd | sgd | 0.3649 | 0.4648 | 0.5387 |
| mmd_grad_cov_sgd | sgd | 0.3873 | **0.4704** | 0.5572 |
| less_adam | adam | 0.3726 | 0.4574 | 0.5418 |
| mmd_grad_rbf_adam | adam | 0.3744 | 0.4542 | 0.5610 |
| mmd_grad_cov_adam | adam | 0.3821 | 0.4527 | **0.5735** |
| mmd_emb_rbf | emb | 0.3916 | 0.4495 | 0.5468 |
| mmd_emb_rbf_stochastic | emb | 0.3930 | 0.4516 | 0.5535 |
| **tsds** | emb (OT) | **0.3944** | 0.4587 | 0.5635 |

### Per-target best
- **BBH:** tsds (0.3944)
- **MMLU:** mmd_grad_cov_sgd (0.4704)
- **TyDiQA:** mmd_grad_cov_adam (0.5735)

## Two clean readings

**1. In embedding space, TSDS (OT) > our MMD-Emb-RBF (greedy MMD), on all 3 targets:**

| Target | tsds | mmd_emb_rbf | mmd_emb_rbf_stoch |
|---|---|---|---|
| BBH | 0.3944 | 0.3916 (−0.003) | 0.3930 (−0.001) |
| MMLU | 0.4587 | 0.4495 (−0.009) | 0.4516 (−0.007) |
| TyDiQA | 0.5635 | 0.5468 (−0.017) | 0.5535 (−0.010) |

With the representation held fixed (same bge embeddings), TSDS's OT solution
beats our greedy MMD coreset. So the embedding-MMD variant is **not** our
selling point — TSDS already does embedding-space matching better.

**2. But the gradient-covariance MMD still beats TSDS on 2 of 3 targets:**

| Target | tsds | best gradient-MMD | winner |
|---|---|---|---|
| BBH | 0.3944 | cov_sgd 0.3873 | tsds |
| MMLU | 0.4587 | cov_sgd **0.4704** | MMD-GradCov |
| TyDiQA | 0.5635 | cov_adam **0.5735** | MMD-GradCov |

The advantage of our method is **not** "MMD vs OT in embedding space" (TSDS wins
there) but **the function space**: matching the target *gradient covariance*
beats both LESS (mean gradient) and TSDS (semantic OT) on MMLU and TyDiQA. On
BBH, semantic/embedding methods (tsds, emb) lead — its CoT reasoning coverage is
better captured semantically than by gradients.

## Takeaway for the story

- Drop embedding-MMD as a headline method (TSDS dominates that lane).
- Position **MMD-GradCov** as the contribution: it beats LESS *and* TSDS on the
  two targets where gradient-distribution structure matters (MMLU broad factual,
  TyDiQA multilingual/heterogeneous), and the win over TSDS shows the gain comes
  from the **training-signal kernel**, not from MMD-vs-OT.
- Caveat: single seed; BBH gaps are within noise. Multi-seed on the GradCov-vs-
  {LESS,TSDS} cells is the priority before any claim.

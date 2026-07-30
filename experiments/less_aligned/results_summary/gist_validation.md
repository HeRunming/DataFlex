# GIST implementation + numerical review (arXiv 2602.18584, official repo github.com/GuanghuiMin/GIST)

Correction to the earlier draft (per code_review_0730): an **official GIST repo exists**; the first
implementation was NOT an exact reproduction. We now keep two scripts:

- `select_gist_jlnorm.py` — **GIST-JL-Norm** (adaptation): unit-normalized 8192-D JL caches, target
  normalized before SVD, plain projector Π=Uᵣᵀ, rank by 95% EVR. Cheap, clearly-labelled ablation.
- `select_gist_faithful.py` — **official algorithm** from `get_gist_gradients.py`: raw target Gram
  `K=G_valG_valᵀ` → eigendecomp → top-k (k=`target_dim`, paper default **150**, chosen offline by
  ~95% EVR) → **isometric whitening projection** `P = G_valᵀ Uₖ Sₖ⁻¹` → project candidates → cosine
  → max over targets → top-k. (The whitening `Sₖ⁻¹` is the "Isometric" in GIST; the paper's Eq-15
  `Π=Uᵀ` text and the official code differ — we follow the official code.)

## Key finding — on OUR normalized caches the fidelity fixes mostly collapse

Selection-only alignment test on STEM80 (candidate=less_output adam, target=stem80, K=13533):

| comparison | Jaccard | what it isolates |
|------------|---------|------------------|
| faithful (rank 62) vs JL-Norm (rank 62) | **1.000** | whitening + Gram vs plain Uᵀ, **same rank** |
| faithful rank 150(→80) vs faithful rank 62 | 0.761 | **rank** effect (the only real lever here) |
| faithful rank 150 vs JL-Norm (r=62) | 0.761 | = rank effect (since whitening is a no-op at equal rank) |
| faithful rank 150 vs DSMC | 0.297 | GIST ≠ DSMC |
| faithful rank 150 vs First-TopK (LESS-like) | 0.466 | closest to relevance top-k |

**Why whitening is a no-op here**: our cached target grads are unit-norm, so the Gram is a
correlation matrix and the `Sₖ⁻¹` whitening rescales axes that the subsequent **cosine** then undoes
— at equal rank the projected-cosine ranking is identical to plain `Uᵣᵀ`. So on normalized caches,
GIST-faithful-math ≡ GIST-JL-Norm; the **only** thing that moves the selection is the **rank k**
(k=80 vs 62 → 24% different) — and, untestable here, whether the target Gram is built from **raw**
vs normalized gradients.

## What remains genuinely un-reproduced (needs a decision)

Exact/byte-faithful GIST needs **raw (un-normalized) LoRA gradients** for the target Gram and for
projecting candidates. Our on-disk caches are unit-normalized and the raw pre-normalization chunks
were deleted, so faithful mode currently runs the official math on normalized inputs
(`exact_reproduction=false` in meta). A truly faithful run requires **re-extracting** raw target
grads (cheap: 80 examples) and ideally raw candidate grads (expensive: 270k). Decision needed:
whether the raw-Gram effect is worth a re-extraction, or whether "official math on the shared
normalized caches + rank sweep" is a fair, controlled baseline for the pilot.

## Numerical checks (JL-Norm, still valid as sanity)

r=62 @95% EVR; basis orthonormality 3.6e-5; rotation-invariance 2e-6; target numerical rank 80=M;
cosine scores in [−0.18, 0.79]. Faithful r150: scores [−0.157, 0.744].

Doc-wording fix (code_review_0730): `G_val` is **numerically full row rank (80)**, and its leading
62 components explain 95% of the spectral energy — *not* a claim that targets are statistically
independent or that discarded directions are "noise".

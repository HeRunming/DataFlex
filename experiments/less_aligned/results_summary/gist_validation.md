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

## Key finding — corrected explanation (per choice_0730_2)

**Correction to an earlier wrong claim.** I previously said "whitening is a no-op because cosine
cancels S⁻¹". That is wrong. The real reason is a pure **algebraic identity**: for the target
matrix `G = UΣVᵀ`, the official projector `P = Gᵀ Uₖ Σₖ⁻¹ = Vₖ` (the top-k **right** singular
vectors) — *always*, independent of normalization or the later cosine. So "official Gram + whitening"
and "plain top-k right singular vectors of the target matrix" are **identical by construction**,
which is why faithful(r=62) ≡ JL-Norm(r=62) gave Jaccard 1.0. It is an identity, not an empirical
coincidence.

What *actually* changes the GIST selection:
1. **rank k** relative to M (see below);
2. **raw vs unit-normalized target gradients** — but *only when k < M*;
3. raw LoRA space vs 8192-D JL space (not testable without full raw re-extraction);
4. paper-vs-official target aggregation details.

### The rank-vs-M scale-invariance (verified empirically)

Rescaling each target row by an arbitrary positive scalar (mimicking raw norms) then running the
full GIST projector, top-13533 Jaccard vs the normalized target:

| k | Jaccard (normalized vs rescaled target) | max score diff |
|---|------------------------------------------|----------------|
| **80 = M** | **1.0000** | 8.7e-6 |
| 64 | 0.9117 | 0.18 |
| 40 | 0.8715 | 0.36 |
| 20 | 0.8538 | 0.47 |

**At k = M the target row-scaling is provably irrelevant** (full row space is recovered); below M it
matters. **Consequence for the pilot**: GIST's official default `target_dim = 150` caps to
`k = min(150, M) = min(150, 64) = 64 = M` for our n=64 draws → **raw-target and normalized-target
GIST select identically**. Re-extracting raw *target* gradients would therefore change nothing at
the official rank; it only matters for the 95%-EVR adaptation (which gives k < M).

## GIST scripts (final)

- `select_gist_faithful.py` — official Gram/eigendecomp/whitening/cosine/max, fixed rank
  (default 150 → capped to M). At the pilot's n=64 this equals the exact official-rank target
  subspace regardless of target normalization (shown above). The remaining gap to byte-exact
  official GIST is only the **space** (our shared 8192-D JL projection vs raw LoRA dim) — reused
  deliberately so every method sits on one identical projection.
- `select_gist_jlnorm.py` — labelled 95%-EVR adaptation (k<M), appendix/ablation only.

**Streaming note**: the official repo streams candidates through the raw-space projector without
saving a raw cache. Our candidates are already the shared 8192-D cache, so no streaming/extra
storage is needed here; the low-storage property is a property of GIST's *own* pipeline, which we
cite rather than reproduce (we hold representation fixed across methods on purpose).

## Numerical checks (still valid)

r=62 @95% EVR (jlnorm); basis orthonormality 3.6e-5; rotation-invariance 2e-6; faithful r150→k=80
scores in [−0.157, 0.744]. Selection overlap: GIST(faithful,r150) vs DSMC 0.297, vs First-TopK
(LESS-like) 0.466 → GIST is a distinct method.

Doc-wording fix (code_review_0730): `G_val` is **numerically full row rank (80)**, and its leading
62 components explain 95% of the spectral energy — *not* a claim that targets are statistically
independent or that discarded directions are "noise".

# GIST implementation + numerical review (arXiv 2602.18584 v2)

`scripts/select_gist.py`. Baseline for the external-validity pilot; the closest conceptual
competitor to DSMC (both recover a target-relevant gradient subspace). Full paper text extracted to
`reviews/gist_paper_extracted.txt`.

## Algorithm implemented (paper Eqs 14–17)

1. **Task subspace** — target gradient matrix `G_val = [g¹…g^M] ∈ R^{d×M}` (column per target
   example); SVD `G_val = UΣVᵀ` (Eq 14). Effective rank r = smallest r with cumulative explained
   variance `Σσ² ≥ τ` (paper τ=0.95). Projector `Π = U_rᵀ` = top-r left singular vectors (Eq 15).
2. **Score** — `Sim(z_i, z_val^j) = cos(Π g_i, Π g_val^j)` (Eq 16).
3. **Aggregate** — `FinalScore(z_i) = max_j Sim(z_i, z_val^j)` (Eq 17, max-relevance like LESS); top-k.

## Adaptations to our caches (all documented, controlled)

- **Projection**: paper SVDs raw d-dim LoRA gradients; we only have the shared **8192-dim
  TRAK/JL-projected** grads (seed 123) used by LESS and DSMC. SVD of projected target grads recovers
  the JL image of the same task subspace; reusing the identical projected caches as every other
  method is the apples-to-apples choice and keeps candidate+target in one common projection (a paper
  requirement). Paper itself uses r≈150 for MMLU at full dim; we get r=62 at 8192-dim (see below).
- **Normalization**: our caches are unit-normalized per example; we SVD the unit-normalized target
  grads so each target contributes equally (consistent with DSMC). Eq-16 cosine makes candidate
  scoring scale-invariant regardless.
- **Optimizer protocol**: Adam-candidate / SGD-target is a CLI choice via cache paths (not
  hardcoded), so GIST rides the same LESS-aligned protocol as the other methods; recorded in meta.

## Numerical checks on STEM80 (candidate=less_output adam, target=stem80, K=13533)

| check | result | expected |
|-------|--------|----------|
| rank r at 95% EVR | **62** (cum_evr@62 = 0.9511) | low-rank ✓ (paper: MMLU rank≈150 at full d, precipitous decay) |
| basis orthonormality `‖UᵣᵀUᵣ−I‖∞` | 3.6e-5 | ≈0 ✓ |
| rotation-invariance `max|score(Uᵣ)−score(UᵣQ)|`, Q random orthonormal | 2.0e-6 | ≈0 ✓ (score depends on subspace, not basis) |
| numerical rank of `G_val` | 80 = M | all targets independent → r=62 genuinely filters noise ✓ |
| score range (cosine) | [−0.175, 0.791], mean 0.261 | valid cosines ✓ |

Selection overlap (Jaccard) vs other selectors — GIST is a **distinct** method, not a re-DSMC:
- GIST vs DSMC = 0.289
- GIST vs Second-TopK = 0.297
- GIST vs First-TopK (LESS-like) = 0.461 (closest, as expected: both are relevance/alignment top-k)

## Open decisions for the pilot (flagged, not hardcoded)

- **Rank rule**: 95% EVR → r=62 here. Could also fix r or sweep; paper uses per-task fixed ranks
  (MMLU 150 at full 4096·… dim). Default 95% EVR is faithful and self-tuning; `--rank` overrides.
- **Efficiency claim**: paper reports GIST at 0.29% storage / 25% compute of LESS. Our version reuses
  the 8192-dim caches (so storage isn't reduced here) — for a fair efficiency comparison we would
  need the low-dim variant; the pilot focuses on **accuracy** parity/ordering, with efficiency noted
  qualitatively.
- Still to build for the pilot: **true round-robin** (First-RR/Second-RR) and **Random-K /
  Random-K-LengthMatched** (per choice_0730).

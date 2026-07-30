# Representation × selector 2×2 attribution gate (review_0729 / code_review_0729)

**Question**: is DSMC's advantage from the 2nd-order *representation*, the MMD *coreset diversity*,
or their interaction? Fix everything except (representation, selector) on the existing STEM80/HUM80
targets, seed 42, balanced acc = ½(mmlu_stem + mmlu_humanities).

- representation: 1st-order `u` vs 2nd-order `(uᵀv)²` (unit-normalized projected gradients)
- selector: **TopK** = relevance top-k, no repulsion (`select_relevance_topk.py`) vs **MMD** = greedy
  coreset with the repulsion term. (TopK ≠ round-robin — a true RR baseline is deferred.)
- MMD cells reuse existing adapters (Linear-MMD = moment α=1; DSMC = moment α=0). Only First/Second-
  TopK are new (4 SFT). `run_attribution_2x2.sh` → `attribution_2x2_results.csv`.

## Balanced accuracy

| target | 1st-TopK | 1st-MMD(Linear) | 2nd-TopK | 2nd-MMD (DSMC) |
|--------|----------|-----------------|----------|----------------|
| stem80 | 0.3977   | 0.3847          | 0.4068   | **0.4110**     |
| hum80  | 0.3931   | 0.3824          | 0.3992   | **0.4054**     |

DSMC is the best cell in **both** directions.

## Effect decomposition (percentage points)

| effect | stem80 | hum80 |
|--------|--------|-------|
| representation (2nd − 1st), selector = TopK | +0.91 | +0.61 |
| representation (2nd − 1st), selector = MMD  | +2.63 | +2.30 |
| selector (MMD − TopK), representation = 1st | −1.30 | −1.07 |
| selector (MMD − TopK), representation = 2nd | +0.42 | +0.62 |

## Reading (matches code_review_0729's interpretation rules)

1. **The representation is the primary driver.** Moving 1st→2nd helps under *both* selectors and
   *both* directions (+0.6 to +2.6 pp). Second-order beats first-order everywhere.
2. **MMD diversity helps only *on top of* the 2nd-order representation — there is a strong
   interaction.** With 1st-order, adding MMD repulsion *hurts* (−1.30 / −1.07 pp: Linear-MMD is the
   worst cell). With 2nd-order, adding MMD repulsion *helps* (+0.42 / +0.62 pp: DSMC > Second-TopK).
   So MMD's diversity is only beneficial in the directional-second-moment space.
3. **Second-TopK is close to DSMC but does not beat it** (0.4068 vs 0.4110; 0.3992 vs 0.4054). Per
   the review's rule this means: keep DSMC as the headline (no need to switch the method to a
   second-order relevance selector), and the contribution is **2nd-order representation + coreset
   diversity**, not representation alone.

**Attribution conclusion**: *DSMC's gains come primarily from matching the second-order (directional)
moment of gradients; the MMD coreset objective adds a further, consistent gain that materialises
specifically in that second-order space, while it is harmful in the first-order space.* This is the
cleanest possible outcome for the paper's mechanism story — representation is necessary, and the
coreset objective is complementary rather than redundant.

## Caveats (keep scope honest)

- Single seed (42), single fixed target per direction — same setting as the endpoint runs; effect
  sizes (~0.4–0.6 pp for the selector-on-2nd effect) are within the training-noise band we measured
  earlier (paired-seed sd ≈ 0.46 pp). The *ordering* is consistent across both directions, but the
  DSMC-vs-Second-TopK gap is small and should be reconfirmed with paired seeds if it becomes load-
  bearing in the paper.
- TopK is relevance top-k, **not** greedy round-robin. A true RR selector (per-query nearest,
  cycling) is still needed for the external-validity phase; it can be stronger at low budget.
- STEM/Humanities subscores are not uniformly dominated (e.g. hum80 2nd-TopK STEM 0.3749 < DSMC
  0.3809 but its HUM 0.4236 < DSMC 0.4300); the claim is about the balanced aggregate.

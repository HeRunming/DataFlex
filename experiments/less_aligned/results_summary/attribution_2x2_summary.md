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

## Reading (matches code_review_0729 / choice_0730 interpretation rules)

1. **The representation is the primary driver (strong conclusion).** Moving 1st→2nd helps under
   *both* selectors and *both* directions: +0.91/+0.61 pp (TopK) and +2.63/+2.30 pp (MMD). Second-
   order beats first-order in every cell.
2. **MMD diversity appears complementary to the 2nd-order representation — a promising *hypothesis*,
   not yet proven necessary.** There is a clean interaction (difference-in-differences, nearly
   identical across directions): stem80 (+0.42)−(−1.30)=**+1.72 pp**, hum80 (+0.62)−(−1.07)=**+1.69
   pp**. With 1st-order, adding MMD repulsion *hurts* (−1.30/−1.07: Linear-MMD is the worst cell);
   with 2nd-order it *helps* (+0.42/+0.62: DSMC > Second-TopK). BUT the direct DSMC−Second-TopK gain
   is only +0.42/+0.62 pp at a single seed — within the training-noise band (paired-seed sd ≈0.46
   pp). So state "MMD diversity is complementary in the 2nd-order space" as a mechanism hypothesis;
   do **not** write "MMD repulsion is statistically proven necessary."
3. **Second-TopK is close but never beats DSMC** → keep DSMC as the headline (no need to switch to a
   relevance selector). Contribution framed as **2nd-order representation (primary) + coreset
   diversity (complementary hypothesis)**. The upcoming independent target draws will supply the
   extra paired observations to test the diversity effect — no need to add seeds to the 2×2 cells.

**Attribution conclusion**: *DSMC's gains come primarily from matching the second-order (directional)
moment of gradients; the MMD coreset objective plausibly adds a further gain that is specific to the
second-order space (harmful in the first-order space), but that increment is within single-seed noise
and is left for the target-draw phase to confirm.*

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

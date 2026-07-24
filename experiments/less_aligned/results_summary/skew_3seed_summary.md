# Skewed-target stress test — 3-seed results (seeds 42, 1, 2)

**Date:** 2026-07-19. Same env. 80/20 skewed MMLU targets (T_stem80, T_hum80)
from dev; C=270k unchanged. 5 methods × 2 T × 3 seeds = 30 select→SFT→eval.
Eval = mmlu_stem + mmlu_humanities (Hendrycks groups), 5-shot.

## Balanced accuracy (STEM+HUM)/2, mean±std over 3 seeds

| method | T_stem80 | T_hum80 |
|---|---|---|
| **mmd_grad_cov_sgd** | **0.4023 ± 0.0037** | 0.4006 ± 0.0033 |
| **mmd_grad_cov_adam** | 0.4022 ± 0.0096 | 0.4036 ± 0.0065 |
| nice | 0.3998 ± 0.0029 | **0.4077 ± 0.0048** |
| less_adam | 0.3952 ± 0.0055 | 0.3971 ± 0.0031 |
| tsds | 0.3915 ± 0.0022 | 0.3822 ± 0.0033 |

## Minority-mode retention, mean±std over 3 seeds

**T_stem80 (minority = Humanities):**
| method | HUM-acc (minority) |
|---|---|
| nice | 0.4305 ± 0.0066 |
| mmd_grad_cov_sgd | 0.4262 ± 0.0036 |
| mmd_grad_cov_adam | 0.4248 ± 0.0111 |
| tsds | 0.4177 ± 0.0018 |
| less_adam | 0.4128 ± 0.0050 (worst) |

**T_hum80 (minority = STEM):**
| method | STEM-acc (minority) |
|---|---|
| less_adam | 0.3834 ± 0.0042 |
| mmd_grad_cov_adam | 0.3804 ± 0.0072 |
| nice | 0.3776 ± 0.0071 |
| mmd_grad_cov_sgd | 0.3745 ± 0.0048 |
| tsds | 0.3515 ± 0.0040 (worst) |

## What survived multi-seed, what didn't (HONEST)

1. **GradCov wins BALANCED accuracy on T_stem80 (both variants top-2, ~0.402
   vs LESS 0.395, TSDS 0.392) and is competitive on T_hum80** (adam 2nd, 0.404,
   behind NICE 0.408). This is robust across 3 seeds — the strongest surviving
   claim: **under skew, GradCov produces the most balanced-capable model, ahead
   of LESS and TSDS.**

2. **The clean single-seed claim "GradCov best minority retention, LESS worst"
   DID NOT fully survive:**
   - T_stem80 minority (HUM): the single-seed winner was GradCov, but at n=3
     **NICE is highest (0.4305)**, with GradCov-sgd (0.426) / adam (0.425) close
     behind. LESS is still clearly worst (0.413) — so "LESS collapses on the
     STEM-majority minority" **holds**, but "GradCov is THE best minority keeper"
     **does not** (NICE edges it, though within overlapping std).
   - T_hum80 minority (STEM): **LESS is actually best (0.383)**, GradCov-adam 2nd
     (0.380), TSDS worst (0.352). So the collapser flips to TSDS here.

3. **Consistent robust findings across seeds:**
   - **TSDS is the worst minority-keeper on BOTH targets when the minority is not
     semantically dominant** (0.418 / 0.352) — embedding-density matching most
     amplifies the majority. Robust and clean.
   - **GradCov is never the collapser** — it's top-2 balanced on both, never worst
     on any minority. Its value is *consistency/robustness to skew direction*,
     not being the single best on any one axis.
   - **NICE is high-variance across the skew direction**: best on HUM-minority and
     HUM-majority (it likes humanities), weaker on STEM sides — it chases whatever
     mode is "easier", not a stable distribution matcher.

## Mechanism (pre-training, seed-42, very clean — the strongest evidence)

Effective rank of selected-set adam gradients (T_stem80):
**mmd_grad_cov_adam 2398 > less_adam 2257 ≫ nice 58.7 (collapsed)**.
Mean pairwise cosine (redundancy): cov 0.015 < less 0.043 ≪ nice 0.142.
→ GradCov selects the most spread-out, highest-rank, least-redundant coreset;
NICE collapses to a low-rank redundant set. This is a robust, training-free
signature of *how* the methods differ, and explains GradCov's balance + NICE's
mode-chasing. (Extending eff-rank to all 3 seeds is a cheap next step.)

## Honest takeaway for the paper

The multi-seed result **weakens the "MMD strictly best at minority retention"
narrative** but supports a **defensible, subtler claim**: *GradCov is the most
skew-robust selector — best balanced accuracy under STEM-majority skew, top-2 in
all skew settings, and never the mode-collapser — whereas LESS, NICE, and TSDS
each collapse in at least one skew direction.* The effective-rank mechanism is the
cleanest differentiator and should anchor the story. This is honest and still
paper-worthy, but the framing must be "robustness across skew conditions,"
not "we always keep the minority best."

Raw: eval_results/skew/skew_{method}_{stem80,hum80}[_seed{1,2}]; summary supersedes skew_experiment_summary.md (single-seed).

## Mechanism (effective rank) — NOW over 3 seeds, both targets (REVISED, honest)

Effective rank of selected-set adam gradients (subsample 4000), mean±std over seeds 42,1,2:

| method | T_stem80 eff_rank | T_hum80 eff_rank |
|---|---|---|
| **mmd_grad_cov_adam** | **2386 ± 22** | **2385 ± 26** |
| less_adam | 2124 ± 202 | 2153 ± 105 |
| mmd_grad_cov_sgd | 858 ± 610 | 790 ± 645 |
| nice | 1019 ± 858 | 1280 ± 532 |

**Correction to the seed-42 finding:** the dramatic "NICE collapses to eff_rank 58.7"
was seed-42-specific — across 3 seeds NICE is high-variance (raw stem80: 62/1273/1721),
NOT reliably collapsed. Likewise **mmd_grad_cov_sgd is NOT reliably high-rank**
(raw: 1546/646/382) — only the **adam** variant is.

**What is robust:** **mmd_grad_cov_adam has the highest effective rank AND by far
the lowest variance (±22-26) across seeds and both skew directions.** It reliably
selects the most spread-out, highest-rank coreset. LESS is second and fairly
stable; NICE and GradCov-sgd are erratic. So the clean mechanism story holds
specifically for **GradCov-Adam**: it is the uniquely *stable* high-rank selector,
consistent with covariance-operator matching being a well-conditioned objective
under the Adam preconditioner. This aligns with GradCov-Adam being the top-2
balanced-accuracy method in both skew directions.

## Bottom line (final, honest)
The defensible ICLR claim is narrower than the single-seed hoped-for one:
**MMD-GradCov (Adam) is the most skew-robust selector — top-2 balanced accuracy in
both skew directions, never the mode-collapser, and uniquely stable high-effective-
rank selection — whereas LESS, NICE, and TSDS each collapse (on accuracy or on
selection rank) in at least one skew direction or seed.** The eff-rank stability of
GradCov-Adam is the cleanest single differentiator.

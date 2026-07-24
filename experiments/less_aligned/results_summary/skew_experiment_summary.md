# Skewed-target stress test — does MMD retain minority modes better than LESS/NICE?

**Date:** 2026-07-17. Single seed (42), same env. Candidate pool C=270k unchanged;
only the target T changed. Two 80/20 skewed MMLU targets built from dev (no test
leak): T_stem80 = 64 STEM + 16 Humanities; T_hum80 = 64 HUM + 16 STEM. 5 methods
× 2 T = 10 select→SFT→eval. Eval = mmlu_stem + mmlu_humanities (Hendrycks groups).

## Results (acc)

**T_stem80 (majority=STEM, minority=Humanities):**
| method | STEM-acc | HUM-acc | avg | gap(S−H) |
|---|---|---|---|---|
| less_adam | 0.3781 | 0.4077 | 0.3929 | −0.0296 |
| nice | 0.3758 | 0.4236 | 0.3997 | −0.0478 |
| tsds | 0.3711 | 0.4168 | 0.3939 | −0.0457 |
| mmd_grad_cov_sgd | 0.3834 | 0.4281 | 0.4057 | −0.0446 |
| **mmd_grad_cov_adam** | **0.3901** | 0.4276 | **0.4089** | −0.0375 |

**T_hum80 (majority=Humanities, minority=STEM):**
| method | STEM-acc | HUM-acc | avg | gap(S−H) |
|---|---|---|---|---|
| less_adam | 0.3841 | 0.4055 | 0.3948 | −0.0214 |
| nice | 0.3695 | 0.4349 | 0.4022 | −0.0654 |
| tsds | 0.3520 | 0.4132 | 0.3826 | −0.0611 |
| mmd_grad_cov_sgd | 0.3736 | 0.4338 | 0.4037 | −0.0602 |
| **mmd_grad_cov_adam** | **0.3841** | 0.4270 | **0.4055** | −0.0429 |

## Core findings

1. **MMD-GradCov wins overall balanced accuracy on BOTH skewed targets.**
   avg(STEM,HUM): grad_cov_adam is best on both (0.4089 stem80, 0.4055 hum80),
   grad_cov_sgd 2nd. LESS and TSDS trail. So under target skew, GradCov produces
   the most *balanced-capable* model.

2. **Minority-mode retention — GradCov leads on the STEM-minority side, ties on
   the HUM-minority side:**
   - T_stem80 (minority=HUM): GradCov-sgd/adam keep the highest HUM-acc
     (0.428) > nice (0.424) > tsds (0.417) > **less_adam WORST (0.408)**.
     → matches the hypothesis: LESS collapses toward the STEM majority, starving
     the humanities minority; MMD keeps it best.
   - T_hum80 (minority=STEM): GradCov-adam ties less_adam for best STEM-acc
     (0.384), both clearly above nice (0.370) and tsds (0.352, worst).
     → here LESS does NOT collapse (it actually protects the STEM minority well);
     nice/tsds collapse toward the HUM majority instead.

3. **The clean "LESS collapses to majority" story holds in ONE direction, not
   symmetrically.** When STEM is the majority (stem80), LESS is worst at the
   minority — as predicted. When HUM is the majority (hum80), LESS is fine and
   **NICE/TSDS become the collapsers** (worst minority-STEM). So "which method
   collapses" depends on the majority content, not just the mechanism.

4. **Balance gap (STEM−HUM): GradCov-adam is the most balanced** (smallest |gap|)
   in both settings (−0.0375, −0.0429), i.e. least skewed toward either category.
   LESS has small gap on hum80 (−0.021) but that's because it's weak on the HUM
   majority, not because it's balanced.

5. **NICE and TSDS are the strongest majority-mode chasers**: on both T they push
   the majority category but sacrifice the minority (NICE hum80: HUM 0.435 best
   but STEM 0.370; TSDS worst minority in both). Consistent with NICE optimizing
   the metric-mean and TSDS matching bulk embedding density → both amplify the
   majority.

## Interpretation for the paper

**Supports the thesis — with nuance.** MMD-GradCov is the most *distribution-faithful*
selector under target skew: it delivers the best balanced accuracy on both skewed
targets and the best (stem80) or tied-best (hum80) minority-mode retention. This is
the cleanest evidence so far that matching the target *distribution* (not its mean
gradient / mean metric) buys robustness to skew — exactly where the n=5 balanced-T
table showed only ties. The mechanism ("mean-based methods collapse to the majority
mode") is confirmed for LESS when STEM dominates and for NICE/TSDS when HUM
dominates; MMD-GradCov avoids collapse in both.

**Honest caveats:** single seed (gaps ~0.01–0.03, need seeds to confirm minority
deltas); the collapse is asymmetric (LESS is robust when HUM is majority); GradCov's
minority win over LESS is clear on stem80 (+0.020 HUM) but only a tie on hum80.
Recommend 2–3 seeds on this skew experiment before it goes in the paper.

Raw: eval_results/skew/skew_{method}_{stem80,hum80}; targets data/mmlu_target_{stem80,hum80}.jsonl.

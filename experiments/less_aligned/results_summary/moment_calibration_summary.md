# Moment-MMD: scale calibration & the joint-vs-second-order question

Setting: `T_stem80` (80/20 STEM/Humanities MMLU dev target), 270k Tulu pool, 5% budget
(13,533), Llama-2-7B LoRA r128 α512, 4ep, eval = mean(mmlu_stem, mmlu_humanities) 5-shot.
Moment kernel on L2-normalized projected gradients:
`k(u,v) = α·(1+⟨u,v⟩)/2 + (1−α)·⟨u,v⟩²` — α=0 = GradCov (2nd-order), α=1 = linear-MMD (1st-order).

## 1. The raw α sweep was mis-calibrated

Diagnostic (`diag_moment_components.py`): the first-order greedy marginal has ~14.5× the
cross-candidate spread of the second-order one (σ_lin=0.00581 vs σ_quad=0.00040). So in
`k_α` the first-order term dominates the ranking well before α reaches the middle of [0,1].
This explains why the earlier **coarse grid** `α∈{0.25,0.5,0.75,1}` never explored a balanced
joint — every tested α>0 was already first-order-dominated. (A very small α need not be
instantly dominated; the point is the tested grid was, not that any α>0 strictly is.)

## 2. Random-MMD normalization fixes the *level* but not the *marginal spread*

Implemented scale-normalized `k̃_β = β/s1·k_lin + (1−β)/s2·k_quad`, with
`s_j = E_random[D_j]` over B=256 same-budget random subsets (`select_moment_normalized.py`).
This equalized the two full MMDs (s1=0.222, s2=0.168, ratio 1.32×, no longer 14×) **but** the
normalized greedy-*marginal* std ratio was still 10.98× — because E_random[D_j] is a level, not
the dispersion that argmax keys on. Consequently β collapsed to the linear endpoint almost
immediately (Jaccard-vs-linear 0.81 already at β=0.1). β is a broken parameterization.

## 3. λ-reparameterization (the clean knob)

Key fact (choice_0725.md): any fixed per-component normalization is just a reparameterization
— a global positive rescale never changes greedy argmax. So drop s1/s2 and use the direct
coefficient ratio (`select_moment_lambda.py`):

    k_λ(u,v) = ⟨u,v⟩² + λ·(1+⟨u,v⟩)/2 ,   self_k = 1+λ

λ=0 = GradCov; λ→∞ = linear. Marginal-balanced point λ ≈ σ_quad/σ_lin ≈ 0.069.

λ sweep on T_stem80 (`diag_moment_lambda.py`), selection-only:

| λ     | D1      | D2      | Jac vs GradCov | Jac vs linear | eff_rank |
|-------|---------|---------|----------------|---------------|----------|
| 0.0   | 0.17194 | 0.14557 | 1.000          | 0.426         | 2398     |
| 0.005 | 0.17010 | 0.14558 | 0.902          | 0.474         | 2380     |
| 0.01  | 0.16861 | 0.14559 | 0.830          | 0.516         | 2385     |
| 0.02  | 0.16598 | 0.14563 | 0.715          | 0.595         | 2407     |
| 0.04  | 0.16271 | 0.14572 | 0.593          | 0.701         | 2425     |
| 0.07  | 0.16031 | 0.14585 | 0.519          | 0.783         | 2440     |
| 0.10  | 0.15908 | 0.14595 | 0.486          | 0.826         | 2463     |
| 0.20  | 0.15780 | 0.14612 | 0.452          | 0.886         | 2465     |

λ interpolates **smoothly** (unlike β). Increasing λ traces a smooth trade-off rather than a
strict Pareto improvement: it substantially reduces D1, incurs only a small increase in D2, and
generally raises effective rank for λ≥0.02 (note eff_rank actually *dips* at λ=0.005, 0.01:
2398→2380→2385, so it is not monotone and not a Pareto improvement everywhere). Two training
candidates: λ=0.02 (GradCov-preserving) and λ=0.07 (marginal-balanced).

## 4. Phase-2 downstream (single seed 42)

| method        | mmlu_stem | mmlu_hum | balanced | Δ vs GradCov |
|---------------|-----------|----------|----------|--------------|
| GradCov (λ=0) | 0.3920    | 0.4300   | 0.4110   | —            |
| joint λ=0.02  | 0.3784    | 0.4251   | 0.4017   | −0.93 pt     |
| joint λ=0.07  | 0.3739    | 0.4117   | 0.3928   | −1.82 pt     |
| linear (α=1)  | 0.3619    | 0.4074   | 0.3847   | −2.63 pt     |

**Monotone ladder**: balanced accuracy falls monotonically with first-order weight
(0.411 > 0.402 > 0.393 > 0.385); STEM and Humanities degrade together (not a minority tradeoff).
Notably the favorable first-moment/second-moment trade-off in the selection diagnostics did
**not** translate into better downstream performance.

Careful statement (single realization; do not overclaim): *Adding a signed first-moment
component improves the measured first-moment discrepancy (D1) while preserving the aggregate
second-moment discrepancy (D2), yet is associated with lower downstream accuracy in this target
regime.* D2 being "≈unchanged" does not mean all second-order structure is controlled — the two
subsets still differ in ~28% of samples.

## 5. Paired-seed confirmation (choice_0726.md) — pivot confirmed

GradCov (λ=0) vs best joint (λ=0.02), **fixed** seed-42 selected subsets, vary only the SFT
seed ∈ {42,1,2}. `run_moment_lambda_seeds.sh`.

| seed | GradCov bal | joint λ=0.02 bal | Δ = joint − GradCov |
|------|-------------|------------------|---------------------|
| 42   | 0.4110      | 0.4017           | −0.93 pt            |
| 1    | 0.4092      | 0.4015           | −0.77 pt            |
| 2    | 0.4057      | 0.4051           | −0.07 pt            |
| mean | 0.4086      | 0.4028           | **−0.59 pt**        |

All three paired diffs are negative; the joint never outperforms GradCov. **But seed 2 is
essentially a tie** (−0.07) and n=3 is small (sd of diffs ≈0.46 pt, se ≈0.26 pt), so a
conventional CI still covers zero. Defensible claim (NOT stronger):

> Pure directional second-moment is at least as strong as the best calibrated joint on average,
> and the joint never outperforms it across three paired seeds in this target regime.

Do NOT claim "first-order info is necessarily harmful", "any nonzero λ strictly hurts",
"second-order is significantly better", or "airtight". This result is enough to **stop a
low-return branch** (freeze λ=0 as default), not to carry the paper's main statistical evidence.

## 6. T_hum80 mirror (skew-direction invariance) — joint branch closed

Same offline greedy code paths as stem80; GradCov (λ=0) & joint (λ=0.02) × seeds {42,1,2},
linear (α=1) × seed 42. Provenance manifest: `eval_results/skew/hum80_mirror_manifest.json`
(candidate+target both from warmup_seed42/ckpt-1692, adapter sha256 44d9c58…, optimizer 10a0169…).
`run_hum80_mirror.sh` → `hum80_mirror_results.csv`.

| method       | seed 42 | seed 1 | seed 2 | mean balanced |
|--------------|---------|--------|--------|---------------|
| GradCov (λ=0)| 0.4054  | 0.4074 | 0.4080 | **0.4069**    |
| joint λ=0.02 | 0.4028  | 0.4012 | 0.4007 | **0.4016**    |
| linear (α=1) | 0.3824  | —      | —      | 0.3824        |

Paired Δ (joint − GradCov) by seed: −0.26 / −0.62 / −0.73 pt, **mean −0.54, all three negative**.
Linear is −2.30 pt vs GradCov (seed 42). The ordering is **identical to stem80**:

| target  | GradCov | joint λ=0.02 | linear | mean Δ (joint−GC) |
|---------|---------|--------------|--------|-------------------|
| stem80  | 0.4086  | 0.4028       | 0.3847 | −0.59 pt          |
| hum80   | 0.4069  | 0.4016       | 0.3824 | −0.54 pt          |

**Conclusion**: pure directional second-moment ≥ the best calibrated joint > linear under **both**
skew directions — the earlier stem80 result is **not a STEM-target artifact**. The joint branch is
closed. Careful wording still holds (n=3 per direction, small effect): "the joint does not stably
improve on the second-order endpoint across two skew directions", not "first-order is harmful".
(Note: on hum80 the joint's loss is monotone across seeds too — no tie this time, unlike stem80 seed 2.)

## 7. Decision & next (per review_0727.md)

**Pivot confirmed. Freeze λ_default = 0. No further λ sweeps or new static normalizations.**

Headline → **Directional Second-Moment Coresets** (candidate names DSMC / GDMC / DM-MMD;
"GradCov" is a misnomer since gradients are per-sample L2-normalized, so it matches
M_P = E_{u~P}[u uᵀ] with u = Πg/‖Πg‖, not raw covariance).

Immediate: slim **T_hum80 mirror** — GradCov (λ=0) and joint (λ=0.02) × seeds {42,1,2}, plus
linear (α=1) × seed 42 only (7 SFT; add linear seeds 1,2 only if it lands near GradCov), re-selected
via the SAME offline greedy (the existing hum80 GradCov used the online selector → different
code path → not an exact-config reuse). Tests skew-direction invariance:
  - GradCov ≥ λ=.02 > linear on hum80 too → close the joint branch with confidence;
  - joint clearly wins on hum80 → value of 1st-order is target-geometry-dependent (conditional method);
  - GradCov ≈ joint → "joint does not stably improve the endpoint" (not "1st-order harmful").

Then shift compute to **second-order external validity** (NOT more Moment-MMD tuning):
  - Statistical unit = independent **target draws**, not just SFT seeds. Min design: 2 skew
    directions × 5 draws × 1 shared seed; methods = DSMC, LESS, NICE, Random, gradient RR, GIST;
    then +2 seeds on one representative draw per direction for training variance.
  - **Representation × selector 2×2 decoupling**: {signed 1st-order, 2nd-order (uᵀv)²} ×
    {relevance/round-robin, MMD coreset} + ablations (target-subspace scoring, 2nd-order top-k,
    2nd-order MMD w/o repulsion) — to attribute gains to representation vs MMD diversity.
  - Keep Random + token-matched Random; validate a lower (1%) budget.
  - **GIST (arXiv 2602.18584)** is the must-address related work (target subspace via SVD +
    alignment scoring). Differentiator: we match the 2nd moment of unit gradient *directions* and
    control target coverage + candidate redundancy jointly via a coreset objective. Add mechanism
    contrasts: subspace-recovery error ‖P̂_S − P̂_T‖_F / principal angles, and skew stability
    (selected-set Jaccard, subspace variance, eff_rank, downstream variance across target draws).

Moment-MMD stays in the paper as the unifying family k_λ(u,v)=(uᵀv)²+λ(1+uᵀv)/2 with the
one-line result "the 2nd-order endpoint is consistently ≥ the best calibrated joint on the
studied skewed target"; the full scale diagnosis / Jaccard / D1,D2 / sweep go to the appendix.

Scripts: `select_moment_normalized.py`, `select_moment_lambda.py`, `diag_moment_normalized.py`,
`diag_moment_lambda.py`, `diag_moment_components.py`, `run_moment_lambda.sh`,
`run_moment_lambda_seeds.sh`.

# Moment-MMD: scale calibration & the joint-vs-second-order question

Setting: `T_stem80` (80/20 STEM/Humanities MMLU dev target), 270k Tulu pool, 5% budget
(13,533), Llama-2-7B LoRA r128 α512, 4ep, eval = mean(mmlu_stem, mmlu_humanities) 5-shot.
Moment kernel on L2-normalized projected gradients:
`k(u,v) = α·(1+⟨u,v⟩)/2 + (1−α)·⟨u,v⟩²` — α=0 = GradCov (2nd-order), α=1 = linear-MMD (1st-order).

## 1. The raw α sweep was mis-calibrated

Diagnostic (`diag_moment_components.py`): the first-order greedy marginal has ~14.5× the
cross-candidate spread of the second-order one (σ_lin=0.00581 vs σ_quad=0.00040). So in
`k_α`, any α>0 lets the first-order term hijack the ranking. This explains why the earlier
`α∈{0,.25,.5,.75,1}` sweep never explored a balanced joint — every α>0 was already
first-order-dominated.

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

λ interpolates **smoothly** (unlike β). Every interior λ is a Pareto improvement in selection
geometry: D1 falls, D2 nearly flat (≤0.4%), eff_rank rises. Two training candidates:
λ=0.02 (GradCov-preserving) and λ=0.07 (marginal-balanced).

## 4. Phase-2 downstream (single seed 42)

| method        | mmlu_stem | mmlu_hum | balanced | Δ vs GradCov |
|---------------|-----------|----------|----------|--------------|
| GradCov (λ=0) | 0.3920    | 0.4300   | 0.4110   | —            |
| joint λ=0.02  | 0.3784    | 0.4251   | 0.4017   | −0.93 pt     |
| joint λ=0.07  | 0.3739    | 0.4117   | 0.3928   | −1.82 pt     |
| linear (α=1)  | 0.3619    | 0.4074   | 0.3847   | −2.63 pt     |

**Monotone ladder**: balanced accuracy falls monotonically with first-order weight
(0.411 > 0.402 > 0.393 > 0.385); STEM and Humanities degrade together (not a minority tradeoff).
Notably the selection-geometry Pareto improvement did **not** translate downstream.

Careful statement (single realization; do not overclaim): *Adding a signed first-moment
component improves the measured first-moment discrepancy (D1) while preserving the aggregate
second-moment discrepancy (D2), yet is associated with lower downstream accuracy in this target
regime.* D2 being "≈unchanged" does not mean all second-order structure is controlled — the two
subsets still differ in ~28% of samples.

## 5. Status / next

Per choice_0726.md: running **paired-seed confirmation** (seeds 1,2 for GradCov λ=0 vs best
joint λ=0.02, fixed subsets, vary only SFT seed) → `run_moment_lambda_seeds.sh`. If all three
paired diffs (s∈{42,1,2}) are negative, pivot the headline to **Directional Second-Moment
Coresets for Robust Targeted Instruction Tuning**, with Moment-MMD retained as the unifying
framework + fully-diagnosed negative ablation. Then a slim `T_hum80` mirror (λ=0, 0.02, linear)
for direction-invariance, and refocus compute on 2nd-order external validity (target draws,
budgets, selector/representation decoupling) rather than further λ tuning.

Scripts: `select_moment_normalized.py`, `select_moment_lambda.py`, `diag_moment_normalized.py`,
`diag_moment_lambda.py`, `diag_moment_components.py`, `run_moment_lambda.sh`,
`run_moment_lambda_seeds.sh`.

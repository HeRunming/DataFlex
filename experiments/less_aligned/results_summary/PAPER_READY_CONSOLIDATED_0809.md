# Directional Second-Moment Coresets for Targeted Instruction Selection
## Paper-ready consolidated evidence, story audit, and ICLR readiness assessment

**Status:** internal consolidation after completion of the 1% equal-step sensitivity

**Snapshot date:** 2026-08-09

**Scope:** this document consolidates the final usable evidence in
`experiments/less_aligned/results_summary/`, distinguishes paper-grade results from historical or
superseded results, and audits the remaining gaps before an ICLR submission.

---

# 1. Executive summary

## 1.1 The strongest defensible story

The current evidence supports the following claim:

> **Directional second-moment matching substantially improves robustness among query-targeted
> instruction selectors under biased finite query sets. However, optimizing target-gradient
> geometry is not sufficient for downstream utility: DSMC matches both the observed query and a
> balanced held-out validation reference substantially better than Random, yet it does not
> outperform Random downstream.**

This is a stronger and more credible paper story than “DSMC is a universally better data selector.”
The positive contribution is a controlled representation result: among the target-aware methods
tested, the directional second moment is consistently the most robust representation. The negative
but scientifically important result is that better target-gradient matching does not necessarily
produce a better instruction-tuning subset.

## 1.2 Current ICLR readiness

The project is **paper-worthy but not yet a solid general ICLR submission**.

Strengths:

- a clean RKHS/MMD formulation;
- calibrated first- versus second-order method development;
- a representation × selector attribution experiment;
- two skew directions;
- ten globally disjoint target sets;
- two nested budgets;
- Random and length-histogram-matched Random controls;
- a no-SFT reference;
- a pre-registered equal-step sensitivity;
- a mechanism analysis that challenges the method’s own surrogate;
- extensive run plans, hashes, and manifests.

Main weaknesses:

- the complete headline evidence is limited to one model, one candidate pool, and one MMLU
  STEM/Humanities setting;
- the effective inferential sample for cross-direction Random comparisons is five paired
  draw-index/training-seed blocks, not ten independent replicates;
- the query realization and SFT seed are coupled;
- LESS, GIST, and NICE are controlled shared-protocol adaptations rather than fully official
  end-to-end reproductions;
- the repository’s alignment/provenance documentation is not fully consistent with the current
  implementation;
- candidate-pool contamination against MMLU test has not been audited.

**Bottom-line assessment:** submitting the current version would likely be viewed as an interesting,
well-controlled case study with limited generality—approximately borderline / weak-reject territory.
A second query-aligned target/evaluation family is the highest-value experiment for solidifying the
paper.

---

# 2. Recommended paper positioning

## 2.1 Candidate titles

Method-forward:

> **Directional Second-Moment Coresets for Robust Targeted Instruction Selection**

Finding-forward, and probably stronger given the Random result:

> **When Target-Gradient Matching Is Not Enough: Directional Second Moments for Instruction Data
> Selection**

Controlled-study framing:

> **Directional Second Moments Improve Targeted Selection, but Not Its Downstream Surrogate**

## 2.2 One-paragraph paper pitch

Targeted instruction selection commonly ranks candidate examples using signed gradient similarity to
a small query set. We instead represent a query distribution by the second moment of unit-normalized
projected-gradient directions and greedily select an MMD coreset that matches this moment while
penalizing candidate redundancy. Controlled calibration and a representation × selector attribution
show that the main gain comes from the second-order representation, with MMD diversity providing a
budget-dependent complementary benefit. Across ten globally disjoint, skewed MMLU query sets,
DSMC consistently outperforms the examined target-aware selectors, but it does not outperform
well-controlled Random selection. A leave-one-draw-out forensic analysis shows that DSMC is
nevertheless closer than Random to both the observed query geometry and a balanced validation
reference on every draw. These results expose a gap between target-gradient coverage and useful
instruction-data coverage: target matching is real and optimizable, but incomplete as a surrogate
for downstream utility.

## 2.3 Recommended contributions

1. **Directional second-moment representation.** We formulate targeted instruction selection as
   matching
   \[
   M_P=\mathbb E_{u\sim P}[uu^\top],
   \]
   where \(u\) is a unit-normalized projected gradient, using the kernel
   \(k_2(u,v)=(u^\top v)^2\).
2. **Representation/selector attribution.** A controlled 2×2 study separates signed first- versus
   second-order representations and relevance TopK versus MMD coreset selection. The representation
   is the main driver; MMD is complementary in the second-order space at the 5% budget.
3. **Robust target-draw evaluation.** We evaluate ten globally disjoint skewed target sets in two
   directions, two nested budgets, and against relevance, round-robin, GIST-SharedProj,
   NICE-MMLU-EM, Random, and length-histogram-matched Random baselines.
4. **Surrogate-failure result.** DSMC dominates Random under the optimized second-moment criterion,
   including a leave-one-draw-out balanced reference, but does not dominate Random downstream.
5. **Transparent negative controls.** A no-SFT reference and a pre-registered equal-step sensitivity
   show that the low-budget result is not rescued by simply repeating the 1% subset for 420 steps.

---

# 3. Method

## 3.1 Targeted selection problem

Let
\[
\mathcal C=\{x_i\}_{i=1}^{N}
\]
be a candidate instruction pool and
\[
\mathcal Q=\{q_j\}_{j=1}^{M}
\]
a small target query set. Given budget \(K\), select
\[
\mathcal S\subset\mathcal C,\qquad |\mathcal S|=K,
\]
such that supervised fine-tuning on \(\mathcal S\) improves a held-out target evaluation.

At a common warm-up checkpoint, an example \(x\) is represented by a projected gradient direction
\[
u(x)=\frac{\Pi g(x)}{\|\Pi g(x)\|_2}.
\]
The main experiments use a shared 8192-dimensional Rademacher projection with seed 123. The
LESS-aligned feature protocol is asymmetric: candidate gradients are Adam-aware, whereas query
gradients are raw SGD gradients. This convention is shared by the gradient-space baselines, but it
must be stated explicitly because candidate and query features are not produced by identical
preconditioning maps.

## 3.2 Directional second moment

DSMC represents a distribution \(P\) by
\[
M_P=\mathbb E_{u\sim P}[uu^\top].
\]
Using
\[
k_2(u,v)=(u^\top v)^2,
\]
the empirical MMD is exactly the squared Frobenius distance between directional second moments:
\[
\operatorname{MMD}_{k_2}^2(\mathcal S,\mathcal Q)
=
\left\|
\frac{1}{|\mathcal S|}\sum_{x\in\mathcal S}u(x)u(x)^\top
-
\frac{1}{|\mathcal Q|}\sum_{q\in\mathcal Q}u(q)u(q)^\top
\right\|_F^2.
\]

Because the gradients are unit-normalized, this object should be called a **directional second
moment**, not raw gradient covariance.

## 3.3 Greedy MMD coreset

Define target relevance
\[
r_{\mathcal Q}(x)=\frac{1}{|\mathcal Q|}
\sum_{q\in\mathcal Q}k_2(u(x),u(q))
\]
and accumulated selected-set similarity
\[
r_{\mathcal S}(x)=\sum_{s\in\mathcal S}k_2(u(x),u(s)).
\]
At step \(m=|\mathcal S|\), the implementation selects
\[
x^\star=\arg\max_{x\notin\mathcal S}
\left[
r_{\mathcal Q}(x)
-
\frac{r_{\mathcal S}(x)+k_2(u(x),u(x))/2}{m+1}
\right].
\]
Since \(\|u(x)\|_2=1\), \(k_2(u(x),u(x))=1\). The first term rewards target
alignment, while the second discourages redundancy.

## 3.4 First/second-order moment family

Method development considered
\[
k_\lambda(u,v)
=(u^\top v)^2+\lambda\frac{1+u^\top v}{2}.
\]
The signed first-order term matches mean direction, while the second-order term matches orientation.
The original convex-mixture parameterization was mis-calibrated: the first-order greedy marginal had
approximately 14.5 times the candidate-wise spread of the second-order marginal. A direct coefficient
ratio \(\lambda\) provided a smoother interpolation.

The final method is the endpoint \(\lambda=0\). This choice is supported by the calibration and
mirror experiments below, but those experiments should be described as method development rather
than fully independent benchmark evidence.

---

# 4. Experimental setup

## 4.1 Model, pool, and training

- **Base model:** Llama-2-7B.
- **Candidate pool:** 270,679 processed Tulu/LESS examples.
- **Sources:** Flan v2 100,000; CoT 100,000; Dolly 15,011; OASST1 55,668.
- **SFT:** LoRA on `q_proj,k_proj,v_proj,o_proj`.
- **Resolved main-run values:** rank 128, alpha 512, dropout 0.05, learning rate \(2\times10^{-5}\),
  linear scheduler, warm-up ratio 0.03, weight decay 0, cutoff 2048, bfloat16, four epochs.
- **Batch:** per-device 4, gradient accumulation 4, eight GPUs, effective global batch 128.

Important provenance note: `train_llama7b_lora.yaml` contains older nominal values in its comments
and body; the main driver overrides alpha, batch, accumulation, and epochs on the command line.
The paper and artifact should therefore report a resolved configuration rather than treating the YAML
alone as the executed recipe.

## 4.2 Warm-up and gradient cache

The target-draw experiments share:

- warm-up checkpoint `warmup_seed42/checkpoint-1692`;
- candidate cache shape \(270{,}679\times8192\);
- candidate gradient type: Adam-aware;
- query gradient type: SGD;
- projection dimension 8192;
- projection seed 123.

The optimizer-state-aware transformation is computed without in-place mutation. However, the final
artifact still needs one authoritative record of the warm-up checkpoint’s resolved command,
world size, batch, LoRA alpha/dropout, and environment.

## 4.3 Skewed target-draw design

The main experiment is a finite-query sampling-bias stress test:

- the latent evaluation target is defined as balanced between MMLU STEM and Humanities;
- each observed query set has 64 validation examples;
- STEM-majority queries contain 51 STEM and 13 Humanities examples;
- Humanities-majority queries contain 13 STEM and 51 Humanities examples;
- five draws are generated per direction;
- all ten query sets are globally non-overlapping;
- validation is used for queries, dev for five-shot demonstrations, and test for evaluation.

Within a domain, query subjects approximately follow lm-eval test micro-weights. The draws are
globally disjoint but **not iid independent**: they are constructed by a joint without-replacement
partition from one finite validation reservoir.

Primary metric:
\[
A_{\mathrm{bal}}=\tfrac12(A_{\mathrm{STEM}}+A_{\mathrm{Humanities}}).
\]

Secondary metric:
\[
A_{\mathrm{tw}}
=\frac{51}{64}A_{\mathrm{majority}}+\frac{13}{64}A_{\mathrm{minority}}.
\]

The target gradients use a zero-shot single-example supervised format, whereas final evaluation uses
five-shot prompts. This is frozen and leakage-free, but it is a limitation because target and
inference contexts are not matched.

## 4.4 Budgets and paired seeds

- **5%:** \(K=13{,}533\), approximately 420 optimizer steps at four epochs.
- **1%:** \(K=2{,}707\), approximately 84 optimizer steps at four epochs.

For ordered selectors, the 1% subset is the first 2,707 examples of the frozen 5% ordering. Random
uses a shared seeded permutation and its prefix. Thus the budget experiment is a **nested-prefix,
fixed-epoch sensitivity**, not two independent selection realizations.

Training seed by draw index:

| Draw index | 0 | 1 | 2 | 3 | 4 |
|---|---:|---:|---:|---:|---:|
| SFT seed | 42 | 1 | 2 | 3 | 4 |

All methods within a draw use the same SFT seed. This yields paired method comparisons, but query
realization and SFT seed are confounded.

## 4.5 Baselines and naming

The main comparison contains:

1. **DSMC:** directional second-order MMD.
2. **LESS-style mean-gradient TopK:** average signed similarity on the shared projected features.
3. **First-RR:** per-query signed-similarity round-robin.
4. **Second-RR:** per-query squared-similarity round-robin.
5. **GIST-SharedProj:** GIST scoring in the shared 8192-dimensional projected-gradient space.
6. **NICE-MMLU-EM:** deterministic MMLU exact-match adaptation of NICE.
7. **Random-K:** uniform fixed-\(K\) sample.
8. **Random-K-LengthMatched:** fixed-\(K\) Random matching five coarse post-tokenization length
   buckets to DSMC.

The unqualified labels “LESS,” “GIST,” and “NICE” would overstate baseline fidelity:

- the main LESS baseline is a shared-checkpoint mean-gradient endpoint rather than an audited
  full-trajectory official reproduction;
- GIST uses the shared projected space, not raw official gradient space, and at \(M=64\) its
  official rank cap is \(k=M\);
- NICE is an MMLU exact-match adaptation and some query examples have zero reward signal.

Random-K is query-independent, so the same subset and trained adapter are reused for the STEM and
Humanities cells with the same draw index. It therefore has five unique trained adapters, despite
appearing in ten table cells.

---

# 5. Consolidated results

## 5.1 Moment calibration selects the second-order endpoint

The calibrated path produced:

| Method | Balanced accuracy | Difference from DSMC |
|---|---:|---:|
| DSMC, \(\lambda=0\) | **0.4110** | — |
| Joint, \(\lambda=0.02\) | 0.4017 | -0.93 pp |
| Joint, \(\lambda=0.07\) | 0.3928 | -1.82 pp |
| Linear endpoint | 0.3847 | -2.63 pp |

Although increasing \(\lambda\) improved the measured first-moment discrepancy while only slightly
changing aggregate second-moment discrepancy, downstream accuracy decreased monotonically in this
single realization.

Paired SFT-seed confirmation for DSMC versus \(\lambda=0.02\):

| Target | Mean DSMC | Mean joint | Joint - DSMC | Paired sign |
|---|---:|---:|---:|---:|
| STEM80 | 0.4086 | 0.4028 | -0.59 pp | 3/3 negative |
| HUM80 | 0.4069 | 0.4016 | -0.54 pp | 3/3 negative |

This supports freezing \(\lambda=0\), but not a universal claim that first-order information is
harmful.

## 5.2 Representation × selector attribution

| Target | First-TopK | First-MMD | Second-TopK | DSMC |
|---|---:|---:|---:|---:|
| STEM80 | 0.3977 | 0.3847 | 0.4068 | **0.4110** |
| HUM80 | 0.3931 | 0.3824 | 0.3992 | **0.4054** |

Effects in percentage points:

| Effect | STEM80 | HUM80 |
|---|---:|---:|
| Second - first under TopK | +0.91 | +0.61 |
| Second - first under MMD | +2.63 | +2.30 |
| MMD - TopK in first-order space | -1.30 | -1.07 |
| MMD - TopK in second-order space | +0.42 | +0.62 |
| Difference-in-differences | +1.72 | +1.69 |

**Interpretation:** the representation is the primary driver. MMD is complementary in the
second-order space, but the direct \(0.42\)–\(0.62\) pp gain is from one seed and is within the
observed training-noise scale. Also, the 2×2 selector factor is TopK versus MMD, whereas the main
benchmark uses true RR versus MMD; these should not be presented as one perfectly closed factorial
experiment.

## 5.3 Main absolute results

Balanced accuracy averaged over the ten direction cells:

| Method | 5% | 1% | 1% - 5% |
|---|---:|---:|---:|
| **DSMC** | **0.4036** | **0.4017** | -0.0019 |
| LESS-style TopK | 0.3941 | 0.3937 | -0.0004 |
| First-RR | 0.3880 | 0.3820 | -0.0060 |
| Second-RR | 0.3948 | 0.4000 | +0.0053 |
| GIST-SharedProj | 0.3885 | 0.3945 | +0.0060 |
| NICE-MMLU-EM | 0.3925 | 0.3942 | +0.0018 |
| Random-K | 0.4037 | **0.4095** | +0.0058 |
| Random-K-LengthMatched | **0.4042** | 0.4080 | +0.0038 |
| No-SFT base | 0.4003 | 0.4003 | — |

Target-weighted means give the same high-level conclusion:

| Method | 5% target-weighted | 1% target-weighted |
|---|---:|---:|
| DSMC | 0.4033 | 0.4014 |
| Random-K | 0.4037 | 0.4095 |
| Random-K-LengthMatched | 0.4036 | 0.4087 |

## 5.4 DSMC versus targeted baselines

### 5% budget

DSMC minus baseline, balanced:

| Baseline | Mean over 10 cells | Cell wins | Mean over 5 blocks | Block wins |
|---|---:|---:|---:|---:|
| First-RR | +1.55 pp | 10/10 | +1.56 pp | 5/5 |
| GIST-SharedProj | +1.51 pp | 9/10 | +1.51 pp | 5/5 |
| NICE-MMLU-EM | +1.11 pp | 9/10 | +1.11 pp | 5/5 |
| LESS-style TopK | +0.95 pp | 10/10 | +0.95 pp | 5/5 |
| Second-RR | +0.88 pp | 10/10 | +0.88 pp | 5/5 |

An exact enumeration of the five-block draw-resampling bootstrap gives descriptive 95% intervals:

- LESS-style TopK: \([+0.63,+1.36]\) pp;
- First-RR: \([+1.17,+1.89]\) pp;
- Second-RR: \([+0.55,+1.16]\) pp;
- GIST-SharedProj: \([+0.66,+2.36]\) pp;
- NICE-MMLU-EM: \([+0.66,+1.56]\) pp.

These intervals are descriptive, not population-level frequentist guarantees, because there are
only five blocks and the draws are not iid.

### 1% budget

| Baseline | Mean over 10 cells | Cell wins | Mean over 5 blocks | Block wins |
|---|---:|---:|---:|---:|
| First-RR | +1.97 pp | 9/10 | +1.97 pp | 5/5 |
| LESS-style TopK | +0.80 pp | 9/10 | +0.81 pp | 5/5 |
| NICE-MMLU-EM | +0.75 pp | 7/10 | +0.75 pp | 5/5 |
| GIST-SharedProj | +0.73 pp | 8/10 | +0.73 pp | 5/5 |
| Second-RR | +0.17 pp | 5/10 | +0.17 pp | 3/5 |

Descriptive five-block bootstrap intervals:

- LESS-style TopK: \([+0.45,+1.16]\) pp;
- First-RR: \([+1.60,+2.34]\) pp;
- GIST-SharedProj: \([+0.55,+0.96]\) pp;
- NICE-MMLU-EM: \([+0.33,+1.17]\) pp;
- Second-RR: \([-0.18,+0.52]\) pp.

**Interpretation:** DSMC is consistently better than the examined first-order and adapted
target-aware baselines. Its additional advantage over Second-RR is clear at 5% but collapses at 1%;
the MMD-coreset contribution is therefore budget-dependent.

## 5.5 DSMC versus Random

Direction-averaged DSMC - Random block differences:

| Draw block | 5% | 1% |
|---|---:|---:|
| 0 | +0.32 pp | -0.88 pp |
| 1 | -0.32 pp | -0.57 pp |
| 2 | +1.22 pp | -1.00 pp |
| 3 | -0.59 pp | -0.41 pp |
| 4 | -0.69 pp | -1.01 pp |
| **Mean** | **-0.01 pp** | **-0.77 pp** |

For Random-K-LengthMatched, the corresponding block means are -0.06 pp at 5% and -0.63 pp at 1%.

Correct claims:

- **5%:** DSMC shows no observed advantage over Random; the mean difference is approximately zero at
  the resolution of five blocks. This is not a demonstrated statistical equivalence.
- **1%:** DSMC is lower than Random in all five direction-averaged blocks and all ten direction
  cells.
- Coarse length-histogram matching does not explain the result, but the length-matched control is not
  an exact total-token or FLOP match.

## 5.6 Absolute performance relative to no-SFT

| Budget | DSMC | Random-K | LESS-style | GIST-SharedProj | NICE-MMLU-EM |
|---|---:|---:|---:|---:|---:|
| 1% | 0.4017 (+0.14 pp) | **0.4095 (+0.92 pp)** | 0.3937 (-0.66 pp) | 0.3945 (-0.58 pp) | 0.3942 (-0.60 pp) |
| 5% | 0.4036 (+0.33 pp) | 0.4037 (+0.34 pp) | 0.3941 (-0.61 pp) | 0.3885 (-1.18 pp) | 0.3925 (-0.78 pp) |

The no-SFT value is a single common reference, not a replicate. Therefore the correct statement is:

> The mean scores of several target-aware baselines fall below the common no-SFT reference under
> this protocol; DSMC is the only examined target-aware method whose mean remains slightly above it.

Do not present these differences as statistically established negative transfer.

## 5.7 Equal-step sensitivity

At 1%:

| Method | 84 steps | 420 steps | Change | 420 steps vs base |
|---|---:|---:|---:|---:|
| DSMC | 0.4017 | 0.3826 | -1.91 pp | -1.77 pp |
| Random-K | 0.4095 | 0.3827 | -2.67 pp | -1.76 pp |

The pre-registered interaction
\[
J_i=[\mathrm{DSMC}-\mathrm{Random}]_{420,i}
-[\mathrm{DSMC}-\mathrm{Random}]_{84,i}
\]
has mean \(+0.76\) pp and is positive in 4/5 blocks. However, the gap closes because Random
degrades more—not because DSMC improves. The 420-step condition over-trains both subsets.

The paper-safe conclusion is:

> Extending the 1% training horizon to the 5% step count does not rescue DSMC in absolute terms.
> The DSMC-Random ordering is horizon-sensitive, but the long-horizon condition is pathologically
> over-trained.

Do not claim that all optimization confounds are eliminated. The result only rejects the simple
explanation that “DSMC loses because 84 steps are insufficient and 420 steps will improve it.”

---

# 6. Mechanism analysis: geometry and utility diverge

## 6.1 DSMC succeeds at its geometric objective

| Budget | D2 to own query: DSMC | Random | D2 to balanced validation reference: DSMC | Random |
|---|---:|---:|---:|---:|
| 1% | **0.16895** | 0.19331 | **0.15206** | 0.17627 |
| 5% | **0.17067** | 0.19306 | **0.15369** | 0.17601 |

Against a domain-balanced, leave-one-draw-out validation reference:

| Budget | DSMC | Random | DSMC closer |
|---|---:|---:|---:|
| 1% | **0.15221** | 0.17642 | 10/10 draws |
| 5% | **0.15384** | 0.17616 | 10/10 draws |

Thus the explanation “DSMC merely overfits its own skewed query and becomes farther from balanced
geometry” is refuted for this empirical validation reference.

This reference should be called a **balanced validation proxy**, not the true latent distribution:
it is constructed from one finite MMLU validation reservoir and is not the test distribution.

## 6.2 Tightened D2-accuracy association

The original pooled Spearman correlation across 70 method × draw cells was:

| Budget | Pooled \(\rho(D2,\mathrm{accuracy})\) |
|---|---:|
| 1% | +0.389 |
| 5% | +0.112 |

Because lower D2 is “better” under the objective, a positive coefficient indicates that lower D2
is associated with lower accuracy. However, pooled cells are not independent and may mix method
identity and draw difficulty.

Using the leave-one-draw-out D2 values for seven methods and recomputing within each draw:

| Budget | Mean within-draw Spearman | Median | Positive draws |
|---|---:|---:|---:|
| 1% | **+0.404** | **+0.429** | **10/10** |
| 5% | +0.111 | +0.232 | 7/10 |

Draw-demeaned residual Spearman:

| Budget | Residual \(\rho\) |
|---|---:|
| 1% | **+0.393** |
| 5% | +0.101 |

This strengthens the descriptive conclusion at 1%:

> Better D2 ranking does not translate into better downstream ranking, even within the same draw.

It remains a descriptive association, not a causal result or a general claim that D2 is universally
anti-predictive.

## 6.3 Source provenance is not a sufficient explanation

At 1%:

| Method | Source entropy | Balanced accuracy | Effective rank | Mean pairwise cosine |
|---|---:|---:|---:|---:|
| Random-K | 1.227 | **0.4095** | 1013 | 0.0942 |
| DSMC | 0.965 | 0.4017 | **1589** | **0.0142** |
| GIST-SharedProj | **1.228** | 0.3945 | 1469 | 0.0333 |
| LESS-style | 1.205 | 0.3937 | 1465 | 0.0368 |

DSMC is source-concentrated, while Random preserves a broader source mix. But source entropy alone
cannot explain utility: GIST has essentially the same entropy as Random and performs much worse.

The defensible mechanism statement is:

> Gradient-geometric coverage and instruction-data coverage are distinct notions. Neither the
> optimized directional second-moment distance nor coarse source entropy is sufficient to explain
> downstream utility.

Do not claim that Random wins “because it is more diverse” without a causal source/format
intervention.

---

# 7. Earlier three-target evidence: useful context, not headline validation

The final five-seed early pipeline produced:

| Method | BBH | MMLU | TyDiQA-F1 |
|---|---:|---:|---:|
| GradCov-SGD | 0.3879 ± 0.0081 | 0.4565 ± 0.0071 | 0.5700 ± 0.0288 |
| GradCov-Adam | 0.3696 ± 0.0056 | 0.4539 ± 0.0095 | **0.5794 ± 0.0073** |
| TSDS | **0.3974 ± 0.0047** | 0.4586 ± 0.0020 | 0.5605 ± 0.0064 |
| NICE | 0.3918 ± 0.0049 | **0.4617 ± 0.0071** | 0.5747 ± 0.0137 |

The winner rotates by target. This supports two contextual observations:

1. second-order gradient matching can be competitive beyond skewed MMLU, especially on TyDiQA;
2. no data selector dominates across target families.

These runs should not be presented as a strict replication of the final DSMC protocol because their
target construction, representation variants, baseline set, and variance decomposition differ.

**Do not use** `three_target_summary.md` as a paper result source. It mixes an older pipeline and
contains values superseded by the corrected unified and five-seed results.

---

# 8. What the evidence does and does not establish

## 8.1 Claims supported by the current evidence

1. In the studied skewed-query regime, pure directional second-moment matching is at least as strong
   as the best calibrated first/second-order joint endpoint on average, and the joint does not win in
   the tested paired seeds.
2. Second-order representations outperform signed first-order representations in both cells of the
   2×2 attribution experiment and both skew directions.
3. At 5%, DSMC outperforms the examined target-aware baselines in every direction-averaged block.
4. At 1%, DSMC remains better than the first-order/adapted targeted baselines, but its advantage over
   Second-RR nearly vanishes.
5. DSMC shows no observed advantage over Random at 5% and is worse at 1%.
6. Repeating the 1% subset for 420 steps over-trains both DSMC and Random and does not improve DSMC.
7. DSMC is closer than Random to both observed-query and leave-one-draw-out balanced validation
   second-moment geometry on every draw.
8. Lower D2 is not sufficient for better downstream ranking in this setting.

## 8.2 Claims not supported

- DSMC is state of the art.
- DSMC significantly beats all baselines.
- DSMC beats Random.
- DSMC and Random are statistically equivalent at 5%.
- first-order gradient information is universally harmful.
- MMD diversity is universally necessary.
- the ten target draws are independent.
- lower D2 is generally anti-predictive across tasks and models.
- Random wins because of source entropy or “diversity.”
- the current GIST result refutes official end-to-end GIST.
- all optimization confounds are closed.
- the method is externally validated across tasks/models/pools.

---

# 9. Reviewer-facing vulnerability audit

## 9.1 Major concern: no advantage over Random

This will be the most immediate reviewer objection. The response should not hide it:

- 5% DSMC 0.4036 versus Random 0.4037;
- 1% DSMC 0.4017 versus Random 0.4095;
- 1%/420 steps DSMC 0.3826 versus Random 0.3827.

The value proposition is therefore not “use an expensive selector to beat Random.” It is:

- second moments repair a substantial robustness failure among target-aware selectors;
- the controlled results reveal a limitation of target-gradient matching itself;
- the paper provides evidence about when a target-aware surrogate fails.

## 9.2 Major concern: external validity

The complete headline chain is limited to:

- Llama-2-7B;
- Tulu/LESS 270k;
- MMLU STEM/Humanities;
- manually constructed 80/20 finite-query skew;
- balanced MMLU evaluation.

The early three-target experiments are informative but are not a clean replication of the final
frozen DSMC protocol. A second query-aligned target/evaluation family is therefore the highest
priority.

## 9.3 Major concern: effective sample size and coupled variation

- five draws per direction;
- draws are globally disjoint but jointly partition one finite reservoir;
- each draw index has one SFT seed;
- Random has five unique adapters shared across direction pairs;
- selection and SFT variance are not separately identified.

The paper should describe the unit as a **paired query-draw/training-seed block**, use block-level
statistics for Random, and keep the analysis descriptive.

## 9.4 Major concern: skew problem definition

The design assumes an observed 80/20 query is a biased finite sample of a balanced latent target.
A reviewer may instead regard the 80/20 query as the true intended distribution. Reporting the
target-weighted metric is therefore important, but a matched 50/50 control or a standard
query/evaluation-aligned task would make the robustness claim much more convincing.

## 9.5 Major concern: baseline fidelity

The baseline labels must be precise:

- LESS-style mean-gradient TopK;
- GIST-SharedProj;
- NICE-MMLU-EM.

Otherwise a reviewer can reasonably argue that the strongest positive comparison is against
adaptations that omit parts of the official pipelines.

## 9.6 Major concern: technical novelty

\((u^\top v)^2\) MMD is mathematically simple. Novelty must come from the combination of:

- orientation-based gradient representation;
- explicit moment interpretation;
- controlled first/second-order calibration;
- representation × selector attribution;
- robust target-draw design;
- the downstream surrogate-failure result.

A paper framed only as “a new polynomial kernel selector” will likely be judged incremental.

---

# 10. Artifact and provenance audit

## 10.1 Strong points

- target/test split separation is explicit;
- the ten target sets have zero example overlap;
- shared candidate-cache/checkpoint/projection hashes are recorded;
- selection/subset hashes are recorded per draw and method;
- 1% prefix relationships were audited;
- Random and RR seeds are recorded;
- evaluation outputs and adapter hashes are pinned;
- the equal-step decision rules were written before the run.

## 10.2 Issues to fix before release

### A. Alignment verifier contradicts the completion reports

At this snapshot, `python verify_alignment.py` reports **4/6**, failing its static checks for Adam
preconditioning and target-dataset handling. The Adam implementation appears to have been revised,
so the verifier is likely stale; target-dataset handling is genuinely different from the old
completion report.

Do not state that the current repository automatically verifies 6/6 alignment. Update or retire the
stale reports and make the verifier describe the current code.

### B. `target_dataset` is not a generic independent loader

`SelectTrainer` currently passes `self.eval_dataset` to the selector and asks users to set
`target_dataset` and `eval_dataset` to the same value. In the main extraction runs, in-training
evaluation is disabled and final MMLU test evaluation is separate, so no direct test leakage is
introduced. Nevertheless, the framework does not currently implement the generic independent
`target_dataset` loader claimed by older alignment documents.

### C. Main YAML is not the resolved training config

`train_llama7b_lora.yaml` says alpha 256, per-device batch 16, accumulation 8, and three epochs.
`run_pilot_sft.sh` overrides these with alpha 512, batch 4, accumulation 4, and four epochs. Export a
resolved config for each run family.

### D. Equal-step manifest omits the shell-injected `max_steps=420`

The driver injects `TRAIN_EXTRA="max_steps=420"`, but `_train_args()` records only the generic
four-epoch recipe. The final artifact should recover and hash the actual global step from
`trainer_state.json` or the authoritative training log.

### E. Warm-up checkpoint provenance is incomplete

Several warm-up configs exist with differing alpha, dropout, and nominal batch descriptions. Produce
one authoritative manifest for `checkpoint-1692` containing the resolved command, world size,
global batch, dataset hash, environment, and adapter/optimizer hashes.

### F. Candidate-pool decontamination is absent

Validation/dev/test roles prevent direct query/test leakage, but no audit establishes that the
Tulu/Flan/CoT candidate pool is free of MMLU test questions or near duplicates. This matters in a
selection paper because methods can differ in how strongly they retrieve benchmark-like examples.

### G. Working tree is not a final frozen snapshot

At consolidation time, the worktree already contained unrelated/uncommitted changes:

- `M data/dataset_info.json`
- `?? reviews/advice_0809.md`

Create a clean final artifact commit after documentation/provenance corrections.

---

# 11. Minimum experiments for a solid submission

## Priority 0: one query-aligned external-validity target family

Keep the Tulu pool and Llama-2-7B fixed so only the target/evaluation axis changes.

Recommended minimal protocol:

- target family: BBH, MMLU-Pro, TyDiQA, or another task with clean query/test splits;
- queries sampled normally from the target distribution, not artificially skewed;
- methods: DSMC, Second-RR, LESS-style TopK, Random-K, no-SFT;
- one frozen budget;
- 3–5 query draws;
- paired SFT seeds;
- report absolute performance and paired differences;
- audit candidate-pool contamination against the target test set.

Interpretation:

- DSMC > Random: target awareness can pay off in a query-aligned setting, while finite-query skew
  exposes brittleness.
- DSMC <= Random but > targeted baselines: a stronger cross-task conclusion that second moments
  improve target-aware selection, while target-aware selection itself remains brittle.

## Priority 1: complete the paper-grade analysis at zero training cost

Already computable from current artifacts:

- add the within-draw D2/accuracy Spearman table;
- add draw-demeaned residual correlations;
- use five direction-averaged blocks as the inferential unit;
- add consistent descriptive block-bootstrap intervals;
- report direction interactions;
- replace unqualified baseline names throughout.

## Priority 1: artifact consistency and decontamination

- fix the alignment verifier/reports;
- export resolved SFT and warm-up configs;
- repair equal-step provenance;
- audit exact, normalized, n-gram, and approximate candidate/test overlaps;
- freeze a clean final commit.

## Priority 2: 50/50 MMLU query control

A matched non-skew control would clarify whether the headline is specifically skew robustness or a
general MMLU target-sampling effect:

- \(n_T=64\);
- same subject weighting;
- 50/50 STEM/Humanities;
- DSMC, Second-RR, LESS-style, Random, no-SFT;
- ideally 3–5 paired blocks.

This is valuable, but a second query-aligned target family is more important for ICLR generality.

## Priority 2: representation symmetry ablation

On one representative draw and one budget, compare:

- current Adam-candidate / SGD-query;
- SGD-candidate / SGD-query;
- if feasible, Adam-candidate / Adam-query.

This addresses whether the DSMC result depends on comparing directions produced by two
preconditioning conventions.

## Optional

- independent Random adapters for both directions, rather than shared adapters;
- minimal multiseed DSMC versus Second-TopK confirmation;
- official raw-space GIST on one representative draw;
- target-prompt 0-shot versus eval-matched 5-shot gradient extraction;
- a second model;
- a second candidate pool;
- target-size sweep.

Do **not** prioritize a post-hoc source-balanced DSMC modification. It would turn a clean controlled
study into outcome-driven method tuning, and source entropy already has strong counterexamples.

---

# 12. Suggested paper structure

## 1. Introduction

- targeted instruction selection relies on small, noisy query sets;
- pointwise signed gradient relevance can collapse to dominant query directions;
- propose directional second moments plus a coreset objective;
- central finding: improved targeted robustness does not guarantee superiority to Random;
- contribution includes exposing the surrogate gap.

## 2. Related work

- influence and gradient-based selection;
- distribution matching and coresets;
- instruction-tuning diversity/coverage;
- critical studies of data selection and strong Random baselines.

## 3. Method

- problem setup;
- unit projected gradients;
- directional second moment;
- MMD identity;
- exact greedy marginal;
- computational complexity;
- relation to signed first-order relevance, RR, and subspace methods.

## 4. Experimental design

- model/pool/warm-up;
- biased finite-query stress test;
- global non-overlap and split roles;
- methods and precise adaptation labels;
- budgets and nested design;
- paired blocks and descriptive analysis.

## 5. Results

1. moment calibration and mirror;
2. 2×2 attribution;
3. 5% target-draw result;
4. 1% budget interaction;
5. no-SFT reference;
6. equal-step sensitivity.

## 6. Why better target matching does not guarantee utility

- query and leave-one-out balanced geometry;
- within-draw D2/accuracy association;
- source provenance counterexample;
- distinction between gradient coverage and instruction-data coverage.

## 7. External validity

- add the second target/evaluation family here;
- place the older three-target benchmark in an appendix unless it can be aligned to the final
  protocol.

## 8. Limitations

- single model/pool in the full controlled chain;
- adapted baselines;
- non-iid finite reservoir;
- draw/seed coupling;
- asymmetric gradient preconditioning;
- prompt mismatch;
- contamination limitations.

---

# 13. Draft abstract

Selecting instruction-tuning data for a target capability is often reduced to ranking candidate
examples by signed gradient similarity to a small query set. Such queries are finite and potentially
biased, while pointwise relevance does not control redundancy. We propose Directional Second-Moment
Coresets (DSMC), which represents candidate and query examples by unit-normalized projected-gradient
directions and greedily minimizes maximum mean discrepancy under the kernel
\(k(u,v)=(u^\top v)^2\). This matches the second moment of gradient orientations while penalizing
selected-set redundancy. Controlled calibration and a representation × selector study show that
second-order representations are the primary source of improvement, with the coreset objective
providing a budget-dependent complementary gain. Across ten globally disjoint MMLU target sets
skewed toward either STEM or Humanities, DSMC consistently outperforms the examined query-targeted
baselines at a 5% budget and remains the strongest targeted method at 1%. However, DSMC does not
outperform well-controlled Random selection: it shows no observed mean advantage at 5% and trails
Random at 1%. A leave-one-draw-out analysis shows that DSMC nevertheless matches both the observed
queries and a balanced validation reference more closely than Random on every draw. These findings
show that target-gradient geometry is an optimizable but incomplete surrogate for instruction-tuning
utility, and distinguish gradient-geometric coverage from the broader data coverage that drives
downstream performance.

The sentence claiming generality across target families should be added only after a clean external
validity experiment.

---

# 14. Final go/no-go recommendation

## Go

Proceed with the paper under a **method + controlled negative-result** framing. Freeze DSMC; do not
resume MMLU/Tulu learning-rate, LoRA, epoch, or source-balancing searches.

## Not yet ready for final submission

Before calling the work “solid ICLR,” complete:

1. one query/evaluation-aligned external target family;
2. zero-cost paper-grade block statistics and within-draw forensic correlations;
3. artifact/provenance reconciliation;
4. candidate-pool decontamination audit.

If resources permit, add the 50/50 MMLU control and the gradient-preconditioning symmetry ablation.

## Final one-sentence assessment

> **The project already demonstrates that directional second moments are a substantially more robust
> target-aware representation, but its most important scientific result is that even accurate
> target-gradient matching can fail to identify the most useful instruction data; one clean external
> target family and a provenance cleanup are the remaining steps toward a solid ICLR submission.**

---

# 15. Authoritative artifact map

Use these files as the primary numerical sources:

- `moment_calibration_summary.md`
- `attribution_2x2_results.csv`
- `attribution_2x2_summary.md`
- `full5draw_5pct_aggregate.csv`
- `full5draw_5pct_results.md`
- `full1pct_aggregate.csv`
- `full1pct_budget_interaction_results.md`
- `equalstep_1pct_aggregate.csv`
- `equalstep_1pct_results.md`
- `base_model_reference.json`
- `forensic_pstar_geometry.json`
- `forensic_robustness.json`
- `forensic_mechanism_analysis.md`
- `target_draw_protocol.md`
- `targetdraw_10draw_master_manifest.json`
- `multiseed_summary.csv`
- `multiseed_final_n5_summary.md`

Treat these as historical/superseded or contextual only:

- `three_target_summary.md` and `three_target_summary.csv`;
- early single-seed unified summaries, except when explicitly discussing the experimental history;
- stale alignment reports claiming that the current verifier passes 6/6.

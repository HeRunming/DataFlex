# Forensic mechanism analysis: why does Random beat DSMC, and why more so at 1%?

Zero-training-cost analysis (advice_0808 step 1 + advice_0808_2 robustness checks).
Artifacts: `forensic_pstar_geometry.json`, `forensic_robustness.json`.
All quantities on unit-normalized 8192-D projected gradients; `M_P = E_{u~P}[u uᵀ]`.

## 1. The proposed geometric mechanism is REFUTED (and it survives a leave-one-out check)

Hypothesis under test: *DSMC over-fits the skewed observed query `Q_d` and therefore ends up FARTHER
from the balanced latent `P*` we evaluate on — especially at the tight 1% budget.*

Balanced **P\*** = union of all 10 disjoint target draws (320 STEM + 320 HUM, validation-only):

| budget | D2 → own query `Q_d` | D2 → balanced `P*` |
|--------|----------------------|--------------------|
| 1% | DSMC **0.16895** < Random 0.19331 | DSMC **0.15206** < Random 0.17627 |
| 5% | DSMC **0.17067** < Random 0.19306 | DSMC **0.15369** < Random 0.17601 |

**Robustness check (advice_0808_2 A).** In the above, each draw's own `Q_d` is 10% of its own `P*`,
which could mechanically flatter DSMC. Recomputed against a **leave-one-draw-out, 50/50-domain-reweighted**
reference `P*₋d = ½·P_STEM,₋d + ½·P_HUM,₋d`:

| budget | DSMC | Random | DSMC closer in |
|--------|------|--------|----------------|
| 1% | 0.15221 | 0.17642 | **10/10 draws** |
| 5% | 0.15384 | 0.17616 | **10/10 draws** |

Identical conclusion. **DSMC is closer to the balanced reference than Random at both budgets, on every
draw, even when the evaluated query is excluded from its own reference.** The gap does not widen at 1%.
The geometric explanation is closed.

## 2. What we can and cannot claim about D2 vs downstream

Precise claim (corrected per advice_0808_2 — the earlier phrasing "our objective does not predict
downstream accuracy" was too strong):

> **Lower directional second-moment distance is not sufficient to predict or rank downstream accuracy
> in this setting.** DSMC dominates Random under the optimized `D2` criterion — its own objective, on
> both the query and the balanced reference — while underperforming it downstream on MMLU.

Descriptive Spearman across the existing (method × draw) points (advice_0808_2 B; **diagnostics only**
— the 70 cells are not independent, no significance claims):

| budget | n | ρ(D2→P\*, balanced) | ρ(source entropy, balanced) |
|--------|---|---------------------|------------------------------|
| 1% | 70 | **+0.389** | +0.281 |
| 5% | 70 | +0.112 | +0.074 |

Since *lower* `D2` is "better" by the objective, a **positive** ρ means lower `D2` goes with **lower**
accuracy. So at 1% `D2` is mildly *anti*-correlated with utility, and at 5% it is near-uninformative.
`D2` is not useless in general — it captures something real about target alignment, and DSMC's
consistent wins over other *targeted* selectors show that — but as a **surrogate for downstream utility
it is incomplete, and at tight budgets can point the wrong way.**

## 3. Source provenance: a candidate axis, NOT a demonstrated mechanism

| method | source entropy 1% | balanced 1% | source entropy 5% | balanced 5% | eff. rank 1% | mean pairwise cos 1% |
|--------|------------------|-------------|-------------------|-------------|--------------|----------------------|
| randk | 1.227 | **0.4095** | 1.223 | 0.4037 | 1013 | 0.0942 |
| randk_lenmatch | 1.217 | 0.4080 | 1.218 | 0.4042 | 1053 | 0.0864 |
| dsmc | **0.965** | 0.4017 | 0.964 | 0.4036 | **1589** | **0.0142** |
| second_rr | 0.958 | 0.4000 | 1.057 | 0.3948 | 866 | 0.0428 |
| gist | **1.228** | **0.3945** | 1.235 | 0.3885 | 1469 | 0.0333 |
| nice | 0.875 | 0.3942 | 1.152 | 0.3925 | 665 | 0.0987 |
| less | 1.205 | 0.3937 | 1.162 | 0.3941 | 1465 | 0.0368 |

Tulu source composition (1%, draw0):
```
dsmc :  flan_v2 1713 | cot 481 | dolly 445 | oasst1  68
randk:  flan_v2 1001 | cot 999 | dolly 159 | oasst1 548
```

DSMC's subset is much more source-concentrated than Random (entropy 0.965 vs 1.227; 63% flan_v2,
oasst1 nearly dropped at 68 vs 548). **But source entropy alone cannot explain Random's advantage —
there is a decisive counterexample in the same table: GIST has the HIGHEST source entropy of any
method (1.228 at 1%, essentially identical to Random's 1.227; 1.235 at 5%) yet the WORST downstream
accuracy (0.3945 / 0.3885).** LESS is similar (entropy 1.205, accuracy 0.3937). The ρ values above
(+0.281 / +0.074) confirm entropy is a weak, non-ranking signal.

Correct statement: **source provenance is a strong candidate explanatory axis distinguishing DSMC from
Random, but the present evidence is correlational and coarse source entropy is insufficient.** Finer
candidates worth testing (not tested here): exact source proportions rather than entropy, source×task
compatibility, format mix (CoT / dialogue / short-QA), answer style, preservation of the pool's native
mixture, quality, and alignment with the pretrained model's distribution. Diversity for instruction
tuning is known not to reduce to a single entropy or pairwise-distance number.

Worth noting the shape of DSMC's diversity: it has the **highest gradient-space effective rank**
(1589 → 1977 from 1%→5%) and the **lowest pairwise cosine** (0.0142) of all methods — maximally
diverse in *gradient directions* while narrow in *data provenance*. Random is the inverse (rank 1013,
pcos 0.0942). Two different senses of "diversity" that come apart here.

## 4. Bottom line

> **Neither target-gradient geometry nor coarse source diversity alone explains downstream utility.**
> DSMC wins the geometric objective decisively (10/10 draws vs Random, both budgets, LOO-robust) and
> still loses downstream; GIST matches Random's source entropy and loses badly. Both candidate
> explanations are individually insufficient.

This is a mechanism result that partly undercuts our own method's premise, and it should be reported
as such. It also sharpens the paper's contribution: DSMC makes targeted selection substantially more
robust *among targeted selectors*, while exposing a real gap between geometric target alignment and
useful instruction-tuning coverage.

## 5. What remains open

The one live confound this analysis cannot address is the **optimization/compute difference**
(84 vs 420 optimizer steps at 1% vs 5%), which requires the pre-registered equal-step sensitivity
(`prereg_1pct_equalstep.md`). These forensic results must not (and did not) change that
pre-registered design.

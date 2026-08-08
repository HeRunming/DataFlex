# Forensic mechanism analysis: why does Random beat DSMC, and why more so at 1%?

Zero-training-cost analysis (advice_0808 step 1). Artifacts: `forensic_pstar_geometry.json`.
Balanced **P\*** reference = union of all 10 target draws (320 STEM + 320 HUM = 640 examples,
globally disjoint by construction, validation-only, never trained on). All quantities on
unit-normalized 8192-D projected gradients; `M_P = E_{u~P}[u uᵀ]`.

## The proposed hypothesis is REFUTED

Hypothesis under test: *DSMC over-fits the skewed observed query `Q_d` and therefore ends up
FARTHER from the balanced latent `P*` we evaluate on — especially at the tight 1% budget.*

| budget | D2 → own query `Q_d` | D2 → balanced `P*` |
|--------|----------------------|--------------------|
| 1% | DSMC **0.16895** < Random 0.19331 | DSMC **0.15206** < Random 0.17627 |
| 5% | DSMC **0.17067** < Random 0.19306 | DSMC **0.15369** < Random 0.17601 |

**DSMC is closer to `P*` than Random at BOTH budgets** (by ≈0.024 at 1%, ≈0.022 at 5%), and the gap
does *not* widen at 1%. So the "target-aware selection wastes tight capacity chasing the observed
skew, losing balanced coverage" story is **not** what is happening in second-moment geometry. DSMC is
simultaneously closer to the query *and* closer to the balanced reference — it is winning the geometry
objective it was designed for, at both budgets.

**Consequence (important for the paper): our own selection objective does not predict downstream
accuracy.** DSMC minimizes exactly `D2` and it wins on `D2` against every baseline — yet loses to
Random on MMLU. This is a direct, quantified dissociation between the matching objective and
downstream performance, not a failure to optimize it.

## What DOES separate DSMC from Random: source concentration / redundancy

| method | source entropy 1% | source entropy 5% | eff. rank 1% | eff. rank 5% | mean pairwise cos 1% |
|--------|------------------|-------------------|--------------|--------------|----------------------|
| dsmc | **0.965** | 0.964 | 1589 | 1977 | **0.0142** |
| randk | **1.227** | 1.223 | 1013 | 1075 | 0.0942 |
| randk_lenmatch | 1.217 | 1.218 | 1053 | 1117 | 0.0864 |
| second_rr | 0.958 | 1.057 | 866 | 1173 | 0.0428 |
| less | 1.205 | 1.162 | 1465 | 1837 | 0.0368 |
| gist | 1.228 | 1.235 | 1469 | 1727 | 0.0333 |
| nice | 0.875 | 1.152 | 665 | 758 | 0.0987 |

Tulu source composition (1%, draw0):

```
dsmc :  flan_v2 1713 | cot 481 | dolly 445 | oasst1  68
randk:  flan_v2 1001 | cot 999 | dolly 159 | oasst1 548
```

DSMC's selection is **substantially more source-concentrated than Random** (entropy 0.965 vs 1.227):
it loads up on `flan_v2` (63% of its 1% subset) and nearly drops `oasst1` (68 vs 548 examples, 8×
fewer). Notably this is *not* a gradient-space collapse — DSMC has the **highest effective rank** and
the **lowest pairwise cosine** of all methods, i.e. it is maximally diverse *in gradient directions*
while being narrow *in data provenance*. Random is the opposite: lower gradient-space rank, higher
redundancy, but broad source coverage.

So the operative contrast is **gradient-direction diversity (DSMC wins) vs data-source/format
coverage (Random wins)** — and on MMLU, the source coverage is what pays off, more so at 1% where
there is less room to have both.

## Secondary observations

- DSMC's effective rank grows 1589 → 1977 from 1% → 5% while Random's barely moves (1013 → 1075):
  DSMC's coverage of gradient space genuinely improves with budget, consistent with its 5%-only
  advantage over Second-RR (+0.0088 at 5% vs +0.0017 at 1%).
- NICE is the most concentrated/redundant selector at 1% (entropy 0.875, eff-rank 665, pcos 0.0987),
  matching its negative transfer vs base.
- Random-K-LengthMatched tracks Random-K closely on every geometry/diversity measure, reconfirming
  that length composition is not the operative variable.

## What this means for the next step

The remaining live explanation from advice_0808 is the **optimization/compute confound** (84 vs 420
optimizer steps), which this analysis cannot address — it needs the pre-registered equal-step
sensitivity (see `prereg_1pct_equalstep.md`). The geometry story is now settled: DSMC is *not*
mis-matching `P*`; the deficit is source/format coverage, and our matching objective does not predict
MMLU accuracy.

Honest framing note: this is a **mechanism result that partly undercuts our own method's premise**
(objective ≠ downstream), and should be reported as such rather than buried.

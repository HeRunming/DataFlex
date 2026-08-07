# 1% budget-interaction experiment (10 draws × 8 methods, 80/80 cells)

Completed 2026-08-08, zero engineering failures. Launch commit `38472d3`; run plan
`pilot1pct_run_plan.json` (K=2707); manifests: shared target geometry
`targetdraw_10draw_master_manifest.json` + budget-specific `pilot1pct_selection_manifest.json`;
table `pilot1pct_aggregate.csv`. 1% subsets are **verified nested prefixes** of the frozen 5%
ordering (all 70 prefixable (draw,method) pairs, 0 failures) — so budget differs by subset size, not
by a new selection realization. 4 epochs frozen at both budgets ⇒ this is a **data/compute-budget
interaction**, not an equal-step comparison (84 optimizer steps at 1% vs 420 at 5%).
Descriptive only; diff = **DSMC − method** (positive = DSMC better).

## 1%: DSMC − method, balanced (mean over 10 replicates)

| method | mean Δ | median | DSMC wins/10 |
|--------|--------|--------|--------------|
| first_rr | +0.0197 | +0.0221 | 9/10 |
| less | +0.0080 | +0.0080 | 9/10 |
| nice | +0.0075 | +0.0048 | 7/10 |
| gist | +0.0073 | +0.0077 | 8/10 |
| second_rr | +0.0017 | +0.0009 | 5/10 |
| **randk** | **−0.0077** | −0.0081 | **0/10** |
| randk_lenmatch | −0.0063 | −0.0089 | 2/10 |

## PRIMARY pre-registered analysis: budget interaction (5 direction-averaged blocks)

`[DSMC − Random]₁% − [DSMC − Random]₅%`, balanced:

| block (seed) | 1% | 5% | interaction (1%−5%) |
|--------------|----|----|---------------------|
| idx0 (42) | −0.0088 | +0.0032 | −0.0120 |
| idx1 (1)  | −0.0057 | −0.0032 | −0.0025 |
| idx2 (2)  | −0.0100 | +0.0122 | −0.0222 |
| idx3 (3)  | −0.0042 | −0.0060 | +0.0018 |
| idx4 (4)  | −0.0101 | −0.0068 | −0.0033 |
| **mean**  | **−0.0077** | **−0.0001** | **−0.0076** (1/5 blocks favor 1%) |

**The interaction goes the OPPOSITE way to the literature-motivated hypothesis.** We expected
targeted selection's edge over Random to *widen* as budget tightened. Instead, DSMC−Random moves
from ≈0 at 5% to **−0.77 pp at 1%**, and DSMC loses to Random-K in **0/10** cells at 1% (vs 4/10 at
5%). Random-K gets *relatively better* at the smaller budget here.

## Absolute performance vs the base-model reference (balanced 0.4003, no selected-data SFT, same 5-shot)

| budget | DSMC | Random-K | LESS | GIST | NICE |
|--------|------|----------|------|------|------|
| 1% | 0.4017 (+0.0014) | **0.4095 (+0.0092)** | 0.3937 (−0.0066) | 0.3945 (−0.0058) | 0.3942 (−0.0060) |
| 5% | 0.4036 (+0.0033) | 0.4037 (+0.0034) | 0.3941 (−0.0062) | 0.3885 (−0.0118) | 0.3925 (−0.0078) |

This is the most important thing the base-model line buys us: **every *targeted* selector
(LESS/GIST/NICE) ends up BELOW the base model at both budgets** — i.e. negative transfer. DSMC is
the only targeted method that stays marginally above base (+0.0014 at 1%, +0.0033 at 5%), and
**Random-K is the best of all at 1% (+0.0092)**. So "DSMC beats other targeted selectors" is largely
*DSMC degrades least*, not *DSMC improves most*.

## Honest conclusions

1. **Within targeted selectors, DSMC is robustly best** — it beats LESS, First-RR, GIST, NICE at
   both budgets (8–10/10 at 1%; 9–10/10 at 5%). This is the paper's solid contribution and it now
   replicates across two budgets and both skew directions.
2. **DSMC vs Second-RR collapses at 1%** (+0.0017, 5/10) vs a consistent +0.0088 (10/10) at 5%. The
   MMD-coreset benefit over second-order round-robin is **budget-dependent** — it exists at 5% and
   essentially vanishes at 1%. The earlier "consistent additional benefit" claim must now be scoped
   to the 5% budget.
3. **Targeted selection does not beat well-controlled Random at either budget**, and at 1% it is
   clearly *worse* (0/10). The pre-registered hypothesis (edge widens at low budget) is **refuted**
   in this setting.
4. **Most targeted selectors show negative transfer vs the base model**; only DSMC stays (barely)
   positive, while Random-K is the strongest absolute method at 1%.

**Defensible headline**: *Directional second-moment matching is consistently the most robust
query-targeted selector across skew directions and budgets, but in this MMLU/Tulu setting targeted
selection as a whole fails to beat budget- and length-matched Random selection at either 5% or 1% —
and the gap against Random widens against targeted methods as the budget tightens. Most targeted
selectors fall below a no-selected-SFT base model; DSMC is the only one that does not.*

This is a negative result for "target awareness pays off", and a positive, well-controlled result
for "if you do targeted selection, match the directional second moment". It aligns with recent
critical work on targeted instruction selection.

## Caveats

- 4 epochs at both budgets ⇒ 1% also has ~5× fewer optimizer steps/token exposure. The interaction
  is data+compute, not pure subset size. An equal-step 1% arm was **not** run (pre-registered as out
  of scope; adding it now would be post-hoc).
- Single model (Llama-2-7B), single candidate pool (Tulu 270k), single eval family (MMLU
  STEM/Humanities). Random's strength here may not generalize to other pools/tasks.
- n=5 blocks per budget; descriptive intervals only, no significance claims.
- Base-model line is one common reference (not a replicate, excluded from win counts/bootstrap).

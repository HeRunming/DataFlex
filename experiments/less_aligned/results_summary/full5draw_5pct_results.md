# Full 5-draw × 5% target-draw results (10 draws, 8 methods, 80/80 cells)

Completed 2026-08-06, zero engineering failures. Launch commit `ec3c2d3`; run plan
`pilot_run_plan.json` (80 cells / 75 adapters); manifest `targetdraw_10draw_master_manifest.json`;
table `pilot_aggregate.csv`. Statistical unit = target-draw/training-seed replicate (5 per
direction, seeds 42/1/2/3/4 by draw index); DSMC frozen. **Descriptive only** — no significance
claims. Diff = **DSMC − method** (positive = DSMC better). Primary = balanced; secondary =
target-weighted (51/64 majority, 13/64 minority).

## DSMC − method, balanced: mean over 10 replicates + DSMC win count

| method | mean Δ | median | wins/10 | stem-mean | hum-mean |
|--------|--------|--------|---------|-----------|----------|
| first_rr       | +0.0155 | +0.0180 | **10/10** | +0.0129 | +0.0182 |
| gist           | +0.0151 | +0.0144 | 9/10 | +0.0109 | +0.0193 |
| nice           | +0.0111 | +0.0080 | 9/10 | +0.0202 | +0.0020 |
| less           | +0.0095 | +0.0080 | **10/10** | +0.0112 | +0.0077 |
| second_rr      | +0.0088 | +0.0095 | **10/10** | +0.0110 | +0.0066 |
| **randk**          | **−0.0001** | −0.0031 | **4/10** | +0.0014 | −0.0017 |
| randk_lenmatch | −0.0006 | −0.0012 | 5/10 | −0.0001 | −0.0010 |

Target-weighted means: first_rr +0.0150 (10/10), gist +0.0160 (8/10), nice +0.0104 (6/10),
less +0.0100 (9/10), second_rr +0.0090 (10/10), randk −0.0004 (4/10), randk_lenmatch −0.0003 (5/10).

## Random-K as 5 unique blocks (direction-averaged DSMC−randk, balanced)

| draw idx (seed) | stem | hum | block mean |
|-----------------|------|-----|-----------|
| 0 (42) | +0.0043 | +0.0021 | +0.0032 |
| 1 (1)  | −0.0043 | −0.0020 | −0.0032 |
| 2 (2)  | +0.0197 | +0.0047 | +0.0122 |
| 3 (3)  | −0.0041 | −0.0079 | −0.0060 |
| 4 (4)  | −0.0084 | −0.0052 | −0.0068 |

**5-block mean = −0.0001; DSMC wins 2/5 blocks.** Per-block DSMC−Random: +0.32, −0.32, +1.22,
−0.60, −0.68 pp. Correct statement: **at 5%, DSMC shows no observed advantage over Random-K; their
mean performance is practically indistinguishable at the resolution of five paired blocks.** This is
NOT a proven statistical tie/equivalence — with only 5 paired blocks and non-trivial spread, we have
not defined a practical-equivalence margin or run an equivalence test; we report "no observed
advantage", not "equivalent". randk_lenmatch gives the **same qualitative conclusion** (mean ≈0), not
identical per-draw.

## Findings (5-draw scale)

1. **DSMC consistently beats every *targeted* selector.** vs LESS, First-RR, Second-RR: **10/10**
   replicates, means +0.009 to +0.016 balanced. vs GIST 9/10 (+0.0151), vs NICE 9/10 (+0.0111) on
   **balanced** — but NICE is only 6/10 on target-weighted, so "highly consistent over NICE" applies
   to the primary balanced endpoint only (NICE's balanced gain is very direction-split: strong on
   STEM +0.0202, ~flat on HUM +0.0020). DSMC > Second-RR 10/10 (+0.0088): the second-order MMD
   objective provides a **consistent additional benefit over second-order round-robin in the primary
   balanced metric** — not a causal proof that MMD repulsion is necessary.
2. **At 5%, DSMC shows no observed advantage over well-controlled Random** (mean −0.0001, 4/10 cells,
   2/5 blocks; randk_lenmatch same qualitative conclusion −0.0006). Reported as "no observed
   advantage", not statistical equivalence.

**Paper headline (defensible)**: *Directional second-moment matching substantially and consistently
improves over existing targeted selectors (LESS, GIST, first/second-order round-robin, NICE) under
skewed query sets in both skew directions, but targeted selection as a whole does not outperform
well-controlled Random selection at a 5% budget.* Matches the "Critical Look" literature.

## Next: 1% budget (separate pre-registered budget-interaction experiment)

The natural mechanism question: does DSMC's (and targeted selection's) advantage over Random emerge
as the budget tightens? Run K=2707 (1%) for the same 10 draws × 8 methods. Both budgets reported;
not an outcome-dependent swap. Selections cheap (rerun selectors at K=2707); SFT is the cost.

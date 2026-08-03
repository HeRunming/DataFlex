# Target-draw pilot results (2 dir × 2 draws, 8 methods, 32/32 cells)

Completed 2026-08-03, zero engineering failures. `pilot_aggregate.csv`; run plan
`pilot_run_plan.json`; master manifest `pilot_4draw_master_manifest.json`. Statistical unit = target
draw; DSMC frozen; **descriptive only** (n=4 draws total, 2 per direction — no significance claims).
Diff convention = **DSMC − method** (positive = DSMC better). Primary = balanced acc; secondary =
target-weighted (51/64 majority, 13/64 minority).

## DSMC − method, balanced, per draw + mean + DSMC win count (out of 4)

| method | stem0 | stem1 | hum0 | hum1 | mean | DSMC wins |
|--------|-------|-------|------|------|------|-----------|
| less           | +0.0093 | +0.0073 | +0.0075 | +0.0032 | **+0.0068** | 4/4 |
| first_rr       | +0.0162 | +0.0117 | +0.0197 | +0.0069 | **+0.0136** | 4/4 |
| second_rr      | +0.0107 | +0.0103 | +0.0098 | +0.0045 | **+0.0088** | 4/4 |
| gist           | +0.0190 | +0.0007 | +0.0382 | +0.0054 | **+0.0158** | 4/4 |
| nice           | +0.0171 | +0.0279 | −0.0095 | +0.0045 | **+0.0100** | 3/4 |
| randk          | +0.0043 | −0.0043 | +0.0021 | −0.0020 | **+0.0000** | 2/4 |
| randk_lenmatch | +0.0065 | −0.0100 | +0.0086 | −0.0054 | **−0.0001** | 2/4 |

Target-weighted means (same direction): less +0.0056, first_rr +0.0125, second_rr +0.0076,
gist +0.0161, nice +0.0098, randk −0.0015, randk_lenmatch −0.0010.

DSMC absolute balanced per draw: stem0 0.4107, stem1 0.3980, hum0 0.4085, hum1 0.4003.

## Reading (descriptive, pilot-scale)

1. **DSMC beats every gradient/geometry baseline on average and consistently**: LESS, First-RR,
   Second-RR, GIST all lose to DSMC in **4/4** draws (means +0.007 to +0.016 balanced). GIST — the
   closest conceptual competitor — is the *most* behind on average (+0.0158), though with high
   draw-to-draw variance (+0.0007 on stem1 to +0.0382 on hum0).
2. **DSMC ≈ Random-K** (mean +0.0000 balanced, −0.0015 target-weighted; DSMC wins 2/4). This is the
   important honest caveat: at 5% budget on these skewed targets, **plain Random-K matches DSMC** —
   consistent with the large-scale-selection literature that Random is a strong baseline at
   non-tiny budgets. randk_lenmatch behaves the same (mean ≈0), so the DSMC-vs-random gap is not a
   length/token-count artifact.
3. NICE is the one baseline DSMC does not always beat (loses on hum0 by 0.0095); otherwise behind.

**Headline caveat for the paper**: the pilot supports "DSMC ≥ other *targeted* selectors (LESS,
GIST, RR, NICE) across both skew directions", but does **NOT** support "DSMC beats Random" at this
5% budget — DSMC and Random-K are tied. This directly motivates the pre-registered **1% budget**
axis (targeted selection's advantage over Random is expected to widen at low budget), and is exactly
the kind of result the "Critical Look at Targeted Instruction Selection" work warns to check.

## Next (per advice_0731 — analysis before expansion)

Do NOT expand to 5 draws yet. Options to weigh: (a) expand these 2-draw results to 5 draws/direction
for tighter draw-clustered intervals; (b) add the **1% budget** condition where DSMC-vs-Random should
separate; (c) both. No method/hyperparameter changes — the pilot is a clean, frozen result.

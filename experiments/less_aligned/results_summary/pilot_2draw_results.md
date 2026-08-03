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

1. **DSMC beats LESS, First-RR, Second-RR, and GIST in 4/4 replicates** (means +0.007 to +0.016
   balanced). GIST — the closest conceptual competitor — is the *most* behind on average (+0.0158),
   though with high draw-to-draw variance (+0.0007 on stem1 to +0.0382 on hum0). **NICE is mixed**:
   DSMC wins both STEM draws (+0.0171, +0.0279) but on HUM the two draws split (−0.0095, +0.0045),
   so NICE is slightly *ahead* of DSMC on the HUM-direction 2-draw average (~−0.25 pp balanced, a bit
   more under target-weighted). So: DSMC > {LESS, First-RR, Second-RR, GIST} in every replicate;
   NICE mixed but DSMC better on the 4-replicate average.
   Also DSMC > Second-RR in 4/4 — the win is not just the 2nd-order representation; the MMD coreset
   selector adds over 2nd-order RR in these replicates too (reinforces the earlier 2×2 mechanism).
2. **DSMC ≈ Random-K** (mean +0.0000 balanced, −0.0015 target-weighted; DSMC wins 2/4). IMPORTANT
   caveat on the evidence strength: Random-K reuses ONE adapter per draw index across both
   directions, so there are only **2 unique Random adapters**, and the sign is perfectly **blocked by
   draw index**: DSMC wins both direction cells at index 0 (+0.43, +0.21 pp) and loses both at index
   1 (−0.43, −0.20 pp). So the apparent tie is largely two blocks cancelling, not 4 independent
   Random comparisons — expanding to draws 2–4 (3 new Random subsets + seeds) is what actually
   resolves whether the tie is stable. randk_lenmatch supports the **same overall conclusion** (mean
   ≈0) — but is not identical per-draw (e.g. stem1 LenMatched +1.00 pp over DSMC, hum0 DSMC +0.86
   pp). So: **the tie is not explained by coarse post-tokenization length-bucket composition** (not a
   claim that all token-exposure confounding is eliminated — the control matches bucket counts, not
   exact total tokens).

**Headline for the paper (pilot-scale)**: DSMC ≥ existing *targeted* selectors (LESS, GIST, RR;
NICE on average) across both skew directions, but does **NOT** beat well-controlled Random at 5%
budget — they are tied within the current 2-block Random evidence. Motivates (a) expanding 5% to 5
draws to test whether the Random tie is stable vs block-cancellation, then (b) the pre-registered
**1% budget** axis as a budget-interaction follow-up (targeted selection's edge over Random is
expected to widen at low budget — an empirical trend, not guaranteed here).

## Next (per advice_0731 / choice_0803 — analysis before 1%)

**Decision: expand the frozen 5% condition to 5 draws/direction FIRST** (draws 2,3,4 × 2 dir already
generated + globally disjoint), THEN the 1% budget axis as a separate pre-registered
budget-interaction experiment. Rationale: the Random tie currently rests on only 2 unique Random
adapters with sign perfectly blocked by draw index — 3 more draw indices directly test whether it's
stable or block-cancellation. Do NOT jump to 1% first (would read as switching to a friendlier budget
after the main budget didn't beat Random). Both budgets will be reported; no outcome-dependent
expansion. Keep all 8 methods frozen. Analysis will treat Random-K as **5 unique Random/seed blocks**,
not 10 direction cells; statistical unit = target-draw/training-seed replicate; descriptive intervals.

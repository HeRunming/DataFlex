# PRE-REGISTRATION: 1% equal-step sensitivity (DSMC vs Random-K)

**Status: PRE-REGISTERED, NOT YET RUN.** Written before any equal-step training, so the judgment
quantity and interpretation rules are fixed in advance. Per advice_0808 steps 2–3.

## Why this specific experiment (and only this one)

The 1% vs 5% comparison holds **4 epochs** fixed at both budgets, so 1% also has ~5× fewer optimizer
steps (**84** vs **420**) and ~5× less cumulative LR exposure under the linear schedule
(`lr=2e-5`, linear decay, `warmup_ratio=0.03` ⇒ ~3 warmup steps at 1% vs ~13 at 5%). That is a real,
*pre-statable* confound in the budget-interaction result, so a narrow sensitivity check is a legitimate
post-hoc analysis.

**This does NOT replace the pre-registered fixed-epoch result.** Both will be reported. It is a
sensitivity analysis of one confound, not a new headline condition.

**Explicitly out of scope** (would be outcome-driven tuning, since we have already seen the 1%
result): any grid over learning rate, LoRA rank/alpha, batch size, or epochs; any change to the
method set, selection, seeds, or evaluation.

## Design

- **Methods: only DSMC and Random-K.** (Not the other 6 — this tests a compute confound, not a
  method comparison.)
- **Same everything else**: identical frozen 1% subsets (verified nested prefixes of the 5%
  ordering), same 10 draws / 5 draw-index blocks, same per-draw training seeds (42/1/2/3/4), same
  LoRA (r=128, α=512), same per-device batch 4 × grad-accum 4 (eff 128), same `lr=2e-5`, linear
  scheduler, `warmup_ratio=0.03`, same eval (`mmlu_stem,mmlu_humanities`, 5-shot).
- **Only change**: `max_steps=420` (≈20 epochs at K=2707) so the optimizer-step count exactly matches
  the 5% condition. Assert the final step count is exactly 420.
- **Adapters**: 10 DSMC (one per draw) + 5 shared Random-K (one per draw index, reused across
  directions, as in the main design) = **15 new adapters**. Separate namespace `pilot1pctES_*` so
  nothing collides with the 84-step 1% results.

## Primary judgment quantity (fixed now)

Over the 5 direction-averaged draw-index blocks *i*:

```
J_i = [DSMC − Random]_{1%, 420 steps, i}  −  [DSMC − Random]_{1%, 84 steps, i}
```

Primary metric = balanced accuracy; secondary = target-weighted (51/64, 13/64). Report per-block
values, mean, median, and the number of blocks with `J_i > 0`. Descriptive only — n=5 blocks, no
significance claims.

**Always additionally report absolute change vs the base-model reference (balanced 0.4003)** for both
DSMC and Random at both step counts, so "who improves on the base model" stays visible.

## Interpretation rules (fixed BEFORE running)

| observation | conclusion |
|---|---|
| DSMC rises at 420 steps, Random roughly flat (`J_i > 0` in most blocks) | under-optimization was a material factor in the 1% result; the budget interaction is partly a compute artifact and must be reported as such |
| Random *falls* at 420 steps (overfitting) while DSMC does not rise | equal-compute changed the ordering; this is **not** evidence DSMC learns better — report as "ordering is compute-dependent", do not claim a DSMC win |
| Random still clearly ahead at 420 steps | the "too few steps" explanation is **closed**; stop optimization-side investigation and move to external validity (other pool / eval family / query-aligned setting) |
| Both degrade | 20 epochs at K=2707 is simply over-training; inconclusive for the confound, report as such |

No other outcome will be used to relabel the main result, and no hyperparameter will be adjusted
based on what we see.

## Cost

15 adapters × 420 steps. At 5%-run rates (~70 min for 420 steps) ≈ **~18h train + ~2h eval ≈ ~20h**.

## Decision this feeds

If the steps explanation is closed (row 3), the project stops tuning the MMLU/Tulu recipe and moves
to external validity. That decision, not a better DSMC number, is the purpose of this run.

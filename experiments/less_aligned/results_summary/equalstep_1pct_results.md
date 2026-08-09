# 1% equal-step sensitivity — RESULT (pre-registered, `prereg_1pct_equalstep.md`)

Completed 2026-08-09. Launch commit `90256c8`; plan `pilot1pctES_run_plan.json` (20 cells / 15
adapters: 10 DSMC + 5 shared Random-K); table `pilot1pctES_aggregate.csv`. 15/15 trained, 15/15
evaled, 20/20 cells, **0 failures**. Verified `max_steps=420` was live (training bar showed `/420`).
Everything else frozen and identical to the 84-step 1% arm (same subsets, draws, seeds, LoRA, batch,
`lr=2e-5`, linear schedule, `warmup_ratio=0.03`, same eval).

## Absolute performance (balanced, mean over 10 draws; base-model reference = 0.4003)

| method | 1% @ 84 steps | 1% @ 420 steps | Δ (420−84) | 420 vs base |
|--------|---------------|----------------|-----------|-------------|
| DSMC | 0.4017 | **0.3826** | **−0.0191** | **−0.0177** |
| Random-K | 0.4095 | **0.3827** | **−0.0267** | **−0.0176** |

## PRIMARY judgment quantity `J_i` (fixed in advance)

`J_i = [DSMC − Random]₍₁%,420,i₎ − [DSMC − Random]₍₁%,84,i₎`, over the 5 direction-averaged blocks:

| block (seed) | @84 | @420 | `J_i` |
|--------------|-----|------|-------|
| idx0 (42) | −0.0088 | +0.0009 | +0.0096 |
| idx1 (1)  | −0.0057 | +0.0033 | +0.0089 |
| idx2 (2)  | −0.0100 | +0.0146 | +0.0246 |
| idx3 (3)  | −0.0042 | −0.0126 | −0.0084 |
| idx4 (4)  | −0.0101 | −0.0068 | +0.0033 |
| **mean**  | **−0.0077** | **−0.0001** | **+0.0076** (J_i>0 in 4/5 blocks, median +0.0089) |

DSMC-vs-Random goes from **0/10 cells** won at 84 steps to **5/10** at 420 steps, i.e. the gap closes
to a dead heat (mean −0.0001, essentially the same as the 5% condition's −0.0001).

## Verdict against the pre-registered rules

The relevant pre-registered row is **#4 — "both degrade ⇒ 20 epochs at K=2707 is simply
over-training; inconclusive for the confound."** That is what happened: `J_i` is positive (+0.0076,
4/5 blocks), but **only because Random degrades MORE (−0.0267) than DSMC (−0.0191)** — DSMC did **not**
improve. Both land at ≈0.3826/0.3827, i.e. **~1.8 pp BELOW the no-SFT base model (0.4003)**.

Per the rules fixed in advance, this must **not** be reported as a DSMC win:

- Rule #1 (DSMC rises, Random flat ⇒ under-optimization mattered): **not satisfied** — DSMC fell.
- Rule #2 (Random falls from overfitting, DSMC doesn't rise ⇒ "ordering is compute-dependent", not a
  DSMC win): partially describes it, and its conclusion applies.
- Rule #4 (both degrade ⇒ over-training, inconclusive for the confound): **this is the case.**

**Conclusion:** repeating 2,707 examples for ~20 epochs over-trains and hurts both methods, pushing
both below the base model. The equal-step arm therefore **does not rescue the 1% result and does not
isolate the compute confound cleanly** — it introduces a new pathology (over-training) instead. The
original fixed-epoch 1% result stands as the primary finding, unchanged:

> at 1%, DSMC shows no advantage over Random-K (mean −0.0077, 0/10 cells).

What we *can* add: the DSMC−Random ordering at 1% is **not robust to optimization horizon** — it moves
from −0.0077 (84 steps) to −0.0001 (420 steps). So the magnitude of the 1% deficit is partly
compute-dependent, even though neither step count makes DSMC beat Random.

## Consequence for the project (per advice_0808 step 4)

The optimization-side explanation is now **closed**: neither the pre-registered fixed-epoch setting
nor the equal-step setting produces a DSMC advantage over Random, and the equal-step setting is itself
degenerate (both below base). **We stop tuning the MMLU/Tulu recipe here** — no LR / LoRA / epoch
grids, as pre-committed. The next meaningful step is external validity: a second candidate pool, a
second eval family, or a setting where the query distribution and the evaluation distribution actually
coincide.

## Reporting commitments honored

Both budgets and both step counts are reported regardless of outcome; the base-model reference is
always shown; n=5 blocks, descriptive only, no significance claims; nothing about the method,
selection, seeds, or evaluation was changed in response to any result.

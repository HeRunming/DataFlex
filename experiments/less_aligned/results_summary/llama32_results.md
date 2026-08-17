# Llama-3.2-3B second model-stack confirmation: RESULTS

**The core geometry→utility conditions of Outcome A replicate.** DSMC minimizes the target second-moment
geometry it optimizes in **3/3 draws** and is nonetheless **last** downstream, while **Random-K is best**.
The geometry–utility reversal reported on Llama-2-7B is **not a Llama-2 pathology.**

> ⚠️ **RETRACTION (advice_0817).** An earlier version of this document, and commit `e0ec06d`, claimed that
> "pre-registered Outcome A fired". **That was overstated.** The prereg defines Outcome A as
> D2(DSMC) < D2(Random) **and** Acc(DSMC) < Acc(Random) **and the operational query surrogate improving**.
> Only the first two were computed here. The third pre-registered condition — wrapped query CE — plus the
> same-query CoT EM and bare-context CE diagnostics had **not been run for Llama-3.2** when that claim was
> made. Until they are, the correct statement is the one above: *the core geometry→utility conditions
> replicate*, not *full Outcome A fired*. See `llama32_diagnostics.md`.

All numbers were sealed until 24/24 adapter evals + 1/1 base eval completed. Zero failures, zero driver
restarts, every evaluation gated at exactly 27 subtasks / 5,209 effective examples.

## Primary outcome

Held-out BBH **micro exact_match**, frozen 5,209-example split. Statistical unit is the query/selection
**draw (n=3)**; the two SFT seeds are averaged **within** a draw first.

**no-SFT reference (this model stack): 0.471108**

| method | draw0 | draw1 | draw2 | mean | Δ vs base |
|---|---|---|---|---|---|
| **Random-K** | 0.4732 | 0.4661 | 0.4717 | **0.4703** | **−0.08 pp** |
| Second-RR | 0.4662 | 0.4623 | 0.4718 | 0.4668 | −0.44 pp |
| First-RR | 0.4611 | 0.4652 | 0.4704 | 0.4656 | −0.55 pp |
| **DSMC** | 0.4598 | 0.4588 | 0.4657 | **0.4614** | **−0.97 pp** |

- **Every method is below the no-SFT base**, as on Llama-2.
- **DSMC − Random = −0.889 pp**, negative in **all three draws** (−0.0134 / −0.0073 / −0.0060).
- **0/3 draw blocks favour DSMC.**
- Ranking is **exactly** the Llama-2 ordering among these four arms: Random > Second-RR > First-RR > DSMC.

## Diagnostic 1: D2 geometry — the dissociation

`D2(S,Q_d) = ‖M_S − M_{Q_d}‖²_F` on unit-normalized 8,192-d projected gradients — the identical
definition used for Llama-2, so the two stacks are directly comparable.

| draw | lowest D2 | best accuracy | Spearman(D2, acc) all-4 | primary-4 | targeted-3 |
|---|---|---|---|---|---|
| 0 | **DSMC** | Random-K | +0.800 | +0.800 | +0.500 |
| 1 | **DSMC** | Random-K | +1.000 | +1.000 | +1.000 |
| 2 | **DSMC** | Second-RR | +0.200 | +0.200 | +0.500 |

**DSMC has the lowest D2 in 3/3 draws.** Pooled Spearman = **+0.400**. Positive Spearman means *lower D2
(better geometry) goes with lower accuracy* — the wrong direction, on a second model stack.

Draw 2 is the weakest cell (+0.200): Random-K and Second-RR are nearly tied there (0.4717 vs 0.4718), so
the ranking is fragile. Reported as-is; **no significance is claimed** — 4 methods × 3 draws is a small
ranking sample and these are descriptive statistics.

## Which pre-registered outcome this is

| # | pre-registered condition | fired? |
|---|---|---|
| **A** | D2(DSMC) < D2(Random) **and** Acc(DSMC) < Acc(Random) **and** the operational query surrogate improves | ⚠️ **first two conditions YES (3/3 draws); the surrogate condition was not yet computed** |
| B | DSMC best on D2 but downstream ≈ Random | no — DSMC is clearly worse |
| C | DSMC beats Random | no |
| D | DSMC no longer minimizes D2 | no — it still does, in 3/3 |

Outcome A was pre-specified as the *strongest replication*. Its **geometry→utility** half is confirmed on
3/3 draws; its **operational-surrogate** half is pending the three evaluation-only diagnostics. B/C/D are
already excluded: DSMC is clearly worse than Random and still minimizes D2.

## What this does and does not license

**Does:** the central claim is no longer a single-model case study. Across the **two tested model stacks** —
Llama-2-7B (7B, 32k vocab, `llama2` template) and Llama-3.2-3B (3.2B, 128k vocab, `llama3` template) — the
method that best minimizes the target second moment is the worst downstream, and the target-independent
baseline is best. Wording: *better target-gradient alignment does **not reliably translate** into better
downstream utility*, **not** "unreliable in general".

**Does not:**

- This is a **model-stack** confirmation, not an architecture-only ablation — model, tokenizer and
  serialization move together, and D2c showed serialization is load-bearing.
- Effect sizes are **smaller** here (DSMC−Random −0.89 pp vs −2.94 pp on Llama-2; all methods within
  1 pp of base vs up to 3.6 pp). The *direction* replicates; the *magnitude* is attenuated. This must be
  stated, not smoothed over.
- 24 cells are **not** 24 independent replicates. n=3 draws carry the inference; seeds show SFT
  stochasticity only.
- No mechanism is identified.

## Frozen provenance

| | |
|---|---|
| base | Llama-3.2-3B, ModelScope `LLM-Research` mirror (**no** bit-equivalence claim to Meta's gated checkpoint) |
| warm-up | 1692 steps / 4 epochs; adapter `6300e3cd…`, optimizer `8d739818…` |
| candidate datastore | (270679, 8192) float32, SHA256 `bcbb3a0f2f2b371f…`, finite, 0 zero rows |
| features | candidate **Adam-aware**, target **SGD**, proj 8192 / seed 123, cutoffs 3072 / 2048 |
| selections | DSMC `alpha=0.0`; RR `perm_seed=6000+d`, shared query_order; Jaccard 0.039–0.062 vs Llama-2 → genuinely model-specific |
| Random-K | **byte-identical frozen Llama-2 indices** (seeds 5000+d), regeneration-verified — one constant data baseline across stacks |
| adapters | 24/24 at 84 steps, r128/α512/dropout 0.05, {q,k,v,o}_proj, all hashes distinct |
| evals | 25/25 at exactly 27 subtasks / 5,209 examples |

## Stop rule

Per the prereg: **all large experiments stop here.** No third model, no third task, no LR sweep, no new
selector, no method or hyperparameter change — whatever the diagnostics show does not license any of them
either. The only remaining scientific work is the three pre-registered evaluation-only diagnostics
(wrapped query CE, same-query CoT EM, bare-context CE); after that, writing.

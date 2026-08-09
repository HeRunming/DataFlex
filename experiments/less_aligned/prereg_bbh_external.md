# PRE-REGISTRATION: BBH external-validation experiment

**Status: PRE-REGISTERED — NO COMPUTE RUN.** Split/draw artifacts are generated
(`data/bbh_external/`, `scripts/gen_bbh_external_split.py`); no gradients, no selection, no SFT.
Frozen per code_review_0809. Awaiting review before any compute.

## Purpose

Every existing result comes from one target family (MMLU STEM/Humanities skew) + one pool (Tulu 270k)
+ one model (Llama-2-7B). The single largest scientific gap is therefore: **is the finding an
MMLU-family artifact?** This experiment answers that on a genuinely different benchmark family, and
simultaneously fixes two inferential weaknesses of the MMLU design (draw/seed confounding; forced
global disjointness).

The question is *not* "can DSMC beat Random somewhere". Both outcomes are informative:

- DSMC beats Random on BBH ⇒ target awareness does pay off in a query-aligned setting; the MMLU
  Random-parity is specific to that setup.
- DSMC still loses to Random but still beats the other targeted selectors ⇒ the unified, stronger
  claim: *second moments improve targeted selection, but target-aware selection is brittle vs Random
  across families.*

## Why BBH (and not MMLU-Pro or TyDiQA)

- **BBH**: different family (multi-step reasoning); data local and verified to be the **official
  23-task suite**; candidate-pool contamination against BBH test 5/270,679 = 0.000018; 113 lm_eval
  configs; used in the recent controlled study.
- **MMLU-Pro**: has an official validation/test split (70 / 12,032) — but **70 validation examples
  cannot support three M=64 draws**, and **6,810 of 12,032 test items are inherited from original
  MMLU**, so it is a weak "second family".
- **TyDiQA**: shipped `dev`/`test` both point at the GoldP dev file, and it is where second-order
  methods looked *favorable* in our old runs ⇒ cherry-picking risk if used as the sole confirmation.

## Data split (generated, deterministic)

Master seed **20260809**. One per-task-stratified split of the official suite:

| | value |
|---|---|
| local files / official tasks | 27 files → **23 tasks** (`logical_deduction`, `tracking_shuffled_objects` each ship 3 size-variants) |
| total examples | 6,511 |
| **query reservoir (20%)** | **1,302** |
| **held-out evaluation (80%)** | **5,209** |
| query ∩ eval | **∅ (verified)** |

Three **independently sampled** query draws, M = **64**, drawn without replacement *within* a draw from
the fixed reservoir; overlap *across* draws allowed and reported (we do **not** force global
disjointness — that induced negative correlation in the MMLU design):

| draw | official tasks covered | pairwise overlap |
|------|------------------------|------------------|
| draw0 | 22 / 23 | 0–1: **2** |
| draw1 | 20 / 23 | 0–2: **3** |
| draw2 | 22 / 23 | 1–2: **2** |

Expected overlap 64²/1302 ≈ **3.1** — observed 2–3, i.e. genuinely independent realizations.

## Design (frozen)

**3 query draws × 2 SFT seeds × 5 methods = 30 adapters**, plus **1 shared no-SFT reference**.

- **Seeds {42, 1} fully CROSSED with draws** — every draw is trained under both seeds. This decouples
  query-realization variance from training-seed variance, which the MMLU design confounded (there each
  draw index was welded to one seed).
- **Methods** (5): `DSMC`, `Second-RR`, `First-RR`, `LESS-style TopK`, `Random-K`.
  First-RR is retained deliberately: recent work finds gradient representation + greedy round-robin is
  a strong low-budget comparator.
- **Budget: K = 2707 only** (one budget; matches our 1% arm and the literature's low-budget regime).
- **4 epochs**, and every other SFT setting identical to the frozen recipe: LoRA r128 / α512 /
  dropout 0.05 on q,k,v,o; per-device 4 × accum 4 × 8 GPUs = eff. batch 128; lr 2e-5, linear,
  warmup_ratio 0.03; bf16; cutoff 2048; base Llama-2-7B; warm-up checkpoint-1692 (hash-pinned).
- **Candidate pool unchanged** (Tulu 270k) so only the target/eval axis moves.

## Prompt alignment (fixes an MMLU limitation)

In the MMLU experiments target gradients were single-example supervised (0-shot) while evaluation was
5-shot. For BBH we align them as far as the supervised target format permits:

- **pin** the lm-eval commit and the official `bbh_cot_fewshot` task/config;
- construct query gradients with the **same task instruction / few-shot context** as the evaluation
  template;
- the gold continuation used to form the loss is fixed **now**, in this document: the task's `target`
  string as the assistant continuation of the pinned prompt;
- **CoT vs direct is fixed to the pinned `bbh_cot_fewshot` setting and will not be changed after seeing
  any base-model BBH accuracy.**

Record `target_num_fewshot` and `evaluation_num_fewshot` in every draw manifest.

## Metrics and analysis (fixed in advance)

- **Primary**: the pinned lm-eval **BBH group metric** on the **held-out BBH evaluation split**
  (5,209 examples). All 23 task-level scores are also saved. **No custom aggregate will be invented.**
- **Secondary / diagnostics logged for every cell**: query loss, `D2` to the query and to a balanced
  reference, downstream score, subset source composition, effective rank, token counts.
- **Paired analysis**: DSMC − method within each (draw, seed) cell ⇒ 6 paired observations per
  comparison. Report per-cell values, mean, median, win counts, and separately the variance
  attributable to query realization vs SFT seed. Descriptive only — **no p-value thresholds**.
- **Absolute reference**: every method also reported as Δ vs the shared no-SFT baseline, so
  "improves on base" and "degrades least" stay distinguishable.

## Naming honesty

Because we carve the query reservoir out of official BBH examples, results are reported as a
**"held-out BBH external-validation split"**, never as an official full-BBH leaderboard score.

## Gates before compute

1. contamination re-run against the **final held-out BBH evaluation split** (preliminary pool-vs-BBH
   13-gram pass: 5/270,679);
2. this pre-registration + the split artifacts reviewed and approved;
3. then, and only then: target-gradient extraction → 5 selections/draw → 30 adapters → eval →
   aggregate.

## Cost estimate

30 adapters × (K=2707, 4 epochs ⇒ ~84 optimizer steps, ~15 min) + 31 evals (CoT generation over 5,209
held-out examples, slower than MC log-likelihood) ⇒ **~15–20 h** total.

## What will NOT be done

No LR / LoRA / epoch / budget sweeps. No modification of DSMC (in particular, **no source-balanced
DSMC variant** — that idea arose from looking at Random's source composition and would be post-hoc
method tuning; it is an exploratory follow-up, not part of this experiment). No second family in this
round.

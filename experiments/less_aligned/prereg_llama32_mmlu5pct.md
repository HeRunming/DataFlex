# STOP-RULE AMENDMENT #1 and PRE-REGISTRATION: Llama-3.2-3B × MMLU at 5%

**Written and committed BEFORE any Llama-3.2 MMLU computation.** No target gradients, no selections, no
SFT, no evaluation had been run for this arm when this document was frozen.

## The amendment, stated openly

The prior stop rule said *all large experiments stop* after the Llama-3.2 BBH arm. This amends it **once**,
transparently and in advance:

> During paper drafting we identified one **asymmetric evidence gap**: the central *negative* result — that
> better target-gradient alignment does not guarantee downstream utility — is now **cross-stack**
> (Llama-2-7B and Llama-3.2-3B on BBH). But the one *positive* method-side result — that directional
> second-moment matching beats first-order targeted selection on MMLU — is still **Llama-2 only**. We
> therefore add exactly one final scoped validation: **Llama-3.2-3B × MMLU at 5%**.

This is **validation of an existing claim**, not a search for a favourable setting. It was chosen for the
axis where our own evidence is *weakest*, and it can only narrow the DSMC claim, never the paper's centre.

**Hard prohibitions, binding regardless of outcome:**

- **No Llama-3.2 MMLU 1% follow-up.** Ever. 1% would mainly re-answer "is Random strong at low budget?",
  which two BBH stacks already settle; and budget sweeps are not this paper's subject.
- No new methods, controls, matched-Random variants, LR/epoch tuning, or third model/task.
- This experiment **may refine the scope of the positive DSMC claim** but **may not change** the paper's
  central *"matching the target is not enough"* framing.

## Frozen design

Reproduce the **historical MMLU design exactly**; only the model stack changes.

| | |
|---|---|
| budget | **K = 13,533** (5%) |
| target draws | the **same ten**: `stem80_draw{0..4}` + `hum80_draw{0..4}` |
| draw → training seed | **{0→42, 1→1, 2→2, 3→3, 4→4}** — the original mapping, **not** the BBH 2-seed crossed design |
| methods | **DSMC, First-RR, Second-RR, Random-K** + one shared no-SFT reference |
| excluded | LESS, GIST, NICE, Random-K-LengthMatched, and anything new |
| SFT | unchanged: 4 epochs, LoRA r128/α512/dropout 0.05, lr 2e-5 linear, warmup 0.03, eff. batch 128, bf16 |

Keeping the original seed mapping is deliberate: it makes the Llama-2 ↔ Llama-3.2 comparison a clean
single-axis change (design fixed, stack varies).

**Cell accounting.** 10 direction cells × 4 methods = **40 analysis cells**, but only **35 unique
adapters**: Random-K is target-independent, so per the original design one Random subset is **shared
between the STEM and HUM directions** of each draw index (`randk_drawidx{0..4}`) → 10×3 = 30 targeted
adapters + 5 Random adapters. Verified against `pilot_run_plan.json`, which shows exactly this sharing and
exactly this seed mapping.

## Reuse vs rebuild

**Reused, already hash-pinned:**

- the Llama-3.2 **warm-up** (adapter `6300e3cd…`, optimizer `8d739818…`);
- the Llama-3.2 **candidate-gradient datastore**, (270679, 8192), SHA256 `bcbb3a0f2f2b371f…`. Candidate
  features depend only on model stack, warm-up and candidate pool — **not** on the target task — so
  reusing it is correct, and it saves the dominant cost;
- **Random-K reuses the exact frozen Llama-2 5% indices** (`randk_drawidx{0..4}`), giving the two stacks
  one genuinely constant data baseline.

**Rebuilt for Llama-3.2:**

- all **ten MMLU target-gradient caches**;
- its own **DSMC / First-RR / Second-RR** selections;
- 35 adapters and the MMLU evaluations.

## Feature/protocol contract — reproduce, do not "improve"

| | value |
|---|---|
| candidate gradients | **Adam-aware** |
| target gradients | **SGD** |
| projection | dim **8192**, seed **123** |
| target draws | the same 64-example draws |
| target-gradient format | the **original MMLU** single-example supervised form, `num_fewshot=0` |
| downstream eval | the **original 5-shot** MMLU |
| serialization | Llama-3.2's own **`llama3`** template |
| RR permutation seed | **`3000 + draw_index`**, shared by First-RR and Second-RR and by the STEM/HUM directions of an index |

**Recovered, not guessed.** The MMLU RR seed is `3000+i`, **not** the BBH arm's `6000+d`; verified by
reading `perm_seed` out of all ten frozen Llama-2 MMLU selections (3000…3004 for both directions, with
First-RR and Second-RR sharing a byte-identical `query_order` within each draw). Reusing the BBH seed here
would silently change the RR arms.

> ⚠️ **Explicitly forbidden:** "fixing" the MMLU target-gradient format to a 5-shot form because the BBH
> work later taught us about prompt alignment. That would change the **stack and the target protocol at the
> same time**, destroying the cross-stack replication this experiment exists to provide.

**Configs are derived structurally, not hand-written**, from the verified Llama-2 MMLU configs. Only
model-stack fields, the Llama-3.2 candidate-cache path, Llama-3.2 target-cache/output paths and the
template may be overridden; the **key set must be unchanged** and drift outside that allowlist must fail.
This is the guard that caught the silent dynamic-selection no-op in the BBH arm.

## Analysis, frozen now

- **Primary metric:** balanced MMLU accuracy = **(STEM + HUM) / 2**.
- **Primary descriptive unit: the 5 draw-index blocks.** For each index, average the stem-majority and
  hum-majority directions first, then report five **paired** block differences. **Do not claim n=10** —
  the two directions of an index are not independent replicates.
- **Three pre-registered comparisons:**
  - $\Delta_{\text{rep}}$ = DSMC − First-RR — does the *second-order representation* advantage transfer?
  - $\Delta_{\text{MMD}}$ = DSMC − Second-RR — does the *MMD coreset* add anything beyond second order?
  - $\Delta_{\text{rand}}$ = DSMC − Random-K — the target-awareness anchor.
- The shared Llama-3.2 no-SFT MMLU reference is reported but does **not** enter block counts.
- Descriptive only: **no p-values, no significance claims.**

## Outcomes: all four pre-registered before any computation

| # | outcome | interpretation |
|---|---|---|
| **1** | DSMC > Second-RR **and** > First-RR | the positive MMLU method result transfers across two model stacks |
| **2** | DSMC ≈ Second-RR, both > First-RR | what replicates is the **second-order representation**, not the extra MMD-coreset gain; scope the DSMC claim to the representation |
| **3** | First-RR or Second-RR ≥ DSMC | DSMC's MMLU method advantage is itself **stack-dependent**; lower the method claim further |
| **4** | Random-K ≥ the targeted methods | further strengthens "target awareness is unreliable" — and **may not** trigger any method change |

Outcome 3 is **not** a failure. It would make the paper *more* unified: *second moments helped on one
stack, but even the representation advantage is model-stack dependent; what replicates robustly is the
failure of geometric target alignment to guarantee utility.*

Outcome 1 gives the strongest combination: *directional second-moment matching is a reproducible targeted
selection improvement on MMLU across two stacks, yet it still does not beat strong Random and does not
transfer to BBH.*

**No outcome may trigger tuning, a 1% follow-up, a new method, or a change to the paper's centre.**

## Drafting continues in parallel

The paper is written now, not after this arm. Per `advice_0817`, the second-model evidence already closed
the "Llama-2 pathology" attack; this arm sharpens the *method* claim's scope only.

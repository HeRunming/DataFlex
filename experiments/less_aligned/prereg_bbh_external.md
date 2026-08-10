# PRE-REGISTRATION: BBH external-validation experiment

**Status: ON HOLD — NO COMPUTE RUN.** Artifacts, pins, and gates are built; **no gradients, no
selection, no SFT, no evaluation**. The previous `GO_FOR_SELECTION_CANARY = true` is **withdrawn**:
code_review_0810 identified a protocol defect that the string-level parity audit could not see, and
adversarial self-review of the new gates then surfaced a second one. Both reproduce; both are recorded
as machine-checked blockers, so the launch manifest now emits
`GO_FOR_SELECTION_CANARY = false`.

| # | blocker | verdict artifact |
|---|---|---|
| 1 | 7/192 query prompts truncated at `cutoff_len=2048`, destroying the query itself | `bbh_token_truncation_audit.json` → **HOLD** |
| 2 | a near-verbatim CoT demonstration carries the **opposite** gold answer to a scored draw1 query | `bbh_fewshot_leakage_audit.json` → **REVIEW** |

Neither was auto-fixed. `cutoff_len`, the few-shot count, the demonstrations, and the prompts are all
untouched; both protocol decisions are deferred to explicit human approval.

## ⛔ BLOCKER: 7/192 query prompts are truncated so badly the query itself is destroyed

`scripts/audit_bbh_token_truncation.py` → `bbh_token_truncation_audit.json`, verdict **HOLD**.

The frozen query-gradient recipe is LlamaFactory `template: llama2` with **`cutoff_len = 2048`**. BBH
CoT prompts are structured `[3 CoT demonstrations] ++ [the actual query LAST]`, and LlamaFactory
truncates the **source tail** (`source_ids[:source_len]`, allocated by `infer_seqlen`). So for an
over-length record the part that gets deleted is exactly the part that matters:

| | |
|---|---|
| records audited | 192 (all 3 draws) |
| **materially truncated** | **7 / 192 — every one in `geometric_shapes`** |
| tokens lost | 493–560 of ~2,540–2,608 |
| the record's own query survives? | **NO** |
| trailing `A: Let's think step by step.` cue survives? | **NO** |
| supervised target survives? | yes (targets are 1–4 tokens) |

For those 7, **the query is 100% absent** — not merely clipped. For `bbh::geometric_shapes::50` the
dropped 560 tokens contain all of demonstration 3's rationale *plus the entire query*: the item's unique
SVG path (`M 15.44,15.80`) does not appear anywhere in the kept text, and what survives of the query
region is only the shared `Options:\n(A) circle…` boilerplate. So the query gradient would be computed
on *~2.5 demonstrations and no query at all*, with no CoT cue — while the paired evaluation sees the
whole thing. Evaluation side is fine (max 2,596 context tokens against a 4096−1024 = 3,072 budget, 0
left-truncated), so this is a **query-gradient-side defect only**, and it is asymmetric between the two
sides: the evaluation side has ~1.5× the budget.

**Why gate B passed anyway:** it compares prompts as **strings**, i.e. before tokenization. Byte-identity
of the pre-tokenization text says nothing about what survives `cutoff_len`. This is the concrete lesson —
string parity was necessary but not sufficient.

**Per the review, nothing was auto-fixed.** `cutoff_len`, the few-shot count, and the prompts are all
untouched; the distribution is reported and the protocol decision is deferred to explicit human
approval. Options exist (raise `cutoff_len` for target extraction; drop to fewer shots for the long
subtask; accept and disclose) but each changes the frozen recipe or the prompt parity, so none is taken
unilaterally.

## ⛔ BLOCKER 2: a near-verbatim CoT demonstration carries the OPPOSITE answer to a scored query

Found by adversarial self-review of the leakage audit's own verdict policy (the first version passed on
exact-identity only, which hid this). `bbh_fewshot_leakage_audit.json` verdict **REVIEW**.

The maximum demo↔item similarity is **J = 0.8929**, `causal_judgement#1` vs
**`bbh::causal_judgement::128`**, and the two differ by exactly one phrase:

| | text | gold answer |
|---|---|---|
| few-shot demonstration | "…triggered if **at least one** person appeared in the room…" | **Yes** |
| evaluated item | "…triggered if **more than one** person appeared in the room…" | **No** |

This is a deliberate minimal-pair probe, and the perturbation **flips the answer**. The item is in
**draw1**, i.e. a scored query record. So for that item the 3-shot prompt contains a near-identical
vignette demonstrating the *opposite* conclusion — the plausible effect is **anti-leakage** (priming the
wrong answer), which is a validity problem in its own right, not a harmless near-duplicate.

5 pairs reach J ≥ 0.85 and **all 5 have a differing gold answer**. The other 4 are the benign
template-generated case (`tracking_shuffled_objects`: same swap sequence, different queried person).

Two corrections this forced in my own work:
- the earlier caveat attributed the maximum to *"template-generated subtasks"*. That was **factually
  wrong** — `causal_judgement` is hand-written. The excuse did not apply to the case it excused.
- the audit claimed to contextualize fuzzy hits against an *"item-vs-item p95 baseline"* that **was never
  computed**. It is now computed per subtask (p50/p95 over 600 sampled pairs) and stored.

The verdict now has three levels — **FAIL** (exact identity), **REVIEW** (any pair ≥ J 0.85, must be
human-cleared), **PASS** — and both non-PASS levels exit non-zero and block the launch manifest. Nothing
was auto-resolved: whether to swap the demonstration, exclude the item, or accept and disclose is a
protocol decision left to explicit approval.

### Artifact index

| artifact | what it fixes |
|---|---|
| `scripts/emit_bbh_execution_contract.py` → `bbh_execution_contract.json` | **P0-1**: the authoritative frozen feature/selector contract (Adam candidates, **SGD targets**, seed 123, dim 8192, exact per-method scripts) |
| `scripts/audit_bbh_token_truncation.py` → `bbh_token_truncation_audit.json` | **P0-2**: execution-level token/truncation gate (currently **HOLD**) |
| `scripts/audit_bbh_fewshot_leakage.py` → `bbh_fewshot_leakage_audit.json` | 81 CoT demos vs raw/reservoir/held-out/drawn populations |
| `results_summary/inherited_context_corrections.md` | corrects two misremembered historical claims |
| `scripts/gen_bbh_external_split.py` → `data/bbh_external/` | 20/80 split, 3 draws, 27/23 accounting, frozen selection seeds |
| `scripts/pin_bbh_lmeval.py` → `bbh_lmeval_pin.json`, `bbh_pip_freeze.txt` | lm-eval pin incl. **Python runtime code**, tokenizer, and 172-package environment |
| `scripts/pin_bbh_eval.py` → `bbh_eval_pin_manifest.json`, `bbh_external_tasks/` | frozen custom held-out suite (dataset source is the only change) |
| `scripts/render_bbh_query_prompts.py` → `bbh_query_prompt_manifest.json` | query prompts built from lm-eval's own `fewshot_context()` |
| `scripts/audit_bbh_prompt_parity.py` → `bbh_prompt_parity_audit.json` | 27-subtask byte-for-byte parity gates A/B/C + gate D disclosure + `--tamper_check` |
| `scripts/contamination_global_lexical.py` → `results_summary/contamination_global_lexical_*.json` | pool-wide lexical screen (32×2, **nominal** recall) |
| `bbh_external_launch_manifest.json` | launch record; asserts the 15-subset invariant, now gated on the truncation verdict |

### Verification bugs found by self-review of this checkpoint (all fixed before commit)

The gate machinery was itself adversarially reviewed, which found four real defects. Recording them
because "the gates passed" is only meaningful if the gates could have failed:

| # | defect | fix | proof |
|---|---|---|---|
| 1 | Gate B `continue`d past a missing prompt file; since `all([]) is True` it then **passed having checked nothing** — and the launch manifest read that as GO | fail-closed: missing file, or any subtask with 0 checked prompts, is a FAILURE; non-vacuity (`n>0`, all 27 covered) is part of the verdict | re-ran with the prompt dir removed ⇒ `0 prompts over 0 subtasks: FAIL`, 108 failures, exit 1 |
| 2 | `--verify` pinned only 9 files, so editing a subtask YAML or deleting held-out rows left it **green** while both the eval prompt and the eval set were corrupted | pin all **27 subtask configs + 27 held-out data files** individually (63 artifacts total) and cross-check each against the pin manifest | deleted 5 rows from `navigate_heldout.jsonl` ⇒ `--verify FAILED: subtask_data:navigate sha256 drifted` |
| 3 | `split_manifest_sha256` in the pin manifest was **stale** (pinned before the metadata enrichment), and this document transcribed the stale value | re-ran `pin_bbh_eval.py`; the document no longer transcribes it at all, so it cannot go stale again | on-disk == pinned, verified |
| 4 | `--verify` **wrote** the run plan — a read-only check mutating the tree | writes are skipped under `--verify` | "run plan NOT rewritten" |
| 5 | the vacuity fix (#1) was **not generalized** to the two new audits: an empty tasks dir gave the leakage audit `PASS`/exit 0, and a 3-record prompt dir gave the truncation audit `PASS` — either would have produced `GO=true` having checked almost nothing | explicit vacuity guards (`--expect_records 192`, `--expect_demos 81`) that refuse to emit a verdict, plus coverage blockers in the manifest | 3-record dir ⇒ `VACUITY GUARD: audited 3, expected 192`; empty tasks dir ⇒ `found 0 demonstrations, expected 81`, no artifact written |
| 6 | leakage verdict blocked on **exact identity only**, hiding Blocker 2; and it cited an "item-vs-item p95 baseline" that was **never computed** | three-level FAIL/REVIEW/PASS with a J ≥ 0.85 human-clearance threshold + answer-flip detection; p50/p95 baseline actually computed per subtask | verdict flipped to REVIEW, surfacing 5 answer-flip pairs |
| 7 | `query_start_survives` used an 80-char substring probe that also matches inside the demo region for 48/192 records — a latent false-pass | replaced with a positional check (kept prefix must extend past the final `Q:` and retain the query's own tail) | same 7 records still caught |
| 8 | the launch-manifest generator still emitted the retracted "no third hidden random axis" line and the "optionally D2(S,P_heldout)" line, contradicting the corrected prereg | both corrected in the generator, so the emitted JSON matches the prereg | re-emitted manifest |

Two overclaims were also scoped down: the chat-wrapper caveat (gate D) and the precise scope of gate B.

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

Master seed **20260809**. One per-subtask-stratified split of the official suite:

| | value |
|---|---|
| local files = **lm-eval subtasks** | **27** (primary reporting unit) |
| conceptual BBH task families | **23** (`logical_deduction`, `tracking_shuffled_objects` each ship 3 size-variants) |
| total examples | 6,511 |
| **query reservoir (20%)** | **1,302** |
| **held-out evaluation (80%)** | **5,209** |
| query ∩ eval | **∅ (verified)** |

### 23 families vs 27 subtasks — task accounting (corrected, choice_0809)

The original BBH paper describes **23 challenging tasks**. But the pinned lm-eval `bbh_cot_fewshot`
group operationally contains **27 subtasks** — `logical_deduction` and `tracking_shuffled_objects` each
enter three times (3-, 5-, 7-object) — and aggregates them with

```yaml
aggregation: mean
weight_by_size: true      # size-weighted / MICRO aggregation over the 27 subtasks
```

So **27 is the unit the primary metric is actually computed over**, and the stratified split was
(correctly) built per local file, i.e. over all 27. Therefore:

- **primary**: all **27** lm-eval subtask scores + their micro aggregate, exactly as the pinned group
  defines it;
- **secondary, diagnostic only**: an optional regroup to the 23 conceptual families. It is never the
  headline number and no custom aggregate is invented;
- every draw manifest records **both** `subtask_composition_27` and
  `conceptual_family_composition_23`.

The split itself is **not** regenerated — it was already aligned with the micro metric. Regenerating
was verified to reproduce byte-identical jsonl (all five sha256 unchanged); only metadata was enriched.

Three **independently sampled** query draws, M = **64**, drawn without replacement *within* a draw from
the fixed reservoir; overlap *across* draws allowed and reported (we do **not** force global
disjointness — that induced negative correlation in the MMLU design):

| draw | lm-eval subtasks covered | conceptual families | pairwise overlap |
|------|--------------------------|---------------------|------------------|
| draw0 | 26 / 27 | 22 / 23 | 0–1: **2** |
| draw1 | 24 / 27 | 20 / 23 | 0–2: **3** |
| draw2 | 25 / 27 | 22 / 23 | 1–2: **2** |

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

### Selection randomness — frozen now (choice_0809 item 4)

Freezing the SFT seeds is not sufficient: two of the five methods draw their own randomness, and if that
randomness moved with the SFT seed we would silently have introduced a **third** random axis
(selection realization) on top of query realization and training stochasticity, breaking the crossed
design. So both are pinned as functions of the **draw only**:

$$\text{random\_k\_seed}(d) = 5000 + d, \qquad \text{rr\_perm\_seed}(d) = 6000 + d$$

| draw | Random-K seed | RR permutation seed |
|------|---------------|---------------------|
| 0 | 5000 | 6000 |
| 1 | 5001 | 6001 |
| 2 | 5002 | 6002 |

- **First-RR and Second-RR share the same RR query visitation order** for a given draw (same
  `rr_perm_seed`), so the only difference between them is first- vs second-order similarity, not the
  visiting order.
- **The selected subset for a draw is bit-identical across the two SFT seeds.** Concretely:
  **3 draws × 5 methods = 15 frozen subsets**, each trained twice ⇒ **30 adapters**. The training seed
  is a training axis only; it must never change which examples are in the subset.
- **What this does and does not eliminate (corrected per code_review_0810).** An earlier draft claimed
  the design is "exactly (query realization) × (training stochasticity) with no third hidden random
  axis". That is too strong. Because the seeds are `5000+d` / `6000+d`, each of the three blocks carries
  a *different* Random-subset realization and a different RR visiting order — the selection randomness
  is not removed, it is **blocked with the draw index**. This is deliberate and preferable: three
  independent Random realizations are more informative than reusing one Random subset three times, which
  would understate Random's own variability. The correct description is therefore:

  > **three draw/selection-realization blocks, crossed with two SFT seeds.**

  When interpreting block spread: for the **targeted** methods it is driven mainly by query realization;
  for **Random** it is driven by the Random-subset realization. Block spread must not be reported as
  pure query-realization variance. What the frozen seeds *do* guarantee is the clean training-seed axis
  above.

Recorded per draw in `bbh_query_draw{d}.meta.json` (`random_k_seed`, `rr_perm_seed`,
`rr_perm_seed_shared_by`) and in `bbh_split_manifest.json` → `frozen_selection_seeds`. The launch
manifest re-asserts the 15-subset invariant, and it will be verified by hash after selection.

## Prompt alignment: what is aligned, and what is not

In the MMLU experiments target gradients were single-example supervised (0-shot) while evaluation was
5-shot. For BBH we remove the *prompt-context* half of that mismatch, and we state precisely the half
that cannot be removed. The earlier phrasing ("query-gradient prompt aligned to eval template", "fixes
MMLU prompt mismatch") overclaimed and is retracted.

**The honest statement, which is what we will write in the paper** (phrasing per code_review_0810):

> Query and evaluation are drawn from the same task distribution and share the same **pre-wrapper** CoT
> prompt context; their **executed token sequences are not identical**. Query gradients supervise only
> the provided final BBH target.

We therefore avoid the bare word "query-aligned", which invites a token-level reading that is not true.
Three distinct gaps, all disclosed: the supervised continuation (below), the llama2 chat wrapper (gate
D), and — pending the gate-3 HOLD — `cutoff_len` truncation on the gradient side only.

Concretely:

| | aligned? |
|---|---|
| task `description` | ✅ byte-identical |
| the 3 hard-coded CoT few-shot exemplars (which *do* contain full rationales) | ✅ byte-identical |
| `Q:` / `A:` delimiters, blank-line separators | ✅ byte-identical |
| trailing `A: Let's think step by step.` cue | ✅ byte-identical |
| **supervised continuation / generation trajectory** | ❌ **not aligned** |
| **chat wrapper at gradient-extraction time** | ⚠️ **differs — disclosed below** |

The ✅ rows are byte-identical **as prompt strings**, which is exactly what gates A and B verify.

**The chat-wrapper caveat (found in self-review; do not drop it).** Query prompts are consumed through
llamafactory with `template: llama2`, whose user slot is `{bos_token}[INST] {{content}} [/INST]`, whereas
lm-eval is invoked **without** `--apply_chat_template` and sends the context verbatim. So the token
sequence a query gradient is actually taken on is the evaluation context **wrapped** in
`<s>[INST] … [/INST]`. This is identical to what the MMLU arm did, so it is **not a new confound** and
not a launch blocker — but the precise claim is *"byte-identical up to the llama2 chat wrapper applied at
gradient-extraction time"*, not unqualified byte-identity. Measured and recorded as **gate D** in
`bbh_prompt_parity_audit.json`.

The gap is intrinsic to BBH: raw BBH items ship only a **final** target (`(C)`, `14`, `Yes`) and there
is **no gold CoT rationale per test item**. Evaluation therefore scores a *generated* rationale ending
in `So the answer is X`, while the query-gradient loss supervises the bare final answer. We do **not**
fabricate teacher rationales to close this — that would inject another model's reasoning style into the
target signal, a strictly larger confound than the mismatch it removes.

Implementation reduces the chance of drift rather than merely asserting alignment:
`scripts/render_bbh_query_prompts.py` builds query prompts by calling **lm-eval's own
`Task.fewshot_context()`** on the pinned task objects, so there is no second copy of the prompt logic.
`scripts/audit_bbh_prompt_parity.py` then re-verifies byte-equality of the stored artifact
independently (gates A/B/C/D below).

`target_num_fewshot` and `evaluation_num_fewshot` are recorded in every draw manifest (both **3**).

**CoT vs direct is fixed to the pinned `bbh_cot_fewshot` setting and will not be changed after seeing
any base-model BBH accuracy.**

## Pinning the evaluation (choice_0809 item 3) — verifiable, not a prose promise

"Pin lm-eval" was previously only a written intention with no SHA. It is now an artifact. This matters
concretely: lm-eval **has** changed the BBH CoT prompts historically (a duplicated
"Let's think step by step." in the few-shot text was removed in 2025), so the installed text — not the
upstream default — is the ground truth for this run.

Recorded in `bbh_lmeval_pin.json` and `bbh_eval_pin_manifest.json`:

| pinned | value |
|---|---|
| lm-eval version | **0.4.5** (installed from wheel; not a git checkout, so no repo SHA exists — recorded as such rather than faked) |
| group YAML `_bbh_cot_fewshot.yaml` | sha256 `4434c2bf…` |
| template `_cot_fewshot_template_yaml` | sha256 `361f566f…` |
| all **27** subtask YAMLs | sha256 each |
| hard-coded few-shot samples | sha256 each (per subtask) |
| raw local BBH data (27 files) | sha256 each |
| split manifest | sha256 recorded in `bbh_eval_pin_manifest.json` → `split_manifest_sha256` (re-pinned after the metadata enrichment, and re-checked by `emit_bbh_launch_manifest.py --verify`; not transcribed here, so this document cannot go stale against it) |
| generation | `num_fewshot=3`, `generate_until`, greedy (`do_sample=false`, `temperature=0`), `max_gen_toks=1024`, `until=["</s>","Q","\n\n"]` |
| scoring | `get-answer` regex filter → `exact_match`, micro group aggregation |

### The custom held-out suite (required, not optional)

We **cannot** call stock `bbh_cot_fewshot` for the final evaluation: its test split is the *full* BBH,
whereas we must evaluate exactly our 5,209-example held-out carve-out (the complementary 1,302 are the
query reservoir). So `scripts/pin_bbh_eval.py` emits a frozen custom suite
(`experiments/less_aligned/bbh_external_tasks/`, 27 subtask YAMLs + group) in which **only the dataset
source changes** — prompt, few-shot samples, generation, filtering, metric, and micro aggregation are
inherited verbatim from the pinned config.

That "only the dataset source changes" claim is **audited, not asserted** — see gates below.

## Metrics and analysis (fixed in advance)

- **Primary**: the pinned lm-eval **micro (`weight_by_size: true`) group metric** on the **held-out BBH
  evaluation split** (5,209 examples), computed over the **27 lm-eval subtasks**. All **27** subtask
  scores are saved. **No custom aggregate will be invented.** The 23-family regroup may be reported as a
  secondary diagnostic only.
- **Secondary / diagnostics logged for every cell**: query loss, downstream score, subset source
  composition, effective rank, token counts, and `D2` — defined explicitly below.

### The `D2` reference, defined explicitly (choice_0809; narrowed per code_review_0810)

The MMLU forensics used "`D2` to a balanced reference", where *balanced* had a concrete meaning
(STEM/HUM 50/50). **BBH has no such natural balance**, so carrying that phrase over would be inventing
an arbitrary reference to preserve a habit. It is dropped. Exactly one geometric diagnostic is
pre-registered:

- $D_2(S, Q_d)$ — the selected subset against **its own query draw**. Well-defined, and it is the
  quantity DSMC's objective actually targets, so it is what the surrogate/outcome question needs.

**$D_2(S, P_{\text{heldout}})$ is NOT part of this experiment.** A previous draft listed it as
"optional, if the extraction cost is acceptable". That is precisely the shape of decision that must not
be left open: a diagnostic whose run/skip choice can be made *after* seeing results is not
pre-registered. It is therefore dropped outright rather than left conditional. If it is ever wanted, it
requires its own pre-registration.

### Paired analysis — descriptive crossed summaries, not variance components

DSMC − method within each (draw, seed) cell ⇒ **6 paired observations** per comparison. The 3×2 crossed
design is genuinely stronger than the MMLU design, but **each (draw, seed) cell holds exactly one
observation**, so it cannot support a variance-component decomposition. The earlier phrasing
("separately the variance attributable to query realization vs SFT seed") overclaimed and is retracted.
Instead we report, descriptively:

- per-cell values, mean, median, and win counts (out of 6);
- the average **over seeds within each query draw** (3 numbers) ⇒ **query-draw spread**;
- the average **over draws within each SFT seed** (2 numbers) ⇒ **seed sensitivity**;
- optionally a purely **descriptive** two-way (draw × seed) table;
- **no variance-component inference, no p-value thresholds, no significance claims.**

- **Absolute reference**: every method also reported as Δ vs the shared no-SFT baseline, so
  "improves on base" and "degrades least" stay distinguishable.

## Naming honesty

Because we carve the query reservoir out of official BBH examples, results are reported as a
**"held-out BBH external-validation split"**, never as an official full-BBH leaderboard score.

## Gates before compute

### Gate 1 — contamination against the final held-out BBH split: **PASSED**

Both a 13-gram containment pass and a **higher-sensitivity** pool-wide fuzzy screen, against the actual
5,209 held-out examples:

| screen | result |
|---|---|
| 13-gram containment | 5 / 270,679 = 0.0000185 — all 5 manually confirmed **false positives** (flan_v2 movie-plot QA / gender-coreference boilerplate) |
| pool-wide MinHash/LSH → exact Jaccard, **64 perms / 32 bands × 2 rows** | **0** at J ≥ 0.5 and **0** at J ≥ 0.3 |

The earlier 16 bands × 4 rows configuration was too weak to support the claim made from it: LSH is
*probabilistic* candidate generation, so detection probability is $1-(1-s^{\text{rows}})^{\text{bands}}$
— only **12.2%** at J=0.3 and **64.4%** at J=0.5. At 32×2 it is **95.1%** and **99.99%**.

**These are NOMINAL figures under the ideal MinHash model, not guarantees** (tightened per
code_review_0810). Two independent reasons: (i) LSH recall is below 100% by construction; and (ii) the
implementation computes `(a*h + b) mod (2^61-1)` in numpy `uint64`, so the **multiply wraps** — verified
directly against exact Python integer arithmetic, which disagrees — meaning the hash family is not
exactly min-wise independent and the textbook recall formula does not strictly apply. Because every
LSH collision is verified with **exact** shingle Jaccard, there are no false positives; it is only
*recall* that is un-guaranteed. The claim is therefore *"no fuzzy lexical near-duplicates detected by a
high-sensitivity screen"* — not a proof that none exist. Making the arithmetic formally safe (or
switching to `datasketch`) would upgrade this from nominal to guaranteed; it is not a launch blocker
because normalized-exact is 0, the long-n-gram hits are 5 manually-confirmed false positives, and this
screen is also 0. Semantic (embedding-NN) contamination remains a disclosed release-time limitation.
Artifacts: `results_summary/contamination_global_lexical_bbh_heldout.json`, `contamination_audit.md`.

### Gate 1b — few-shot demonstration leakage: ⚠️ **REVIEW** (see Blocker 2)

`scripts/audit_bbh_fewshot_leakage.py` → `bbh_fewshot_leakage_audit.json`. Gate C only proved that the
reservoir and held-out split are disjoint *from each other*; it never asked whether the **81 hard-coded
CoT demonstrations** (27 subtasks × 3) are themselves BBH evaluation or query items — which would put an
evaluated item's gold answer (or, as it turns out, the *opposite* answer) into its own prompt.

| population | normalized-exact | fuzzy J ≥ 0.5 | J ≥ 0.85 (needs clearance) |
|---|---|---|---|
| raw suite (6,511) | **0** | 14 | — |
| query reservoir (1,302) | **0** | 5 | 1 |
| held-out eval (5,209) | **0** | 14 | 3 |
| drawn queries (192) | **0** | 1 | 1 |

**Zero exact identities anywhere** — that part is clean. But exact identity is not the only leakage
channel, so the verdict is **REVIEW**: 5 pairs reach J ≥ 0.85 and all 5 carry a differing gold answer.
Details in Blocker 2. Moderate fuzzy overlap below the threshold does *not* block, because several BBH
subtasks are template-generated and two independent items legitimately share most of their text — that
judgement is now backed by the computed per-subtask item-vs-item p50/p95 baseline rather than by
assertion.

Three matcher bugs were found and fixed while building this, all of which had manufactured *false*
leakage: stripping non-alphanumerics erased bracket-only `dyck_languages` payloads (spurious J=1.0);
an `Options:`/`Input:` payload heuristic kept the wrong side for `hyperbaton`/`logical_deduction`
(spurious J=1.0); and cross-subtask comparison with un-subtracted boilerplate inflated the n-gram
counts. The audit now compares within-subtask with per-subtask boilerplate shingles removed. A fourth
bug was in the *verdict* rather than the matcher: exact-only blocking, which is what hid Blocker 2.

### Gate 2 — 27-subtask prompt parity audit: **PASSED**

`scripts/audit_bbh_prompt_parity.py` → `bbh_prompt_parity_audit.json`. Three byte-for-byte gates:

| gate | claim tested | result |
|---|---|---|
| **A** | custom held-out task **==** stock pinned `bbh_cot_fewshot`, same doc: full request string + `description`, `doc_to_text`, `doc_to_target`, few-shot samples, `num_fewshot`, `generation_kwargs`, `until`, filters, metrics, output type. Also flags **any** non-exempt config field that differs, so a silent behavioural change cannot slip through. | **27 / 27 identical** |
| **B** | rendered query-gradient prompt **==** the evaluation prompt prefix, checked against the **stock** task object so it does not inherit gate A's assumption | **192 / 192 rows identical, all 27 subtasks covered** |
| **C** | each custom task loads exactly the intended held-out rows, and **no** held-out id appears in the query reservoir | **27 / 27, 5,209 examples, 0 leakage** |
| **D** | *disclosure, not pass/fail*: quantifies the llama2 `[INST]` chat-wrapper delta between the stored prompt and the actual training-time sequence | resolved and recorded |

**Scope of gate B, stated precisely.** Gate B compares against the **stock** task, so it is genuinely
independent of the custom config — but both sides ultimately call lm-eval's own `fewshot_context()`. It
therefore verifies (i) the stored artifact was not corrupted after rendering and (ii) stock and custom
agree on those docs. It does **not** independently re-derive lm-eval's prompt-construction logic, and is
not claimed to. The 192 figure counts prompt **rows** over **185 distinct** query examples (7 ids recur
across draws, consistent with the reported 2/3/2 overlap).

Gate B's coverage exceeded the requested ≥1 example per subtask: all 27 subtasks are represented among
the 192 rows. Both gates are **fail-closed**: a missing prompt file, or any subtask with zero checked
prompts, is now a FAILURE rather than a skip. This mattered — an earlier version of the audit `continue`d
past a missing prompt file, and since `all([]) is True` that made gate B silently pass having checked
**nothing**. Verified by re-running with the prompt directory removed: gate B now reports
`0 prompts over 0 subtasks: FAIL`, 108 failures, exit 1.

**Negative control**, now a recorded mode (`--tamper_check`) rather than a claim in prose: perturbing one
subtask's `description` in memory makes gate A's prompt comparison differ, and the verdict is written to
`bbh_prompt_parity_audit.json` → `negative_control`. So a green gate A is demonstrably informative. Gate
B correctly stays green under that perturbation — it compares against the **stock** task, so it is by
construction insensitive to custom-config edits; that independence is the point of splitting A from B.

The known, intended asymmetries (supervised continuation; chat wrapper) are recorded in the audit
artifact under `known_intended_differences` rather than glossed.

**Gate 2 is necessary but NOT sufficient** — see gate 3. It certifies the prompt *strings*; it says
nothing about what survives tokenization and `cutoff_len`.

### Gate 3 — execution-level token/truncation audit: ⛔ **HOLD**

`scripts/audit_bbh_token_truncation.py` → `bbh_token_truncation_audit.json`. This is the gate that
string parity could not provide, and it **fails**: 7/192 records (all `geometric_shapes`) lose 493–560
source tokens, and what is dropped is the tail — i.e. the record's **own query** plus the trailing CoT
cue. Details and the deferred protocol decision are in the blocker section at the top of this document.

Both regimes, measured with the installed tokenizer and LlamaFactory's own `infer_seqlen`:

| side | budget | max observed | truncated |
|---|---|---|---|
| query gradient (llama2 wrapper, `cutoff_len`) | 2,048 | 2,608 tokens | **7 / 192** |
| evaluation (lm-eval, 4096 − `max_gen_toks` 1024) | 3,072 | 2,596 tokens | 0 / 192 |

Note the asymmetry: the evaluation side has ~1.5× the budget of the gradient side, so the two sides are
not merely differently truncated — one is truncated and the other is not.

### Gate 4 — execution contract restored: **PASSED**

`scripts/emit_bbh_execution_contract.py` → `bbh_execution_contract.json`. The frozen protocol is
recorded and machine-verified rather than restated in prose (see "Frozen execution contract" below).

### Gate 5 — remaining, in order

1. **resolve the gate-3 HOLD** — explicit human decision on the truncation protocol; re-run the audit
   to `PASS`;
2. this pre-registration + the contract/split/pin/parity/leakage artifacts reviewed and approved;
3. then a **selection-only, no-SFT canary**: base-model held-out evaluation + draw0 target-gradient
   extraction + all five selectors run to K=2707, checking target cache integrity, selection sizes,
   hashes, determinism, RR order, Random seed, and Jaccard/source/token diagnostics — **no training**;
4. only then: 15 frozen subsets → 30 adapters → eval → aggregate.

## Frozen execution contract (P0-1, code_review_0810)

A fresh session summarised the method as *"MMD over Adam-preconditioned gradients"*. That omits the one
asymmetry that defines it, and building BBH from that summary would have produced a **different
experiment** — because the generic online `MMDSelector` defaults to `target_gradient_type = same`, which
would make target gradients Adam-aware too. The frozen protocol, read from
`targetdraw_10draw_master_manifest.json` and machine-checked:

| | frozen value |
|---|---|
| candidate gradients | **Adam-aware** |
| target / query gradients | **SGD** ← *differs from the candidate side by design* |
| projection | dim **8192**, seed **123** |
| candidate cache | the existing 270,679 × 8,192 artifact, **reused verbatim**, tensor-content sha256 `4bef1bf8…` **verified** |
| warm-up checkpoint | `warmup_seed42/checkpoint-1692`, adapter sha256 **verified** |

Per-method endpoints — the offline scripts actually used by the completed experiments, not
re-derivations:

| method | exact invocation |
|---|---|
| **DSMC** | `scripts/select_moment_mmd.py --alpha 0.0` — $k(u,v)=\langle u,v\rangle^2$, exact marginal greedy. **NOT** the generic online `MMDSelector`. |
| LESS-style TopK | `scripts/select_relevance_topk.py --order first` (not official trajectory-LESS) |
| First-RR | `scripts/select_round_robin.py --order first --perm_seed 6000+d` |
| Second-RR | `scripts/select_round_robin.py --order second --perm_seed 6000+d` (same script *and* same seed as First-RR; only the representation order differs) |
| Random-K | `torch.randperm(N, generator=manual_seed(5000+d))[:K]` — plain uniform; **not** `randk_lenmatch`, which was an MMLU-only 9th arm |

A note on hashing that cost a false alarm: the master manifest hashes gradient caches by **tensor
content** (`v.numpy().tobytes()`), not file bytes. Hashing the file instead reports a mismatch on an
identical cache. The contract now uses the same convention and the cache verifies.

## Cost estimate (anchored to a measured run, not assumed throughput)

Evaluation, not training, dominates — so the eval term is taken from an **actual completed full-BBH
CoT run** in this repo rather than a guess:

| | value | source |
|---|---|---|
| measured full-BBH CoT eval | **1,457 s** (24.3 min) for 6,511 examples, batch 16, Llama-2-7B | `eval_results/bbh/less_sgd_bbh_seed42/.../results_2026-07-06T11-23-25.json` → `total_evaluation_time_seconds` |
| ⇒ per example | 0.224 s | derived |
| ⇒ held-out split (5,209) | ~19.4 min per eval | derived |
| **31 evals** (30 adapters + shared no-SFT reference) | **~10.0 h** | derived |
| 30 adapters × ~15 min (K=2707, 4 epochs ⇒ ~84 optimizer steps) | **~7.5 h** | same recipe as the 1% arm |
| **total** | **~17.5 h** | |

So the earlier **~15–20 h** figure is confirmed by measurement rather than assumed. Caveat: the measured
run used batch 16 on the same hardware and model; a different batch size or contention will move this
roughly proportionally.

## What will NOT be done

No LR / LoRA / epoch / budget sweeps. No modification of DSMC (in particular, **no source-balanced
DSMC variant** — that idea arose from looking at Random's source composition and would be post-hoc
method tuning; it is an exploratory follow-up, not part of this experiment). No second family in this
round.

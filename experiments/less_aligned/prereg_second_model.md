# PRE-REGISTRATION: second model-stack confirmation (Llama-3.2-3B)

**Status: PRE-REGISTERED — NO SUBSTANTIVE COMPUTE RUN.** Checkpoint pinned, tokenizer truncation audit
passed. No warm-up, no candidate gradients, no selection, no SFT. Frozen per `choice_0814_2`.

**Decision frozen:** Llama-3.2-3B. Qwen3-4B-Base is **not** revisited (it needs `transformers ≥ 4.51`
against a frozen 4.50.0 env) and **no third model will be considered**. This is the **last large
experiment before submission**, whatever the outcome.

## What this is, and what it is NOT

> This is a **second model-stack confirmation**, not an architecture-only ablation.

D2c established that **serialization is load-bearing** — the CE improvement was specific to the
targeting pipeline's wrapper. Llama-2 uses the `llama2` template; Llama-3.2 legitimately uses `llama3`;
and the tokenizer changes from 32k to 128,256 vocab. So the model, tokenizer and model-appropriate
serialization move **together**. We must **not** write "we isolate model architecture while holding
everything else identical."

That does not weaken the test: it asks whether the central phenomenon survives on a *second realistic
model stack*, which is exactly the reviewer question ("is the double reversal a Llama-2-7B pathology?").

## Checkpoint provenance

`second_model_checkpoint_pin.json`. HuggingFace `meta-llama/Llama-3.2-3B` is **gated** (HTTP 401), so the
checkpoint comes from the **ModelScope `LLM-Research/Llama-3.2-3B` mirror** (`modelscope 1.37.1`).

| | |
|---|---|
| `model_type` | `llama` — natively supported by the **frozen** transformers 4.50.0 |
| layers / hidden / heads | 28 / 3072 / 24 (8 KV) |
| vocab | **128,256** (Llama-2: 32,000) |
| context | 131,072 |
| params | 3.21B (measured on bf16 load) |
| weight shards | `584d8d3e3f82f796…` (4.97 GB), `4719a04514ec2f06…` (1.46 GB) |

All 10 weight/config/tokenizer artifacts are SHA256-pinned.

**We do NOT claim bit-equivalence to Meta's gated HuggingFace checkpoint** — that comparison is
impossible without access to the gated weights. The paper will say only *"Llama-3.2-3B checkpoint
obtained from the ModelScope `LLM-Research` mirror"*, with our hashes as the provenance record. Meta's
model card and the mirror metadata agree on the spec (3.21B, 128k context, base pretrained), and
transformers ≥ 4.43 is the stated requirement, so 4.50.0 is in range.

## Design (frozen)

**3 query draws × 2 SFT seeds × 4 methods = 24 adapters**, plus **1 shared no-SFT reference**.

| | |
|---|---|
| methods | **DSMC, First-RR, Second-RR, Random-K** |
| excluded | LESS-style TopK, GIST, NICE, Random-K-SeqLabelMatched, and any new method |
| BBH held-out split | **reused unchanged** (5,209 examples, 27 subtasks, frozen suite) |
| query draws | **reused unchanged** (the same three M=64 draws) |
| budget | K = **2707** |
| SFT seeds | **{42, 1}**, fully crossed with draws |

The purpose is to **confirm the central phenomenon**, not to re-run a benchmark. Four methods suffice:
DSMC tests the central claim, First-RR is the strongest gradient-targeted comparator in the recent
literature, Second-RR preserves the first- vs second-order representation contrast, and Random-K is the
target-awareness baseline.

## What must be rebuilt vs reused

**Rebuilt from scratch for Llama-3.2-3B** — the central claim is about *model-specific gradient
geometry*, so reusing Llama-2 artifacts would answer a different question:

- its own **warm-up checkpoint** (same warm-up data and recipe; adapter + optimizer hash-pinned);
- its own **candidate-gradient datastore**, all 270,679 candidates;
- its own **three target-gradient sets**;
- its own **DSMC / First-RR / Second-RR selections**.

**Reused exactly:**

- **Random-K uses the identical Llama-2 candidate indices** (seeds `5000+d`). Random-K is
  target-independent, so holding its subsets fixed gives a genuinely constant data baseline across the
  two model stacks.

> **Explicitly forbidden:** training Llama-3.2 on Llama-2-selected DSMC/RR subsets and calling it
> replication. That is *cross-model transfer of Llama-2 selections* — a different scientific question.

## Frozen feature and training contract

| | value |
|---|---|
| candidate gradients | **Adam-aware** |
| target/query gradients | **SGD** |
| projection | dim **8192**, seed **123** |
| target-gradient cutoff | **3072** |
| SFT cutoff | **2048** |
| SFT recipe | unchanged: 4 epochs, LoRA r128/α512/dropout 0.05, lr 2e-5 linear, warmup 0.03, eff. batch 128, bf16 |

**No hyperparameter tuning for the 3B model.** Re-tuning would confound the model axis with a tuning axis.

**Token gate — passed.** Llama-3.2 supports 131k context, but that is **not** a reason to widen our
budgets. Cutoffs stay at 3072/2048, and the 192-query truncation audit was re-run with the Llama-3.2
tokenizer: **192/192 records clean, 0 materially truncated, max 2,007 tokens** (the Llama-3 tokenizer is
more efficient than Llama-2's 2,581 max, so headroom increases).

## Outcomes: all four pre-registered before any training

| # | outcome | interpretation |
|---|---|---|
| **A** | D2(DSMC) < D2(Random) **and** Acc(DSMC) < Acc(Random), with the operational query surrogate improving | **Strongest replication**: better target alignment is not sufficient for downstream utility **across two model stacks**. |
| **B** | DSMC best on D2 but downstream ≈ Random | Still supports the core claim — *"better matching ⇒ better utility"* still fails; negative transfer is merely milder than on Llama-2-7B. |
| **C** | DSMC genuinely **beats** Random on Llama-3.2 | **Not a failed experiment.** Conclusion becomes: target matching *can* help, but its utility is **model-dependent rather than reliable** — consistent with the cross-model inconsistency reported in neighbouring work. |
| **D** | DSMC no longer even minimizes D2 | Weakens "DSMC stably optimizes the geometry across models", but still shows the MMLU/Llama-2 method advantage does not generalize. Report as-is. |

**No outcome may trigger tuning, a third model, a method change, or any protocol change.** All four are
reportable.

## Analysis, frozen now

- **Primary outcome**: held-out BBH **micro exact_match** on the frozen 5,209-example split.
- **Primary statistical unit**: the query/selection **draw (n=3)** — seeds averaged within draw first, as
  in the Llama-2 arm. Six seed-level cells are secondary stability evidence.
- **Diagnostics (exactly four, all theory-motivated and pre-specified):**
  1. `D2(S, Q_d)` — second-moment geometry;
  2. **operational wrapped** query CE;
  3. **same-query 64-item CoT EM**;
  4. **bare-context CE** — a *serialization-sensitivity* diagnostic only. Per D2c it may **not** be
     promoted to a primary criterion.
- Descriptive only: no p-values, no variance-component inference.
- Every method reported as Δ vs the model's **own** shared no-SFT reference (cross-model absolute
  accuracy is not comparable).

## Engineering gates before the full run

1. ✅ pin ModelScope snapshot + all weight/tokenizer SHA256;
2. ✅ 192/192 target-prompt token audit with the Llama-3.2 tokenizer;
3. ☐ Llama-3.2 warm-up checkpoint + small canary;
4. ☐ small candidate-gradient canary (tens of examples): Adam preconditioning, 8192 projection,
   finite/non-zero, determinism;
5. ☐ draw0 target-gradient canary — shape `(64, 8192)`, finite, no zero rows;
6. ☐ all three target-aware selectors reach exactly K=2707, unique and in range;
7. **no BBH accuracy is inspected at any gate.**

Only when these are green: build the full candidate datastore → 24 adapters → eval → analyse.

## Cost

~45–50 GPU-h: warm-up ~1.5 h; candidate gradients ~6–8 h (dominant); target gradients ~1 h; selections
~1.5 h; 24 adapters ~6 h; 25 evals ~26–29 h. Parallelizable across the 8 H20s.

## Stop rule

After this arm, **all large experiments stop**: no third model, no third task, no method changes, no
hyperparameter changes — regardless of the result.

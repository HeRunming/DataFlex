# Artifact & provenance audit — resolution

Closes items **A–G** from `PAPER_READY_CONSOLIDATED_0809.md` §10.2, per advice_0809
("先把 Codex 暴露出的 artifact/decontamination 问题全部清掉"). **No experiments were run.**

| item | status | resolution |
|------|--------|-----------|
| A. verifier contradicts completion reports (4/6) | **FIXED** | verifier was stale; now **6/6** and it describes the current code |
| B. `target_dataset` is not a generic loader | **DOCUMENTED** | verifier now checks the real contract and prints the limitation |
| C. main YAML is not the resolved config | **FIXED** | `resolved_run_provenance.json` exports the resolved recipe per run family |
| D. equal-step manifest omits `max_steps=420` | **FIXED** | actual `global_step` recovered from `trainer_state.json` for all 165 adapters + hashed |
| E. warm-up checkpoint provenance incomplete | **FIXED** | one authoritative record (hashes + trainer_state + config caveat) |
| F. candidate-pool decontamination absent | **DONE** | see `contamination_audit.md` — gate **PASSED** |
| G. working tree not a frozen snapshot | **DONE** | clean snapshot commit at the end of this round |

## A. The verifier was stale, not the code (now 6/6)

`verify_alignment.py` failed `adam_preconditioning` because it string-matched one exact expression
(`'numerator = beta1 * m + (1.0 - beta1) * vectorized_grads'`) that a later revision renamed. I read
the actual implementations first: **both `less_selector.py` and `mmd_selector.py` compute the
LESS-official form non-destructively**

```
grad = (β1·m + (1−β1)·g) / sqrt(β2·v + (1−β2)·g² + eps)      # eps INSIDE sqrt, fp32, m/v not mutated
```

so the code was correct and the check was wrong. The check is now **semantic** (verifies the β1/β2
moment ingredients, `torch.sqrt`, and that `eps` sits inside the sqrt) plus a word-boundary regex for
genuine in-place `m`/`v` mutation.

Worth recording as a caution: my first version of that regex used the bare substring `m.add_(`, which
false-positived on `selected_kernel_sum.add_(k_col)` — the greedy MMD kernel accumulator, unrelated to
optimizer state. Fixed with `(?<![\w.])[mv]\.add_\(`. A verifier that produces false alarms is as
harmful as one that goes stale.

## B. `target_dataset` contract, stated honestly

DataFlex does **not** implement a generic independent `target_dataset` loader. `SelectTrainer` does
`target_dataset_for_selector = self.eval_dataset` and requires `target_dataset` and `eval_dataset` to
be set to the same value. The verifier now asserts what the code actually guarantees — the field is
defined, read, passed to the selector, and target-aware selectors **fail loudly** with no target set —
and prints the limitation rather than claiming a loader that does not exist.

No leakage follows from this in our runs: in-training eval is disabled (`eval_strategy: no`) and final
MMLU test evaluation is a separate `lm_eval` process.

## C + D. Resolved configs and ACTUAL step counts (`resolved_run_provenance.json`)

The YAML's nominal values are superseded by the driver's CLI overrides, so the artifact now exports
the resolved recipe explicitly:

| | YAML (nominal) | resolved (authoritative) |
|---|---|---|
| lora_alpha | 256 | **512** (driver override) |
| per_device_train_batch_size | 16 | **4** (driver override) |
| gradient_accumulation_steps | 8 | **4** (driver override) |
| num_train_epochs | 3 | **4** (driver override) |
| lora_dropout | 0.05 | **0.05** — driver does NOT override it, so the YAML value is what ran |

Effective batch = 4 × 4 × 8 GPUs = **128** examples/optimizer step; lr 2e-5, linear, warmup_ratio 0.03,
bf16, cutoff 2048, LoRA r128 / α512 / dropout **0.05** on q,k,v,o.

*Correction (code_review_0809):* an earlier version of this artifact said `dropout 0.1` and also carried
a self-contradictory `effective_global_batch: 16` (a stray `//8` in the emitter). Both were **my
metadata errors, not run errors** — the executed adapters are unaffected, and the authoritative step
counts below come from `trainer_state.json`. Now: dropout **0.05**, single `effective_batch: 128`.

**Actual optimizer steps recovered from `trainer_state.json` (ground truth, not the shell command),
verified across every adapter:**

| run family | K | adapters | `TRAIN_EXTRA` | actual steps / epochs | all identical | matches expected |
|---|---|---|---|---|---|---|
| 5% fixed-epoch | 13533 | 75 | — | **420 / 3.99** | ✅ 75/75 | ✅ |
| 1% fixed-epoch | 2707 | 75 | — | **84 / 3.85** | ✅ 75/75 | ✅ |
| 1% equal-step | 2707 | 15 | `max_steps=420` | **420 / 19.09** | ✅ 15/15 | ✅ |

This independently confirms the equal-step arm did match the 5% step count (420) while running ~19
epochs over 2,707 examples — the over-training that produced pre-registered rule #4.

Minor note: the committed 5% plan predates budget parameterization so its `budget` field is `null`
(K=13533 is implied by its subset row counts, which `register` hash-validates). The 1% and equal-step
plans carry `budget` explicitly.

## E. Warm-up checkpoint — one authoritative record

`checkpoint-1692` is pinned by hash rather than by any single YAML (several warm-up configs disagree on
nominal alpha/dropout/batch):

- `adapter_model.safetensors` sha256 `44d9c580…`
- `optimizer.pt` sha256 `10a0169e…`
- verified **byte-identical** to `sft_results/random_selected/checkpoint-1692` (the path referenced by
  the candidate-gradient config), which is what makes candidate and target gradients same-state
- `trainer_state.json` step/epoch + hash recorded; candidate config files listed with an explicit
  caveat that the hashes, not the YAMLs, are authoritative

## What this does and does not change

It changes **no result**. Every number in the consolidated document stands. What changes is that the
repository can now be used to reconstruct what was actually run — which was the real gap, and the one
most likely to be challenged. Remaining disclosed limitation: contamination layer 4 (semantic NN) was
not run; see `contamination_audit.md`.

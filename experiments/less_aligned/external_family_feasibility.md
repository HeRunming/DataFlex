# External target-family feasibility audit (no accuracy consulted)

Prepared per advice_0809: *"先不跑实验 … 拿出 2–3 个 external target family 的 split / contamination /
query-reservoir / runtime feasibility 表；然后我们只选一次，不看任何新的 accuracy。"*

**Selection criteria fixed BEFORE looking at any result** (from advice_0809):
clean query/test split · query reservoir large enough for independent draws · low candidate-pool
contamination · reliable existing eval pipeline. **No downstream accuracy from any family was
consulted in producing this table.**

Target protocol this must support (frozen shape from advice_0809): **3 independent query draws × 2 SFT
seeds × 5 methods = 30 adapters**, one budget (K ≈ 2500–2707), 4 epochs, plus one shared no-SFT
reference. Methods: DSMC, Second-RR, First-RR, LESS-style TopK, Random-K.

## Feasibility table

| criterion | **BBH** | **TyDiQA (GoldP)** | **MMLU-Pro** |
|---|---|---|---|
| local data present | ✅ `eval/bbh/` (27 task files) + HF `lukaemon___bbh` | ✅ `eval/tydiqa/` + HF `google-research-datasets___tydiqa` | ❌ **not cached**; needs download |
| test pool size | **6,511** examples across 27 tasks | **5,077** GoldP dev-as-test QAs | ~12k (not verified locally) |
| clean query/test split available? | ⚠️ **no official held-out query split** — the 6,511 are the eval set; a query reservoir must be carved out of it (per-task disjoint split) | ⚠️ **worse**: the repo's `test/` and `dev/` both point at `tydiqa-goldp-v1.1-dev.json`; the "dev" dir also holds 1-shot prompt files (9 QAs). A query/test split must be constructed from the same GoldP dev file | ⚠️ unknown until downloaded; MMLU-Pro has a validation split upstream |
| query reservoir sufficient for 3 draws? | ✅ yes if we reserve e.g. 3×64 = 192 from 6,511 (3%) and evaluate on the remaining ~6.3k | ✅ numerically yes (192 of 5,077) but see split concern | ✅ likely, unverified |
| candidate-pool contamination (L2 13-gram) | **5 / 270,679 = 0.000018** ✅ negligible | not yet measured | not measurable offline |
| eval pipeline reliability | ✅ **113 lm_eval yaml task configs** (`bbh/cot_fewshot/...`); also a LESS-native 3-shot harness we already used | ❌ **0 lm_eval task configs** — we'd rely on the custom LESS TyDiQA F1 path (works, but bespoke) | ✅ **15 lm_eval yaml configs** (`mmlu_pro_*`), but data must be fetched |
| literature comparability | ✅ used in the recent controlled study | ✅ used in the recent controlled study | ✅ used in the recent controlled study |
| prior exposure in *our* runs (cherry-pick risk) | we have old BBH numbers, mixed/rotating winners | ⚠️ **HIGH RISK** — 2nd-order methods looked *favorable* on TyDiQA in old runs; choosing it as the sole confirmation target invites a cherry-picking objection | ✅ **none** — never run here |
| runtime for 30 adapters @K≈2707, 4 ep | ~84 steps/adapter ≈ 15 min train + ~7 min eval ⇒ **~11 h** (eval is multi-task CoT generation, so likely more: est. **~15–20 h**) | similar train; F1 gen eval cheaper ⇒ **~10–14 h** | similar train; 15-subject eval ⇒ **~15–20 h** |

## Observations that matter for the choice

1. **None of the three has an off-the-shelf clean query/test split.** For MMLU we had the natural
   dev/validation/test three-way split; BBH and TyDiQA do not. Any of these families requires
   *constructing* a query reservoir disjoint from the eval set (per-task, seeded, hash-recorded) —
   which is exactly what our existing `gen_target_draws.py` machinery already does, so this is
   engineering-feasible but must be pre-registered rather than assumed.
2. **TyDiQA has two strikes**: the shipped `dev`/`test` layout is confusing (both reference the same
   GoldP dev file), and it is the family where our *old* results were most favorable to second-order
   methods. Per advice_0809 I would **not** propose it as the sole confirmation target.
3. **BBH is the strongest on infrastructure**: data local, contamination negligible (0.000018), 113
   lm_eval configs, and a harness we have already run. Its weakness is that CoT generation eval is
   slower and noisier than multiple-choice log-likelihood.
4. **MMLU-Pro is the cleanest on cherry-pick risk** (never run here) and has lm_eval support, but the
   data is not cached — it needs a download, and its contamination cannot be audited until then. It is
   also the closest sibling to MMLU, which slightly weakens the "different family" argument while
   strengthening comparability.

## What I have NOT done (deliberately)

- No accuracy from BBH/TyDiQA/MMLU-Pro was computed or consulted.
- No family has been selected — per the instruction, the choice is made once, by you, from this table.
- No SFT, no gradient extraction, no draw generation for any external family.

## Recommended next decision (yours)

If the priority is **infrastructure certainty + audited cleanliness** → BBH.
If the priority is **avoiding any cherry-picking appearance** → MMLU-Pro (accepting a download +
contamination audit first).
TyDiQA is not recommended as the sole confirmation target for the reasons above.

Whichever is chosen, the pre-registration must fix: the per-task query/test split rule, 3 independent
draws (sampled independently with overlap reported, *not* forced globally disjoint), the crossed
3 draws × 2 seeds design, K, the 5 methods + no-SFT reference, and the metrics logged
(query loss, D2, downstream, no-SFT).

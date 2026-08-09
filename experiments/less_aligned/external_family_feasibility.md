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
| local data present | ✅ `eval/bbh/` (27 files = **23 official tasks**, see note) + HF `lukaemon___bbh` | ✅ `eval/tydiqa/` + HF `google-research-datasets___tydiqa` | ❌ **not cached**; needs download |
| test pool size | **6,511** examples over the official 23 tasks | **5,077** GoldP dev-as-test QAs | **12,032** official test |
| clean query/test split available? | ⚠️ **no** official held-out query split — a reservoir must be carved from the 6,511 (per-task stratified) | ⚠️ **no** — the repo's `test/` and `dev/` both point at `tydiqa-goldp-v1.1-dev.json`; official GoldP ships train/dev, not a 3-way split | ✅ **YES** — official **validation 70 / test 12,032** |
| query reservoir sufficient for 3 draws? | ✅ carve 20% ≈ **1,300**; three M=64 draws overlap only ≈64²/1300 ≈ **3.2** examples | ✅ numerically, but split concern above | ❌ **validation is only 70** — three M=64 draws would almost entirely coincide; carving from test forfeits the split's whole advantage |
| candidate-pool contamination (L2 13-gram) | **5 / 270,679 = 0.000018** ✅ negligible | not yet measured | not measurable offline |
| eval pipeline reliability | ✅ **113 lm_eval yaml configs** (`bbh/cot_fewshot/...`) + a LESS-native 3-shot harness we have run | ❌ **0 lm_eval configs** — bespoke LESS F1 path only | ✅ **15 lm_eval configs** (`mmlu_pro_*`), data must be fetched |
| how "external" is it really? | ✅ a **different** benchmark family (multi-step reasoning) | ✅ different (multilingual extractive QA) | ⚠️ **6,810 of 12,032 test questions come from original MMLU** — a weak second family for our purpose |
| literature comparability | ✅ used in the recent controlled study | ✅ used in the recent controlled study | ✅ used in the recent controlled study |
| prior exposure in *our* runs (cherry-pick risk) | old BBH numbers exist, winners rotated | ⚠️ **HIGH** — 2nd-order looked *favorable* on TyDiQA before | ✅ none |
| runtime, 30 adapters @K=2707, 4 ep | ~84 steps ⇒ ~15 min train + CoT-generation eval ⇒ est. **~15–20 h** | ~10–14 h | ~15–20 h |

### Note on "27 files vs 23 tasks" (corrected per code_review_0809)

The official BBH suite is **23 tasks**. Our local layout has 27 JSON files because two tasks ship as
three size-variants each: `logical_deduction_{three,five,seven}_objects` and
`tracking_shuffled_objects_{three,five,seven}_objects`. Collapsing those: 27 − 2 − 2 = **23**. So the
local data *is* the official suite, not a non-standard benchmark — but the experiment will pin the
official 23-task list explicitly so no ad-hoc aggregate can creep in.

## Observations that matter for the choice (updated per code_review_0809)

1. **Corrected claim.** An earlier version of this table said *"none of the three has an off-the-shelf
   clean query/test split."* That is **wrong for MMLU-Pro**, which does ship an official
   **validation (70) / test (12,032)** split. The reason MMLU-Pro is still not preferred is different:
   **70 validation examples cannot support three meaningful M=64 draws** (they would almost entirely
   coincide), and carving a reservoir out of `test` instead would forfeit the very split that is its
   advantage. Additionally, **6,810 of its 12,032 test questions are inherited from original MMLU**, so
   it is a weak choice for answering *"is our finding an MMLU-family artifact?"*.
2. **TyDiQA has two strikes**: the shipped `dev`/`test` layout both reference the same GoldP dev file
   (official GoldP provides train/dev, not a natural 3-way split), and it is the family where our *old*
   results were most favorable to second-order methods. Not suitable as the sole confirmation target.
3. **BBH is the strongest choice**: genuinely different benchmark family (multi-step reasoning), data
   local and confirmed to be the official 23-task suite, contamination negligible (0.000018), 113
   lm_eval configs plus a harness we have run. Its costs are that a query reservoir must be carved
   (feasible: 20% ≈ 1,300 examples ⇒ three M=64 draws overlap only ~3 examples in expectation) and that
   CoT-generation eval is slower and noisier than multiple-choice log-likelihood.

**Selected family: BBH** (per code_review_0809). See `prereg_bbh_external.md` for the frozen protocol.

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

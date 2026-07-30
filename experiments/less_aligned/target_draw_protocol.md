# Preregistered target-draw protocol (review_0729)

**Status: DRAFT for review — no gradients/selection/training run yet.**
This defines how independent skewed target sets are sampled so that the statistical unit of the
main DSMC evaluation is the *target draw*, not the training seed or the eval item.

## 1. What "skew" means here (the claim being tested)

We adopt the **sampling-bias** interpretation (review_0729 §"先定义 skew"):

> The latent evaluation distribution P\* is **balanced** (STEM and Humanities equally weighted).
> The observed few-shot target set is a **skewed finite sample** Q_ρ of P\*, with majority
> fraction ρ. We test robustness of selection to biased finite target samples.

Consequence: the **primary metric is balanced accuracy** (½·mmlu_stem + ½·mmlu_humanities),
because P\* is balanced. We *also* report the target-weighted score (ρ·maj + (1−ρ)·min) as a
secondary metric so a reader who prefers the "80/20 is the true distribution" reading can see it.
Claim form: *DSMC is robust when the observed target set is a skewed finite sample of a broader
balanced target-capability distribution.*

ρ ∈ {0.5, 0.8, 0.9} is the eventual sweep; the **pilot fixes ρ = 0.8** (matches existing runs).

## 2. Data sources (no leakage) — three disjoint MMLU splits, distinct roles

| split | role | why |
|-------|------|-----|
| **validation** | **target-draw reservoir** (build Q_ρ) | large enough for disjoint draws; never evaluated on |
| **dev** | fixed 5-shot demonstrations for eval only | standard Hendrycks few-shot; already used by lm-eval |
| **test** | evaluation (mmlu_stem, mmlu_humanities), untouched | 18,738 items; the reported numbers, comparable to all prior runs |

Target reservoir = validation only ⇒ **fully disjoint from the test eval set** (no leakage) and
disjoint from the dev demonstrations. This matches the directive: val→targets, dev→demos,
test→eval.

## 3. Measured reservoir sizes (offline, `hails/mmlu_no_train` validation, lm-eval subject map)

STEM = 19 subjects, **335** validation examples. Humanities = 13 subjects, **518**.
(Per-subject counts saved in `target_draws/reservoir_counts.json` when generated.)

Feasibility of **5 disjoint draws per direction** at n_T = 80, ρ = 0.8 (64 majority + 16 minority):

| direction | needs (5 draws) | reservoir | slack |
|-----------|-----------------|-----------|-------|
| stem80 | 320 STEM + 80 HUM | 335 STEM / 518 HUM | STEM tight (15 spare) |
| hum80  | 320 HUM + 80 STEM | 518 HUM / 335 STEM | comfortable |

→ **5 fully-disjoint draws are feasible within each skew direction.** stem80 consumes almost the
entire STEM validation pool (320/335), so its minority (HUM) and the hum80 draws still have slack,
but a 6th disjoint stem80 draw would not fit. **Decision: 5 draws/direction, disjoint WITHIN a
direction.** Across opposite directions overlap is unavoidable (both dip into both groups) and is
**recorded** (§5), per the review.

## 4. Sampling procedure (deterministic, seeded)

For draw d ∈ {0..4} in direction dir ∈ {stem80, hum80}:
1. Fix a global RNG seed = `1000 + 100·dir_id + d` (recorded).
2. Within the majority group: sample 64 examples **without replacement across the 5 draws**
   (partition the shuffled majority reservoir into 5 disjoint blocks, take block d), balancing
   across subjects as evenly as the per-subject counts allow (round-robin over subjects, then
   fill). Same for 16 minority examples.
3. Format identical to existing `data/mmlu_target_stem80.jsonl` (sharegpt messages, the exact
   Hendrycks 5-shot prompt template already in `build_skewed_mmlu_target.py`).
4. Write `data/target_draws/{dir}_draw{d}.jsonl`.

Training seed is **rotated across draws** (not fixed to 42) so results don't all ride one
training trajectory, but is **shared by all methods within a draw** (paired comparison):
draw 0→42, 1→1, 2→2, 3→42, 4→1. Recorded per draw.

## 5. Provenance recorded per draw (`target_draws/{dir}_draw{d}.meta.json`)

- target example IDs (subject + validation row index) and per-subject composition
- target data sha256, and (after gen) target gradient cache sha256
- shared training seed for that draw
- candidate cache sha256 (same 270k seed-42 cache throughout)
- **pairwise overlap matrix** across all 10 draws (example-ID Jaccard) → `target_draws/overlap_matrix.csv`
- selection indices sha256 per method (added at selection time)

## 6. Statistical analysis (unit = target draw)

For each method m vs DSMC, per direction, Δ_d = score_DSMC,d − score_m,d over the 5 draws:
- mean and **median** paired Δ; number of draws DSMC wins
- **bootstrap CI clustered on target draw** (resample draws, not eval items — do not inflate n)
- report the **direction × method interaction** (is any effect symmetric across stem80/hum80?)
- both balanced (primary) and target-weighted (secondary) metrics

Explicit guardrail: **DSMC is frozen; no hyperparameter is tuned on these 10 draws.**

## 7. Scope / sequencing (from review_0729 + choice_0730)

1. **(prereq) rep×selector 2×2 attribution gate** — DONE (`attribution_2x2_summary.md`). DSMC best
   in both directions; 2nd-order representation is the primary driver, MMD-diversity complementary
   (hypothesis). Gate passed → DSMC stays the headline.
2. **This protocol → pilot** (LAUNCH ONLY AFTER: protocol approved+frozen AND GIST passes numerical
   review): 2 directions × **2 draws** × 7 method-rows:
   1. DSMC
   2. true **Second-RR** (per-query nearest, cycling — NOT Second-TopK; that stays in the 2×2 ablation)
   3. LESS (+ true First-RR as the 1st-order relevance/RR reference)
   4. GIST (arXiv 2602.18584)
   5. NICE
   6. **Random-K** (uniform fixed-K, PRIMARY random baseline)
   7. **Random-K-LengthMatched** (fixed-K, length-histogram-matched; compute/length control)
3. Expand to **5 draws/direction** only after the pilot is clean.
4. One representative draw per direction → **+3 paired training seeds** for training variance.
5. **Later axes** (separate, not now): target size n_T ∈ {16,64,128}, budget K ∈ {1%,5%}, keeping
   only {DSMC, strongest gradient baseline, GIST, Random-K}.

## 8. Baseline definitions (resolved per choice_0730)

- **GIST** (arXiv 2602.18584, v2): SVD of target validation gradients → low-rank task subspace;
  project candidates onto it; score by target-direction alignment. No official repo — implement from
  v2 formulas and pass a numerical review on STEM80/HUM80 (SVD input/centering/normalization, rank
  rule, exact score formula, shared projection, orthonormal basis, rotation-invariance, full-rank
  degeneration, LESS-aligned Adam/SGD alignment) BEFORE the pilot. Also report selection storage /
  FLOPs / wall-clock, since GIST claims efficiency and is the closest conceptual competitor.
- **Random-K** (primary): uniformly sample exactly K=13,533 without replacement; multiple random
  seeds. Target-independent → the SAME adapter can be reused across target draws at a fixed training
  seed (only depends on random-subset seed + training seed + pool). Fixed selection budget, matched
  optimizer steps to all methods.
- **Random-K-LengthMatched** (compute/length control): still exactly K examples, but the
  post-tokenization length histogram (buckets [0,256),[256,512),[512,1024),[1024,1536),[1536,2048])
  is matched per-bucket to the DSMC subset for that draw. Draw-specific (must be regenerated per
  draw). Uses effective length after tokenizer+template+cutoff_len=2048, NOT raw string length.
  Do **not** change K to match tokens — match the distribution at fixed K.
- **true First-RR / Second-RR**: greedy round-robin (per query, nearest unpicked candidate, cycle
  over queries until K). Distinct from relevance top-k; often strong at low budget. Second-TopK from
  the 2×2 is retained ONLY as a mechanism ablation, not a pilot baseline.
- **Second-TopK / Linear-MMD / DSMC**: already computed in the 2×2 gate.

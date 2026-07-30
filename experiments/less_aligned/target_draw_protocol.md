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
| **test** | evaluation (mmlu_stem, mmlu_humanities), untouched | the reported numbers, comparable to all prior runs |

Target reservoir = validation only ⇒ **fully disjoint from the test eval set** (no leakage) and
disjoint from the dev demonstrations. This matches the directive: val→targets, dev→demos,
test→eval. (Do NOT hardcode a test item count; record the STEM/Humanities eval item counts that
the current lm-eval version actually reports at run time. Standard cais/mmlu has dev 285 /
validation 1,531 / test 14,042 total across all 57 subjects.)

## 3. Measured reservoir sizes (offline, `hails/mmlu_no_train` validation, lm-eval subject map)

STEM = 19 subjects, **335** validation examples. Humanities = 13 subjects, **518**.
(Per-subject counts in `target_draws/reservoir_counts.json`.)

**Target size n_T = 64 (revised from 80, per code_review_0730)** so that all **10 draws are
GLOBALLY disjoint** (not just within-direction). At ρ=0.8, n_T=64 → **51 majority + 13 minority**
(= 79.7/20.3%, reported precisely, not called exactly 64/16). Five stem-majority + five
hum-majority draws need, in total:

| group | consumed by 5 stem80 draws | consumed by 5 hum80 draws | total | reservoir |
|-------|----------------------------|---------------------------|-------|-----------|
| STEM  | 5×51 = 255 (majority)      | 5×13 = 65 (minority)      | 320   | 335 ✓ (15 spare) |
| HUM   | 5×13 = 65 (minority)       | 5×51 = 255 (majority)     | 320   | 518 ✓ (comfortable) |

→ **10 globally-disjoint draws fit** (STEM 320≤335, HUM 320≤518). This makes each draw a genuinely
independent statistical unit and cleans up the direction-interaction analysis. The existing n=80
STEM80/HUM80 runs remain valid as preliminary/mechanism experiments; n=64 is also a planned
target-size point, so this is not an extra axis.

## 4. Sampling procedure — JOINT generation, deterministic (revised per code_review_0730)

All 10 draws are generated **in one pass** (not per-draw independent sampling + dedup), so global
disjointness is guaranteed by construction:
1. One **master seed** (recorded).
2. Shuffle the STEM reservoir once and the HUM reservoir once (with the master seed).
3. Allocate blocks in a single pass: 5 stem80 draws take disjoint majority blocks of 51 from the
   STEM shuffle and disjoint minority blocks of 13 from the HUM shuffle; 5 hum80 draws take
   disjoint majority blocks of 51 from HUM and minority blocks of 13 from STEM — all drawn from the
   *remaining* (unused) portions, so STEM usage = 255+65=320 and HUM usage = 65+255=320, both ≤
   reservoir with no overlap across any of the 10 draws. Subjects balanced round-robin within each
   block as counts allow.
4. Write `data/target_draws/{dir}_draw{d}.jsonl` + `.meta.json`; format identical to
   `data/mmlu_target_stem80.jsonl` (sharegpt, the Hendrycks 5-shot template in
   `build_skewed_mmlu_target.py`).

**Training seeds**: 5 **distinct** values, one per draw index, shared by all methods within a draw
(paired design): draw 0→42, 1→1, 2→2, 3→3, 4→4. (No seed repeats — paired design already controls
the seed nuisance, so there's no reason to reuse 42/1.)

**Representative draw for the +3-seed variance study is pre-registered as draw 0** (chosen now, not
after seeing downstream results, to avoid post-hoc selection).

## 5. Provenance recorded per draw (`target_draws/{dir}_draw{d}.meta.json`)

- target example IDs (subject + validation row index) and per-subject composition
- target data sha256, and (after gen) target gradient cache sha256
- shared training seed for that draw
- candidate cache sha256 (same 270k seed-42 cache throughout)
- **pairwise overlap matrix** across all 10 draws (example-ID Jaccard) → `target_draws/overlap_matrix.csv`
  (expected ≈0 off-diagonal by construction; reported to confirm global disjointness)
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
2. **This protocol → pilot** (LAUNCH ONLY AFTER: protocol approved+frozen AND GIST fidelity gate
   resolved): 2 directions × **2 draws** × **8 method rows** (LESS and First-RR are distinct
   selectors — not collapsed into one parenthetical):
   1. DSMC
   2. true **Second-RR** (per-query nearest, cycling — NOT Second-TopK; that stays in the 2×2 ablation)
   3. **LESS** (Adam-preconditioned mean-gradient relevance top-k)
   4. true **First-RR** (1st-order round-robin)
   5. **GIST** (faithful variant; see §8 fidelity gate)
   6. **NICE**
   7. **Random-K** (uniform fixed-K, PRIMARY random baseline)
   8. **Random-K-LengthMatched** (fixed-K, length-histogram-matched; compute/length control)
3. Expand to **5 draws/direction** only after the pilot is clean.
4. Pre-registered representative draw (draw 0) per direction → **+3 paired training seeds** for
   training variance.
5. **Later axes** (separate, not now): target size n_T ∈ {16,64,128}, budget K ∈ {1%,5%}, keeping
   only {DSMC, strongest gradient baseline, GIST, Random-K}.

## 8. Baseline definitions (resolved per choice_0730 + code_review_0730)

- **GIST**: official repo = github.com/GuanghuiMin/GIST. Two scripts exist: `select_gist_faithful.py`
  (official Gram/eigendecomp + isometric whitening `P=G_valᵀUₖSₖ⁻¹` + fixed rank, default 150) and
  `select_gist_jlnorm.py` (labelled adaptation). **Finding (gist_validation.md)**: on our unit-norm
  caches the whitening is a no-op at equal rank (cosine undoes the Sₖ⁻¹ rescale) → faithful ≡ JL-Norm
  at the same rank; only the **rank** and **raw-vs-normalized target Gram** actually move the
  selection. **Fidelity gate before pilot**: decide (with user) whether to re-extract RAW target
  grads (cheap, 80 ex) — and possibly raw candidate grads (expensive, 270k) — for a byte-faithful
  Gram, or accept "official math on shared normalized caches + a small rank setting" as the
  controlled baseline. Report selection storage / FLOPs / wall-clock (GIST claims efficiency).
- **Random-K** (primary): uniformly sample exactly K=13,533 without replacement. **Random-subset
  seed varies by draw index** (e.g. 2000+d) so we capture random-selection variance; the STEM/HUM
  draws sharing a draw index reuse the same Random-K adapter (same training seed) → only **5**
  Random-K adapters, not 10. Do NOT reuse a single random subset across all draws (would understate
  variance).
- **Random-K-LengthMatched** (compute/length control): still exactly K examples, but the
  post-tokenization length histogram (buckets [0,256),[256,512),[512,1024),[1024,1536),[1536,2048])
  is matched per-bucket to the DSMC subset for that draw. Draw-specific (must be regenerated per
  draw). Uses effective length after tokenizer+template+cutoff_len=2048, NOT raw string length.
  Do **not** change K to match tokens — match the distribution at fixed K.
- **true First-RR / Second-RR**: greedy round-robin (per query, nearest unpicked candidate, cycle
  over queries until K). Distinct from relevance top-k; often strong at low budget. Second-TopK from
  the 2×2 is retained ONLY as a mechanism ablation, not a pilot baseline.
- **Second-TopK / Linear-MMD / DSMC**: already computed in the 2×2 gate.

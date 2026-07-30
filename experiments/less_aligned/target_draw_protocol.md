# Preregistered target-draw protocol (review_0729 → code_review_0730/_2/_3)

**Status: FROZEN (2026-07-30).** Sampling design, methods, seeds, metrics, and GIST rank rule are
fixed. Next step is artifact generation only (10 draw JSON/meta + allocation + overlap matrices) for
a data-level review; NO gradients/selection/SFT until those artifacts are approved.
This defines how globally non-overlapping skewed target sets are sampled so that the statistical unit of the
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

→ **10 globally-disjoint draws fit** (STEM 320≤335, HUM 320≤518). This makes each draw a
**globally non-overlapping replicate unit** — NOT a statistically independent sample: all draws come
from one finite reservoir via a single joint partition, so they are (mildly) negatively correlated
(using a question in one draw forbids it in others). The statistical unit is still the target draw,
but we do not claim independence. The existing n=80
STEM80/HUM80 runs remain valid as preliminary/mechanism experiments; n=64 is also a planned
target-size point, so this is not an extra axis.

## 3b. Latent distribution P\* and within-category subject composition (per choice_0730_2)

lm-eval's `mmlu_stem` / `mmlu_humanities` group scores are **micro-averages** (weighted by each
subject's test-doc count), not subject-uniform. To keep the target sampling distribution consistent
with the primary metric, we define P\* as:
- **50/50 across the two domains** (balanced latent eval), and
- **within each domain, subjects in proportion to their lm-eval test-doc counts** (micro weights).

Each draw's majority (51) / minority (13) quota is allocated across subjects by these micro weights
with constrained rounding to hit 51 / 13 exactly. Before freezing, generate a **per-subject
allocation feasibility table** confirming each subject's *validation* reservoir count supports the
required per-subject quota across all 10 globally-disjoint draws (a subject with few validation
examples may cap its achievable share — if so, document the deviation). Primary metric stays the
lm-eval micro-average; we additionally report a **subject-macro** robustness number (equal per
subject) since sampling and metric are never perfectly matched.

**Joint allocation computed** (`scripts/allocate_target_subjects.py` →
`data/target_draws/subject_allocation_plan.json`): a deterministic joint integer allocator assigns
**exactly 320 STEM and 320 Humanities** slots across the ten 51/13 blocks — per-block
largest-remainder rounding (exact block totals) followed by a deterministic cap-repair that moves
units off over-cap subjects. All block totals (51/13) and reservoir caps are satisfied; the two
tight STEM subjects `college_chemistry` (cap 8) and `high_school_computer_science` (cap 9) sit
exactly at cap with the deficit absorbed by subjects that still have slack. The realized aggregate
subject distribution deviates from the ideal micro-weighted distribution by column-TVD ≈ 0.017
(STEM) / 0.046 (HUM). The allocator does not solve for a global optimum — it is a deterministic
heuristic that keeps the deviation small. (The earlier per-subject `round(w·320)` table summed to
318 and is superseded: `superseded_subject_allocation_feasibility.json`.)

## 4. Sampling procedure — JOINT generation, deterministic (revised per code_review_0730)

All 10 draws are generated **in one pass** (not per-draw independent sampling + dedup), so global
disjointness is guaranteed by construction:
1. One **master seed** (recorded).
2. Shuffle the STEM reservoir once and the HUM reservoir once (with the master seed).
3. Allocate blocks in a single pass: 5 stem80 draws take disjoint majority blocks of 51 from the
   STEM shuffle and disjoint minority blocks of 13 from the HUM shuffle; 5 hum80 draws take
   disjoint majority blocks of 51 from HUM and minority blocks of 13 from STEM — all drawn from the
   *remaining* (unused) portions, so STEM usage = 255+65=320 and HUM usage = 65+255=320, both ≤
   reservoir with no overlap across any of the 10 draws. **Within-category subject composition**
   (see §3b) matches lm-eval's micro-average evaluation weights, not uniform-over-subjects.
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
- per-draw paired differences (reported individually), **mean**, **median**, and **win count** —
  these are the PRIMARY stability evidence
- a **descriptive** draw-clustered bootstrap interval (resample draws, not eval items). Treated as
  descriptive only: with 5 paired draws/direction the exact two-sided sign-flip test has just
  2⁵=32 arrangements, so even all-same-sign gives min two-sided p≈0.0625 — we therefore do **not**
  make p<0.05 significance claims from the pilot; the goal is effect-consistency across draws.
- the **direction × method interaction** (is any effect symmetric across stem80/hum80?)
- both balanced (primary) and target-weighted (secondary) metrics

Explicit guardrail: **DSMC is frozen; no hyperparameter is tuned on these 10 draws.**

## 7. Scope / sequencing (from review_0729 + choice_0730)

1. **(prereq) rep×selector 2×2 attribution gate** — DONE (`attribution_2x2_summary.md`). DSMC best
   in both directions; 2nd-order representation is the primary driver, MMD-diversity complementary
   (hypothesis). Gate passed → DSMC stays the headline.
2. **This protocol → pilot** (LAUNCH ONLY AFTER the draw artifacts are generated and data-reviewed):
   2 directions × **2 draws** × **8 method rows** (LESS and First-RR are distinct
   selectors — not collapsed into one parenthetical):
   1. DSMC
   2. true **Second-RR** (per-query nearest, cycling — NOT Second-TopK; that stays in the 2×2 ablation)
   3. **LESS** (Adam-preconditioned mean-gradient relevance top-k)
   4. true **First-RR** (1st-order round-robin)
   5. **GIST-SharedProj** (GIST scoring on the shared LESS-aligned projected features; see §8)
   6. **NICE**
   7. **Random-K** (uniform fixed-K, PRIMARY random baseline)
   8. **Random-K-LengthMatched** (fixed-K, length-histogram-matched; compute/length control)
3. Expand to **5 draws/direction** only after the pilot is clean.
4. Pre-registered representative draw (draw 0) per direction → **+3 paired training seeds** for
   training variance.
5. **Later axes** (separate, not now): target size n_T ∈ {16,64,128}, budget K ∈ {1%,5%}, keeping
   only {DSMC, strongest gradient baseline, GIST, Random-K}.

## 8. Baseline definitions (resolved per choice_0730 + code_review_0730)

- **GIST-SharedProj** (main GIST baseline = `select_gist_faithful.py`): the GIST scoring algorithm
  (official Gram/eigendecomp + isometric whitening `P=G_valᵀUₖSₖ⁻¹`, max-relevance top-k) run on our
  **shared LESS-aligned projected-gradient features**. It is **algorithm-faithful, not byte-faithful
  official GIST** — two deliberate adaptations remain: (1) raw LoRA gradient space → shared 8192-D
  JL/TRAK projection; (2) our LESS-aligned **Adam-preconditioned candidate / SGD target** protocol
  (this can change gradient *direction*, not just scale, so it is NOT undone by cosine). Using the
  same cache for all methods makes this a controlled comparison; we do NOT claim GIST's paper
  storage/compute numbers. Math notes (verified): `P = G_valᵀUₖSₖ⁻¹ = Vₖ` is an algebraic identity
  (top-k right singular vectors), so whitening ≡ plain top-k by construction; and at **k = M** target
  row-scaling is provably irrelevant (normalized-vs-rescaled Jaccard 1.0000 at k=M; <1 only k<M) —
  so re-extracting raw-target norms would not change selection at the official rank. **Rank rule =
  fixed at the official default `k = min(150, M) = M`** (no downstream selection). We do NOT run a
  test-accuracy rank gate: the pilot reports on MMLU test, so choosing GIST's rank by test accuracy
  would be test-set tuning while DSMC is frozen. `select_gist_jlnorm.py` (EVR95, k<M) is a
  **predeclared sensitivity ablation**, not a test-selected alternative. Caveat: at M=64,
  `k=min(150,64)=64` is the full row span (no spectral truncation) — a faithful consequence of the
  official fixed `target_dim`, and the EVR95 ablation is what exercises GIST's spectral filtering.
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
  over queries until K). Distinct from relevance top-k; often strong at low budget. **Per-draw RR
  permutation seed** = `3000 + draw_id` (recorded in meta); First-RR and Second-RR use the SAME
  target-query visiting order within a draw. Second-TopK from the 2×2 is retained ONLY as a mechanism
  ablation, not a pilot baseline. (RR top-K-per-query memory optimization is exact, not approximate.)
- **Second-TopK / Linear-MMD / DSMC**: already computed in the 2×2 gate.

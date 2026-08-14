# BBH external validation: 36 cells, 36/36 complete

Pre-registered in `prereg_bbh_external.md`; executed exactly as frozen. Comparative accuracy was sealed
until 36/36 train + 36/36 eval finished, then unsealed once. Descriptive only — **no p-values, no
variance-component inference**.

**Primary metric**: the pinned lm-eval `bbh_cot_fewshot` **micro** (`weight_by_size: true`) `exact_match`
over the **5,209-example held-out split**. **Shared no-SFT reference: 0.396429** (results JSON sha
`bb4006ada919…`).

**Primary statistical unit: the query/selection draw (n=3).** The two SFT seeds share the *same* query
draw *and* the *same* selected subset, so the six seed-level cells are **not** six independent selection
replicates. Seeds are averaged within each draw first; the six cells are retained as secondary stability
evidence.

Integrity, re-verified before unsealing: all 36 cells at exactly **84 optimizer steps**, all evals
**27/27 subtasks and 5,209 examples**, all **18 subset hashes unchanged**.

---

## Headline

> **Every targeted selector, including DSMC, loses to plain Random-K on BBH.** DSMC is not even the best
> targeted selector here — both round-robin variants beat it. And the post-hoc geometry diagnostic shows
> **DSMC achieves the lowest D2 to its own query set in 3/3 draws while ranking at the bottom
> downstream**, with the query-loss diagnostic showing every targeted selector *improving* the surrogate
> it optimizes while *losing* on the task. The MMLU surrogate/outcome dissociation therefore **replicates
> externally and is much stronger here.**

## 1. Absolute performance (micro exact_match)

| method | mean | Δ vs base |
|---|---|---|
| **randk** | **0.3925** | −0.39 pp |
| randk_seqlabelmatch *(secondary control)* | 0.3805 | −1.59 pp |
| first_rr | 0.3715 | −2.49 pp |
| second_rr | 0.3687 | −2.77 pp |
| **dsmc** | **0.3631** | −3.33 pp |
| less | 0.3601 | −3.63 pp |

**All method means fall below the shared no-SFT reference.** Random-K stays close to base (−0.39 pp) and
**one of its six cells (0.3974) is actually above base**, so it is wrong to say every Random run degrades
the model. Every *target-aware* method shows a substantially larger mean degradation — DSMC's −3.33 pp is
an order of magnitude beyond Random's −0.39 pp.

## 2. Paired analysis — PRIMARY: three draw-level blocks

DSMC − method, seeds averaged within each draw.

| comparator | draw0 | draw1 | draw2 | mean | DSMC wins |
|---|---|---|---|---|---|
| randk | −2.94 | −2.31 | −3.57 | **−2.94 pp** | **0/3** |
| randk_seqlabelmatch | −1.14 | −1.81 | −2.27 | **−1.74 pp** | **0/3** |
| second_rr | −0.44 | −0.54 | −0.70 | **−0.56 pp** | **0/3** |
| first_rr | +0.12 | −1.14 | −1.51 | **−0.84 pp** | **1/3** |
| less | +0.83 | +0.11 | −0.05 | **+0.29 pp** | **2/3** |

*Secondary stability evidence* (six seed-level cells): the same signs hold, at 0/6, 0/6, 0/6, 1/6 and 4/6
respectively — every cell agrees with its draw-level block except where the block is already near zero.

**This contradicts the MMLU within-targeted-selector ordering.** On MMLU, DSMC beat every targeted
selector at 5% (10/10). On BBH it loses to Second-RR **0/3** and First-RR 1/3, beating only LESS-style
TopK. The claim *"directional second moments substantially improve robustness among target-aware
selectors"* must be re-scoped:

> Directional second moments improved target-aware selection **on the MMLU family, particularly at 5%**,
> but that advantage **did not transfer to BBH**.

## 3. Query-draw spread and seed sensitivity

| method | draw0 | draw1 | draw2 | spread | seed42 | seed1 | \|diff\| |
|---|---|---|---|---|---|---|---|
| dsmc | 0.3636 | 0.3647 | 0.3610 | 0.0036 | 0.3648 | 0.3614 | 0.0035 |
| second_rr | 0.3680 | 0.3700 | 0.3680 | 0.0020 | 0.3685 | 0.3688 | 0.0003 |
| first_rr | 0.3624 | 0.3761 | 0.3761 | 0.0137 | 0.3724 | 0.3706 | 0.0019 |
| less | 0.3553 | 0.3636 | 0.3615 | 0.0083 | 0.3583 | 0.3620 | 0.0037 |
| randk | 0.3930 | 0.3878 | 0.3967 | 0.0089 | 0.3940 | 0.3910 | 0.0030 |
| randk_seqlabelmatch | 0.3750 | 0.3828 | 0.3838 | 0.0087 | 0.3827 | 0.3784 | 0.0043 |

Draw spread exceeds seed sensitivity for every method. Note that for the **targeted** methods block
variation is driven mainly by query realization, whereas for **Random** and **SeqLabelMatched** it is
driven by their own subset realization (seeds `5000+d` / `7000+d`) — *three draw/selection-realization
blocks crossed with two SFT seeds*, not pure query variance.

## 4. Secondary control: Random-K-SeqLabelMatched

Coarse joint (sequence-length × loss-bearing-label-position) matched Random, matching **DSMC's** 5×5 bin
histogram at fixed K=2707. Realized: sequence 0.969–0.976× DSMC, label positions 1.15–1.19× DSMC (plain
Random is 5.67×).

| | mean |
|---|---|
| DSMC | 0.3631 |
| SeqLabelMatched Random | 0.3805 |
| plain Random-K | 0.3925 |

Pre-registered interpretation, applied:

- **DSMC ≤ both Random variants (0/3 against each)** ⇒ *"strengthens the negative target-awareness
  result."* This is the case that obtained.
- **SeqLabelMatched < plain Random (−1.20 pp)** ⇒ *"over-matching BBH's short-response format may cause
  specialization / negative transfer."* Also obtained.

**Stated carefully.** Constraining Random to DSMC's coarse length profile costs 1.20 pp, which is
**numerically ~41% of the raw 2.94 pp DSMC−Random gap**. That is **consistent with instruction-format
composition being a contributor** — but it is **not a causal decomposition**, because matching
sequence/label length also shifts correlated source composition, provenance entropy, and lexical/content
distribution (this control's source entropy is 0.885, closer to DSMC's 0.833 than to Random's 1.201). We
therefore do **not** claim "format explains 41% of the gap", and we describe the SeqLabelMatched
degradation as *consistent with* harmful specialization toward the BBH-like long-context/short-response
regime, not as causal identification. DSMC still trails the format-matched control by 1.74 pp, so format
is at most part of the story.

## 5. 27-subtask diagnostic breakdown

*Not a statistical unit* — the primary metric remains the 5,209-example micro aggregate. This is a
diagnostic breakdown.

DSMC beats Random on **4 / 27** subtasks. Largest deficits, all on tasks the base model already handled:

| subtask | base | dsmc | randk | dsmc − randk |
|---|---|---|---|---|
| movie_recommendation | 0.7000 | 0.4908 | 0.6742 | −0.1833 |
| navigate | 0.5300 | 0.4767 | 0.5925 | −0.1158 |
| sports_understanding | 0.9000 | 0.7800 | 0.8675 | −0.0875 |
| geometric_shapes | 0.4100 | 0.2942 | 0.3683 | −0.0742 |
| logical_deduction_five_objects | 0.2950 | 0.2767 | 0.3375 | −0.0608 |
| … | | | | |
| object_counting | 0.4800 | 0.4758 | 0.4500 | +0.0258 |
| tracking_shuffled_objects_seven_objects | 0.1550 | 0.1450 | 0.1200 | +0.0250 |

Secondary 23-family regroup: DSMC 0.3824, Random 0.4118 — same ordering, reported for completeness.

---

# Post-hoc diagnostics (exploratory, existing artifacts only)

No new training. None of this was pre-registered; no method or protocol decision depends on it.

## D1. The surrogate/outcome dissociation REPLICATES — and is stronger than on MMLU

`bbh_forensic_geometry.json`. Same definition as the MMLU forensics: `M_P = E[u uᵀ]` on unit-normalized
projected gradients, `D2(S, Q_d) = ‖M_S − M_{Q_d}‖²_F`.

| method | D2 draw0 | draw1 | draw2 | accuracy |
|---|---|---|---|---|
| **dsmc** | **0.08654** | **0.09835** | **0.07390** | 0.3631 |
| less | 0.08813 | 0.09982 | 0.07535 | 0.3601 |
| first_rr | 0.08841 | 0.10020 | 0.07558 | 0.3715 |
| second_rr | 0.09062 | 0.10171 | 0.07714 | 0.3687 |
| randk | 0.11042 | 0.12155 | 0.09733 | **0.3925** |
| randk_seqlabelmatch | 0.11109 | 0.12504 | 0.09953 | 0.3805 |

**DSMC attains the lowest D2 in 3/3 draws**, and the D2 ordering is *identical* across all three draws.
The accuracy ordering is close to its reverse: **Spearman(D2, accuracy) = +0.771 / +0.829 / +0.886**
(pooled +0.829). Positive ρ means *lower D2 → lower accuracy*.

> **The method that best matches the target second moment is the method that performs worst.** DSMC
> demonstrably optimizes the geometry it claims to optimize — this is not a failure of the optimizer —
> yet on a query-aligned external family better matching predicts *worse* downstream utility. On MMLU the
> same anti-correlation was +0.389 (1%) and +0.112 (5%); on BBH it is +0.77 to +0.89.

## D2. The selection surrogate is misaligned with the downstream objective

`bbh_forensic_query_loss.json`. Query loss `L_Q` = CE of the final answer under *exactly* the supervision
used for target-gradient extraction. Base `L_Q` ≈ 4.48–4.80.

| method | Δ L_Q vs base | held-out EM |
|---|---|---|
| first_rr | **−1.400** | 0.3715 |
| less | **−1.356** | 0.3601 |
| dsmc | **−1.159** | 0.3631 |
| second_rr | **−0.685** | 0.3687 |
| randk | **+0.329** | **0.3925** |
| randk_seqlabelmatch | **+0.351** | 0.3805 |

**A perfect sign split.** All four target-aware selectors *improve* the query final-answer loss; both
Random variants *worsen* it — and the two that worsen it are the two that score best. Spearman(ΔL_Q,
EM) = **+0.600**.

This is the most direct statement of the mechanism available from these artifacts:

> The selectors succeed at the objective they optimize (final-answer cross-entropy on the query set) and
> fail at the objective the task rewards (generate a chain of thought, *then* answer). BBH targets are
> single tokens like `(C)`, `14`, `Yes`, while evaluation scores generated CoT — so **final-answer CE
> alignment is not reasoning-generation utility.** The negative transfer needs no appeal to anything
> mysterious about gradient matching.

## D3. It is NOT task-level specialization

`bbh_forensic_specialization.json`.

**H1 — does query exposure protect a task?** Correlating query-draw task frequency `n_{d,t}` with
seed-averaged subtask deltas gives Spearman **−0.210 / −0.280 / −0.224** vs Random (mean −0.238), and
−0.164 vs base. Consistent in sign across all three draws — and **negative**, i.e. more exposure goes with
slightly *more* damage.

> This is the **opposite** of the task-level specialization prediction. If DSMC were narrowly specializing
> toward the tasks its 64 queries happen to contain, those tasks should have been *protected*. They were
> not. Whatever is being over-fit lives at the **format/response-style level**, not the BBH-task level —
> consistent with the SeqLabelMatched control moving 41% of the gap while task exposure predicts nothing
> protective.

**H2 — does base accuracy predict degradation?** Spearman(base accuracy, post-SFT delta) = **−0.432** for
DSMC and **−0.233** for Random. Higher-base tasks fall further, about twice as steeply for DSMC.
**Heavily confounded** by ceiling effects and regression to the mean (a task at 0.90 has far more room to
fall than one at 0.005), so this is a descriptive association only. The interesting part is that DSMC's
coefficient is roughly double Random's despite both sharing the same ceiling structure.

---

## Honest conclusions

1. **The negative result replicates externally and gets stronger.** −2.94 pp vs Random on BBH, against
   −0.01 pp (5%) and −0.77 pp (1%) on MMLU. The earlier "skewed query misled the selector" explanation is
   unavailable here: the BBH queries are same-family, held-out, and normally sampled.
2. **"DSMC is the best targeted selector" does not generalize.** It held on MMLU; on BBH it loses to both
   round-robin variants. Scope that claim to the MMLU family.
3. **All method means fall below base**, though Random stays within 0.39 pp and one Random cell exceeds
   base. Target-aware methods degrade an order of magnitude more.
4. **The surrogate/outcome dissociation is now a cross-family result with a mechanism.** DSMC minimizes
   D2 in 3/3 draws and still loses; every targeted selector improves query final-answer CE and still
   loses. The selection objective is measurably achieved and measurably fails to transfer.
5. **Format, not task, is the plausible locus.** Task exposure does not protect; length/format matching
   moves ~41% of the gap. Neither is causally identified.

**What this does not show.** One pool, one budget, one model (Llama-2-7B), 3 draws × 2 seeds. The
diagnostics are exploratory and post-hoc. Nothing here isolates a cause; D1–D3 are consistent evidence,
not identification.

## Provenance

36/36 train + 36/36 eval, zero failures, zero driver restarts. `bbh_full_run_state.json`;
`bbh_external_run_plan.json` (36 cells / 18 subsets); `bbh_execution_contract.json`;
`bbh_eval_pin_manifest.json`; `bbh_canary_report.json`; `bbh_sft_canary_launch_receipt.json`.
All 18 subset hashes verified unchanged after the run. Diagnostics:
`bbh_forensic_geometry.json`, `bbh_forensic_query_loss.json`, `bbh_forensic_specialization.json`.

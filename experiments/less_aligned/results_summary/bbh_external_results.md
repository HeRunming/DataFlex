# BBH external validation: 36 cells, 36/36 complete

Pre-registered in `prereg_bbh_external.md`; executed exactly as frozen. Comparative accuracy was sealed
until 36/36 train + 36/36 eval finished, then unsealed once. Descriptive only — **no p-values, no
variance-component inference**.

**Primary metric**: the pinned lm-eval `bbh_cot_fewshot` **micro** (`weight_by_size: true`) `exact_match`
over the **27 lm-eval subtasks** on the **5,209-example held-out split**. The 23-family regroup is a
secondary diagnostic only. **Shared no-SFT reference: 0.396429** (results JSON sha `bb4006ada919…`).

Integrity, re-verified before unsealing: all 36 cells at exactly **84 optimizer steps**, all evals
**27/27 subtasks and 5,209 examples**, all **18 subset hashes unchanged**.

## Headline

> **Every targeted selector, including DSMC, loses to plain Random-K on BBH — and every one of them,
> including Random, lands BELOW the no-SFT base model.** DSMC is not even the best targeted selector
> here: First-RR and Second-RR beat it, and it beats only LESS-style TopK. The MMLU finding therefore
> replicates and strengthens on an external family: **target-aware selection does not pay off, and the
> gap against Random is larger on BBH (−0.0294) than it was on MMLU (−0.0001 at 5%, −0.0077 at 1%).**

## Absolute performance (micro exact_match)

| method | d0s42 | d0s1 | d1s42 | d1s1 | d2s42 | d2s1 | **mean** | **Δ vs base** |
|---|---|---|---|---|---|---|---|---|
| **randk** | 0.3924 | 0.3935 | 0.3922 | 0.3834 | 0.3974 | 0.3960 | **0.3925** | **−0.0039** |
| randk_seqlabelmatch *(secondary)* | 0.3765 | 0.3736 | 0.3857 | 0.3799 | 0.3859 | 0.3816 | 0.3805 | −0.0159 |
| first_rr | 0.3659 | 0.3588 | 0.3768 | 0.3753 | 0.3745 | 0.3776 | 0.3715 | −0.0249 |
| second_rr | 0.3701 | 0.3659 | 0.3688 | 0.3713 | 0.3667 | 0.3694 | 0.3687 | −0.0277 |
| **dsmc** | 0.3649 | 0.3623 | 0.3680 | 0.3613 | 0.3615 | 0.3605 | **0.3631** | **−0.0333** |
| less | 0.3534 | 0.3573 | 0.3628 | 0.3644 | 0.3586 | 0.3644 | 0.3601 | −0.0363 |

**All six arms are below base (0.396429).** Random-K is closest (−0.0039); DSMC is −0.0333. Every method
degrades a 7B base model on held-out BBH at K=2707 — consistent with the MMLU 1% arm, where most targeted
selectors also showed negative transfer, but here it extends to Random as well.

## Paired analysis: DSMC − method, per (draw, seed) cell

6 paired observations per comparison, as pre-registered.

| method | d0s42 | d0s1 | d1s42 | d1s1 | d2s42 | d2s1 | mean | median | DSMC wins/6 |
|---|---|---|---|---|---|---|---|---|---|
| less | +0.0115 | +0.0050 | +0.0052 | −0.0031 | +0.0029 | −0.0038 | **+0.0029** | +0.0039 | **4/6** |
| second_rr | −0.0052 | −0.0036 | −0.0008 | −0.0100 | −0.0052 | −0.0088 | **−0.0056** | −0.0052 | **0/6** |
| first_rr | −0.0010 | +0.0035 | −0.0088 | −0.0140 | −0.0131 | −0.0171 | **−0.0084** | −0.0109 | **1/6** |
| randk_seqlabelmatch | −0.0115 | −0.0113 | −0.0177 | −0.0186 | −0.0244 | −0.0211 | **−0.0174** | −0.0181 | **0/6** |
| randk | −0.0275 | −0.0313 | −0.0242 | −0.0221 | −0.0359 | −0.0355 | **−0.0294** | −0.0294 | **0/6** |

**This contradicts the MMLU within-targeted-selector ordering.** On MMLU, DSMC beat every targeted
selector at 5% (10/10) and most at 1%. On BBH it beats only LESS (4/6, +0.0029, and the sign flips in 2
cells), while **losing to Second-RR 0/6 and First-RR 1/6**. The claim "DSMC is the most robust targeted
selector" does **not** survive this external family and must be scoped to MMLU.

## Query-draw spread and seed sensitivity (descriptive)

Per-draw means (averaged over the 2 seeds) and per-seed means (averaged over the 3 draws). With one
observation per cell this design cannot support variance decomposition, so these are reported as spreads
only.

| method | draw0 | draw1 | draw2 | spread | seed42 | seed1 | \|diff\| |
|---|---|---|---|---|---|---|---|
| dsmc | 0.3636 | 0.3647 | 0.3610 | 0.0036 | 0.3648 | 0.3614 | 0.0035 |
| second_rr | 0.3680 | 0.3700 | 0.3680 | 0.0020 | 0.3685 | 0.3688 | 0.0003 |
| first_rr | 0.3624 | 0.3761 | 0.3761 | **0.0137** | 0.3724 | 0.3706 | 0.0019 |
| less | 0.3553 | 0.3636 | 0.3615 | 0.0083 | 0.3583 | 0.3620 | 0.0037 |
| randk | 0.3930 | 0.3878 | 0.3967 | 0.0089 | 0.3940 | 0.3910 | 0.0030 |
| randk_seqlabelmatch | 0.3750 | 0.3828 | 0.3838 | 0.0087 | 0.3827 | 0.3784 | 0.0043 |

Query-draw spread (0.002–0.014) exceeds seed sensitivity (0.000–0.004) for every method, so which queries
you draw matters more than training stochasticity. Note the caveat from `inherited_context_corrections.md`:
for the **targeted** methods block-to-block variation is driven mainly by query realization, but for
**Random** and **SeqLabelMatched** it is driven by their own subset realization (seeds are `5000+d` /
`7000+d`), so this is *three draw/selection-realization blocks crossed with two SFT seeds*, not pure
query variance.

## Secondary control: Random-K-SeqLabelMatched

Coarse joint (sequence-length × loss-bearing-label-position) matched Random, matching **DSMC's** 5×5 bin
histogram at fixed K=2707, seed `7000+d`. Realized: sequence 0.969–0.976× DSMC, label positions
1.15–1.19× DSMC (plain Random is 5.67×).

| | mean |
|---|---|
| DSMC | 0.3631 |
| SeqLabelMatched Random | 0.3805 |
| plain Random-K | 0.3925 |

- DSMC − plain Random = **−0.0294**
- DSMC − SeqLabelMatched = **−0.0174**
- SeqLabelMatched − plain Random = **−0.0120**

Applying the pre-registered interpretation table:

- **DSMC ≤ both Random variants (0/6 against each)** ⇒ *"strengthens the negative target-awareness
  result."* This is the outcome that obtained.
- **SeqLabelMatched < plain Random** ⇒ *"over-matching BBH's short-response format may cause
  specialization / negative transfer."* Also obtained, and it is informative: simply forcing Random to
  adopt DSMC's long-context/short-answer length profile **costs 1.2 points** relative to unconstrained
  Random.

So the format axis explains a real part of DSMC's deficit — moving Random onto DSMC's length profile
recovers 41% of the DSMC−Random gap (0.0120 / 0.0294) — but **not all of it**. DSMC remains 0.0174 below
even the format-matched control, so its deficit is not *only* a length/format artifact. Equally, this rules
out the reading that DSMC's targeting is helping in a way that length masks.

## 27 lm-eval subtasks (primary unit)

DSMC beats Random on **4 / 27** subtasks. Largest deficits are concentrated in the tasks the base model
was already good at:

| subtask | base | dsmc | randk | dsmc − randk |
|---|---|---|---|---|
| movie_recommendation | 0.7000 | 0.4908 | 0.6742 | **−0.1833** |
| navigate | 0.5300 | 0.4767 | 0.5925 | **−0.1158** |
| sports_understanding | 0.9000 | 0.7800 | 0.8675 | **−0.0875** |
| geometric_shapes | 0.4100 | 0.2942 | 0.3683 | −0.0742 |
| logical_deduction_five_objects | 0.2950 | 0.2767 | 0.3375 | −0.0608 |
| … | | | | |
| object_counting | 0.4800 | 0.4758 | 0.4500 | +0.0258 |
| tracking_shuffled_objects_seven_objects | 0.1550 | 0.1450 | 0.1200 | +0.0250 |
| temporal_sequences | 0.1250 | 0.1458 | 0.1325 | +0.0133 |
| hyperbaton | 0.5250 | 0.5208 | 0.5150 | +0.0058 |

The three worst cases (`movie_recommendation`, `navigate`, `sports_understanding`) are high-base-accuracy
tasks where DSMC loses 9–18 points to Random and 12–21 points to the base model — i.e. targeted SFT is
actively damaging capabilities the base model already had. That is the clearest signature of the
specialization the length-matched control also hints at.

**Secondary** 23-family regroup: DSMC 0.3824, Random 0.4118. Same ordering; reported for completeness
only, and never as the headline.

## Honest conclusions

1. **The negative result replicates externally and gets stronger.** Targeted selection does not beat
   well-controlled Random on BBH, and the margin (−0.0294) is much larger than on MMLU (−0.0001 at 5%,
   −0.0077 at 1%). This was pre-registered as an informative outcome either way.
2. **"DSMC is the best targeted selector" does not generalize.** It held on MMLU; on BBH DSMC loses to
   both round-robin variants (0/6 and 1/6) and beats only LESS-style TopK. That claim must now be scoped
   to the MMLU family.
3. **Everything degrades the base model at K=2707**, Random included (−0.0039). Any framing that presents
   BBH selection as an improvement over no-SFT would be wrong.
4. **Format/length explains part, not all.** Forcing Random onto DSMC's length profile costs 1.2 points,
   recovering 41% of the DSMC−Random gap; DSMC still trails the matched control by 0.0174.
5. **Query realization dominates training stochasticity** in this design, which is why the 3×2 crossed
   structure was worth having.

**What this does not show.** These are 6 cells per method at one budget on one pool and one model; nothing
here establishes a mechanism, and the earlier forensic result stands — DSMC optimizes its own D2 objective
successfully while that does not translate into downstream utility. See
`inherited_context_corrections.md` for the corrected MMLU claims this summary is consistent with.

## Provenance

36/36 train + 36/36 eval, zero failures, zero driver restarts. Run state:
`bbh_full_run_state.json`. Frozen artifacts: `bbh_external_run_plan.json` (36 cells / 18 subsets),
`bbh_execution_contract.json`, `bbh_eval_pin_manifest.json`, `bbh_canary_report.json`,
`bbh_sft_canary_launch_receipt.json`. All 18 subset hashes verified unchanged after the run.

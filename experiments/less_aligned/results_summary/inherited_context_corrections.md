# Inherited-context corrections (code_review_0810 item 4)

A fresh session summarised the completed MMLU results from its own condensed context rather than from
the result artifacts, and got two load-bearing claims wrong. Both are corrected here against the
committed summaries, and this file — not any conversational summary — is the reference for the BBH
round's related-work and framing text.

**Rule going forward: quote the artifacts.** `results_summary/*.md` and the manifests are the record;
a model-written recap of them is not.

---

## Correction 1 — DSMC does NOT beat every targeted selector at 1%

**What was wrongly claimed:** *"DSMC beats every targeted selector at both 5% (10/10) and 1% (9/10)."*

**What the artifacts say** (`results_summary/full1pct_budget_interaction_results.md`, DSMC − method,
balanced, 10 replicates):

| method | mean Δ | median | DSMC wins/10 |
|---|---|---|---|
| first_rr | +0.0197 | +0.0221 | 9/10 |
| less | +0.0080 | +0.0080 | 9/10 |
| nice | +0.0075 | +0.0048 | **7/10** |
| gist | +0.0073 | +0.0077 | **8/10** |
| **second_rr** | **+0.0017** | **+0.0009** | **5/10** |
| randk | −0.0077 | −0.0081 | 0/10 |
| randk_lenmatch | −0.0063 | −0.0089 | 2/10 |

So at 1% the win counts are **9/10, 9/10, 8/10, 7/10 and 5/10** — not a uniform 9/10. The
Second-RR comparison is the one that matters: mean advantage **+0.0017 (+0.17 pp)**, **5/10 cells**,
**3/5 direction-averaged blocks**, with a descriptive interval that crosses zero.

**The correct statement**, which is also the repo's own conclusion #2:

> At 1%, DSMC retains a clear advantage over the first-order and adapted targeted baselines
> (First-RR, LESS-style, GIST, NICE), but the additional MMD-coreset gain **over Second-RR
> essentially vanishes**. The "consistent additional benefit over second-order round-robin" claim is
> scoped to the **5%** budget (+0.0088, 10/10), where it does hold.

This matters for BBH because Second-RR is in the 5-method set: it is the sharpest comparator, and the
BBH round must not be written up as if DSMC were already known to beat it at K=2707.

## Correction 2 — the forensic analysis was not "refuted"; one *mechanism hypothesis* was

**What was wrongly claimed:** *"the forensic mechanism analysis was refuted."*

**What was actually refuted** (`results_summary/forensic_mechanism_analysis.md` §1) is one specific
hypothesis:

> *DSMC over-fits the skewed observed query `Q_d` and therefore ends up FARTHER from the balanced
> latent `P*`, especially at the tight 1% budget.*

That skew-capacity explanation is dead: DSMC is **closer** to the balanced reference than Random at
both budgets (D2 0.15206 vs 0.17627 at 1%; 0.15369 vs 0.17601 at 5%), on 10/10 draws, and it survives
a leave-one-draw-out 50/50-reweighted recomputation.

The forensic analysis itself **stands and produced a substantive positive result**:

> DSMC optimizes its own geometric objective decisively — and better D2 does **not** translate into
> better downstream utility. Source entropy alone does not explain it either (GIST matches Random's
> source entropy at 1.228 and still has the worst accuracy).

That is a finding about the surrogate/outcome gap, not a failed analysis. Calling it "refuted" both
understates the result and misstates which claim died.

## Correction 3 — there IS a selection-randomness axis; it is blocked with the draws

**What was wrongly claimed:** the BBH design is *"exactly (query realization) × (training
stochasticity) with no third hidden random axis."*

That is too strong. Random-K's seed is `5000 + draw_id` and the RR permutation seed is
`6000 + draw_id`, so the three draw blocks each carry a **different Random-subset realization** and a
different RR visiting order. The randomness has not been eliminated — it has been **blocked with the
draw index**.

Keeping it this way is deliberate and better than the alternative: three independent Random
realizations are more informative than reusing one Random subset three times, which would understate
Random's own variability.

**The correct statistical language:**

> Three draw/selection-realization blocks, crossed with two SFT seeds.

And when interpreting block spread: for the **targeted** methods the block-to-block variation is driven
mainly by query realization; for **Random** it is driven by the Random-subset realization. Block spread
must therefore not be reported as pure query-realization variance. What the frozen seeds *do*
guarantee — and this part was right — is that a given (draw, method) subset is **bit-identical across
the two SFT seeds**, so the training-seed axis is clean.

---

## Verified-correct claims (no change needed)

- 5% budget: DSMC beats LESS 10/10 (+0.0095), First-RR 10/10 (+0.0155), Second-RR 10/10 (+0.0088),
  GIST 9/10 (+0.0151), NICE 9/10 (+0.0111); **vs Random-K mean −0.0001, 4/10** (no observed advantage).
- 1% vs Random-K: mean **−0.0077**, **0/10** cells; the pre-registered "edge widens at low budget"
  hypothesis is refuted in this setting.
- Budget interaction: **−0.0076**, 1/5 blocks favour 1%.
- Equal-step arm: J = +0.0076, 4/5 blocks favour 420 steps, but **both methods degrade**
  (DSMC −0.0191, Random −0.0267) ⇒ pre-registered rule #4, inconclusive; the fixed-epoch 1% result
  stands unchanged.
- Most targeted selectors show negative transfer vs the no-selected-SFT base model (0.4003); only DSMC
  stays barely positive (+0.0014 at 1%, +0.0033 at 5%), while Random-K is strongest at 1% (+0.0092).

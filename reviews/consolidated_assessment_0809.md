# Consolidated assessment and recommendations — 2026-08-09

This note preserves the concise assessment delivered after consolidating the repository results.
The detailed paper-ready analysis, numerical tables, method description, vulnerability audit, and
experiment roadmap are in:

`experiments/less_aligned/results_summary/PAPER_READY_CONSOLIDATED_0809.md`

## Core assessment

The strongest paper story is **not**:

> DSMC is a generally superior instruction-data selector.

It is:

> **Directional second moments substantially improve robustness among target-aware selectors, but
> accurate target-gradient matching is not sufficient for downstream utility and does not
> outperform strong Random selection.**

This supports a paper framed as a positive representation result plus a controlled negative
mechanism result.

## Most solid findings

- At 5%, DSMC beats every examined target-aware baseline in all five direction-averaged blocks:
  - LESS-style mean-gradient TopK: `+0.95 pp`;
  - First-RR: `+1.56 pp`;
  - Second-RR: `+0.88 pp`;
  - GIST-SharedProj: `+1.51 pp`;
  - NICE-MMLU-EM: `+1.11 pp`.
- At 1%, DSMC remains stronger than the first-order/adapted targeted baselines, but its advantage
  over Second-RR nearly disappears.
- DSMC does not beat Random:
  - 5%: DSMC minus Random is approximately `-0.01 pp`;
  - 1%: `-0.77 pp`, with DSMC lower in all five direction-averaged blocks;
  - 1% at 420 steps: both DSMC and Random fall to approximately `0.3826`, below the no-SFT
    reference `0.4003`.
- DSMC is closer than Random to both the observed query geometry and a leave-one-draw-out balanced
  validation reference on every draw, despite failing to outperform Random downstream.
- Tightened within-draw D2/accuracy analysis:
  - 1% mean Spearman `+0.404`, median `+0.429`, positive in 10/10 draws;
  - 5% mean `+0.111`, median `+0.232`;
  - because lower D2 is the desired objective direction, this shows that better D2 ranking does not
    translate into better downstream ranking at 1%.

## ICLR readiness

The project is paper-worthy, but a submission in its current form would likely be borderline because:

1. the complete headline chain uses one model, one candidate pool, and one MMLU target family;
2. the effective unit for cross-direction Random comparisons is five paired blocks, not ten
   independent replicates;
3. query draw and SFT seed are coupled;
4. LESS, GIST, and NICE are shared-protocol adaptations rather than fully official end-to-end
   reproductions;
5. candidate-pool contamination against MMLU test has not been audited;
6. alignment/provenance documents are not fully consistent with the current code.

The main risk is limited external validity, not the negative Random result itself. The Random result
can become one of the paper's strongest scientific findings if reported directly.

## Highest-priority next experiment

If only one large experiment is added:

- keep Llama-2-7B and the Tulu candidate pool fixed;
- use a second target/evaluation family where query and evaluation are drawn from the same task
  distribution;
- compare only DSMC, Second-RR, LESS-style TopK, Random-K, and no-SFT;
- use one frozen budget and 3–5 paired query-draw/training-seed blocks;
- perform candidate-pool versus target-test decontamination.

Possible outcomes are both informative:

- DSMC beats Random: target awareness pays off in a query-aligned setting, while skewed finite
  queries expose its brittleness;
- DSMC still does not beat Random but beats other targeted methods: second moments robustly improve
  targeted selection, while target-aware selection itself remains brittle across tasks.

## Other priorities

1. Add the block-level intervals and within-draw forensic correlations already computed in the
   consolidated document.
2. Reconcile the alignment verifier, alignment reports, actual `target_dataset` loading behavior,
   resolved SFT configuration, warm-up checkpoint provenance, and equal-step `max_steps=420`
   provenance.
3. Audit exact, normalized, n-gram, and approximate candidate/test contamination.
4. If resources permit, add:
   - a matched 50/50 MMLU query control;
   - an Adam-candidate/SGD-query versus symmetric-gradient representation ablation.

Do not resume MMLU/Tulu learning-rate, LoRA, epoch, or source-balanced DSMC tuning. Those would be
outcome-driven modifications after observing the Random result.

## Final one-sentence assessment

> **The project already demonstrates that directional second moments are a substantially more
> robust target-aware representation, but its most important scientific result is that even accurate
> target-gradient matching can fail to identify the most useful instruction data; one clean external
> target family and a provenance cleanup are the remaining steps toward a solid ICLR submission.**

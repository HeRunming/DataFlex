# ICLR 2027 paper gap log — updated 2026-08-19

This file records open scientific, writing, and submission risks while the
paper is being drafted. It is not a list of requests to expand the project
indefinitely: the pre-registered stop rules still apply.

## Completed since the first draft

### Pre-registered Llama-3.2-3B × MMLU 5% arm

Completed in commit `c514298`: 35/35 adapters and 36/36 evaluations.
The pre-registered Outcome 3 fired. DSMC's Llama-2 MMLU advantage does not
transfer: DSMC−First-RR changes from +1.55 pp (10/10) to −0.18 pp (2/5
blocks), and DSMC−Second-RR from +0.88 pp (10/10) to −0.31 pp (1/5).
DSMC−Random remains near zero on both stacks. This result is now included in
the main paper and scopes DSMC as an instrument rather than a generally
superior selector.

### Evaluation-matched target-gradient sensitivity

Completed in commit `8873daf`. Only target gradients were re-extracted; frozen
subsets and downstream models were not changed. Operational versus bare target
gradients have mean row cosine 0.25–0.36. Among the frozen compared subsets,
DSMC remains lowest-D2 in 3/3 draws on both stacks under the tested bare
serialization, while losing to Random downstream in 3/3. The full non-DSMC
ranking changes in all Llama-3.2 draws, so only the DSMC-vs-Random
counterexample—not the complete ranking—is called serialization-robust.

## P0 — blocks a submission-ready draft

### 1. Verify every bibliography entry

`paper/iclr2027/references.bib` has primary metadata for the core papers.
Before submission:

- confirm whether the 2026 Critical Look and GIST papers acquire proceedings
  citations or remain arXiv citations;
- add model, benchmark, MMD, and LoRA citations that survive the page budget.

### 2. Compile and enforce the 9-page main-text limit

The draft has been compiled with a temporary Tectonic installation. Recompile
after every substantive revision and check:

- initial-submission main text ≤ 9 pages;
- references and appendix begin after the main text;
- figures/tables are legible in grayscale and at 100% zoom;
- no overfull boxes or unresolved references;
- line numbers and anonymity are preserved.

## P1 — high-value scientific clarity

### 3. Statistical reporting is intentionally descriptive

Keep the following units explicit:

- MMLU shared-Random comparisons: five direction-averaged draw-index blocks;
- BBH: three query/selection draws, after averaging two SFT seeds within draw;
- no-SFT: one shared reference, not a replicate;
- subtask rows: diagnostic, not independent samples.

Do not add “significant”, p-values, or confidence claims without a defensible
analysis plan. “Observed mean difference”, “0/3 draws”, and “no observed
advantage” are the current supported language.

### 4. Baseline fidelity must remain qualified

The main paper uses:

- LESS-style Mean-Gradient TopK;
- GIST-SharedProj;
- NICE-MMLU-EM / task-metric adaptation.

These share the experiment's datastore and protocol and are not full official
end-to-end reproductions. The appendix needs a compact comparison to official
implementations and the existing fidelity checks. Never silently shorten the
main claims to “we beat LESS/GIST/NICE”.

### 5. The theory is exact but narrow

Supported:

- quadratic-kernel MMD equals empirical directional second-moment discrepancy;
- the implemented deterministic greedy rule is the exact one-step marginal for
  the biased empirical MMD;
- DSMC measurably lowers its own D2 objective.

Unsupported and forbidden:

- calling the object a centered covariance;
- claiming recovery of the true update subspace or empirical Fisher;
- claiming the polynomial kernel matches the full distribution;
- claiming a stochastic-greedy \(1-1/e-\epsilon\) guarantee;
- claiming lower D2 should generally imply higher downstream utility.

### 6. Mechanism remains unidentified

Seq×Label-matched Random is consistent with format/provenance composition being
a contributor, but does not yield a causal percentage decomposition. The
task-exposure diagnostic does not support a simple task-frequency specialization
story. The conclusion should remain:

> We identify a failure of the surrogate assumption, not a unique failure
> mechanism.

## P2 — reproducibility and artifact quality

### 7. Semantic contamination audit remains missing

Exact, fuzzy, and long n-gram lexical checks are complete. A semantic
embedding-nearest-neighbor audit was not performed. Either:

- disclose the omission in the appendix and limitations, as the current draft
  does. The experimental stop rule is binding; do not reopen experiments solely
  to fill this audit layer.

Do not call the current audit a complete proof of no contamination.

### 8. Reproducibility drift in code/docs

Known issues to fix or disclose before releasing anonymous code:

- `scripts/build_less_pool.py` contains a stale hard-coded output path;
- `tests/test_mmd_basic.py` targets an older selector API;
- the available environments did not run the current MMD tests end to end;
- `MMD_IMPLEMENTATION_GUIDE.md`, `experiments/less_aligned/README.md`, and root
  usage docs contain stale settings;
- root README does not document the paper's MMD/DSMC extension.

These do not invalidate the frozen experiment artifacts, but they weaken a
reproducibility claim if left unresolved in the anonymous release.

### 9. Generate all paper displays from source artifacts

The new Figure 1 is generated by:
`paper/iclr2027/scripts/make_surrogate_chain.py`.

Still needed: convert remaining manually maintained LaTeX tables into generated
artifacts, especially the cross-stack MMLU table and appendix diagnostics.

Avoid manually maintained numbers in the final submission.

## ICLR 2027 submission checklist

Official requirements re-verified on 2026-08-19:

- genuine abstract deadline: **September 18, 2026 AOE**;
- full paper deadline: **September 25, 2026 AOE**;
- no authors may be added after the abstract deadline;
- all authors need current OpenReview profiles;
- initial main text: **9 pages maximum**;
- double blind: main paper and supplementary material must be anonymous;
- appendix is unlimited but reviewers are not required to read it;
- AI use statement is mandatory and does not count toward the page limit;
- reproducibility statement is strongly recommended;
- at least one author must register for the reciprocal-reviewing requirement
  unless the official exemption applies; author and submission quotas must be
  checked against the final author list.

The old internal date “September 16” is stale and must not reappear.

## Current title and central claim

Working title:

> Matching the Target Is Not Enough: Surrogate Failure in Gradient-Based
> Targeted Instruction Selection

Central claim:

> Better alignment under the studied operational gradient-based targeting
> construction is not sufficient for better downstream utility.

Forbidden inflation:

- “targeted selection fails”;
- “unreliable in general”;
- “DSMC is SOTA”;
- “Random significantly outperforms DSMC”;
- “cross-entropy is inherently misaligned with CoT”;
- “we found the mechanism”;
- “format explains 41%”.

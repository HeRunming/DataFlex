# ICLR 2027 paper gap log — 2026-08-18

This file records open scientific, writing, and submission risks while the
paper is being drafted. It is not a list of requests to expand the project
indefinitely: the pre-registered stop rules still apply.

## P0 — blocks a factual final draft

### 1. Finish the pre-registered Llama-3.2-3B × MMLU 5% arm

Current source:
`experiments/less_aligned/llama32_mmlu5pct_run_state.json`.

At the last drafting check, 22/35 adapters were trained, 0/35 evaluated, and
the base reference was not evaluated. The paper currently contains one explicit
red placeholder in Section 4. This result determines only the scope of the
positive method claim:

- DSMC > First-RR and Second-RR: full second-order coreset result transfers;
- DSMC ≈ Second-RR > First-RR: second-order representation transfers, extra
  coreset gain is stack dependent;
- first-order/Random ≥ second-order: the transferable result is the negative
  sufficiency finding, not the MMLU ranking.

Do not unseal or summarize partial accuracy.

### 2. Verify every bibliography entry

`paper/iclr2027/references.bib` has primary metadata for the core papers.
Before submission:

- confirm whether the 2026 Critical Look and GIST papers acquire proceedings
  citations or remain arXiv citations;
- add model, benchmark, MMD, and LoRA citations that survive the page budget.

### 3. Compile and enforce the 9-page main-text limit

The environment used to draft the paper currently has no TeX engine. Install or
use a separate TeX Live environment, run `make` in `paper/iclr2027`, and check:

- initial-submission main text ≤ 9 pages;
- references and appendix begin after the main text;
- figures/tables are legible in grayscale and at 100% zoom;
- no overfull boxes or unresolved references;
- line numbers and anonymity are preserved.

## P1 — high-value scientific clarity

### 4. Statistical reporting is intentionally descriptive

Keep the following units explicit:

- MMLU shared-Random comparisons: five direction-averaged draw-index blocks;
- BBH: three query/selection draws, after averaging two SFT seeds within draw;
- no-SFT: one shared reference, not a replicate;
- subtask rows: diagnostic, not independent samples.

Do not add “significant”, p-values, or confidence claims without a defensible
analysis plan. “Observed mean difference”, “0/3 draws”, and “no observed
advantage” are the current supported language.

### 5. Baseline fidelity must remain qualified

The main paper uses:

- LESS-style Mean-Gradient TopK;
- GIST-SharedProj;
- NICE-MMLU-EM / task-metric adaptation.

These share the experiment's datastore and protocol and are not full official
end-to-end reproductions. The appendix needs a compact comparison to official
implementations and the existing fidelity checks. Never silently shorten the
main claims to “we beat LESS/GIST/NICE”.

### 6. The theory is exact but narrow

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

### 7. Mechanism remains unidentified

Seq×Label-matched Random is consistent with format/provenance composition being
a contributor, but does not yield a causal percentage decomposition. The
task-exposure diagnostic does not support a simple task-frequency specialization
story. The conclusion should remain:

> We identify a failure of the surrogate assumption, not a unique failure
> mechanism.

## P2 — reproducibility and artifact quality

### 8. Semantic contamination audit is missing

Exact, fuzzy, and long n-gram lexical checks are complete. A semantic
embedding-nearest-neighbor audit was not performed. Either:

- run it only if it fits the already-approved remaining work and freeze its
  protocol before looking at selector outcomes; or
- disclose the omission in the appendix and limitations, as the current draft
  does.

Do not call the current audit a complete proof of no contamination.

### 9. Reproducibility drift in code/docs

Known issues to fix or disclose before releasing anonymous code:

- `scripts/build_less_pool.py` contains a stale hard-coded output path;
- `tests/test_mmd_basic.py` targets an older selector API;
- the available environments did not run the current MMD tests end to end;
- `MMD_IMPLEMENTATION_GUIDE.md`, `experiments/less_aligned/README.md`, and root
  usage docs contain stale settings;
- root README does not document the paper's MMD/DSMC extension.

These do not invalidate the frozen experiment artifacts, but they weaken a
reproducibility claim if left unresolved in the anonymous release.

### 10. Generate all paper displays from source artifacts

The new Figure 1 is generated by:
`paper/iclr2027/scripts/make_surrogate_chain.py`.

Still needed:

- a script/table generator for MMLU means and paired block differences;
- BBH main-table generation from `bbh_forensic_geometry.json` and
  `llama32_results.json`;
- appendix tables for equal-step, matched Random, bare CE, contamination, and
  provenance.

Avoid manually maintained numbers in the final submission.

## ICLR 2027 submission checklist

Official requirements verified on 2026-08-18:

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

> Matching the Target Is Not Enough: A Controlled Study of Targeted
> Instruction Selection

Central claim:

> Better target-gradient alignment is not sufficient for downstream
> improvement across the two tested model stacks.

Forbidden inflation:

- “targeted selection fails”;
- “unreliable in general”;
- “DSMC is SOTA”;
- “Random significantly outperforms DSMC”;
- “cross-entropy is inherently misaligned with CoT”;
- “we found the mechanism”;
- “format explains 41%”.

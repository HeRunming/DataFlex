# Frozen post-hoc protocol: held-out BBH proxy geometry

**Frozen before any proxy-gradient extraction or geometry result is observed.**

This is a target-side, no-training analysis motivated by the concern that the
primary BBH geometry is evaluated on the same 64 examples that define each
selection signal. It does not redefine any primary endpoint.

## Question

Do the frozen selected subsets preserve their target-geometry ordering on BBH
query examples that were never used by any of the three selectors?

## Proxy-test construction

- Population: the 1,302-example BBH query reservoir.
- Exclusion: every example appearing in any of the three 64-example selection
  draws.
- Remaining population: 1,117 examples.
- Proxy-test size: 256.
- Sampling: deterministic proportional stratification over the 27 harness
  subtasks, using seed `20260827` and largest-remainder allocation.
- The resulting IDs and all source/output hashes are frozen in
  `data/bbh_external/bbh_proxy_test_manifest.json`.

## Feature protocol

For each model stack:

- identical warm-up checkpoint and candidate datastore as the frozen BBH arm;
- candidate features: Adam-aware;
- proxy-test features: raw SGD gradients;
- projection dimension 8,192, seed 123;
- operational stack-specific serialization (`llama2` / `llama3`);
- cutoff 3,072;
- pinned BBH few-shot prompt rendering.

## Frozen subsets and metrics

No subset is regenerated. For each original draw and each of:

- DSMC,
- First-RR,
- Second-RR,
- Random-K,

compute against the common held-out proxy set:

\[
D_1(S,Q_{\rm proxy})
=
\|\mathbb E_S[u]-\mathbb E_{Q_{\rm proxy}}[u]\|_2,
\]

\[
D_2(S,Q_{\rm proxy})
=
\|M_S-M_{Q_{\rm proxy}}\|_F^2.
\]

Report per-draw rankings, DSMC-vs-Random differences, and the number of draws
in which each targeted subset is closer than Random.

## Interpretation rules

- If DSMC remains closer than Random under held-out \(D_2\), this weakens the
  explanation that its geometry advantage is only selection-query overfitting.
- If the ordering disappears or reverses, the geometry claim is restricted to
  the selection queries.
- First-RR held-out \(D_1\) is a sign-sensitive diagnostic.
- No outcome triggers reselection, SFT, a new kernel, a new model, or another
  proxy set.

This analysis is post-hoc and must be labelled as such in the paper.

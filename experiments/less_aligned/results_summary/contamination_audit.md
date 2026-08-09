# Candidate-pool ↔ MMLU-test **lexical** contamination screen

Hard gate before any new experiment (advice_0809), with scope corrected per code_review_0809.
Artifacts: `contamination_audit.json` (L1–L3), `contamination_global_lexical.json` (pool-wide fuzzy).
Scripts: `scripts/contamination_audit.py`, `scripts/contamination_global_lexical.py`.
**Pool**: 270,679 Tulu/Flan-v2/CoT/Dolly/OASST1 candidates. **Target**: 7,858 MMLU **test** items from
the STEM + Humanities subjects we evaluate on (`hails/mmlu_no_train`, offline cache).

**Naming, stated precisely:** this is a **lexical** contamination screen that passed. It is *not* a
complete decontamination certificate, because semantic (embedding) near-duplicate detection was not
run. See "Limitations" below.

## Layers run

| layer | definition | scope | hits | rate |
|-------|-----------|-------|------|------|
| L1 normalized exact | canonicalized question, and question+choices, hashed | pool-wide | **0** | 0.000000 |
| L2 long n-gram containment | any shared 13-gram with a canonicalized test question | pool-wide | **7** | 0.000026 |
| L3 fuzzy Jaccard (original) | Jaccard ≥ 0.5 over word 5-shingles | ⚠️ **L2 suspects only** | 0 | — |
| **L3-global (added)** | **MinHash/LSH over the whole pool → exact shingle Jaccard on every collision** | **pool-wide** | **0** at ≥0.5, **0** at ≥0.3 | 0.00000000 |
| L4 semantic NN | bge-base cosine nearest neighbour | — | **NOT RUN** | — |

### Why L3 had to be redone (code_review_0809)

The original L3 only evaluated candidates that had already passed the L2 13-gram filter. So "L3 = 0"
established only *"none of the seven 13-gram suspects also passes the fuzzy criterion"* — **not** that
the pool is free of fuzzy near-duplicates. A candidate with no contiguous 13-gram overlap but high
shingle similarity would never have been examined. That was a real scope error in my audit.

`contamination_global_lexical.py` closes it: 5-word shingles, 64-permutation MinHash, 16-band LSH
(122,215 buckets), every candidate probed against all 7,858 test items, and **every LSH collision
verified with exact shingle Jaccard**. Result: **0 candidates at Jaccard ≥ 0.5, and 0 even at the
deliberately loose ≥ 0.3** — pool-wide, not on a pre-filtered subset. Per-selector exposure is 0.00
examples for all 7 selectors at both budgets.

## The 7 L2 hits are false positives, not contamination

| cand idx | source | what it actually is |
|----------|--------|---------------------|
| 56741 | flan_v2 | reading-comprehension article about Winston Churchill |
| 89663 | flan_v2 | "write the sentence numbers needed to answer" boilerplate instruction |
| 248756 | oasst1 | coin/tube SI-density word problem |
| 263364–67 | oasst1 | four near-duplicate "differences between capitalism and communism" answers |

None is an MMLU test question; each shares a 13-gram only through generic instruction phrasing. L1 = 0,
pool-wide fuzzy = 0 corroborate this.

## Per-selector exposure: no enrichment (and the L2 test is underpowered by construction)

| budget | dsmc | randk | randk_lenmatch | second_rr | less | gist | nice |
|--------|------|-------|----------------|-----------|------|------|------|
| 1% (K=2707) | 0.00 | 0.00 | 0.10 | 0.20 | 0.00 | 0.00 | 0.00 |
| 5% (K=13533) | 1.50 | 0.60 | 0.40 | 1.00 | 0.00 | 0.20 | 0.10 |

(expected L2-flagged examples per selected subset, mean over 10 draws; pool-wide fuzzy is 0.00 for all)

With only 7 flagged candidates pool-wide, every selector's exposure is 0–1.5 examples out of
2,707–13,533. Those differences are single-example noise, not enrichment — and the flagged items are
false positives. The enrichment test is *underpowered by construction* because **there is essentially
nothing to enrich.**

## Verdict against the pre-stated decision rule

Rule: *near-zero overlap ⇒ MMLU results more credible; overlap with equal exposure ⇒ disclose as
limitation; method-differential contamination ⇒ downgrade the MMLU comparison.*

**Outcome: near-zero lexical overlap** (L1 = 0; pool-wide fuzzy = 0 at ≥0.3; L2 = 7/270,679, all
manually confirmed false positives). The MMLU downstream comparison is **not** downgraded on
contamination grounds, and the external family is not forced to become primary evidence for this
reason. **Lexical gate: PASSED.**

## Limitations (do not overstate this audit)

1. **L4 semantic / embedding NN was not run.** Candidate embeddings exist
   (`embeddings/candidate_270k.npy`, 270679×768, bge-base-en-v1.5) but embedding the 7,858 MMLU test
   items needs the `BAAI/bge-base-en-v1.5` encoder, unavailable offline in this pass. Residual risk is
   *paraphrase-level* overlap with no lexical trace. Kept as an explicit **release-time checklist item**;
   it is cheap once the encoder is available (one 7,858-item encode + a 270k×768 matmul, then hand/rule
   review of the closest few hundred pairs).
2. **"Standard pool" is not an argument.** An earlier draft of this document said residual risk was
   bounded because this is the standard LESS pool. That is not a bound — reusing prior work's data does
   not mathematically limit contamination. Removed.
3. When the BBH external split is fixed, contamination must be **re-run against the final held-out BBH
   evaluation subset** (a preliminary pool-vs-BBH 13-gram pass already gives 5/270,679 = 0.000018).

## Reproduction

```
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 python scripts/contamination_audit.py            # L1-L3
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 python scripts/contamination_global_lexical.py   # pool-wide fuzzy
```
Parameters recorded in the artifacts: `ngram=13`, `jaccard_thr=0.5`; MinHash `perms=64`, `bands=16`,
`rows=4`, `shingle_k=5`, `report_thr=0.3`.


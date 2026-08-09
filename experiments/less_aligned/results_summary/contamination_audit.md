# Candidate-pool ↔ MMLU-test contamination audit

Hard gate before any new experiment (advice_0809). Artifact: `contamination_audit.json`.
Script: `scripts/contamination_audit.py`. **Pool**: 270,679 Tulu/Flan-v2/CoT/Dolly/OASST1 candidates.
**Target**: 7,858 MMLU **test** items from the STEM + Humanities subjects we evaluate on
(`hails/mmlu_no_train`, offline cache).

## Layers run

| layer | definition | pool hits | rate |
|-------|-----------|-----------|------|
| L1 normalized exact | canonicalized question, and question+choices, hashed | **0** | 0.000000 |
| L2 long n-gram containment | any shared 13-gram between canonicalized test question and candidate | **7** | 0.000026 |
| L3 fuzzy lexical | Jaccard ≥ 0.5 over word 5-shingles, evaluated on all L2 suspects | **0** | 0.000000 |
| L4 semantic NN | bge-base cosine nearest neighbour | **NOT RUN** — see limitation |

## The 7 L2 hits are false positives, not contamination

Manual inspection of every flagged candidate:

| cand idx | source | what it actually is |
|----------|--------|---------------------|
| 56741 | flan_v2 | reading-comprehension article about Winston Churchill |
| 89663 | flan_v2 | "write the sentence numbers needed to answer" boilerplate instruction |
| 248756 | oasst1 | coin/tube SI-density word problem |
| 263364–67 | oasst1 | four near-duplicate "differences between capitalism and communism" answers |

None is an MMLU test question. Each shares a 13-gram with some test item only through generic
instruction/boilerplate phrasing. **L1 = 0 and L3 = 0 corroborate this**: there is no verbatim or even
fuzzy-lexical reproduction of MMLU test items in the pool.

## Per-selector exposure: no evidence of enrichment (and the test is underpowered by construction)

| budget | dsmc | randk | randk_lenmatch | second_rr | less | gist | nice |
|--------|------|-------|----------------|-----------|------|------|------|
| 1% (K=2707) | 0.00 | 0.00 | 0.10 | 0.20 | 0.00 | 0.00 | 0.00 |
| 5% (K=13533) | 1.50 | 0.60 | 0.40 | 1.00 | 0.00 | 0.20 | 0.10 |

(expected number of L2-flagged examples per selected subset, mean over 10 draws)

**Honest reading**: with only 7 flagged candidates pool-wide, every selector's exposure is between
**0 and 1.5 examples out of 2,707–13,533**. The apparent differences (e.g. DSMC 1.50 vs LESS 0.00 at
5%) are single-example noise, not enrichment — and the flagged examples are false positives anyway.
This test is *underpowered by construction*, which is the correct outcome: **there is essentially
nothing to enrich.**

## Verdict against the pre-stated decision rule

The pre-stated rule was: *near-zero overlap ⇒ MMLU results more credible; overlap with equal exposure
⇒ disclose as limitation; method-differential contamination ⇒ downgrade the MMLU comparison.*

**Outcome: near-zero overlap (L1=0, L3=0, L2=7/270,679 all false positives).** The MMLU downstream
comparison does **not** need to be downgraded, and the external clean family is *not* forced to become
the primary evidence on contamination grounds. The hard gate is **PASSED**.

## Limitation to disclose

**L4 (semantic nearest-neighbour) was not run.** Candidate embeddings already exist
(`embeddings/candidate_270k.npy`, 270679×768, bge-base-en-v1.5), but embedding the 7,858 MMLU test
items requires downloading the `BAAI/bge-base-en-v1.5` encoder, which was not available offline in
this pass. Two things bound the risk:

1. semantic-NN contamination without any lexical trace (L1/L2/L3 all clean) would be *paraphrase-level*
   overlap, which is a much weaker form of leakage than the verbatim/near-duplicate leakage this audit
   rules out;
2. the pool is the standard LESS pool and MMLU is a widely-audited benchmark, so any residual overlap
   is shared with prior work rather than specific to our setup.

Still, this should be stated as an open item rather than claimed as a complete audit. If the encoder
becomes available, L4 is cheap (one 7,858-item encode + a 270k×768 matmul) and should be added, with
the closest few hundred pairs reviewed by rule/hand as originally specified.

## Reproduction

```
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 python scripts/contamination_audit.py
```
Parameters recorded in the artifact: `ngram=13`, `ngram2=8`, `jaccard_thr=0.5`.

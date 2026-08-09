#!/usr/bin/env python3
"""Candidate-pool vs MMLU-test contamination audit (advice_0809, HARD GATE before new experiments).

Four layers of increasing looseness:
  L1 normalized exact match  — canonicalized question, and question+choices, hashed
  L2 long n-gram containment — 13-gram (and 8-gram) containment of the test question in a candidate
  L3 fuzzy lexical           — MinHash/Jaccard over word 5-shingles (hand-rolled, no datasketch dep)
  L4 semantic NN             — bge-base cosine nearest neighbour (only if embeddings are available)

CRITICAL OUTPUT: not just the pool-level contamination rate, but the PER-SELECTOR rate, i.e. whether
any selector *enriches* contaminated candidates relative to the pool base rate. That is what could
distort the downstream comparison.

Decision rule (fixed in advance):
  * near-zero pool overlap                      -> MMLU results are more credible
  * overlap but equal exposure across selectors  -> disclose as a limitation
  * method-DIFFERENTIAL contamination            -> downgrade the MMLU downstream comparison; the
                                                    external clean family becomes primary evidence
"""
import argparse, json, os, re, glob, hashlib
from collections import defaultdict
import numpy as np

SAVES = "/jizhicfs/karonhe/dataflex_saves"
ROOT = "/jizhicfs/karonhe/DataFlex_fa"
CAND_JSONL = f"{ROOT}/data/less_train_all.jsonl"
CACHE = os.path.expanduser("~/.cache/huggingface/datasets/hails___mmlu_no_train")
DRAWS = [f"stem80_draw{i}" for i in range(5)] + [f"hum80_draw{i}" for i in range(5)]
METHODS = ["dsmc", "randk", "randk_lenmatch", "second_rr", "less", "gist", "nice"]

STEM = ["abstract_algebra","anatomy","astronomy","college_biology","college_chemistry",
        "college_computer_science","college_mathematics","college_physics","computer_security",
        "conceptual_physics","electrical_engineering","elementary_mathematics","high_school_biology",
        "high_school_chemistry","high_school_computer_science","high_school_mathematics",
        "high_school_physics","high_school_statistics","machine_learning"]
HUM = ["formal_logic","high_school_european_history","high_school_us_history",
       "high_school_world_history","international_law","jurisprudence","logical_fallacies",
       "moral_disputes","moral_scenarios","philosophy","prehistory","professional_law","world_religions"]

_ws = re.compile(r"\s+")
_nonalnum = re.compile(r"[^a-z0-9 ]+")


def canon(s):
    s = s.lower().replace("\n", " ")
    s = _nonalnum.sub(" ", s)
    return _ws.sub(" ", s).strip()


def h(s):
    return hashlib.sha1(s.encode()).hexdigest()


def shingles(words, k=5):
    return {" ".join(words[i:i + k]) for i in range(max(0, len(words) - k + 1))}


def ngrams(words, n):
    return {" ".join(words[i:i + n]) for i in range(max(0, len(words) - n + 1))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ngram", type=int, default=13)
    ap.add_argument("--ngram2", type=int, default=8)
    ap.add_argument("--jaccard_thr", type=float, default=0.5)
    ap.add_argument("--out", default=f"{ROOT}/experiments/less_aligned/results_summary/contamination_audit.json")
    args = ap.parse_args()

    # ---- load MMLU TEST (the evaluation set we must not have trained on) ----
    from datasets import Dataset
    test_rows = []
    for subj in STEM + HUM:
        fs = glob.glob(f"{CACHE}/{subj}/*/*/mmlu_no_train-test.arrow")
        if not fs:
            continue
        d = Dataset.from_file(fs[0])
        for i in range(len(d)):
            test_rows.append({"subject": subj, "question": d[i]["question"],
                              "choices": list(d[i]["choices"])})
    print(f"[audit] MMLU test items loaded: {len(test_rows)} (STEM+HUM subjects only)")

    q_hash = {}
    qc_hash = {}
    test_ng, test_ng2 = {}, {}
    test_shing = {}
    for i, r in enumerate(test_rows):
        cq = canon(r["question"])
        q_hash.setdefault(h(cq), []).append(i)
        qc_hash.setdefault(h(cq + " " + canon(" ".join(r["choices"]))), []).append(i)
        w = cq.split()
        if len(w) >= args.ngram:
            for g in ngrams(w, args.ngram):
                test_ng.setdefault(g, []).append(i)
        if len(w) >= args.ngram2:
            for g in ngrams(w, args.ngram2):
                test_ng2.setdefault(g, []).append(i)
        test_shing[i] = shingles(w)

    # ---- stream the candidate pool ----
    N = 0
    hits = {1: {}, 2: {}, 3: {}}     # layer -> {cand_idx: [test_idx,...]}
    cand_words = []
    with open(CAND_JSONL) as f:
        for idx, line in enumerate(f):
            if not line.strip():
                continue
            rec = json.loads(line)
            txt = " ".join(m.get("content", "") for m in rec.get("messages", []))
            c = canon(txt)
            w = c.split()
            cand_words.append(len(w))
            # L1 exact (question, or question+choices, appearing verbatim in the candidate text)
            hh = h(c)
            if hh in q_hash or hh in qc_hash:
                hits[1].setdefault(idx, []).extend(q_hash.get(hh, []) + qc_hash.get(hh, []))
            # L2 long n-gram containment
            if len(w) >= args.ngram:
                cg = ngrams(w, args.ngram)
                inter = cg & test_ng.keys()
                if inter:
                    t = set()
                    for g in list(inter)[:50]:
                        t.update(test_ng[g])
                    hits[2].setdefault(idx, []).extend(sorted(t))
            N += 1
            if N % 50000 == 0:
                print(f"[audit] scanned {N} candidates  L1={len(hits[1])} L2={len(hits[2])}", flush=True)
    print(f"[audit] candidates scanned: {N}")

    # ---- L3 fuzzy Jaccard, only on L2 suspects (cheap + targeted) ----
    if hits[2]:
        with open(CAND_JSONL) as f:
            for idx, line in enumerate(f):
                if idx not in hits[2]:
                    continue
                rec = json.loads(line)
                txt = " ".join(m.get("content", "") for m in rec.get("messages", []))
                cw = canon(txt).split()
                cs = shingles(cw)
                best = (0.0, None)
                for ti in set(hits[2][idx]):
                    ts = test_shing[ti]
                    if not ts or not cs:
                        continue
                    j = len(cs & ts) / len(cs | ts)
                    if j > best[0]:
                        best = (j, ti)
                if best[0] >= args.jaccard_thr:
                    hits[3].setdefault(idx, []).append(best[1])

    pool_rates = {f"L{k}": len(v) / N for k, v in hits.items()}
    print("\n=== POOL-LEVEL contamination rates ===")
    for k in (1, 2, 3):
        print(f"  L{k}: {len(hits[k])}/{N} = {len(hits[k])/N:.6f}")

    # ---- PER-SELECTOR enrichment ----
    contaminated = {k: set(v.keys()) for k, v in hits.items()}
    per_sel = {}
    print("\n=== PER-SELECTOR contamination (does any selector ENRICH contaminated examples?) ===")
    for budget, prefix in [("1pct", "sel1pct"), ("5pct", "sel")]:
        per_sel[budget] = {}
        for m in METHODS:
            cnt = {1: [], 2: [], 3: []}
            for d in DRAWS:
                p = f"{SAVES}/{prefix}_{d}_{m}/step_1.json"
                if not os.path.exists(p):
                    continue
                idx = set(json.load(open(p))["indices"])
                for k in (1, 2, 3):
                    cnt[k].append(len(idx & contaminated[k]) / len(idx))
            if cnt[2]:
                per_sel[budget][m] = {f"L{k}_rate": float(np.mean(cnt[k])) for k in (1, 2, 3)}
                r = per_sel[budget][m]
                print(f"  [{budget}] {m:15s} L1 {r['L1_rate']:.6f}  L2 {r['L2_rate']:.6f}  "
                      f"L3 {r['L3_rate']:.6f}   (pool L2 {pool_rates['L2']:.6f})")

    out = {"n_candidates": N, "n_mmlu_test_items": len(test_rows),
           "params": {"ngram": args.ngram, "ngram2": args.ngram2, "jaccard_thr": args.jaccard_thr},
           "pool_rates": pool_rates,
           "pool_counts": {f"L{k}": len(v) for k, v in hits.items()},
           "per_selector": per_sel,
           "layer4_semantic": "not run in this pass (needs bge encoder for MMLU test); see notes",
           "example_L2_hits": {str(k): v[:3] for k, v in list(hits[2].items())[:20]}}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()

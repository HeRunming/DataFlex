#!/usr/bin/env python3
"""GLOBAL approximate lexical near-duplicate screen (code_review_0809 fix to contamination L3).

Why this exists: in `contamination_audit.py`, the L3 fuzzy-Jaccard test only ran on candidates that had
already passed the L2 13-gram filter. So "L3 = 0" only established

    none of the seven 13-gram suspects also passes the fuzzy criterion

and NOT "the 270k pool contains no fuzzy lexical near-duplicates of the evaluation items". A candidate
with no contiguous 13-gram overlap but high shingle similarity would never have been examined.

This script closes that gap with a pool-wide MinHash/LSH screen (hand-rolled; no datasketch dep):
  * 5-word shingles, MinHash with P permutations
  * banded LSH so every candidate is compared against test items sharing at least one band
  * every LSH-colliding pair then gets an EXACT shingle-Jaccard check
So the fuzzy criterion is evaluated over the whole pool, not over a pre-filtered subset.

HIGHER-SENSITIVITY banding (choice_0809, wording tightened per code_review_0810). LSH only *generates*
candidates, so under the ideal MinHash model a true pair of similarity s is detected with NOMINAL
probability 1-(1-s^rows)^bands. The first run used 16 bands x 4 rows = 12.2% at J=0.3 and 64.4% at
J=0.5 — far too weak for a null result to support a pool-wide exclusion claim. The default is now
32 bands x 2 rows: nominally 95.1% at J=0.3, 99.99% at J=0.5.

These are NOMINAL figures, not guarantees: see the caveat in `sig()` — the (a*h+b) mod (2^61-1) step
runs in numpy uint64, so the multiply wraps and the hash family is not exactly min-wise independent.
Exact-Jaccard verification of every collision means reported hits are never false positives; it is
recall, not precision, that is un-guaranteed.

`--target` selects the evaluation set screened against: `mmlu` (the 7,858 STEM+HUM test items) or
`bbh_heldout` (the 5,209-example BBH external-validation held-out split).

Reports pool-level rate AND per-selector rates (to test enrichment), same as the lexical audit.
"""
import argparse, json, os, re, glob, hashlib
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

_ws = re.compile(r"\s+"); _na = re.compile(r"[^a-z0-9 ]+")
MERSENNE = (1 << 61) - 1


def canon(s):
    s = s.lower().replace("\n", " ")
    return _ws.sub(" ", _na.sub(" ", s)).strip()


def shingles(words, k=5):
    if len(words) < k:
        return {" ".join(words)} if words else set()
    return {" ".join(words[i:i + k]) for i in range(len(words) - k + 1)}


def sig(shs, a, b):
    """MinHash-style signature: for each of P hash functions, min over shingles.

    CAVEAT (code_review_0810): the (a*h + b) mod (2^61-1) step is computed in numpy uint64, so the
    multiplication WRAPS rather than being exact modular arithmetic. Verified: exact Python
    (a*h) % M disagrees with the uint64 result. The family is therefore not a true min-wise independent
    permutation family, so the banded-LSH detection probability 1-(1-s^rows)^bands is a NOMINAL figure
    under the ideal MinHash model, not a guarantee for this implementation. It remains a useful
    high-sensitivity screen, and every collision is verified with EXACT shingle Jaccard, so reported
    hits are never false; only recall is un-guaranteed.
    """
    if not shs:
        return None
    h = np.array([int(hashlib.blake2b(s.encode(), digest_size=8).hexdigest(), 16) for s in shs],
                 dtype=np.uint64)
    # (a*h + b) mod Mersenne61 -> P x len(h), take min over shingles  [uint64: wraps, see docstring]
    v = (np.outer(a, h) + b[:, None]) % MERSENNE
    return v.min(axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--perms", type=int, default=64)
    ap.add_argument("--bands", type=int, default=32,
                    help="LSH bands (rows = perms/bands). HIGH-RECALL default 32x2: P_detect(J=0.3)="
                         "95.1%%, P_detect(J=0.5)=99.99%%. The earlier 16x4 gave only 12.2%%/64.4%%, "
                         "which is too weak to interpret a null result as a pool-wide exclusion.")
    ap.add_argument("--target", choices=["mmlu", "bbh_heldout"], default="mmlu",
                    help="which evaluation set to screen the candidate pool against")
    ap.add_argument("--jaccard_thr", type=float, default=0.5)
    ap.add_argument("--report_thr", type=float, default=0.3, help="also log weaker matches for review")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    assert args.perms % args.bands == 0
    if args.out is None:
        args.out = (f"{ROOT}/experiments/less_aligned/results_summary/"
                    f"contamination_global_lexical_{args.target}.json")
    rows = args.perms // args.bands
    rng = np.random.default_rng(12345)
    a = rng.integers(1, MERSENNE, size=args.perms, dtype=np.uint64)
    b = rng.integers(0, MERSENNE, size=args.perms, dtype=np.uint64)

    # ---- evaluation-set items -> signatures + LSH buckets ----
    test = []
    if args.target == "mmlu":
        from datasets import Dataset
        for subj in STEM + HUM:
            fs = glob.glob(f"{CACHE}/{subj}/*/*/mmlu_no_train-test.arrow")
            if not fs:
                continue
            d = Dataset.from_file(fs[0])
            for i in range(len(d)):
                test.append(canon(d[i]["question"] + " " + " ".join(d[i]["choices"])))
        label = "MMLU test (STEM+HUM)"
    else:
        p = f"{ROOT}/data/bbh_external/bbh_eval_heldout.jsonl"
        for line in open(p):
            if line.strip():
                r = json.loads(line)
                test.append(canon(r["input"]))
        label = "BBH held-out eval split"
    print(f"[global] target = {label}: {len(test)} items")
    # NOMINAL detection probability under the ideal MinHash model (see sig() docstring: the uint64
    # multiply wraps, so this is not a guarantee for this implementation).
    P_detect = lambda s: 1 - (1 - s ** rows) ** args.bands
    print(f"[global] NOMINAL LSH recall (ideal MinHash model): P_detect(0.3)={P_detect(0.3):.4f} "
          f"P_detect(0.5)={P_detect(0.5):.4f}")
    test_sh = [shingles(t.split()) for t in test]
    buckets = {}
    test_sig = []
    for i, sh in enumerate(test_sh):
        s = sig(sh, a, b)
        test_sig.append(s)
        if s is None:
            continue
        for bi in range(args.bands):
            key = (bi, hash(s[bi * rows:(bi + 1) * rows].tobytes()))
            buckets.setdefault(key, []).append(i)
    print(f"[global] LSH buckets: {len(buckets)} (perms={args.perms} bands={args.bands} rows={rows})")

    # ---- stream candidates, LSH-probe, exact-Jaccard verify collisions ----
    hits, weak = {}, {}
    N = 0
    with open(CAND_JSONL) as f:
        for idx, line in enumerate(f):
            if not line.strip():
                continue
            N += 1
            rec = json.loads(line)
            w = canon(" ".join(m.get("content", "") for m in rec.get("messages", []))).split()
            sh = shingles(w)
            s = sig(sh, a, b)
            if s is None:
                continue
            cands = set()
            for bi in range(args.bands):
                key = (bi, hash(s[bi * rows:(bi + 1) * rows].tobytes()))
                if key in buckets:
                    cands.update(buckets[key])
            if not cands:
                continue
            best = (0.0, None)
            for ti in cands:
                ts = test_sh[ti]
                if not ts:
                    continue
                j = len(sh & ts) / len(sh | ts)
                if j > best[0]:
                    best = (j, ti)
            if best[0] >= args.jaccard_thr:
                hits[idx] = {"jaccard": round(best[0], 4), "test_idx": int(best[1])}
            elif best[0] >= args.report_thr:
                weak[idx] = {"jaccard": round(best[0], 4), "test_idx": int(best[1])}
            if N % 50000 == 0:
                print(f"[global] scanned {N}  strong={len(hits)} weak={len(weak)}", flush=True)
    print(f"[global] candidates scanned: {N}")
    print(f"\n=== POOL-WIDE fuzzy lexical near-duplicates (exact Jaccard on all LSH collisions) ===")
    print(f"  Jaccard >= {args.jaccard_thr}: {len(hits)}/{N} = {len(hits)/N:.8f}")
    print(f"  Jaccard in [{args.report_thr},{args.jaccard_thr}): {len(weak)}/{N} = {len(weak)/N:.8f}")

    contaminated = set(hits)
    per_sel = {}
    # NOTE: the only selection subsets that exist are the MMLU-target ones. For --target bbh_heldout
    # these are therefore NOT the BBH experiment's subsets (those are not selected yet); the number is
    # a pool-composition check only. It is also trivially 0 whenever the pool-wide count is 0.
    print("\n=== PER-SELECTOR pool-wide fuzzy exposure (MMLU-target selection subsets) ===")
    for budget, prefix in [("1pct", "sel1pct"), ("5pct", "sel")]:
        per_sel[budget] = {}
        for m in METHODS:
            rates, counts = [], []
            for d in DRAWS:
                p = f"{SAVES}/{prefix}_{d}_{m}/step_1.json"
                if not os.path.exists(p):
                    continue
                idx = set(json.load(open(p))["indices"])
                inter = idx & contaminated
                rates.append(len(inter) / len(idx)); counts.append(len(inter))
            if rates:
                per_sel[budget][m] = {"rate": float(np.mean(rates)),
                                      "mean_examples": float(np.mean(counts))}
                print(f"  [{budget}] {m:16s} rate {np.mean(rates):.8f}  ({np.mean(counts):.2f} examples)")

    out = {"target": args.target, "target_label": label,
           "lsh_recall_nominal": {"P_detect_0.3": P_detect(0.3), "P_detect_0.5": P_detect(0.5),
                                  "basis": "1-(1-s^rows)^bands under the IDEAL MinHash model"},
           "n_candidates": N, "n_test": len(test),
           "params": {"perms": args.perms, "bands": args.bands, "rows": rows,
                      "shingle_k": 5, "jaccard_thr": args.jaccard_thr, "report_thr": args.report_thr},
           "pool_strong_count": len(hits), "pool_strong_rate": len(hits) / N,
           "pool_weak_count": len(weak), "pool_weak_rate": len(weak) / N,
           "strong_hits": {str(k): v for k, v in list(hits.items())[:100]},
           "weak_hits_sample": {str(k): v for k, v in list(weak.items())[:50]},
           "per_selector": per_sel,
           "per_selector_scope": ("Selection subsets exist only for the MMLU-target experiments. For "
                                  "--target bbh_heldout these are a pool-composition check, NOT the BBH "
                                  "experiment's subsets, which are not selected yet."),
           "scope_note": (f"This screen is POOL-WIDE: every one of the {N} candidates is MinHash/LSH-"
                          f"probed against all {len(test)} {label} items and every collision is verified "
                          f"with exact shingle Jaccard. It supersedes the earlier L3, which only examined "
                          f"L2 13-gram suspects."),
           "recall_caveat": (f"LSH is probabilistic candidate generation: at {args.bands} bands x {rows} "
                             f"rows the NOMINAL detection probability is 1-(1-s^rows)^bands, i.e. "
                             f"{P_detect(0.3):.3f} at J=0.3 and {P_detect(0.5):.5f} at J=0.5. Two "
                             f"separate reasons this is not a proof of absence: (1) LSH recall is <100% "
                             f"by construction; (2) the (a*h+b) mod (2^61-1) step runs in numpy uint64, "
                             f"so the multiply WRAPS and the family is not exactly min-wise independent "
                             f"-- the nominal figure assumes the ideal MinHash model. Every reported "
                             f"collision is still verified by EXACT shingle Jaccard, so there are no "
                             f"false positives; only recall is un-guaranteed.")}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()

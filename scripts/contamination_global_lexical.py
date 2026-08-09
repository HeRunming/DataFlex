#!/usr/bin/env python3
"""GLOBAL approximate lexical near-duplicate screen (code_review_0809 fix to contamination L3).

Why this exists: in `contamination_audit.py`, the L3 fuzzy-Jaccard test only ran on candidates that had
already passed the L2 13-gram filter. So "L3 = 0" only established

    none of the seven 13-gram suspects also passes the fuzzy criterion

and NOT "the 270k pool contains no fuzzy lexical near-duplicates of MMLU test items". A candidate with
no contiguous 13-gram overlap but high shingle similarity would never have been examined.

This script closes that gap with a pool-wide MinHash/LSH screen (hand-rolled; no datasketch dep):
  * 5-word shingles, MinHash with P permutations
  * banded LSH so every candidate is compared against test items sharing at least one band
  * every LSH-colliding pair then gets an EXACT shingle-Jaccard check
So the fuzzy criterion is evaluated over the whole pool, not over a pre-filtered subset.

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
    """MinHash signature: for each of P hash functions, min over shingles."""
    if not shs:
        return None
    h = np.array([int(hashlib.blake2b(s.encode(), digest_size=8).hexdigest(), 16) for s in shs],
                 dtype=np.uint64)
    # (a*h + b) mod Mersenne61 -> P x len(h), take min over shingles
    v = (np.outer(a, h) + b[:, None]) % MERSENNE
    return v.min(axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--perms", type=int, default=64)
    ap.add_argument("--bands", type=int, default=16, help="LSH bands (rows = perms/bands)")
    ap.add_argument("--jaccard_thr", type=float, default=0.5)
    ap.add_argument("--report_thr", type=float, default=0.3, help="also log weaker matches for review")
    ap.add_argument("--out", default=f"{ROOT}/experiments/less_aligned/results_summary/contamination_global_lexical.json")
    args = ap.parse_args()
    assert args.perms % args.bands == 0
    rows = args.perms // args.bands
    rng = np.random.default_rng(12345)
    a = rng.integers(1, MERSENNE, size=args.perms, dtype=np.uint64)
    b = rng.integers(0, MERSENNE, size=args.perms, dtype=np.uint64)

    # ---- MMLU test items -> signatures + LSH buckets ----
    from datasets import Dataset
    test = []
    for subj in STEM + HUM:
        fs = glob.glob(f"{CACHE}/{subj}/*/*/mmlu_no_train-test.arrow")
        if not fs:
            continue
        d = Dataset.from_file(fs[0])
        for i in range(len(d)):
            test.append(canon(d[i]["question"] + " " + " ".join(d[i]["choices"])))
    print(f"[global] MMLU test items: {len(test)}")
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
    print("\n=== PER-SELECTOR pool-wide fuzzy exposure ===")
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

    out = {"n_candidates": N, "n_test": len(test),
           "params": {"perms": args.perms, "bands": args.bands, "rows": rows,
                      "shingle_k": 5, "jaccard_thr": args.jaccard_thr, "report_thr": args.report_thr},
           "pool_strong_count": len(hits), "pool_strong_rate": len(hits) / N,
           "pool_weak_count": len(weak), "pool_weak_rate": len(weak) / N,
           "strong_hits": {str(k): v for k, v in list(hits.items())[:100]},
           "weak_hits_sample": {str(k): v for k, v in list(weak.items())[:50]},
           "per_selector": per_sel,
           "scope_note": "This screen is POOL-WIDE: every candidate is MinHash/LSH-probed against all "
                         "MMLU test items and every collision is verified with exact shingle Jaccard. "
                         "It supersedes the earlier L3, which only examined L2 13-gram suspects."}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()

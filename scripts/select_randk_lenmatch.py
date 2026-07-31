#!/usr/bin/env python3
"""
Random-K-LengthMatched selection (choice_0730 / review_0731 item 6): a fixed-K=13533 random subset
whose POST-TOKENIZATION length histogram matches the DSMC subset's, bucket-by-bucket. Controls for
the possibility that a method's gain comes merely from selecting longer/shorter (more/fewer training
tokens) examples. K is NOT changed to match tokens — we match the length DISTRIBUTION at fixed K.

Lengths = Llama-2 tokenizer length of the full sharegpt example (user+assistant, template applied),
truncated at cutoff_len=2048 (same as training). Buckets: [0,256),[256,512),[512,1024),[1024,1536),
[1536,2048]. For each bucket we sample (without replacement, seeded) exactly as many candidates as
DSMC has in that bucket. Fails loudly if any bucket lacks enough candidates.

Inputs: the full candidate jsonl (to tokenize) + the DSMC selection indices (step_1.json) as the
target histogram. Output: step_1.json with the matched indices + per-bucket diagnostics in metric.
"""
import argparse, json, os
import numpy as np

BUCKETS = [(0, 256), (256, 512), (512, 1024), (1024, 1536), (1536, 2049)]  # last inclusive of 2048


def bucket_of(n):
    for i, (lo, hi) in enumerate(BUCKETS):
        if lo <= n < hi:
            return i
    return len(BUCKETS) - 1  # >=2048 clamped to last bucket


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate_data", required=True, help="full candidate jsonl (270k)")
    ap.add_argument("--dsmc_step1", required=True, help="DSMC selection step_1.json (target histogram)")
    ap.add_argument("--out_cache_dir", required=True)
    ap.add_argument("--num_select", type=int, required=True)
    ap.add_argument("--tokenizer", default="/jizhicfs/karonhe/models/shakechen/Llama-2-7b-hf")
    ap.add_argument("--cutoff_len", type=int, default=2048)
    ap.add_argument("--length_cache", default=None,
                    help="optional .npy of precomputed per-candidate token lengths (order = jsonl)")
    ap.add_argument("--seed", type=int, required=True)
    args = ap.parse_args()

    # candidate texts
    rows = [json.loads(l) for l in open(args.candidate_data) if l.strip()]
    N = len(rows)
    K = args.num_select
    if K > N:
        raise ValueError(f"K {K} > N {N}")

    # token lengths (post-template, truncated at cutoff_len)
    if args.length_cache and os.path.exists(args.length_cache):
        lengths = np.load(args.length_cache)
        if len(lengths) != N:
            raise ValueError(f"length cache size {len(lengths)} != N {N}")
        print(f"[randk-lm] loaded cached lengths {args.length_cache}")
    else:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
        def ex_text(r):
            msgs = r.get("messages", [])
            return "\n".join(m.get("content", "") for m in msgs)
        lengths = np.zeros(N, dtype=np.int32)
        for i, r in enumerate(rows):
            ids = tok(ex_text(r), truncation=True, max_length=args.cutoff_len,
                      add_special_tokens=True)["input_ids"]
            lengths[i] = len(ids)
            if (i + 1) % 20000 == 0:
                print(f"[randk-lm] tokenized {i+1}/{N}", flush=True)
        if args.length_cache:
            os.makedirs(os.path.dirname(args.length_cache) or ".", exist_ok=True)
            np.save(args.length_cache, lengths)
            print(f"[randk-lm] saved lengths -> {args.length_cache}")

    cand_bucket = np.array([bucket_of(int(x)) for x in lengths])
    # target histogram from DSMC selection
    dsmc_idx = json.load(open(args.dsmc_step1))["indices"]
    dsmc_hist = np.bincount(cand_bucket[np.array(dsmc_idx)], minlength=len(BUCKETS))
    assert int(dsmc_hist.sum()) == len(dsmc_idx)
    # match per bucket
    rng = np.random.RandomState(args.seed)
    selected = []
    per_bucket = []
    for b in range(len(BUCKETS)):
        want = int(dsmc_hist[b])
        pool = np.where(cand_bucket == b)[0]
        if want > len(pool):
            raise RuntimeError(f"bucket {b} ({BUCKETS[b]}): need {want} but only {len(pool)} candidates")
        pick = rng.choice(pool, size=want, replace=False)
        selected.extend(int(x) for x in pick)
        per_bucket.append({"bucket": list(BUCKETS[b]), "dsmc": want, "picked": int(want),
                           "candidates_available": int(len(pool))})
    assert len(selected) == len(set(selected)) == K == int(dsmc_hist.sum()), \
        f"len {len(selected)} unique {len(set(selected))} K {K} dsmc_sum {int(dsmc_hist.sum())}"

    # token totals for the report
    tok_dsmc = int(lengths[np.array(dsmc_idx)].sum())
    tok_sel = int(lengths[np.array(selected)].sum())
    os.makedirs(args.out_cache_dir, exist_ok=True)
    meta = {"kernel": "random_k_lengthmatched", "seed": args.seed, "num_select": K,
            "n_candidates": N, "cutoff_len": args.cutoff_len,
            "dsmc_step1": os.path.abspath(args.dsmc_step1),
            "per_bucket": per_bucket, "total_tokens_dsmc": tok_dsmc,
            "total_tokens_selected": tok_sel, "token_diff": tok_sel - tok_dsmc}
    json.dump({"indices": selected, "metric": meta},
              open(os.path.join(args.out_cache_dir, "step_1.json"), "w"))
    print(f"[randk-lm] K={K} matched buckets={list(dsmc_hist)} tok(sel/dsmc)={tok_sel}/{tok_dsmc} "
          f"diff={tok_sel-tok_dsmc}")
    print(f"[randk-lm] wrote {args.out_cache_dir}/step_1.json")


if __name__ == "__main__":
    main()

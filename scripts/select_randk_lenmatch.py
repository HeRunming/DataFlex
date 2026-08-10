#!/usr/bin/env python3
"""
Random-K-LengthMatched selection (choice_0730 / review_0731 item 6): a fixed-K=13533 random subset
whose POST-TOKENIZATION length histogram matches the DSMC subset's, bucket-by-bucket. Controls for
the possibility that a method's gain comes merely from selecting longer/shorter (more/fewer training
tokens) examples. K is NOT changed to match tokens — we match the length DISTRIBUTION at fixed K.

LENGTH DEFINITION -- corrected in advice_0810_2. Buckets: [0,256),[256,512),[512,1024),[1024,1536),
[1536,2048]. For each bucket we sample (without replacement, seeded) exactly as many candidates as DSMC
has in that bucket. Fails loudly if any bucket lacks enough candidates.

  PREFERRED (use this): --length_npz data/candidate_length_cache_llamafactory.npz, which holds the EXACT
  executed length -- llama2 template `<s>[INST] u [/INST] a</s>` with LlamaFactory's own `infer_seqlen`
  budget split at cutoff_len=2048, i.e. len(input_ids) actually trained on.

  LEGACY FALLBACK (inexact): if no npz is given, lengths are computed as
  `"\n".join(m["content"] for m in messages)` then tokenized. That does NOT apply the llama2 template,
  despite what this docstring previously claimed. On a 300-candidate probe it disagrees with the true
  post-template bucket for ~0.3% of examples. The completed MMLU length-matched arm used this legacy
  path, so it should be described as COARSE content-length matched, NOT exact post-template matched.
  The legacy path is retained only so that arm stays reproducible; new arms must pass --length_npz.

NOTE what is and is not controlled: this matches the SEQUENCE-length distribution. It does NOT match
loss-bearing supervised tokens (#labels != IGNORE_INDEX), which can differ in the opposite direction --
on BBH draw0, DSMC has 1.51x Random's sequence tokens but only 0.18x its supervised tokens. Report both.

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
                    help="LEGACY .npy of naive-join lengths (order = jsonl). Inexact; see docstring.")
    ap.add_argument("--length_npz", default=None,
                    help="PREFERRED: candidate_length_cache_llamafactory.npz with the EXACT executed "
                         "post-template post-cutoff length. Required for new (BBH) arms.")
    ap.add_argument("--length_field", default="sequence_tokens_after_cutoff",
                    choices=["sequence_tokens_after_cutoff", "sequence_tokens_before_cutoff",
                             "supervised_label_tokens"],
                    help="which cached quantity to bucket on; default is the trained sequence length")
    ap.add_argument("--seed", type=int, required=True)
    args = ap.parse_args()

    # candidate texts
    rows = [json.loads(l) for l in open(args.candidate_data) if l.strip()]
    N = len(rows)
    K = args.num_select
    if K > N:
        raise ValueError(f"K {K} > N {N}")

    # ---- token lengths ----
    length_source = None
    if args.length_npz and os.path.exists(args.length_npz):
        z = np.load(args.length_npz)
        lengths = z[args.length_field]
        if len(lengths) != N:
            raise ValueError(f"length npz size {len(lengths)} != N {N}")
        length_source = f"EXACT LlamaFactory {args.length_field} from {args.length_npz}"
        print(f"[randk-lm] {length_source}")
    elif args.length_cache and os.path.exists(args.length_cache):
        lengths = np.load(args.length_cache)
        if len(lengths) != N:
            raise ValueError(f"length cache size {len(lengths)} != N {N}")
        length_source = f"LEGACY naive-join cache {args.length_cache} (NOT post-template)"
        print(f"[randk-lm] {length_source}")
    else:
        length_source = "LEGACY naive-join, computed now (NOT post-template; see docstring)"
        print(f"[randk-lm] WARNING: {length_source}. Pass --length_npz for the exact definition.")
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
            "total_tokens_selected": tok_sel, "token_diff": tok_sel - tok_dsmc,
            "length_source": length_source,
            "length_field": args.length_field if args.length_npz else "naive_join(legacy)",
            "matched_axis": "SEQUENCE length distribution only",
            "not_matched": ("loss-bearing supervised tokens (#labels != IGNORE_INDEX) are NOT matched and "
                            "can differ in the opposite direction; report them separately")}
    # supervised-label exposure of both subsets, when the exact cache is available
    if args.length_npz and os.path.exists(args.length_npz):
        _z = np.load(args.length_npz)
        if "supervised_label_tokens" in _z:
            _lab = _z["supervised_label_tokens"]
            meta["supervised_label_tokens_selected"] = int(_lab[np.array(selected)].sum())
            meta["supervised_label_tokens_dsmc"] = int(_lab[np.array(dsmc_idx)].sum())
    json.dump({"indices": selected, "metric": meta},
              open(os.path.join(args.out_cache_dir, "step_1.json"), "w"))
    print(f"[randk-lm] K={K} matched buckets={list(dsmc_hist)} tok(sel/dsmc)={tok_sel}/{tok_dsmc} "
          f"diff={tok_sel-tok_dsmc}")
    print(f"[randk-lm] wrote {args.out_cache_dir}/step_1.json")


if __name__ == "__main__":
    main()

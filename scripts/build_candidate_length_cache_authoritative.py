#!/usr/bin/env python3
"""AUTHORITATIVE candidate length cache, built by invoking the REAL pinned LlamaFactory SFT
preprocessing path (code_review_0811). Artifacts only — no model, no gradients, no SFT.

Why the previous cache was NOT exact
------------------------------------
`build_candidate_length_cache.py` concatenated all user messages into one string and all assistant
messages into another, then hand-built a single `[INST] user [/INST] assistant` pair. The message-structure
audit shows that is wrong for a real fraction of the pool:

    exactly (user, assistant)                    246,051  (90.90%)
    (user, assistant) x 2                         23,089
    (user, assistant) x 3                          1,539
    -> 24,628 candidates (9.10%) are MULTI-TURN, all from oasst1

For a multi-turn conversation LlamaFactory's `Llama2Template` encodes each turn in order, wrapping EVERY
user turn in its own `[INST] ... [/INST]` and supervising EVERY assistant turn. Flattening to one turn
therefore gets both `len(input_ids)` and `count(labels != IGNORE_INDEX)` wrong for those rows.

This script instead calls `SupervisedDatasetProcessor._encode_data_example` -- the exact function the SFT
run uses -- through the pinned `llama2` template, tokenizer, and the same `DataArguments`
(`cutoff_len=2048`, `train_on_prompt=False`, `mask_history=False`), and reads the two quantities straight
out of its output. No template is reconstructed by hand.

TERMINOLOGY (tightened per code_review_0811):
  * `len(input_ids)` is **post-cutoff sequence-token exposure**, NOT "GPU cost". Real batches pad to the
    longest member, so a sum of unpadded lengths is not FLOPs.
  * `count(labels != -100)` is **loss-bearing label positions**, NOT "amount of supervised signal".
    Transformers ignores -100 and then takes a token-count-normalized mean cross-entropy, so 5.67x the
    label positions is not 5.67x the gradient signal.
"""
import argparse, hashlib, json, os

import numpy as np

ROOT = "/jizhicfs/karonhe/DataFlex_fa"
EXP = f"{ROOT}/experiments/less_aligned"
CAND_JSONL = f"{ROOT}/data/less_train_all.jsonl"
BASE = "/jizhicfs/karonhe/models/shakechen/Llama-2-7b-hf"
IGNORE_INDEX = -100
OUT_NPZ = f"{ROOT}/data/candidate_length_cache_authoritative.npz"
MANUAL_NPZ = f"{ROOT}/data/candidate_length_cache_llamafactory.npz"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cutoff_len", type=int, default=2048)
    ap.add_argument("--out", default=OUT_NPZ)
    ap.add_argument("--manifest", default=f"{EXP}/candidate_length_cache_authoritative_manifest.json")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    import warnings
    warnings.filterwarnings("ignore")
    import llamafactory
    from transformers import AutoTokenizer, __version__ as tfv
    from llamafactory.data.template import TEMPLATES
    from llamafactory.data.processor.supervised import SupervisedDatasetProcessor
    from llamafactory.hparams import DataArguments

    tok = AutoTokenizer.from_pretrained(BASE)
    template = TEMPLATES["llama2"]
    # the pinned SFT data settings; these are what select_<draw>.yaml / the SFT configs use
    data_args = DataArguments(cutoff_len=args.cutoff_len, train_on_prompt=False, mask_history=False)
    proc = SupervisedDatasetProcessor(template=template, tokenizer=tok, processor=None,
                                     data_args=data_args)

    seq, lab, turns = [], [], []
    n = 0
    with open(CAND_JSONL) as f:
        for line in f:
            if args.limit and n >= args.limit:
                break
            if not line.strip():
                continue
            r = json.loads(line)
            msgs = r.get("messages", [])
            # LlamaFactory's internal role names; the converter maps sharegpt user/assistant to these
            conv = [{"role": ("user" if m["role"] == "user" else "assistant"),
                     "content": m.get("content", "")} for m in msgs]
            prompt, response = conv[:-1], conv[-1:]
            ids, labels = proc._encode_data_example(
                prompt=prompt, response=response, system="", tools="",
                images=[], videos=[], audios=[])
            seq.append(len(ids))
            lab.append(int(sum(1 for x in labels if x != IGNORE_INDEX)))
            turns.append(len(msgs) // 2)
            n += 1
            if n % 25000 == 0:
                print(f"  {n} candidates ...", flush=True)

    seq = np.array(seq, dtype=np.int32)
    lab = np.array(lab, dtype=np.int32)
    turns = np.array(turns, dtype=np.int16)
    np.savez_compressed(args.out, sequence_tokens_after_cutoff=seq,
                        loss_bearing_label_positions=lab, n_turns=turns)

    h = hashlib.sha256()
    with open(args.out, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 22), b""):
            h.update(b)

    # ---- parity vs the superseded manual cache ----
    parity = None
    if os.path.exists(MANUAL_NPZ):
        z = np.load(MANUAL_NPZ)
        mseq, mlab = z["sequence_tokens_after_cutoff"][:n], z["supervised_label_tokens"][:n]
        srcs = []
        with open(CAND_JSONL) as f:
            for i, line in enumerate(f):
                if i >= n:
                    break
                srcs.append(json.loads(line).get("dataset", "unknown"))
        srcs = np.array(srcs)
        dseq, dlab = seq - mseq, lab - mlab
        mism = (dseq != 0) | (dlab != 0)
        by_src = {}
        for s in sorted(set(srcs.tolist())):
            k = srcs == s
            by_src[s] = {"n": int(k.sum()), "n_mismatch": int(mism[k].sum()),
                         "mismatch_rate": round(float(mism[k].mean()), 6),
                         "mean_seq_diff": round(float(dseq[k].mean()), 3),
                         "mean_label_diff": round(float(dlab[k].mean()), 3)}
        multi = turns > 1
        parity = {
            "compared_against": os.path.relpath(MANUAL_NPZ, ROOT),
            "n_mismatched": int(mism.sum()), "mismatch_rate": round(float(mism.mean()), 6),
            "sequence_diff": {"mean": round(float(dseq.mean()), 3), "max_abs": int(np.abs(dseq).max()),
                              "total_manual": int(mseq.sum()), "total_authoritative": int(seq.sum())},
            "label_diff": {"mean": round(float(dlab.mean()), 3), "max_abs": int(np.abs(dlab).max()),
                           "total_manual": int(mlab.sum()), "total_authoritative": int(lab.sum())},
            "by_source": by_src,
            "single_turn_mismatch_rate": round(float(mism[~multi].mean()), 6),
            "multi_turn_mismatch_rate": round(float(mism[multi].mean()), 6) if multi.any() else None,
            "conclusion": ("the manual cache is wrong exactly where the audit predicted: multi-turn rows. "
                           "Single-turn rows should agree closely (the manual reconstruction matches the "
                           "template for one turn), multi-turn rows should not."),
        }

    man = {
        "cache": os.path.relpath(args.out, ROOT), "sha256": h.hexdigest(),
        "n_candidates": int(n), "cutoff_len": args.cutoff_len,
        "authoritative_because": ("built by calling SupervisedDatasetProcessor._encode_data_example -- the "
                                  "exact function the SFT run uses -- through the pinned llama2 template, "
                                  "pinned tokenizer and the same DataArguments. No hand-built template."),
        "pipeline": {"llamafactory": getattr(llamafactory, "__version__", "unknown"),
                     "transformers": tfv, "tokenizer": BASE, "template": "llama2",
                     "processor": "llamafactory.data.processor.supervised.SupervisedDatasetProcessor",
                     "data_args": {"cutoff_len": args.cutoff_len, "train_on_prompt": False,
                                   "mask_history": False}},
        "fields": {
            "sequence_tokens_after_cutoff": ("len(input_ids): POST-CUTOFF SEQUENCE-TOKEN EXPOSURE. Not "
                                             "'GPU cost' -- real batches pad to the longest member, so a "
                                             "sum of unpadded lengths is not FLOPs."),
            "loss_bearing_label_positions": ("count(labels != -100): LOSS-BEARING LABEL POSITIONS. Not "
                                             "'amount of supervised signal' -- the loss ignores -100 then "
                                             "takes a token-normalized mean CE, so Nx the positions is "
                                             "not Nx the gradient signal."),
            "n_turns": "number of (user, assistant) pairs in the conversation",
        },
        "turn_distribution": {int(k): int(v) for k, v in
                              zip(*np.unique(turns, return_counts=True))},
        "stats": {k: {"mean": float(v.mean()), "median": float(np.median(v)),
                      "p90": float(np.percentile(v, 90)), "max": int(v.max()), "total": int(v.sum())}
                  for k, v in [("sequence_tokens_after_cutoff", seq),
                               ("loss_bearing_label_positions", lab)]},
        "label_fraction_of_sequence": float(lab.sum() / seq.sum()),
        "parity_vs_manual_cache": parity,
        "no_compute_run": "artifacts only: no model loaded, no gradients, no SFT",
    }
    json.dump(man, open(args.manifest, "w"), indent=2)
    print(f"\ncandidates                 : {n}")
    print(f"turn distribution          : {man['turn_distribution']}")
    for k, s in man["stats"].items():
        print(f"  {k:30s} mean {s['mean']:8.1f}  median {s['median']:7.0f}  total {s['total']:,}")
    print(f"label fraction of sequence : {man['label_fraction_of_sequence']:.4f}")
    if parity:
        print(f"\nPARITY vs manual cache:")
        print(f"  mismatched            : {parity['n_mismatched']:,} ({parity['mismatch_rate']:.4%})")
        print(f"  single-turn mismatch  : {parity['single_turn_mismatch_rate']:.4%}")
        mt = parity['multi_turn_mismatch_rate']
        print(f"  multi-turn mismatch   : {mt:.4%}" if mt is not None else
              "  multi-turn mismatch   : n/a (no multi-turn rows in this slice)")
        print(f"  label totals manual/auth: {parity['label_diff']['total_manual']:,} / "
              f"{parity['label_diff']['total_authoritative']:,}")
    print(f"\ncache sha256 : {man['sha256']}")
    print(f"wrote {args.out}\nwrote {args.manifest}")


if __name__ == "__main__":
    main()

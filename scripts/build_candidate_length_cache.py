#!/usr/bin/env python3
"""SUPERSEDED — do not use for new work. Kept only to reproduce the provisional numbers it produced.
Use `build_candidate_length_cache_authoritative.py` instead.

WHY SUPERSEDED (code_review_0811): this script concatenates all user messages into one string and all
assistant messages into another, then hand-builds ONE `[INST] user [/INST] assistant` pair. A
message-structure audit of the pool found 24,628 / 270,679 candidates (9.10%, all oasst1) are MULTI-TURN,
and LlamaFactory wraps EVERY user turn in its own `[INST] ... [/INST]` and supervises EVERY assistant
turn. Measured parity against the authoritative cache: 0.0000% mismatch on single-turn rows, 100% on
multi-turn rows. So this file is right for 90.9% of the pool and wrong for the rest, and it must not be
called "exact".

The finding it produced did survive the rebuild (DSMC/Random sequence ratio 1.5082 -> 1.5050, label ratio
0.1764 unchanged), but the authoritative cache is the one to cite.

Original docstring follows.
---
Immutable candidate-level LENGTH cache using a HAND-RECONSTRUCTED LlamaFactory template (advice_0810_2).
Artifacts only — no model, no gradients, no SFT.

Why this exists
---------------
Phase B reported `post_template_tokens` and I described the DSMC/Random ratio as "~1.5x more supervised
tokens". That conflated two different quantities, and the correction matters:

  sequence_tokens_after_cutoff   the full `input_ids` length -- post-cutoff sequence-token EXPOSURE
  supervised_label_tokens        count(labels != IGNORE_INDEX) -- LOSS-BEARING LABEL POSITIONS

Under the llama2 template the prompt is MASKED, so only the assistant continuation is supervised. On a
300-candidate probe that is ~18.5% of the sequence. So a 1.5x sequence-token ratio does NOT imply a 1.5x
supervised-token ratio, and the two must be reported separately.

A second reason: `scripts/select_randk_lenmatch.py` documents "template applied" but actually computes
`"\n".join(m["content"] for m in messages)` and tokenizes that. This cache provides the real thing, so
BBH length matching buckets on the length the model is actually trained on.

Emits three quantities per candidate, computed with the pinned tokenizer and LlamaFactory's own
`infer_seqlen` budget split at the SFT `cutoff_len=2048`:
  * sequence_tokens_before_cutoff   (source + target, untruncated)
  * sequence_tokens_after_cutoff    (== len(input_ids) actually trained on)
  * supervised_label_tokens         (== #labels != IGNORE_INDEX)
"""
import argparse, hashlib, json, os

import numpy as np

ROOT = "/jizhicfs/karonhe/DataFlex_fa"
EXP = f"{ROOT}/experiments/less_aligned"
CAND_JSONL = f"{ROOT}/data/less_train_all.jsonl"
BASE = "/jizhicfs/karonhe/models/shakechen/Llama-2-7b-hf"
SFT_CUTOFF = 2048
OUT_NPZ = f"{ROOT}/data/candidate_length_cache_llamafactory.npz"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cutoff_len", type=int, default=SFT_CUTOFF,
                    help="the SFT cutoff (2048). NOT the target-gradient cutoff (3072).")
    ap.add_argument("--out", default=OUT_NPZ)
    ap.add_argument("--manifest", default=f"{EXP}/candidate_length_cache_manifest.json")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    import warnings
    warnings.filterwarnings("ignore")
    from transformers import AutoTokenizer, __version__ as tfv
    from llamafactory.data.processor.processor_utils import infer_seqlen
    import llamafactory
    tok = AutoTokenizer.from_pretrained(BASE)

    before, after, labels = [], [], []
    n = 0
    with open(CAND_JSONL) as f:
        for line in f:
            if args.limit and n >= args.limit:
                break
            if not line.strip():
                continue
            r = json.loads(line)
            m = r.get("messages", [])
            u = " ".join(x.get("content", "") for x in m if x.get("role") == "user")
            a = " ".join(x.get("content", "") for x in m if x.get("role") == "assistant")
            # exactly the llama2 template LlamaFactory applies: <s>[INST] u [/INST] a </s>
            src = tok(f"[INST] {u} [/INST]", add_special_tokens=True)["input_ids"]
            tgt = tok(a, add_special_tokens=False)["input_ids"] + [tok.eos_token_id]
            ks, kt = infer_seqlen(len(src), len(tgt), args.cutoff_len)
            before.append(len(src) + len(tgt))
            after.append(ks + kt)
            labels.append(kt)          # prompt is masked; only the continuation carries loss
            n += 1
            if n % 25000 == 0:
                print(f"  {n} candidates ...", flush=True)

    before = np.array(before, dtype=np.int32)
    after = np.array(after, dtype=np.int32)
    labels = np.array(labels, dtype=np.int32)
    np.savez_compressed(args.out, sequence_tokens_before_cutoff=before,
                        sequence_tokens_after_cutoff=after, supervised_label_tokens=labels)

    h = hashlib.sha256()
    with open(args.out, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 22), b""):
            h.update(b)
    man = {
        "cache": os.path.relpath(args.out, ROOT), "sha256": h.hexdigest(),
        "n_candidates": int(n), "cutoff_len": args.cutoff_len,
        "pipeline": {
            "template": "llama2 -> <s>[INST] {user} [/INST] {assistant}</s>",
            "budget_split": "llamafactory.data.processor.processor_utils.infer_seqlen",
            "llamafactory_version": getattr(llamafactory, "__version__", "unknown"),
            "transformers": tfv, "tokenizer": BASE,
        },
        "fields": {
            "sequence_tokens_before_cutoff": "source+target tokens, untruncated",
            "sequence_tokens_after_cutoff": ("len(input_ids): post-cutoff sequence-token EXPOSURE. NOT \"GPU cost\" -- batches pad to the longest member, so unpadded sums are not FLOPs."),
            "supervised_label_tokens": ("#labels != IGNORE_INDEX: LOSS-BEARING LABEL POSITIONS. NOT "
                                       "\"amount of supervised signal\" -- the loss ignores -100 then "
                                       "takes a token-normalized mean CE. The prompt is masked under "
                                       "llama2, so this counts assistant continuations only."),
        },
        "stats": {k: {"mean": float(v.mean()), "median": float(np.median(v)),
                      "p90": float(np.percentile(v, 90)), "max": int(v.max()), "total": int(v.sum())}
                  for k, v in [("sequence_tokens_before_cutoff", before),
                               ("sequence_tokens_after_cutoff", after),
                               ("supervised_label_tokens", labels)]},
        "label_fraction_of_sequence": float(labels.sum() / after.sum()),
        "n_truncated_by_cutoff": int((before > args.cutoff_len).sum()),
        "why": ("supersedes the naive '\\n'.join(contents) length used by select_randk_lenmatch.py, which "
                "does NOT apply the llama2 template despite its docstring. Length matching must bucket on "
                "the length actually trained on."),
        "no_compute_run": "artifacts only: no model loaded, no gradients, no SFT",
    }
    json.dump(man, open(args.manifest, "w"), indent=2)
    print(f"\ncandidates            : {n}")
    for k in man["stats"]:
        s = man["stats"][k]
        print(f"  {k:32s} mean {s['mean']:8.1f}  median {s['median']:7.0f}  total {s['total']:,}")
    print(f"label fraction of seq : {man['label_fraction_of_sequence']:.4f}")
    print(f"truncated by cutoff   : {man['n_truncated_by_cutoff']:,}")
    print(f"cache sha256          : {man['sha256']}")
    print(f"wrote {args.out}\nwrote {args.manifest}")


if __name__ == "__main__":
    main()

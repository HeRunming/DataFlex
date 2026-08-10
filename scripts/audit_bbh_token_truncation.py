#!/usr/bin/env python3
"""EXECUTION-LEVEL tokenization / truncation audit (code_review_0810 P0-2). Artifacts only.
No model is loaded, no gradients, no selection, no SFT, no accuracy is computed.

WHY STRING PARITY WAS NOT ENOUGH
--------------------------------
`audit_bbh_prompt_parity.py` compares the query prompt to the lm-eval prompt as STRINGS, i.e. BEFORE
tokenization. But the two sides are executed under different length regimes:

  query-gradient side : LlamaFactory `template: llama2`, `cutoff_len = 2048`
                        -> `<s>[INST] {ctx} [/INST] {target}</s>`, then truncated to 2048
  evaluation side     : pinned lm-eval `generate_until`, `max_gen_toks = 1024` inside Llama-2's
                        4096-token context -> ~3072 tokens of context, LEFT-truncated by lm-eval

So gate B can pass byte-for-byte while the sequence the gradient is actually taken on has been cut.
Worse, BBH prompts are structured as [3 CoT demonstrations] ++ [the actual query LAST], and
LlamaFactory truncates the source TAIL (`source_ids[:source_len]`, allocation by `infer_seqlen`).
Cutting the tail therefore removes THE QUERY ITSELF, keeping only the demonstrations.

This script measures that directly with the REAL installed tokenizer and the REAL LlamaFactory
allocation function, and it FAILS LOUD. It deliberately does NOT change `cutoff_len`, the few-shot
count, or the prompts: per the review, if anything is materially truncated we stop and report the
distribution so the protocol correction is made once, explicitly, by a human.
"""
import argparse, glob, hashlib, json, os, warnings

warnings.filterwarnings("ignore")

ROOT = "/jizhicfs/karonhe/DataFlex_fa"
PROMPTS_DIR = f"{ROOT}/data/bbh_external/query_prompts"
BASE_MODEL = "/jizhicfs/karonhe/models/shakechen/Llama-2-7b-hf"
CUTOFF_LEN = 2048          # frozen SFT/query-gradient recipe
EVAL_CTX = 4096            # Llama-2 native context
MAX_GEN_TOKS = 1024        # pinned bbh_cot_fewshot generation reservation
CUE = "Let's think step by step."


def sha(s):
    return hashlib.sha256(s.encode()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=f"{ROOT}/experiments/less_aligned/bbh_token_truncation_audit.json")
    ap.add_argument("--cutoff_len", type=int, default=CUTOFF_LEN)
    ap.add_argument("--expect_records", type=int, default=192,
                    help="required record count; auditing fewer is a FAILURE, not a pass. Generalizes "
                         "the gate-B vacuity fix: an audit that inspected 3 records must never read as "
                         "clean to the launch manifest.")
    args = ap.parse_args()

    from transformers import AutoTokenizer, __version__ as tfv
    from llamafactory.data.processor.processor_utils import infer_seqlen
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)

    rows, truncated = [], []
    for p in sorted(glob.glob(f"{PROMPTS_DIR}/*_prompts.jsonl")):
        draw = os.path.basename(p).split("_")[2]
        for line in open(p):
            if not line.strip():
                continue
            r = json.loads(line)
            ctx, tgt = r["messages"][0]["content"], r["messages"][1]["content"]

            # ---- query-gradient side: exactly what LlamaFactory llama2 + cutoff_len produces ----
            src_ids = tok(f"[INST] {ctx} [/INST]", add_special_tokens=True)["input_ids"]
            tgt_ids = tok(tgt, add_special_tokens=False)["input_ids"] + [tok.eos_token_id]
            keep_src, keep_tgt = infer_seqlen(len(src_ids), len(tgt_ids), args.cutoff_len)
            kept_text = tok.decode(src_ids[:keep_src])

            # The ACTUAL query is the final "Q: ..." block; the demos precede it. Use a POSITIONAL
            # test, not a substring search: an 80-char probe also occurs inside the demo region for 48
            # of these records (shared task boilerplate), so `probe in kept_text` is a false-pass
            # mechanism waiting to fire. Instead require that the kept prefix extends past where the
            # final query block begins, and that the query's own tail survives verbatim.
            q_start_char = ctx.rfind("\nQ: ")
            query_block = ctx[q_start_char + 1:] if q_start_char >= 0 else ctx
            n_src_full = len(src_ids)
            # chars of ctx retained, measured by re-decoding the kept prefix
            kept_ctx = kept_text.rsplit("[/INST]", 1)[0]
            query_start_present = (keep_src == n_src_full) or (
                q_start_char >= 0 and len(kept_ctx) > q_start_char + len("\nQ: ") + 1
                and query_block.rstrip()[-40:] in kept_ctx)
            # The prompt must still carry the trailing CoT cue for the continuation to be well-posed.
            # NB: the llama2 wrapper appends " [/INST]" AFTER the cue, so we test that the cue is the
            # last thing before that closing tag -- not that the decoded string ends with the cue.
            head = kept_text.rsplit("[/INST]", 1)[0].rstrip()
            ends_with_cue = head.endswith(CUE)
            target_intact = keep_tgt == len(tgt_ids)

            # ---- evaluation side: lm-eval reserves max_gen_toks inside the context window ----
            eval_ids = tok(ctx, add_special_tokens=False)["input_ids"]
            eval_budget = EVAL_CTX - MAX_GEN_TOKS
            eval_trunc = max(0, len(eval_ids) - eval_budget)

            rec = {
                "id": r["id"], "draw": draw,
                "prompt_chars": len(ctx),
                "grad_side": {
                    "source_tokens": len(src_ids), "target_tokens": len(tgt_ids),
                    "source_kept": keep_src, "target_kept": keep_tgt,
                    "source_tokens_dropped": len(src_ids) - keep_src,
                    "target_tokens_dropped": len(tgt_ids) - keep_tgt,
                    "exceeds_cutoff": len(src_ids) + len(tgt_ids) > args.cutoff_len,
                    "query_start_survives": query_start_present,
                    "ends_with_cot_cue": ends_with_cue,
                    "target_intact": target_intact,
                },
                "eval_side": {
                    "context_tokens": len(eval_ids), "budget": eval_budget,
                    "tokens_left_truncated": eval_trunc,
                    "fits": eval_trunc == 0,
                },
                "materially_truncated": not (query_start_present and ends_with_cue and target_intact),
            }
            rows.append(rec)
            if rec["materially_truncated"]:
                truncated.append(rec)

    n = len(rows)
    if n != args.expect_records:
        raise SystemExit(f"VACUITY GUARD: audited {n} records, expected {args.expect_records}. "
                         f"Refusing to emit a verdict — run render_bbh_query_prompts.py first.")
    gmax = max(r["grad_side"]["source_tokens"] + r["grad_side"]["target_tokens"] for r in rows)
    emax = max(r["eval_side"]["context_tokens"] for r in rows)
    by_subtask = {}
    for r in truncated:
        st = r["id"].split("::")[1]
        by_subtask[st] = by_subtask.get(st, 0) + 1

    verdict = "PASS" if not truncated else "HOLD"
    out = {
        "audit": "execution-level tokenization / truncation over all query records",
        "verdict": verdict,
        "n_records": n,
        "n_materially_truncated": len(truncated),
        "env": {"tokenizer": BASE_MODEL, "transformers": tfv,
                "llamafactory_allocator": "llamafactory.data.processor.processor_utils.infer_seqlen"},
        "grad_side_regime": {"template": "llama2", "wrapper": "<s>[INST] {ctx} [/INST] {target}</s>",
                            "cutoff_len": args.cutoff_len,
                            "truncation_direction": "source TAIL (source_ids[:source_len])",
                            "max_observed_total_tokens": gmax},
        "eval_side_regime": {"context_window": EVAL_CTX, "max_gen_toks": MAX_GEN_TOKS,
                            "effective_context_budget": EVAL_CTX - MAX_GEN_TOKS,
                            "truncation_direction": "LEFT-truncated by lm-eval",
                            "max_observed_context_tokens": emax,
                            "all_fit": all(r["eval_side"]["fits"] for r in rows)},
        "truncated_by_subtask": by_subtask,
        "why_string_parity_missed_this": (
            "gate B compares prompts BEFORE tokenization. BBH prompts are [3 CoT demos] ++ [query LAST] "
            "and LlamaFactory truncates the source TAIL, so an over-length record loses the query itself "
            "while the byte-for-byte string comparison still passes."),
        "records": rows,
    }
    if truncated:
        out["HOLD_reason"] = (
            f"{len(truncated)}/{n} query records are materially truncated at cutoff_len="
            f"{args.cutoff_len}. In each case the tail of the record's OWN query is dropped and the "
            f"trailing \"A: {CUE}\" cue is cut, so the query gradient would be taken on "
            f"[demonstrations + a partial query] with no CoT cue. Per code_review_0810 this is a HOLD: "
            f"cutoff_len / few-shot count / prompts are NOT altered automatically.")
    json.dump(out, open(args.out, "w"), indent=2)

    print(f"records                     : {n}")
    print(f"grad side  max total tokens : {gmax}  (cutoff_len={args.cutoff_len})")
    print(f"eval side  max ctx tokens   : {emax}  (budget={EVAL_CTX - MAX_GEN_TOKS}, "
          f"all fit={out['eval_side_regime']['all_fit']})")
    print(f"materially truncated        : {len(truncated)}/{n}")
    for r in truncated:
        g = r["grad_side"]
        print(f"   {r['id']:40s} src {g['source_tokens']:5d} -> {g['source_kept']:5d} "
              f"(lost {g['source_tokens_dropped']:4d})  query_survives={g['query_start_survives']} "
              f"cue={g['ends_with_cot_cue']} target_ok={g['target_intact']}")
    if by_subtask:
        print(f"   concentrated in: {by_subtask}")
    print(f"\nVERDICT: {verdict}")
    if truncated:
        print(out["HOLD_reason"])
    print(f"wrote {args.out}")
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())

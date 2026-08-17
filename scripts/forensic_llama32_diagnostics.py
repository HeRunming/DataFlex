#!/usr/bin/env python3
"""The three pre-registered Llama-3.2 diagnostics that were frozen but not yet reported (advice_0817).

EVALUATION ONLY. No training, no selection, no protocol change. Uses the shared no-SFT base plus the
24 existing adapters. This is pre-registered closure, not a new experiment.

  D2  operational WRAPPED query CE  -- the surrogate the targeting pipeline actually optimizes.
      This is the third condition of pre-registered Outcome A, which my earlier write-up claimed had
      fired without computing it.
  D2b same-query 64-item CoT exact-match -- fixes the EXAMPLES, so a pure query->held-out shift
      cannot explain a drop here.
  D2c bare-context final-answer CE -- serialization sensitivity ONLY. Per the Llama-2 D2c result it
      may NOT be promoted to a primary criterion whatever it shows.

Definitions are the frozen Llama-2 ones. The one deliberate difference: the wrapper is this stack's
own `llama3` template, obtained from LlamaFactory's OWN template encoder rather than hand-written, so
"operational" means what this pipeline actually fed the model. Hand-copying the llama2 `[INST]` form
would measure a serialization this stack never used.

Reporting follows the prereg: the two SFT seeds are averaged WITHIN a draw, and the draw (n=3) is the
unit. No significance is claimed.
"""
import argparse, json, os
import statistics as st

ROOT = "/jizhicfs/karonhe/DataFlex_fa"
EXP = f"{ROOT}/experiments/less_aligned"
SAVES = "/jizhicfs/karonhe/dataflex_saves"
BASE = "/jizhicfs/karonhe/models/modelscope/LLM-Research/Llama-3___2-3B"
METHODS = ["dsmc", "first_rr", "second_rr", "randk"]
DRAWS = [0, 1, 2]
SEEDS = [42, 1]
CUTOFF = 3072          # the frozen target-gradient extraction cutoff


def qfile(d):
    return f"{ROOT}/data/bbh_external/query_prompts/bbh_query_draw{d}_prompts.jsonl"


def rows(d):
    return [json.loads(l) for l in open(qfile(d)) if l.strip()]   # readlines, never splitlines


def build_wrapped(tok, tmpl, d):
    """Operational encoding: LlamaFactory's own llama3 template, prompt masked, cutoff 3072."""
    from llamafactory.data.processor.processor_utils import infer_seqlen
    out = []
    for r in rows(d):
        ctx, tgt = r["messages"][0]["content"], r["messages"][1]["content"]
        src, tg = tmpl.encode_oneturn(
            tok, [{"role": "user", "content": ctx}, {"role": "assistant", "content": tgt}])
        ks, kt = infer_seqlen(len(src), len(tg), CUTOFF)
        out.append((r["id"], src[:ks] + tg[:kt], [-100] * ks + tg[:kt]))
    return out


def build_bare(tok, d):
    """Bare pinned lm-eval context + gold final answer, NO chat wrapper, delimiter ' '."""
    out = []
    for r in rows(d):
        ctx, tgt = r["messages"][0]["content"], r["messages"][1]["content"]
        src = tok(ctx, add_special_tokens=True)["input_ids"]
        tg = tok(" " + tgt, add_special_tokens=False)["input_ids"]
        out.append((r["id"], src + tg, [-100] * len(src) + tg))
    return out


def ce(model, batches, dev):
    import torch
    tot = 0.0
    for _i, ids, labels in batches:
        x = torch.tensor([ids], device=dev)
        y = torch.tensor([labels], device=dev)
        tot += float(model(input_ids=x, labels=y).loss)
    return tot / len(batches)


def cot_em(model, tok, d, dev):
    """Greedy CoT generation on the bare pinned prompt, scored like the frozen BBH suite: take the
    text after the final 'the answer is' and compare to the gold final answer."""
    import re, torch
    hit = 0
    n = 0
    for r in rows(d):
        ctx = r["messages"][0]["content"]
        gold = r["messages"][1]["content"].strip()
        ids = tok(ctx, add_special_tokens=True, return_tensors="pt").to(dev)
        with torch.no_grad():
            o = model.generate(**ids, max_new_tokens=512, do_sample=False,
                               pad_token_id=tok.eos_token_id)
        gen = tok.decode(o[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)
        gen = gen.split("\n\nQ:")[0]                       # stop at the next question, as lm-eval does
        m = re.findall(r"[Tt]he answer is\s*(.*?)\s*(?:\.|\n|$)", gen)
        pred = m[-1].strip() if m else gen.strip().split("\n")[0].strip()
        hit += int(pred == gold)
        n += 1
    return hit / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=f"{EXP}/results_summary/llama32_diagnostics.json")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    stt = json.load(open(f"{EXP}/llama32_full_run_state.json"))
    if sum(1 for c in stt["cells"] if c.get("evaluated")) != 24 or not stt.get("base_eval"):
        raise SystemExit("refusing to run: the Llama-3.2 arm is not fully evaluated")

    import warnings
    warnings.filterwarnings("ignore")
    import torch
    from transformers import AutoTokenizer
    from llamafactory.data.template import TEMPLATES
    from peft import PeftModel

    dev = a.device
    tok = AutoTokenizer.from_pretrained(BASE)
    tmpl = TEMPLATES["llama3"]
    wrapped = {d: build_wrapped(tok, tmpl, d) for d in DRAWS}
    bare = {d: build_bare(tok, d) for d in DRAWS}
    print(f"[diag] {[len(wrapped[d]) for d in DRAWS]} queries/draw; "
          f"max wrapped {max(len(b[1]) for d in DRAWS for b in wrapped[d])} tokens, "
          f"max bare {max(len(b[1]) for d in DRAWS for b in bare[d])} tokens", flush=True)

    from transformers import AutoModelForCausalLM
    base = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.bfloat16, device_map=dev)
    base.eval()

    rep = {"arm": "Llama-3.2-3B", "scope": "evaluation only; pre-registered closure",
           "definitions": {
               "wrapped_query_CE": ("operational surrogate: LlamaFactory's OWN llama3 template "
                                    "(this stack's actual serialization), prompt masked, cutoff 3072"),
               "query_cot_EM": "greedy CoT on the bare pinned lm-eval prompt, same 64 items",
               "bare_CE": ("gold final-answer CE on the bare pinned context, no chat wrapper -- "
                           "SERIALIZATION SENSITIVITY ONLY, may not be promoted to primary")},
           "statistical_unit": "draw (n=3); the two SFT seeds averaged within draw first",
           "base": {}, "cells": {}}

    with torch.no_grad():
        for d in DRAWS:
            rep["base"][str(d)] = {"wrapped_ce": ce(base, wrapped[d], dev),
                                   "bare_ce": ce(base, bare[d], dev)}
            rep["base"][str(d)]["cot_em"] = cot_em(base, tok, d, dev)
            b = rep["base"][str(d)]
            print(f"[diag] base draw{d}: wrapped {b['wrapped_ce']:.4f}  bare {b['bare_ce']:.4f}  "
                  f"cotEM {b['cot_em']:.4f}", flush=True)

    for c in stt["cells"]:
        aid, d = c["adapter_id"], c["draw"]
        model = PeftModel.from_pretrained(base, c["sft_out"], torch_dtype=torch.bfloat16)
        model.eval()
        with torch.no_grad():
            w = ce(model, wrapped[d], dev)
            br = ce(model, bare[d], dev)
            em = cot_em(model, tok, d, dev)
        model = model.unload()
        bb = rep["base"][str(d)]
        rep["cells"][aid] = {"draw": d, "method": c["method"], "seed": c["train_seed"],
                             "wrapped_ce": w, "d_wrapped_ce": w - bb["wrapped_ce"],
                             "bare_ce": br, "d_bare_ce": br - bb["bare_ce"],
                             "cot_em": em, "d_cot_em": em - bb["cot_em"]}
        r = rep["cells"][aid]
        print(f"[diag] {aid:28s} dWrapped {r['d_wrapped_ce']:+.4f}  dBare {r['d_bare_ce']:+.4f}  "
              f"dCoT {r['d_cot_em']:+.4f}", flush=True)
        json.dump(rep, open(a.out, "w"), indent=2)

    summ = {}
    for m in METHODS:
        per = {k: [] for k in ("d_wrapped_ce", "d_bare_ce", "d_cot_em")}
        for d in DRAWS:                                   # seeds averaged WITHIN draw first
            for k in per:
                per[k].append(st.mean([rep["cells"][f"l32_draw{d}_{m}_seed{s}"][k]
                                       for s in SEEDS]))
        summ[m] = {f"{k}_per_draw": per[k] for k in per}
        summ[m].update({f"{k}_mean": st.mean(per[k]) for k in per})
    rep["method_summary"] = summ

    tgt = ["dsmc", "first_rr", "second_rr"]
    rep["VERDICT"] = {
        "targeted_reduce_wrapped_ce": all(summ[m]["d_wrapped_ce_mean"] < 0 for m in tgt),
        "random_increases_wrapped_ce": summ["randk"]["d_wrapped_ce_mean"] > 0,
        "targeted_reduce_cot_em": all(summ[m]["d_cot_em_mean"] < 0 for m in tgt),
        "no_method_improves_bare_ce": all(summ[m]["d_bare_ce_mean"] > 0 for m in METHODS),
    }
    v = rep["VERDICT"]
    rep["OUTCOME_A_SURROGATE_CONDITION"] = {
        "condition": "the operational query surrogate improves for the targeted methods",
        "met": v["targeted_reduce_wrapped_ce"],
        "reading": (
            "The third pre-registered condition of Outcome A IS met: the targeted selectors move the "
            "operational surrogate toward the target while downstream utility falls. Full Outcome A "
            "replicates across two model stacks."
            if v["targeted_reduce_wrapped_ce"] else
            "The third pre-registered condition of Outcome A is NOT met on this stack: the targeted "
            "selectors do not reduce the operational wrapped query CE. The geometry->utility "
            "dissociation still replicates, but the surrogate-improvement level is model-stack "
            "dependent, and the paper must NOT claim a full cross-stack double dissociation.")}
    json.dump(rep, open(a.out, "w"), indent=2)

    print(f"\n{'method':12s} {'d wrapped CE':>13s} {'d bare CE':>11s} {'d CoT EM':>10s}")
    for m in METHODS:
        s_ = summ[m]
        print(f"{m:12s} {s_['d_wrapped_ce_mean']:+13.4f} {s_['d_bare_ce_mean']:+11.4f} "
              f"{s_['d_cot_em_mean']:+10.4f}")
    print(f"\noutcome-A surrogate condition met: {v['targeted_reduce_wrapped_ce']}")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()

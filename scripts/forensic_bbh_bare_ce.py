#!/usr/bin/env python3
"""FINAL forensic check (advice_0814_3): does the surrogate/task dissociation survive removal of the
chat-wrapper confound? Evaluation only — forward passes over existing adapters, no training.

The gap this closes
-------------------
D2b fixed the *examples* (same 64 query items) but not the *executable input*:

    operational target CE : LlamaFactory llama2 template -> `<s>[INST] ctx [/INST] answer</s>`
    CoT exact-match       : pinned lm-eval generation    -> bare `ctx`, no wrapper

So a reviewer could argue the dissociation is an input-serialization artifact rather than a property of
cross-entropy vs the task metric. This computes a THIRD quantity on the same items:

    L_Q^bare : teacher-forced CE of the gold final answer, conditioned on the BARE pinned lm-eval prompt
               context -- no [INST] wrapper, no CoT generated, same gold answer, same token-level loss

Outcomes, decided before running:
  * targeted methods still reduce BARE-context CE while query CoT EM falls
        -> the dissociation survives removal of the chat-wrapper confound
  * bare-context CE does NOT improve
        -> scope the claim to the exact operational targeting surrogate, NOT to cross-entropy generically

Either way this is the LAST forensic diagnostic; no protocol, method or result changes.
"""
import argparse, glob, json, os
import statistics as st

ROOT = "/jizhicfs/karonhe/DataFlex_fa"
EXP = f"{ROOT}/experiments/less_aligned"
SAVES = "/jizhicfs/karonhe/dataflex_saves"
BASE = "/jizhicfs/karonhe/models/shakechen/Llama-2-7b-hf"
METHODS = ["dsmc", "second_rr", "first_rr", "less", "randk", "randk_seqlabelmatch"]
DRAWS = [0, 1, 2]


def build(tok, draw):
    """BARE lm-eval context + gold final answer. Deliberately NO [INST] wrapper and no truncation:
    max prompt is 2,581 tokens, well inside the 4,096 context, so nothing is cut."""
    out = []
    p = f"{ROOT}/data/bbh_external/query_prompts/bbh_query_draw{draw}_prompts.jsonl"
    for line in open(p):
        if not line.strip():
            continue
        r = json.loads(line)
        ctx, tgt = r["messages"][0]["content"], r["messages"][1]["content"]
        # lm-eval concatenates context + continuation with target_delimiter " "
        src = tok(ctx, add_special_tokens=True)["input_ids"]
        tg = tok(" " + tgt, add_special_tokens=False)["input_ids"]
        out.append((r["id"], src + tg, [-100] * len(src) + tg))
    return out


def ce(model, batches, device):
    import torch
    tot = 0.0
    for _id, ids, labels in batches:
        x = torch.tensor([ids], device=device)
        y = torch.tensor([labels], device=device)
        tot += float(model(input_ids=x, labels=y).loss)
    return tot / len(batches)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=f"{EXP}/results_summary/bbh_forensic_bare_ce.json")
    args = ap.parse_args()

    import warnings
    warnings.filterwarnings("ignore")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    dev = "cuda"
    tok = AutoTokenizer.from_pretrained(BASE)
    batches = {d: build(tok, d) for d in DRAWS}
    print(f"[bare] {[len(batches[d]) for d in DRAWS]} query items per draw; "
          f"max prompt tokens {max(len(b[1]) for d in DRAWS for b in batches[d])}", flush=True)

    base = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.bfloat16, device_map=dev)
    base.eval()
    prev = json.load(open(f"{EXP}/results_summary/bbh_forensic_query_loss.json"))
    cot = json.load(open(f"{EXP}/results_summary/bbh_forensic_query_cot.json"))

    rep = {"diagnostic": "bare-context final-answer CE (no [INST] wrapper) on the same 64 query items",
           "why": ("D2b fixed the examples but not the executable input: the operational target CE uses "
                   "the LlamaFactory llama2 wrapper while CoT generation uses the bare lm-eval context. "
                   "This removes that confound."),
           "definition": ("teacher-forced CE of the gold final answer conditioned on the BARE pinned "
                          "lm-eval prompt context, target_delimiter ' ', no wrapper, no truncation"),
           "base_bare_ce": {}, "cells": {}}
    with torch.no_grad():
        for d in DRAWS:
            rep["base_bare_ce"][str(d)] = ce(base, batches[d], dev)
            print(f"[bare] base draw{d}: {rep['base_bare_ce'][str(d)]:.4f}", flush=True)

    plan = json.load(open(f"{EXP}/bbh_external_run_plan.json"))
    for c in plan["cells"]:
        aid, d = c["adapter_id"], c["draw"]
        model = PeftModel.from_pretrained(base, c["sft_out"], torch_dtype=torch.bfloat16)
        model.eval()
        with torch.no_grad():
            L = ce(model, batches[d], dev)
        model = model.unload()
        rep["cells"][aid] = {"draw": d, "method": c["method"], "seed": c["train_seed"],
                             "bare_ce": L, "delta_vs_base": L - rep["base_bare_ce"][str(d)]}
        print(f"[bare] {aid:38s} {L:.4f}  d={L - rep['base_bare_ce'][str(d)]:+.4f}", flush=True)

    summ = {}
    for m in METHODS:
        dl = [st.mean([rep["cells"][f"bbhx_draw{d}_{m}_seed{s}"]["delta_vs_base"] for s in (42, 1)])
              for d in DRAWS]
        summ[m] = {
            "delta_bare_ce_per_draw": dl,
            "delta_bare_ce_mean": st.mean(dl),
            "delta_wrapped_ce_mean": prev["method_summary"][m]["delta_query_loss_mean"],
            "delta_query_cot_em": cot["methods"][m]["delta_query_cot_em_vs_base"],
            "delta_heldout_em": cot["methods"][m]["delta_heldout_em_vs_base"],
        }
    rep["method_summary"] = summ
    tgt = ["dsmc", "second_rr", "first_rr", "less"]
    rnd = ["randk", "randk_seqlabelmatch"]
    survives = (all(summ[m]["delta_bare_ce_mean"] < 0 for m in tgt)
                and all(summ[m]["delta_query_cot_em"] < 0 for m in tgt))
    rep["VERDICT"] = {
        "targeted_reduce_bare_ce": all(summ[m]["delta_bare_ce_mean"] < 0 for m in tgt),
        "random_increase_bare_ce": all(summ[m]["delta_bare_ce_mean"] > 0 for m in rnd),
        "dissociation_survives_wrapper_removal": survives,
        "READING": (
            "The dissociation SURVIVES removal of the chat-wrapper confound: with the bare pinned lm-eval "
            "context -- the exact serialization the task metric uses -- the target-aware selectors still "
            "reduce final-answer CE while their CoT exact-match on those same items falls. The mismatch "
            "is therefore not an artifact of the [INST] serialization."
            if survives else
            "Bare-context CE does NOT improve for the target-aware methods. The claim must be scoped to "
            "the exact operational targeting surrogate (wrapped final-answer CE) rather than to "
            "cross-entropy generically."),
        "scope_note": ("This is the LAST forensic diagnostic. No protocol, method, or result changes "
                       "follow from it."),
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(rep, open(args.out, "w"), indent=2)
    print(f"\n{'method':22s} {'d bare CE':>10s} {'d wrapped CE':>13s} {'d qCoT EM':>10s} {'d heldout':>10s}")
    for m in METHODS:
        v = summ[m]
        print(f"{m:22s} {v['delta_bare_ce_mean']:+10.4f} {v['delta_wrapped_ce_mean']:+13.4f} "
              f"{v['delta_query_cot_em']:+10.4f} {v['delta_heldout_em']:+10.4f}")
    print(f"\ndissociation survives wrapper removal: {survives}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

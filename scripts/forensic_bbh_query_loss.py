#!/usr/bin/env python3
"""Is the selection surrogate misaligned with the downstream objective? (advice_0814 diagnostic 2)
Read-only over frozen artifacts: loads existing adapters for a forward pass only. No training.

The BBH query gradients supervise the BARE FINAL ANSWER -- `(C)`, `14`, `Yes` -- because BBH ships no
gold rationale per test item. But the evaluation requires GENERATING a chain of thought and only then the
answer. So there are two different objectives, and this measures both on the same 64 queries:

    L_Q(theta) = mean token CE of the final answer under the exact same supervision used for
                 target-gradient extraction (llama2 template, cutoff 3072, prompt masked)
    held-out exact_match = what the paper actually reports

If the target-aware selectors IMPROVE L_Q most while scoring WORST on held-out generation, the failure is
not mysterious: the surrogate the selection optimizes is not the objective the task rewards.

DIAGNOSTIC ONLY. No model, method, or protocol decision may depend on this output.
"""
import argparse, glob, json, os

ROOT = "/jizhicfs/karonhe/DataFlex_fa"
EXP = f"{ROOT}/experiments/less_aligned"
SAVES = "/jizhicfs/karonhe/dataflex_saves"
BASE = "/jizhicfs/karonhe/models/shakechen/Llama-2-7b-hf"
CUTOFF = 3072                     # the target-gradient extraction cutoff, not the SFT 2048
METHODS = ["dsmc", "second_rr", "first_rr", "less", "randk", "randk_seqlabelmatch"]


def build_batches(tok, draw):
    """Exactly the extraction-time encoding: <s>[INST] ctx [/INST] target</s>, prompt masked."""
    from llamafactory.data.processor.processor_utils import infer_seqlen
    out = []
    p = f"{ROOT}/data/bbh_external/query_prompts/bbh_query_draw{draw}_prompts.jsonl"
    for line in open(p):
        if not line.strip():
            continue
        r = json.loads(line)
        ctx, tgt = r["messages"][0]["content"], r["messages"][1]["content"]
        src = tok(f"[INST] {ctx} [/INST]", add_special_tokens=True)["input_ids"]
        tg = tok(tgt, add_special_tokens=False)["input_ids"] + [tok.eos_token_id]
        ks, kt = infer_seqlen(len(src), len(tg), CUTOFF)
        ids = src[:ks] + tg[:kt]
        labels = [-100] * ks + tg[:kt]
        out.append((r["id"], ids, labels))
    return out


def query_loss(model, batches, device):
    """Mean per-token CE over the supervised (final-answer) positions, macro-averaged over queries."""
    import torch
    tot, per = 0.0, []
    for _id, ids, labels in batches:
        x = torch.tensor([ids], device=device)
        y = torch.tensor([labels], device=device)
        out = model(input_ids=x, labels=y)
        per.append(float(out.loss))
        tot += float(out.loss)
    return tot / len(batches), per


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=f"{EXP}/results_summary/bbh_forensic_query_loss.json")
    args = ap.parse_args()

    import warnings
    warnings.filterwarnings("ignore")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    dev = "cuda"
    tok = AutoTokenizer.from_pretrained(BASE)
    batches = {d: build_batches(tok, d) for d in (0, 1, 2)}
    print(f"[qloss] built {sum(len(v) for v in batches.values())} query examples "
          f"({[len(batches[d]) for d in (0,1,2)]} per draw)", flush=True)

    plan = json.load(open(f"{EXP}/bbh_external_run_plan.json"))
    acc = {}
    for c in plan["cells"]:
        r = json.load(open(sorted(glob.glob(f"{c['eval_out']}/*/results_*.json"))[-1]))
        acc[(c["draw"], c["method"], c["train_seed"])] = \
            r["results"]["bbh_external_heldout"]["exact_match,get-answer"]

    print("[qloss] loading base model ...", flush=True)
    base = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.bfloat16, device_map=dev)
    base.eval()
    rep = {"diagnostic": "query final-answer loss vs held-out CoT exact-match",
           "supervision": ("identical to target-gradient extraction: llama2 template, cutoff 3072, "
                           "prompt masked, loss on the bare final BBH answer only"),
           "why": ("BBH targets are single tokens like (C)/14/Yes while evaluation requires generating a "
                   "chain of thought and then the answer -- two different objectives"),
           "DIAGNOSTIC_ONLY": "no model, method, or protocol decision may depend on this",
           "base_query_loss": {}, "cells": {}}

    with torch.no_grad():
        for d in (0, 1, 2):
            L, _ = query_loss(base, batches[d], dev)
            rep["base_query_loss"][str(d)] = L
            print(f"[qloss] base draw{d}: L_Q = {L:.4f}", flush=True)

    for c in plan["cells"]:
        aid, d, m, s = c["adapter_id"], c["draw"], c["method"], c["train_seed"]
        model = PeftModel.from_pretrained(base, c["sft_out"], torch_dtype=torch.bfloat16)
        model.eval()
        with torch.no_grad():
            L, _ = query_loss(model, batches[d], dev)
        model = model.unload()                       # restore the shared base in place
        b = rep["base_query_loss"][str(d)]
        rep["cells"][aid] = {"draw": d, "method": m, "seed": s,
                             "query_loss": L, "delta_vs_base": L - b,
                             "heldout_exact_match": acc[(d, m, s)]}
        print(f"[qloss] {aid:38s} L_Q={L:.4f}  dL={L-b:+.4f}  EM={acc[(d,m,s)]:.4f}", flush=True)

    # method-level summary: seed-averaged within draw, then averaged over draws
    summ = {}
    for m in METHODS:
        dl = [sum(rep["cells"][f"bbhx_draw{d}_{m}_seed{s}"]["delta_vs_base"] for s in (42, 1)) / 2
              for d in (0, 1, 2)]
        em = [sum(rep["cells"][f"bbhx_draw{d}_{m}_seed{s}"]["heldout_exact_match"] for s in (42, 1)) / 2
              for d in (0, 1, 2)]
        summ[m] = {"delta_query_loss_per_draw": dl, "delta_query_loss_mean": sum(dl) / 3,
                   "heldout_em_per_draw": em, "heldout_em_mean": sum(em) / 3}
    rep["method_summary"] = summ
    by_dl = sorted(METHODS, key=lambda m: summ[m]["delta_query_loss_mean"])       # most improved first
    by_em = sorted(METHODS, key=lambda m: -summ[m]["heldout_em_mean"])            # best accuracy first
    rep["ranking_by_query_loss_improvement"] = by_dl
    rep["ranking_by_heldout_accuracy"] = by_em
    rep["INTERPRETATION"] = (
        "If the ordering by query-loss improvement is roughly the REVERSE of the ordering by held-out "
        "accuracy, the selection surrogate (final-answer CE on the query set) is misaligned with the "
        "downstream objective (generate a CoT, then answer). That would explain the negative transfer "
        "without appealing to anything mysterious about gradient matching.")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(rep, open(args.out, "w"), indent=2)
    print(f"\n{'method':22s} {'dL_Q':>9s} {'held-out EM':>12s}")
    for m in by_dl:
        print(f"{m:22s} {summ[m]['delta_query_loss_mean']:+9.4f} {summ[m]['heldout_em_mean']:12.4f}")
    print(f"\nby query-loss improvement : {by_dl}")
    print(f"by held-out accuracy      : {by_em}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

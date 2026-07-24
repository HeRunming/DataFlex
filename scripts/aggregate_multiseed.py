#!/usr/bin/env python3
"""
Aggregate multi-seed eval results into a mean±std comparison table.

Scans eval_results/{bbh,mmlu,tydiqa}/{method}_{target}_seed{s} (or the single-
seed naming without _seed for seed 42 legacy) and produces per-(method,target)
mean ± std over seeds, plus a markdown/json/csv table with error bars.

Metrics: BBH = bbh_cot_fewshot exact_match (macro over 27 subtasks); MMLU =
mmlu acc,none (5-shot); TyDiQA = macro_f1 (from eval_tydiqa.py json).
"""
import argparse
import glob
import json
import os
import statistics
from collections import defaultdict

EVAL = "/jizhicfs/karonhe/dataflex_saves/eval_results"

METHODS = [
    "less_sgd", "mmd_grad_rbf_sgd", "mmd_grad_cov_sgd",
    "less_adam", "mmd_grad_rbf_adam", "mmd_grad_cov_adam",
    "mmd_emb_rbf", "mmd_emb_rbf_stochastic", "tsds", "nice",
]
TARGETS = ["bbh", "mmlu", "tydiqa"]


def read_bbh(path_dir):
    """path_dir contains a nested results_*.json (lm_eval)."""
    files = glob.glob(os.path.join(path_dir, "**", "results_*.json"), recursive=True)
    if not files:
        return None
    r = json.load(open(files[0]))["results"]
    subs = [v.get("exact_match,get-answer", v.get("exact_match,none"))
            for k, v in r.items() if k.startswith("bbh_cot_fewshot_") and isinstance(v, dict)]
    subs = [s for s in subs if s is not None]
    if subs:
        return sum(subs) / len(subs)
    agg = r.get("bbh_cot_fewshot", {})
    return agg.get("exact_match,get-answer") or agg.get("exact_match,none")


def read_mmlu(path_dir):
    files = glob.glob(os.path.join(path_dir, "**", "results_*.json"), recursive=True)
    if not files:
        return None
    r = json.load(open(files[0]))["results"]
    return r.get("mmlu", {}).get("acc,none")


def read_tydiqa(path_json):
    if not os.path.exists(path_json):
        return None
    return json.load(open(path_json)).get("macro_f1")


def collect(method, target, seeds):
    """Return list of scores across seeds for (method, target)."""
    scores = []
    for s in seeds:
        # naming: seed 42 may be legacy without suffix; try both
        candidates = [f"{method}_{target}_seed{s}", f"{method}_{target}"] if s == 42 else [f"{method}_{target}_seed{s}"]
        # nice uses method name == target-specific dir nice_{target}
        if method == "nice":
            candidates = [f"nice_{target}_seed{s}"] + ([f"nice_{target}"] if s == 42 else [])
        val = None
        for name in candidates:
            if target == "bbh":
                val = read_bbh(os.path.join(EVAL, "bbh", name))
            elif target == "mmlu":
                val = read_mmlu(os.path.join(EVAL, "mmlu", name))
            else:
                val = read_tydiqa(os.path.join(EVAL, "tydiqa", f"{name}.json"))
            if val is not None:
                break
        if val is not None:
            scores.append(val)
    return scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 1, 2])
    ap.add_argument("--out", default="/jizhicfs/karonhe/DataFlex_fa/experiments/less_aligned/results_summary/multiseed_summary")
    args = ap.parse_args()

    table = {}  # method -> target -> (mean, std, n, raw)
    for m in METHODS:
        table[m] = {}
        for t in TARGETS:
            sc = collect(m, t, args.seeds)
            if sc:
                mean = sum(sc) / len(sc)
                std = statistics.stdev(sc) if len(sc) > 1 else 0.0
                table[m][t] = {"mean": mean, "std": std, "n": len(sc), "raw": sc}
            else:
                table[m][t] = None

    # markdown
    lines = ["# Multi-seed comparison (mean ± std over seeds {})".format(args.seeds), ""]
    lines.append("| Method | BBH | MMLU | TyDiQA-F1 |")
    lines.append("|---|---|---|---|")
    for m in METHODS:
        row = [m]
        for t in TARGETS:
            c = table[m][t]
            row.append(f"{c['mean']:.4f} ± {c['std']:.4f} (n={c['n']})" if c else "—")
        lines.append("| " + " | ".join(row) + " |")
    md = "\n".join(lines)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    open(args.out + ".md", "w").write(md + "\n")
    json.dump(table, open(args.out + ".json", "w"), indent=2)
    # csv
    with open(args.out + ".csv", "w") as f:
        f.write("method,target,mean,std,n,raw\n")
        for m in METHODS:
            for t in TARGETS:
                c = table[m][t]
                if c:
                    f.write(f"{m},{t},{c['mean']:.5f},{c['std']:.5f},{c['n']},\"{c['raw']}\"\n")
    print(md)
    print(f"\n[aggregate] saved {args.out}.{{md,json,csv}}")


if __name__ == "__main__":
    main()

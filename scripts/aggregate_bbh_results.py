#!/usr/bin/env python3
"""Aggregate BBH CoT few-shot results across all methods."""
import json
import glob
import os
from pathlib import Path

EVAL_DIR = "/jizhicfs/karonhe/dataflex_saves/eval_results/bbh"

METHODS = [
    "base_model",
    "random_selected",
    "less_sgd_selected",
    "mmd_grad_rbf_sgd_selected",
    "mmd_grad_cov_sgd_selected",
    "mmd_emb_rbf_selected",
    "less_adam_selected",
    "mmd_grad_rbf_adam_selected",
    "mmd_grad_cov_adam_selected",
]

def load_results(method):
    pattern = f"{EVAL_DIR}/{method}/**/results_*.json"
    files = sorted(glob.glob(pattern, recursive=True))
    if not files:
        return None
    with open(files[-1]) as f:
        return json.load(f)

def get_metric(task_data):
    """Extract primary metric (exact_match or acc) from a task entry."""
    for k in ("exact_match,get-answer", "exact_match,none", "acc,none", "acc_norm,none"):
        if k in task_data:
            return task_data[k]
    return None

print("=" * 90)
print("BBH CoT Few-shot (3-shot) — LESS-aligned setting, Llama-2-7B + LoRA")
print("=" * 90)

all_data = {}
for m in METHODS:
    r = load_results(m)
    if r is None:
        print(f"[MISSING] {m}")
        continue
    all_data[m] = r

# Build subtask comparison table
results = {m: d.get("results", {}) for m, d in all_data.items()}

# Aggregate (macro avg)
print(f"\n{'Method':<35}  {'BBH Avg (CoT 3-shot)':>22}")
print("-" * 60)
agg_scores = {}
for m in METHODS:
    if m not in results:
        continue
    # bbh_cot_fewshot is the group; the group-level metric is sometimes there.
    if "bbh_cot_fewshot" in results[m]:
        agg = get_metric(results[m]["bbh_cot_fewshot"])
        if agg is not None:
            agg_scores[m] = agg
            print(f"{m:<35}  {agg:>22.4f}")
            continue
    # Fallback: macro-average across subtasks
    subtask_scores = []
    for k, v in results[m].items():
        if k.startswith("bbh_cot_fewshot_"):
            s = get_metric(v)
            if s is not None:
                subtask_scores.append(s)
    if subtask_scores:
        macro = sum(subtask_scores) / len(subtask_scores)
        agg_scores[m] = macro
        print(f"{m:<35}  {macro:>22.4f}  (macro avg of {len(subtask_scores)} subtasks)")

# Per-subtask breakdown
print("\n" + "=" * 90)
print("Per-subtask breakdown:")
print("=" * 90)

# Get all subtask names
all_subtasks = set()
for m in METHODS:
    if m in results:
        for k in results[m].keys():
            if k.startswith("bbh_cot_fewshot_"):
                all_subtasks.add(k)
all_subtasks = sorted(all_subtasks)

# Header
hdr = f"{'Subtask':<55}"
for m in METHODS:
    label = m.replace("_selected", "").replace("_sgd", "").replace("base_model", "base")[:10]
    hdr += f"  {label:>10}"
print(hdr)
print("-" * len(hdr))

for st in all_subtasks:
    short = st.replace("bbh_cot_fewshot_", "")
    line = f"{short:<55}"
    for m in METHODS:
        s = get_metric(results.get(m, {}).get(st, {}))
        if s is not None:
            line += f"  {s:>10.4f}"
        else:
            line += f"  {'N/A':>10}"
    print(line)

# Summary CSV
csv_path = os.path.join(EVAL_DIR, "bbh_summary.csv")
with open(csv_path, "w") as f:
    f.write("method," + ",".join(s.replace("bbh_cot_fewshot_","") for s in all_subtasks) + ",macro_avg\n")
    for m in METHODS:
        if m not in results: continue
        scores = []
        for st in all_subtasks:
            s = get_metric(results[m].get(st, {}))
            scores.append(f"{s:.4f}" if s is not None else "")
        avg = agg_scores.get(m, "")
        avg_str = f"{avg:.4f}" if avg != "" else ""
        f.write(f"{m}," + ",".join(scores) + f",{avg_str}\n")
print(f"\nCSV saved to: {csv_path}")

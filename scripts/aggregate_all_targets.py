#!/usr/bin/env python3
"""
Aggregate all three-target results (BBH / MMLU / TyDiQA) into a single
summary (CSV + JSON + Markdown).

Reads eval outputs from /jizhicfs/karonhe/dataflex_saves/eval_results/{bbh,mmlu,tydiqa}
and writes to experiments/less_aligned/results_summary/.

Metrics:
  - BBH:    macro-avg exact_match over 27 CoT few-shot (3-shot) subtasks
  - MMLU:   acc (5-shot), aggregate over 57 subjects
  - TyDiQA: macro-F1 over languages (also reports macro-EM)
"""
import json
import glob
import os
import csv

SAVES = "/jizhicfs/karonhe/dataflex_saves/eval_results"
OUT_DIR = os.path.join(os.path.dirname(__file__), os.pardir,
                       "experiments", "less_aligned", "results_summary")
OUT_DIR = os.path.abspath(OUT_DIR)
os.makedirs(OUT_DIR, exist_ok=True)

# (display_name, dataset_suffix) — order matters for table
METHODS = [
    "less_sgd",
    "less_adam",
    "mmd_grad_rbf_sgd",
    "mmd_grad_rbf_adam",
    "mmd_grad_cov_sgd",
    "mmd_grad_cov_adam",
    "mmd_emb_rbf",
    "mmd_emb_rbf_stochastic",
]


def get_bbh(method):
    """Macro-avg exact_match over BBH CoT few-shot subtasks."""
    fs = glob.glob(f"{SAVES}/bbh/{method}_selected/**/results_*.json", recursive=True)
    if not fs:
        return None
    results = json.load(open(sorted(fs)[-1]))["results"]
    scores = []
    for k, v in results.items():
        if k.startswith("bbh_cot_fewshot_"):
            for mk in ("exact_match,get-answer", "exact_match,none", "acc,none"):
                if mk in v:
                    scores.append(v[mk])
                    break
    return sum(scores) / len(scores) if scores else None


def get_mmlu(method):
    fs = glob.glob(f"{SAVES}/mmlu/{method}_mmlu_selected/**/results_*.json", recursive=True)
    if not fs:
        return None
    r = json.load(open(sorted(fs)[-1]))["results"]
    return r.get("mmlu", {}).get("acc,none")


def get_tydiqa(method):
    f = f"{SAVES}/tydiqa/{method}_tydiqa_selected.json"
    if not os.path.exists(f):
        return None, None
    d = json.load(open(f))
    return d.get("macro_f1"), d.get("macro_em")


def main():
    rows = []
    for m in METHODS:
        bbh = get_bbh(m)
        mmlu = get_mmlu(m)
        tyf1, tyem = get_tydiqa(m)
        rows.append({
            "method": m,
            "bbh_cot_3shot": bbh,
            "mmlu_5shot_acc": mmlu,
            "tydiqa_macro_f1": tyf1,
            "tydiqa_macro_em": tyem,
        })

    # base model (BBH only — no target-specific SFT)
    base_bbh = get_bbh("base_model")
    if base_bbh is not None:
        rows.insert(0, {
            "method": "base_model (no SFT)",
            "bbh_cot_3shot": base_bbh,
            "mmlu_5shot_acc": None,
            "tydiqa_macro_f1": None,
            "tydiqa_macro_em": None,
        })

    # ---- JSON ----
    json_path = os.path.join(OUT_DIR, "three_target_summary.json")
    with open(json_path, "w") as f:
        json.dump(rows, f, indent=2)

    # ---- CSV ----
    csv_path = os.path.join(OUT_DIR, "three_target_summary.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["method", "bbh_cot_3shot", "mmlu_5shot_acc",
                                          "tydiqa_macro_f1", "tydiqa_macro_em"])
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else "") for k, v in r.items()}
                       | {"method": r["method"]})

    # ---- Markdown ----
    md_path = os.path.join(OUT_DIR, "three_target_summary.md")
    with open(md_path, "w") as f:
        f.write("# Three-Target Data Selection Results\n\n")
        f.write("**Setup:** Llama-2-7B + LoRA (r=128, α=512), 5% selection from 270k LESS pool, "
                "4-epoch SFT.\n\n")
        f.write("**Metrics:** BBH = macro-avg exact_match (CoT 3-shot, 27 subtasks); "
                "MMLU = acc (5-shot, 57 subjects); "
                "TyDiQA = macro-F1 over languages (macro-EM in parens).\n\n")
        f.write("| Method | BBH | MMLU | TyDiQA-F1 | TyDiQA-EM |\n")
        f.write("|---|---|---|---|---|\n")
        for r in rows:
            def fmt(v):
                return f"{v:.4f}" if isinstance(v, float) else "—"
            f.write(f"| {r['method']} | {fmt(r['bbh_cot_3shot'])} | {fmt(r['mmlu_5shot_acc'])} "
                    f"| {fmt(r['tydiqa_macro_f1'])} | {fmt(r['tydiqa_macro_em'])} |\n")
        f.write("\n## Per-target best method\n\n")
        for col, label in [("bbh_cot_3shot", "BBH"), ("mmlu_5shot_acc", "MMLU"),
                           ("tydiqa_macro_f1", "TyDiQA-F1")]:
            cand = [(r["method"], r[col]) for r in rows
                    if isinstance(r[col], float) and "base_model" not in r["method"]]
            if cand:
                best = max(cand, key=lambda x: x[1])
                f.write(f"- **{label}**: {best[0]} ({best[1]:.4f})\n")

    print(f"Wrote:\n  {json_path}\n  {csv_path}\n  {md_path}")
    print("\n=== Summary ===")
    for r in rows:
        def fmt(v):
            return f"{v:.4f}" if isinstance(v, float) else "  --  "
        print(f"{r['method']:<26} BBH={fmt(r['bbh_cot_3shot'])} "
              f"MMLU={fmt(r['mmlu_5shot_acc'])} TyDiQA-F1={fmt(r['tydiqa_macro_f1'])}")


if __name__ == "__main__":
    main()

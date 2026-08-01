#!/usr/bin/env python3
"""Aggregate the target-draw pilot eval results into a per-cell table (advice_0731 / code_review_0801).
Expands the 30 unique adapters back to 32 (method x draw) cells (Random-K shared adapter appears in
both same-index directions), computes balanced + target-weighted accuracy, and the FROZEN-convention
paired difference DSMC - method within each draw. Reads only the eval_manifest-pinned result file.
Descriptive only (2 draws/direction in the pilot; no significance). Requires 32/32 cells unless
--allow-partial (canary)."""
import argparse, json, glob, os

W_MAJ = 51.0 / 64.0    # exact target-weighted majority weight (rho = 51/64)
W_MIN = 13.0 / 64.0


def acc(r, k):
    for kk in r:
        if kk == k or kk.endswith(k):
            return r[kk].get("acc,none", r[kk].get("acc"))
    return None


def scores(saves, aid):
    # prefer the eval_manifest-pinned authoritative result; fall back to a single results file
    base = f"{saves}/eval_results/skew/pilot_{aid}"
    mf = f"{base}/eval_manifest.json"
    path = None
    if os.path.exists(mf):
        path = json.load(open(mf)).get("result_path")
    if not path or not os.path.exists(path):
        fs = glob.glob(f"{base}/**/results_*.json", recursive=True)
        if len(fs) != 1:
            return None
        path = fs[0]
    r = json.load(open(path))["results"]
    return acc(r, "mmlu_stem"), acc(r, "mmlu_humanities")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", required=True)
    ap.add_argument("--saves", default="/jizhicfs/karonhe/dataflex_saves")
    ap.add_argument("--out", default=None)
    ap.add_argument("--allow-partial", action="store_true",
                    help="canary mode: permit <32 cells with results (full pilot requires 32/32)")
    args = ap.parse_args()
    plan = json.load(open(args.plan))
    rows = []
    for c in plan["cells"]:
        sc = scores(args.saves, c["adapter_id"])
        row = dict(c)
        if sc and sc[0] is not None and sc[1] is not None:
            stem, hum = sc
            row["stem"], row["hum"] = stem, hum
            row["balanced"] = (stem + hum) / 2
            maj_is_stem = c["direction"] == "stem80"
            row["target_weighted"] = W_MAJ * (stem if maj_is_stem else hum) + W_MIN * (hum if maj_is_stem else stem)
        else:
            row["stem"] = row["hum"] = row["balanced"] = row["target_weighted"] = None
        rows.append(row)
    by_draw = {}
    for r in rows:
        by_draw.setdefault(r["draw"], {})[r["method"]] = r
    out = args.out or f"{args.saves}/eval_results/skew/pilot_aggregate.csv"
    with open(out, "w") as f:
        # FROZEN convention: paired difference = DSMC - method
        f.write("draw,direction,method,adapter_id,shared,stem,hum,balanced,target_weighted,"
                "dsmc_minus_method_balanced,dsmc_minus_method_target_weighted\n")
        for r in rows:
            dsmc = by_draw[r["draw"]].get("dsmc")
            dbal = dtw = ""
            if r["balanced"] is not None and dsmc and dsmc["balanced"] is not None:
                dbal = f"{dsmc['balanced'] - r['balanced']:.4f}"
                dtw = f"{dsmc['target_weighted'] - r['target_weighted']:.4f}"
            def fmt(x):
                return "" if x is None else f"{x:.4f}"
            nl = "\n"
            f.write(f"{r['draw']},{r['direction']},{r['method']},{r['adapter_id']},"
                    f"{int(r['shared_adapter'])},{fmt(r['stem'])},{fmt(r['hum'])},"
                    f"{fmt(r['balanced'])},{fmt(r['target_weighted'])},{dbal},{dtw}{nl}")
    done = sum(1 for r in rows if r["stem"] is not None)
    print(f"wrote {out}  (cells with results: {done}/32)")
    if done < 32 and not args.allow_partial:
        raise SystemExit(f"[FATAL] full pilot requires 32/32 cells, have {done}. Use --allow-partial for canary.")


if __name__ == "__main__":
    main()

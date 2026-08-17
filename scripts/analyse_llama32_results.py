#!/usr/bin/env python3
"""Pre-registered analysis of the Llama-3.2-3B second model-stack arm.

Run ONLY after 24/24 adapter evals + 1/1 base eval. Everything here was fixed in
`prereg_second_model.md` before any training.

Statistical unit, per the prereg and code_review_0815
----------------------------------------------------
The query/selection DRAW is the primary unit (n=3). The two SFT seeds are averaged WITHIN a draw
first. The 24 adapter cells are NOT 24 independent replicates, and n=6 must never be reported as
six independent draws: the seeds share a draw AND a selected subset, and exist only to expose SFT
stochasticity.

Every method is reported as a delta against this model stack's OWN no-SFT reference; cross-model
absolute accuracy is not comparable.
"""
import argparse, json, statistics as st

ROOT = "/jizhicfs/karonhe/DataFlex_fa"
EXP = f"{ROOT}/experiments/less_aligned"
METHODS = ["dsmc", "first_rr", "second_rr", "randk"]
DRAWS = [0, 1, 2]
SEEDS = [42, 1]


def spearman(x, y):
    def rank(v):
        s = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(s):
            j = i
            while j + 1 < len(s) and v[s[j + 1]] == v[s[i]]:
                j += 1
            avg = (i + j) / 2 + 1
            for k in range(i, j + 1):
                r[s[k]] = avg
            i = j + 1
        return r
    rx, ry = rank(x), rank(y)
    n = len(x)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return num / den if den else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=f"{EXP}/results_summary/llama32_results.json")
    a = ap.parse_args()

    stt = json.load(open(f"{EXP}/llama32_full_run_state.json"))
    ne = sum(1 for c in stt["cells"] if c.get("evaluated"))
    if ne != 24 or not stt.get("base_eval"):
        raise SystemExit(f"refusing to unseal: {ne}/24 adapter evals, base_eval={stt.get('base_eval')}")
    if not all(c["n_subtasks"] == 27 and c["n_examples"] == 5209
               for c in stt["cells"] if c.get("evaluated")):
        raise SystemExit("refusing to unseal: a cell is not 27 subtasks / 5209 examples")

    base = stt["base_cell"]["_micro_sealed"]
    cell = {c["adapter_id"]: c["_micro_sealed"] for c in stt["cells"]}

    rep = {"arm": "Llama-3.2-3B second model-stack confirmation",
           "primary_outcome": "held-out BBH micro exact_match, frozen 5209-example split",
           "statistical_unit": ("the query/selection DRAW (n=3); the two SFT seeds are averaged "
                                "WITHIN a draw first. The 24 cells are NOT 24 independent "
                                "replicates and n=6 is never reported as six draws."),
           "base_noSFT_micro": base, "cells": cell, "per_draw": {}, "methods": {}}

    # seeds averaged within draw -> the draw-level values that carry the inference
    draw_means = {m: [] for m in METHODS}
    for d in DRAWS:
        row = {}
        for m in METHODS:
            vals = [cell[f"l32_draw{d}_{m}_seed{s}"] for s in SEEDS]
            mu = st.mean(vals)
            draw_means[m].append(mu)
            row[m] = {"seed_values": vals, "seed_spread": round(max(vals) - min(vals), 6),
                      "draw_mean": mu, "delta_vs_base": mu - base}
        best = min(row, key=lambda m: -row[m]["draw_mean"])
        row["best_method_this_draw"] = best
        row["dsmc_beats_randk"] = row["dsmc"]["draw_mean"] > row["randk"]["draw_mean"]
        rep["per_draw"][str(d)] = row

    for m in METHODS:
        dm = draw_means[m]
        rep["methods"][m] = {
            "per_draw_means": dm,
            "mean_over_draws": st.mean(dm),
            "delta_vs_base": st.mean(dm) - base,
            "sd_over_draws": st.stdev(dm) if len(dm) > 1 else 0.0,
            "n_draws": len(dm)}

    # the central pre-registered contrast
    diffs = [draw_means["dsmc"][i] - draw_means["randk"][i] for i in range(len(DRAWS))]
    rep["dsmc_vs_randk"] = {
        "per_draw_diff": diffs,
        "mean_diff": st.mean(diffs),
        "mean_diff_pp": round(st.mean(diffs) * 100, 3),
        "draw_blocks_favouring_dsmc": sum(1 for x in diffs if x > 0),
        "n_draw_blocks": len(diffs)}

    order = sorted(METHODS, key=lambda m: -rep["methods"][m]["mean_over_draws"])
    rep["ranking_best_to_worst"] = order
    rep["all_below_base"] = all(rep["methods"][m]["delta_vs_base"] < 0 for m in METHODS)

    # geometry association, using the same draw-level accuracies (D2 is computed separately)
    d2p = f"{EXP}/results_summary/llama32_forensic_geometry.json"
    try:
        g = json.load(open(d2p))
        rep["geometry"] = {}
        for d in DRAWS:
            gd = g["per_draw"][str(d)]
            D2 = {m: gd["methods"][m]["D2_to_Q"] for m in METHODS}
            acc = {m: draw_means[m][d] for m in METHODS}
            ms = sorted(METHODS)
            # cross-check: the geometry script's own seed-averaged accuracy must match the value
            # computed here from the sealed state, or the two views have drifted apart
            for m in METHODS:
                if abs(gd["methods"][m]["acc_seed_avg"] - acc[m]) > 1e-9:
                    raise SystemExit(f"draw{d}/{m}: geometry acc {gd['methods'][m]['acc_seed_avg']} "
                                     f"!= analysis acc {acc[m]}")
            rep["geometry"][str(d)] = {
                "D2": D2, "spearman_D2_vs_acc": spearman([D2[m] for m in ms], [acc[m] for m in ms]),
                "dsmc_has_lowest_D2": min(D2, key=D2.get) == "dsmc",
                "best_acc_method": max(acc, key=acc.get)}
    except FileNotFoundError:
        rep["geometry"] = "not computed yet (run the D2 diagnostic separately)"

    json.dump(rep, open(a.out, "w"), indent=2)

    print(f"no-SFT base micro EM: {base:.6f}\n")
    print(f"{'method':12s} {'draw0':>9s} {'draw1':>9s} {'draw2':>9s} {'mean':>9s} {'d vs base':>10s}")
    for m in order:
        v = rep["methods"][m]
        print(f"{m:12s} " + " ".join(f"{x:9.4f}" for x in v["per_draw_means"]) +
              f" {v['mean_over_draws']:9.4f} {v['delta_vs_base']:+10.4f}")
    dv = rep["dsmc_vs_randk"]
    print(f"\nDSMC - Random per draw: {[round(x,4) for x in dv['per_draw_diff']]}")
    print(f"DSMC - Random mean    : {dv['mean_diff_pp']:+.3f} pp "
          f"({dv['draw_blocks_favouring_dsmc']}/{dv['n_draw_blocks']} draw blocks favour DSMC)")
    print(f"all methods below base: {rep['all_below_base']}")
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Two zero-cost forensic robustness checks (advice_0808_2).

(A) LEAVE-ONE-DRAW-OUT reweighted P*: in the first analysis P* was the union of all 10 draws, so each
    draw's own query Q_d was 10% of its own reference — mechanically helping "closer to Q_d" look like
    "closer to P*". Here, for each draw d we build
        P*_{-d} = 1/2 * P_{STEM,-d} + 1/2 * P_{HUM,-d}
    i.e. exclude draw d, then reweight the two domains to exactly 50/50, and recompute D2(S_d, P*_{-d}).

(B) DESCRIPTIVE Spearman correlations across existing (method x draw) points between downstream
    balanced accuracy and (a) D2-to-P*, (b) source entropy — separately at 1% and 5%.
    Diagnostics only: 80 cells are NOT independent, so no significance claims.
"""
import argparse, json, os, csv
import numpy as np
import torch

SAVES = "/jizhicfs/karonhe/dataflex_saves"
ROOT = "/jizhicfs/karonhe/DataFlex_fa"
STEM_DRAWS = [f"stem80_draw{i}" for i in range(5)]
HUM_DRAWS = [f"hum80_draw{i}" for i in range(5)]
DRAWS = STEM_DRAWS + HUM_DRAWS
METHODS = ["dsmc", "randk", "randk_lenmatch", "second_rr", "less", "gist", "nice"]
CAND_GRAD = f"{SAVES}/less_output/train/1/all_projected_grads.pt"


def unit(X):
    return X / X.norm(dim=1, keepdim=True).clamp_min(1e-12)


def spearman(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    den = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / den) if den > 0 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=f"{ROOT}/experiments/less_aligned/results_summary/forensic_robustness.json")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    X = unit(torch.load(CAND_GRAD, map_location="cpu").float().to(dev))

    # per-draw target grads (unit)
    T = {d: unit(torch.load(f"{SAVES}/draw_{d}_output/target/1/all_projected_grads.pt",
                            map_location="cpu").float().to(dev)) for d in DRAWS}
    # Each draw is 51 majority + 13 minority. For a domain-pure pool we use, per draw, the whole
    # target set of that direction; leave-one-out at the DRAW level then reweight domains 50/50 by
    # averaging the STEM-majority draws' second moment with the HUM-majority draws'.
    def M_of(rows):
        U = torch.cat(rows, 0)
        return (U.T @ U) / U.shape[0]

    res = {"leave_one_draw_out": {}, "spearman": {}}
    # (A) LOO reweighted P*_{-d}
    for d in DRAWS:
        stem_rows = [T[x] for x in STEM_DRAWS if x != d]
        hum_rows = [T[x] for x in HUM_DRAWS if x != d]
        M_loo = 0.5 * M_of(stem_rows) + 0.5 * M_of(hum_rows)   # 50/50 domain reweight
        entry = {}
        for m in METHODS:
            for budget, prefix in [("1pct", "sel1pct"), ("5pct", "sel")]:
                p = f"{SAVES}/{prefix}_{d}_{m}/step_1.json"
                if not os.path.exists(p):
                    continue
                idx = torch.tensor(json.load(open(p))["indices"], device=dev)
                U = X.index_select(0, idx)
                M_S = (U.T @ U) / U.shape[0]
                entry.setdefault(budget, {})[m] = float(((M_S - M_loo) ** 2).sum())
        res["leave_one_draw_out"][d] = entry

    print("=== (A) D2 to LEAVE-ONE-DRAW-OUT, 50/50-reweighted P*_{-d} ===")
    for budget in ["1pct", "5pct"]:
        print(f"  [{budget}]")
        for m in METHODS:
            vals = [res["leave_one_draw_out"][d][budget][m] for d in DRAWS
                    if budget in res["leave_one_draw_out"][d] and m in res["leave_one_draw_out"][d][budget]]
            if vals:
                print(f"    {m:16s} mean {np.mean(vals):.5f}")
        # does DSMC still beat Random on every draw?
        wins = sum(1 for d in DRAWS
                   if res["leave_one_draw_out"][d][budget]["dsmc"] < res["leave_one_draw_out"][d][budget]["randk"])
        print(f"    -> DSMC closer to P*_-d than Random in {wins}/10 draws")
        res["spearman"].setdefault(budget, {})["dsmc_closer_than_randk_draws"] = wins

    # (B) descriptive Spearman: downstream vs D2->P* and vs source entropy
    geo = json.load(open(f"{ROOT}/experiments/less_aligned/results_summary/forensic_pstar_geometry.json"))
    agg = {"1pct": f"{SAVES}/eval_results/skew/pilot1pct_aggregate.csv",
           "5pct": f"{SAVES}/eval_results/skew/pilot_aggregate.csv"}
    print("\n=== (B) descriptive Spearman across (method x draw) points ===")
    for budget in ["1pct", "5pct"]:
        rows = list(csv.DictReader(open(agg[budget])))
        acc = {(r["draw"], r["method"]): float(r["balanced"]) for r in rows if r["balanced"]}
        d2, ent, bal = [], [], []
        for m in METHODS:
            gm = geo["budgets"][budget].get(m)
            if not gm:
                continue
            for pr in gm["per_draw"]:
                k = (pr["draw"], m)
                if k in acc:
                    d2.append(pr["D2_to_Pstar"]); ent.append(pr["source_entropy"]); bal.append(acc[k])
        r_d2 = spearman(d2, bal); r_ent = spearman(ent, bal)
        res["spearman"][budget].update({"n_points": len(bal),
                                        "spearman_D2toPstar_vs_balanced": r_d2,
                                        "spearman_source_entropy_vs_balanced": r_ent})
        print(f"  [{budget}] n={len(bal)}  rho(D2->P*, balanced) = {r_d2:+.3f}   "
              f"rho(source_entropy, balanced) = {r_ent:+.3f}")
        print(f"           (note: lower D2 is 'better' by the objective, so a POSITIVE rho here means "
              f"lower D2 -> LOWER accuracy)")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(res, open(args.out, "w"), indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()

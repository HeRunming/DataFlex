#!/usr/bin/env python3
"""Derive the 1% (K=2707) selections as nested PREFIXES of the frozen 5% (K=13533) selections
(decision_0806). For every draw and every deterministic method (dsmc, less, first_rr, second_rr,
gist, nice) AND Random-K, the 1% subset = the first 2707 entries of the frozen 5% step_1.json index
ordering (greedy/RR append order or topk score-descending order; Random-K = same seeded permutation
prefix). This guarantees the cross-budget comparison is paired (budget differs only by subset size,
not by a new selection realization). randk_lenmatch is NOT prefixable (its 1% must match the 1% DSMC
length histogram) — it is rebuilt separately at K=2707 by the driver, not here.

Writes 1% caches to sel1pct_<draw>_<method>/step_1.json with the K=1% budget recorded in metric.
"""
import argparse, json, os

SAVES = "/jizhicfs/karonhe/dataflex_saves"
DRAWS = ["stem80_draw0", "stem80_draw1", "stem80_draw2", "stem80_draw3", "stem80_draw4",
         "hum80_draw0", "hum80_draw1", "hum80_draw2", "hum80_draw3", "hum80_draw4"]
PREFIX_METHODS = ["dsmc", "less", "first_rr", "second_rr", "gist", "nice"]  # + randk handled below
K1 = 2707


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k1", type=int, default=K1)
    args = ap.parse_args()
    n = 0
    for d in DRAWS:
        for m in PREFIX_METHODS + ["randk"]:
            src = f"{SAVES}/sel_{d}_{m}/step_1.json"
            if not os.path.exists(src):
                raise FileNotFoundError(src)
            full = json.load(open(src))
            idx5 = full["indices"]
            if len(idx5) < args.k1:
                raise ValueError(f"{src}: only {len(idx5)} < {args.k1}")
            idx1 = idx5[:args.k1]
            assert len(idx1) == args.k1 and len(set(idx1)) == args.k1, f"{d}/{m}: bad prefix"
            out = f"{SAVES}/sel1pct_{d}_{m}"
            os.makedirs(out, exist_ok=True)
            meta = dict(full.get("metric", {}))
            meta.update({"derived_from": os.path.abspath(src), "selection_budget": args.k1,
                         "budget_pct": 1, "nested_prefix_of_5pct": True, "num_select": args.k1})
            json.dump({"indices": idx1, "metric": meta}, open(f"{out}/step_1.json", "w"))
            n += 1
    print(f"derived {n} 1% prefix selections (K={args.k1}) for {len(DRAWS)} draws "
          f"x {len(PREFIX_METHODS)+1} methods (randk_lenmatch rebuilt separately)")


if __name__ == "__main__":
    main()

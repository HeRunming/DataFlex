#!/usr/bin/env python3
"""
Build the explicit target-draw pilot RUN PLAN: 32 (method x draw) cells -> 30 unique SFT adapters
(Random-K reused across the two directions that share a draw index). Emits a JSON plan the SFT/eval
driver consumes, and a human-readable table. review_0731 / advice_0731.

Pilot draws (2 dir x 2 draw): stem80_draw0, stem80_draw1, hum80_draw0, hum80_draw1.
Methods (8): dsmc, less, first_rr, second_rr, gist, nice, randk, randk_lenmatch.
Random-K is target-INDEPENDENT: for a given draw index it depends only on (subset seed 2000+idx,
train seed) -> the STEM and HUM draws of the same index reuse ONE adapter. All others are
draw-specific. => 7*4 + 2 = 30 unique adapters, but the aggregation table keeps all 32 cells.
Per-draw train seed: draw0->42, draw1->1 (from each draw's frozen meta).
"""
import argparse, json, os, hashlib

# Full 5%-primary experiment: 5 draws per direction (choice_0803). draws 0,1 = original pilot.
DRAWS = ["stem80_draw0", "stem80_draw1", "stem80_draw2", "stem80_draw3", "stem80_draw4",
         "hum80_draw0", "hum80_draw1", "hum80_draw2", "hum80_draw3", "hum80_draw4"]
METHODS = ["dsmc", "less", "first_rr", "second_rr", "gist", "nice", "randk", "randk_lenmatch"]
SAVES = "/jizhicfs/karonhe/dataflex_saves"
ROOT = "/jizhicfs/karonhe/DataFlex_fa"
# expected counts: 7 draw-specific x 10 draws + 5 shared Random-K (one per draw index 0..4) = 75
N_CELLS_EXPECT = len(DRAWS) * len(METHODS)          # 80
N_ADAPTERS_EXPECT = 7 * len(DRAWS) + 5              # 75


def fsha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=f"{ROOT}/experiments/less_aligned/pilot_run_plan.json")
    ap.add_argument("--tag", default="pilot", help="namespace tag/prefix for adapter+eval dirs (pilot=5%, pilot1pct=1%)")
    ap.add_argument("--budget", type=int, default=13533, help="selection budget K (5%=13533, 1%=2707)")
    ap.add_argument("--subset_tmpl", default=f"{SAVES}/sft_subsets/{{draw}}_{{m}}_sel.jsonl",
                    help="subset jsonl template (5%); 1% uses ..._1pct_sel.jsonl")
    ap.add_argument("--randk_subset_tmpl", default=f"{SAVES}/sft_subsets/{{draw}}_randk_sel.jsonl")
    args = ap.parse_args()
    TAG = args.tag
    cells = []           # method-draw cells (aggregation rows)
    adapters = {}        # adapter_id -> {dataset_key, train_seed, subset_jsonl, subset_sha256}
    randk_hash_by_idx = {}   # idx -> {draw: sha} to assert shared subsets are byte-identical
    for draw in DRAWS:
        meta = json.load(open(f"{ROOT}/data/target_draws/{draw}.meta.json"))
        seed = meta["train_seed"]; idx = draw.split("draw")[-1]
        for m in METHODS:
            if m == "randk":
                # shared across same-draw-index directions -> one adapter id keyed by draw index
                adapter_id = f"{TAG}_randk_drawidx{idx}_seed{seed}"
                dataset_key = f"{TAG}_randk_drawidx{idx}_sel"    # identical subset across directions
                subset = args.randk_subset_tmpl.format(draw=draw)
            else:
                adapter_id = f"{TAG}_{draw}_{m}_seed{seed}"
                dataset_key = f"{TAG}_{draw}_{m}_sel"
                subset = args.subset_tmpl.format(draw=draw, m=m)
            subset_sha = fsha(subset)
            if m == "randk":
                randk_hash_by_idx.setdefault(idx, {})[draw] = subset_sha
            cells.append({"draw": draw, "direction": meta["direction"], "method": m,
                          "train_seed": seed, "adapter_id": adapter_id,
                          "dataset_key": dataset_key, "budget": args.budget,
                          "sft_out": f"{SAVES}/sft_results/{adapter_id}",
                          "eval_out": f"{SAVES}/eval_results/skew/{adapter_id}",
                          "shared_adapter": (m == "randk")})
            if adapter_id not in adapters:
                adapters[adapter_id] = {"dataset_key": dataset_key, "train_seed": seed,
                                        "subset_jsonl": subset, "subset_sha256": subset_sha,
                                        "method": m, "budget": args.budget}
    # shared Random-K: assert the two directional subsets at a draw index are byte-identical
    for idx, hs in randk_hash_by_idx.items():
        uniq = set(hs.values())
        assert len(uniq) == 1, f"Random-K drawidx{idx} subsets differ across directions: {hs}"
    plan = {"draws": DRAWS, "methods": METHODS, "tag": TAG, "budget": args.budget,
            "n_cells": len(cells), "n_unique_adapters": len(adapters),
            "cells": cells, "adapters": adapters}
    json.dump(plan, open(args.out, "w"), indent=2)
    print(f"[{TAG} K={args.budget}] cells={len(cells)} (expect {N_CELLS_EXPECT})  unique_adapters={len(adapters)} (expect {N_ADAPTERS_EXPECT})")
    assert len(cells) == N_CELLS_EXPECT and len(adapters) == N_ADAPTERS_EXPECT, "run plan cell/adapter count mismatch"
    # human table
    print(f"\n{'draw':16s} {'method':16s} {'seed':4s} {'shared':6s} adapter_id")
    for c in cells:
        print(f"{c['draw']:16s} {c['method']:16s} {c['train_seed']:<4d} "
              f"{'Y' if c['shared_adapter'] else '':6s} {c['adapter_id']}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()

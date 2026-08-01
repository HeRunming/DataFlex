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

DRAWS = ["stem80_draw0", "stem80_draw1", "hum80_draw0", "hum80_draw1"]
METHODS = ["dsmc", "less", "first_rr", "second_rr", "gist", "nice", "randk", "randk_lenmatch"]
SAVES = "/jizhicfs/karonhe/dataflex_saves"
ROOT = "/jizhicfs/karonhe/DataFlex_fa"


def fsha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=f"{ROOT}/experiments/less_aligned/pilot_run_plan.json")
    args = ap.parse_args()
    cells = []           # 32 method-draw cells (aggregation rows)
    adapters = {}        # adapter_id -> {dataset_key, train_seed, subset_jsonl, subset_sha256}
    randk_hash_by_idx = {}   # idx -> {draw: sha} to assert shared subsets are byte-identical
    for draw in DRAWS:
        meta = json.load(open(f"{ROOT}/data/target_draws/{draw}.meta.json"))
        seed = meta["train_seed"]; idx = draw.split("draw")[-1]
        for m in METHODS:
            if m == "randk":
                # shared across same-draw-index directions -> one adapter id keyed by draw index
                adapter_id = f"randk_drawidx{idx}_seed{seed}"
                dataset_key = f"randk_drawidx{idx}_sel"          # points at whichever dir built it (identical subset)
                subset = f"{SAVES}/sft_subsets/{draw}_randk_sel.jsonl"
            else:
                adapter_id = f"{draw}_{m}_seed{seed}"
                dataset_key = f"{draw}_{m}_sel"
                subset = f"{SAVES}/sft_subsets/{draw}_{m}_sel.jsonl"
            subset_sha = fsha(subset)
            if m == "randk":
                randk_hash_by_idx.setdefault(idx, {})[draw] = subset_sha
            cells.append({"draw": draw, "direction": meta["direction"], "method": m,
                          "train_seed": seed, "adapter_id": adapter_id,
                          "dataset_key": dataset_key,
                          "sft_out": f"{SAVES}/sft_results/pilot_{adapter_id}",
                          "eval_out": f"{SAVES}/eval_results/skew/pilot_{adapter_id}",
                          "shared_adapter": (m == "randk")})
            if adapter_id not in adapters:
                adapters[adapter_id] = {"dataset_key": dataset_key, "train_seed": seed,
                                        "subset_jsonl": subset, "subset_sha256": subset_sha,
                                        "method": m}
    # shared Random-K: assert the two directional subsets at a draw index are byte-identical
    for idx, hs in randk_hash_by_idx.items():
        uniq = set(hs.values())
        assert len(uniq) == 1, f"Random-K drawidx{idx} subsets differ across directions: {hs}"
    plan = {"draws": DRAWS, "methods": METHODS, "n_cells": len(cells),
            "n_unique_adapters": len(adapters), "cells": cells, "adapters": adapters}
    json.dump(plan, open(args.out, "w"), indent=2)
    print(f"cells={len(cells)} (expect 32)  unique_adapters={len(adapters)} (expect 30)")
    assert len(cells) == 32 and len(adapters) == 30, "run plan cell/adapter count mismatch"
    # human table
    print(f"\n{'draw':16s} {'method':16s} {'seed':4s} {'shared':6s} adapter_id")
    for c in cells:
        print(f"{c['draw']:16s} {c['method']:16s} {c['train_seed']:<4d} "
              f"{'Y' if c['shared_adapter'] else '':6s} {c['adapter_id']}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()

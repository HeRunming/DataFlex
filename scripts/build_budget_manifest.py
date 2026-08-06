#!/usr/bin/env python3
"""Budget-specific SELECTION/SUBSET manifest for a target-draw experiment (decision_0806 /
user checkpoint 0806). Deliberately SEPARATE from the shared target-geometry manifest
(targetdraw_10draw_master_manifest.json), which records budget-independent facts: target draws,
target gradients, checkpoint, candidate cache, projection, environment.

This manifest records the budget-DEPENDENT layer:
  * selection_budget K
  * per (draw, method): selection hash, subset hash, subset row count
  * for nested-prefix budgets: parent (5%) selection file hash + parent ordered-index hash +
    prefix hash + ordering semantics, so the nesting is auditable
  * LengthMatched Random artifacts (per-bucket counts, token diff) at this budget
  * NICE artifacts at this budget (derived-from parent, zero-signal ids, reward diag)
  * environment
Usage: build_budget_manifest.py --budget 2707 --sel_prefix sel1pct --subset_suffix _1pct_sel \
         --parent_sel_prefix sel --out .../pilot1pct_selection_manifest.json
"""
import argparse, json, os, hashlib, glob
import torch, transformers

SAVES = "/jizhicfs/karonhe/dataflex_saves"
ROOT = "/jizhicfs/karonhe/DataFlex_fa"
DRAWS = ["stem80_draw0", "stem80_draw1", "stem80_draw2", "stem80_draw3", "stem80_draw4",
         "hum80_draw0", "hum80_draw1", "hum80_draw2", "hum80_draw3", "hum80_draw4"]
METHODS = ["dsmc", "less", "first_rr", "second_rr", "gist", "nice", "randk", "randk_lenmatch"]
PREFIXABLE = ["dsmc", "less", "first_rr", "second_rr", "gist", "nice", "randk"]


def fsha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def isha(idx):
    return hashlib.sha256(json.dumps(list(idx)).encode()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, required=True)
    ap.add_argument("--sel_prefix", required=True, help="selection dir prefix, e.g. sel1pct or sel")
    ap.add_argument("--subset_suffix", required=True, help="subset filename suffix, e.g. _1pct_sel or _sel")
    ap.add_argument("--parent_sel_prefix", default=None,
                    help="if this budget is a nested prefix of another, that budget's selection prefix (e.g. sel)")
    ap.add_argument("--parent_budget", type=int, default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    man = {"selection_budget": args.budget,
           "nested_prefix_of": args.parent_budget,
           "ordering_semantics": {
               "dsmc": "greedy append order (prefix = first-k greedy picks)",
               "first_rr": "round-robin append order", "second_rr": "round-robin append order",
               "less": "topk score-descending", "gist": "topk score-descending",
               "nice": "topk score-descending", "randk": "seeded randperm order",
               "randk_lenmatch": "rebuilt at this budget (NOT a prefix)"},
           "env": {"torch": torch.__version__, "transformers": transformers.__version__,
                   "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
                   "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"},
           "shared_target_geometry_manifest": "experiments/less_aligned/targetdraw_10draw_master_manifest.json",
           "draws": {}}
    for d in DRAWS:
        e = {"selections": {}, "subsets": {}, "subset_rows": {}}
        for m in METHODS:
            sel = f"{SAVES}/{args.sel_prefix}_{d}_{m}/step_1.json"
            sub = f"{SAVES}/sft_subsets/{d}_{m}{args.subset_suffix}.jsonl"
            idx = json.load(open(sel))["indices"]
            assert len(idx) == args.budget, f"{d}/{m}: {len(idx)} != {args.budget}"
            e["selections"][m] = {"step1_sha256": fsha(sel), "ordered_indices_sha256": isha(idx)}
            e["subsets"][m] = fsha(sub)
            e["subset_rows"][m] = sum(1 for _ in open(sub))
            assert e["subset_rows"][m] == args.budget, f"{d}/{m} subset rows mismatch"
            # nested-prefix audit fields
            if args.parent_sel_prefix and m in PREFIXABLE:
                psel = f"{SAVES}/{args.parent_sel_prefix}_{d}_{m}/step_1.json"
                pidx = json.load(open(psel))["indices"]
                e["selections"][m].update({
                    "parent_step1_sha256": fsha(psel),
                    "parent_ordered_indices_sha256": isha(pidx),
                    "prefix_verified": pidx[:args.budget] == idx})
                assert e["selections"][m]["prefix_verified"], f"{d}/{m}: NOT a prefix of parent"
        # LengthMatched artifacts at this budget
        lm = json.load(open(f"{SAVES}/{args.sel_prefix}_{d}_randk_lenmatch/step_1.json"))["metric"]
        e["randk_lenmatch"] = {"per_bucket": lm["per_bucket"], "token_diff": lm["token_diff"],
                               "seed": lm["seed"]}
        # NICE artifacts at this budget
        nm = json.load(open(f"{SAVES}/{args.sel_prefix}_{d}_nice/step_1.json"))["metric"]
        e["nice"] = {k: nm[k] for k in ("n_zero_signal", "zero_signal_target_ids", "reward_mean_overall",
                                        "reward_hist_counts", "mc", "gen_seed") if k in nm}
        if "derived_from" in nm:
            e["nice"]["derived_from"] = nm["derived_from"]
        man["draws"][d] = e
    json.dump(man, open(args.out, "w"), indent=2)
    print(f"wrote {args.out}  budget={args.budget} draws={len(man['draws'])} "
          f"methods={len(METHODS)} (all subsets/selections hash-recorded, prefix audit "
          f"{'ON' if args.parent_sel_prefix else 'n/a'})")


if __name__ == "__main__":
    main()

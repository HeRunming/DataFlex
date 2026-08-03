#!/usr/bin/env python3
"""Prelaunch validator for the full 5-draw (5%) target-draw run (choice_0803_02). Verifies the
frozen state before launching 45 new adapters. Exits nonzero on any failure."""
import json, os, glob, hashlib, sys

SAVES = "/jizhicfs/karonhe/dataflex_saves"
ROOT = "/jizhicfs/karonhe/DataFlex_fa"
DRAWS = ["stem80_draw0", "stem80_draw1", "stem80_draw2", "stem80_draw3", "stem80_draw4",
         "hum80_draw0", "hum80_draw1", "hum80_draw2", "hum80_draw3", "hum80_draw4"]
METHODS = ["dsmc", "less", "first_rr", "second_rr", "gist", "nice", "randk", "randk_lenmatch"]


def fsha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    fails = []
    def check(cond, msg):
        print(("  OK  " if cond else "  FAIL ") + msg)
        if not cond:
            fails.append(msg)

    # 1. 10/10 target draws present
    check(all(os.path.exists(f"{ROOT}/data/target_draws/{d}.jsonl") for d in DRAWS),
          "10/10 target draw jsonl present")
    # 2. 80/80 selections + subsets present
    nsel = sum(os.path.exists(f"{SAVES}/sel_{d}_{m}/step_1.json") for d in DRAWS for m in METHODS)
    nsub = sum(os.path.exists(f"{SAVES}/sft_subsets/{d}_{m}_sel.jsonl") for d in DRAWS for m in METHODS)
    check(nsel == 80, f"80/80 selections present (got {nsel})")
    check(nsub == 80, f"80/80 subsets present (got {nsub})")
    # 3. run plan: 10 draws, 80 cells, 75 adapters
    plan = json.load(open(f"{ROOT}/experiments/less_aligned/pilot_run_plan.json"))
    check(len(plan["draws"]) == 10 and plan["n_cells"] == 80 and plan["n_unique_adapters"] == 75,
          f"run plan 10 draws/80 cells/75 adapters (got {len(plan['draws'])}/{plan['n_cells']}/{plan['n_unique_adapters']})")
    # 4. shared Random-K subset hashes equal within each draw index
    ok_rk = True
    for idx in range(5):
        hs = {}
        for direc in ["stem80", "hum80"]:
            j = f"{SAVES}/sft_subsets/{direc}_draw{idx}_randk_sel.jsonl"
            hs[direc] = fsha(j)
        if len(set(hs.values())) != 1:
            ok_rk = False
    check(ok_rk, "shared Random-K subset hashes equal within all 5 draw-index pairs")
    # 5. run plan subset hashes match on disk
    ok_h = all(fsha(a["subset_jsonl"]) == a["subset_sha256"] for a in plan["adapters"].values())
    check(ok_h, "all 75 run-plan subset hashes match on-disk files")
    # 6. exactly 45 adapters remain (30 already trained w/ valid manifest)
    trained = 0
    for aid, a in plan["adapters"].items():
        ck = f"{SAVES}/sft_results/pilot_{aid}/adapter_model.safetensors"
        mf = f"{SAVES}/sft_results/pilot_{aid}/train_manifest.json"
        if os.path.exists(ck) and os.path.exists(mf):
            m = json.load(open(mf))
            if m.get("subset_sha256") == a["subset_sha256"] and m.get("adapter_sha256") == fsha(ck):
                trained += 1
    check(trained == 30, f"exactly 30 existing adapters validate (got {trained}) -> 45 remain")
    # 7. master manifest has 10 draws
    mm = json.load(open(f"{ROOT}/experiments/less_aligned/targetdraw_10draw_master_manifest.json"))
    check(len(mm["draws"]) == 10, f"10-draw master manifest present (got {len(mm['draws'])})")
    # 8. target-jsonl hash matches frozen draw meta (no drift)
    ok_drift = True
    for d in DRAWS:
        meta = json.load(open(f"{ROOT}/data/target_draws/{d}.meta.json"))
        if fsha(f"{ROOT}/data/target_draws/{d}.jsonl") != meta["target_file_sha256"]:
            ok_drift = False
    check(ok_drift, "all 10 target jsonl hashes match frozen draw meta (no drift)")

    print(f"\n{'PRELAUNCH OK' if not fails else 'PRELAUNCH FAILED: ' + str(len(fails)) + ' checks'}")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()

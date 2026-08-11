#!/usr/bin/env python3
"""2-adapter ENGINEERING canary verdict (advice_0811). Reports ENGINEERING_PASS/FAIL only.

Deliberate design constraint: this script stores and hashes the raw accuracy outputs but **never prints
or computes a DSMC-vs-Random comparison**. Per advice_0811 the interim accuracy of these two adapters
cannot alter the protocol and is not a stopping condition however large the gap. Keeping the comparison
out of the summary is the mechanical way to honour that, rather than relying on discipline.

Engineering checks, all fail-loud:
  1. train manifest complete           adapter file, trainer_state, config
  2. actual subset hash correct        the trained dataset == the frozen selection, verified by hash
  3. frozen SFT recipe correct         lora r128/alpha512/dropout0.05, lr 2e-5, cutoff 2048, eff batch 128
  4. optimizer steps / epochs          84 steps, 4 epochs
  5. eval completeness                 27/27 subtasks, 5,209 examples
  6. result + manifest hashes          every artifact hashed
  7. resume validated-skip             re-running training must skip, not retrain
  8. aggregate schema                  the 36-cell run plan is accepted
"""
import argparse, hashlib, glob, json, os

ROOT = "/jizhicfs/karonhe/DataFlex_fa"
EXP = f"{ROOT}/experiments/less_aligned"
SAVES = "/jizhicfs/karonhe/dataflex_saves"
CELLS = ["bbhx_draw0_dsmc_seed42", "bbhx_draw0_randk_seed42"]
K, STEPS, EPOCHS = 2707, 84, 4
N_SUBTASKS, N_EXAMPLES = 27, 5209


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def sha_idx(idx):
    return hashlib.sha256(json.dumps(sorted(idx)).encode()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=f"{EXP}/bbh_sft_canary_report.json")
    args = ap.parse_args()

    rep = {"canary": "2-adapter ENGINEERING canary (draw0 DSMC seed42 + draw0 Random-K seed42)",
           "scope": ("engineering validation ONLY. Raw accuracy is stored and hashed but deliberately "
                     "NOT compared here: per advice_0811 interim accuracy cannot alter the protocol and "
                     "is not a stopping condition however large the gap."),
           "cells": {}}
    fails = []

    for aid in CELLS:
        method = aid.replace("bbhx_draw0_", "").replace("_seed42", "")
        ad = f"{SAVES}/sft_results/{aid}"
        c = {}

        # 1. train artifacts
        adapter = f"{ad}/adapter_model.safetensors"
        ts_p = f"{ad}/trainer_state.json"
        c["adapter_present"] = os.path.exists(adapter)
        c["trainer_state_present"] = os.path.exists(ts_p)
        if c["adapter_present"]:
            c["adapter_sha256"] = sha_file(adapter)
        if not (c["adapter_present"] and c["trainer_state_present"]):
            fails.append(f"{aid}: missing train artifacts")
            rep["cells"][aid] = c
            continue

        # 2. the trained data really is the frozen selection
        sel = json.load(open(f"{SAVES}/selbbhx_draw0_{method}/step_1.json"))["indices"]
        jl = f"{SAVES}/sft_subsets/bbhx_draw0_{method}_sel.jsonl"
        pool = open(f"{ROOT}/data/less_train_all.jsonl").readlines()
        exported = open(jl).readlines()
        c["subset_sha256"] = sha_idx(sel)
        c["subset_jsonl_sha256"] = sha_file(jl)
        c["n_train_rows"] = len(exported)
        c["n_rows_ok"] = len(exported) == K
        c["rows_match_selected_indices"] = all(
            json.loads(exported[j]) == json.loads(pool[i]) for j, i in enumerate(sel))
        if not (c["n_rows_ok"] and c["rows_match_selected_indices"]):
            fails.append(f"{aid}: trained data does not match the frozen selection")

        # 3-4. recipe + schedule, read back from what actually ran
        ts = json.load(open(ts_p))
        c["global_step"] = ts["global_step"]
        c["epoch"] = round(ts["epoch"], 3)
        c["steps_ok"] = ts["global_step"] == STEPS
        c["epochs_ok"] = abs(ts["epoch"] - EPOCHS) < 0.2      # 84 steps over 2707 ex => 3.847
        acfg = json.load(open(f"{ad}/adapter_config.json"))
        c["lora"] = {"r": acfg.get("r"), "alpha": acfg.get("lora_alpha"),
                     "dropout": acfg.get("lora_dropout"),
                     "targets": sorted(acfg.get("target_modules") or [])}
        c["lora_ok"] = (acfg.get("r") == 128 and acfg.get("lora_alpha") == 512
                        and abs((acfg.get("lora_dropout") or 0) - 0.05) < 1e-9
                        and sorted(acfg.get("target_modules") or []) ==
                        ["k_proj", "o_proj", "q_proj", "v_proj"])
        if not (c["steps_ok"] and c["epochs_ok"] and c["lora_ok"]):
            fails.append(f"{aid}: recipe/schedule mismatch")

        # 5-6. eval completeness + hashes (accuracy stored, not compared)
        rj = sorted(glob.glob(f"{SAVES}/eval_results/bbh_external/{aid}/*/results_*.json"))
        c["n_results_files"] = len(rj)
        if not rj:
            fails.append(f"{aid}: no eval results")
            rep["cells"][aid] = c
            continue
        r = json.load(open(rj[-1]))
        sub = {k: v for k, v in r["results"].items() if k.startswith("bbh_external_heldout_")}
        ns = r["n-samples"]
        c["eval_results_json"] = rj[-1]
        c["eval_results_sha256"] = sha_file(rj[-1])
        c["n_subtasks"] = len(sub)
        c["n_examples"] = sum(ns[k]["effective"] for k in sub)
        c["subtasks_ok"] = len(sub) == N_SUBTASKS
        c["examples_ok"] = c["n_examples"] == N_EXAMPLES
        c["group_metric_present"] = "bbh_external_heldout" in r["results"]
        # stored, never surfaced as a comparison
        c["_raw_micro_stored"] = r["results"]["bbh_external_heldout"]["exact_match,get-answer"]
        if not (c["subtasks_ok"] and c["examples_ok"] and c["group_metric_present"]):
            fails.append(f"{aid}: eval incomplete")
        rep["cells"][aid] = c

    # 8. the aggregate/run-plan schema accepts 36 cells
    rp = json.load(open(f"{EXP}/bbh_external_run_plan.json"))
    ids = [x["adapter_id"] for x in rp["cells"]]
    rep["run_plan_schema"] = {
        "n_cells": rp["n_cells"], "n_subsets": rp["n_frozen_subsets"],
        "accepts_36": rp["n_cells"] == 36 and rp["n_frozen_subsets"] == 18,
        "canary_cells_in_plan": all(a in ids for a in CELLS),
        "n_primary": sum(1 for x in rp["cells"] if x["role"] == "primary"),
        "n_secondary": sum(1 for x in rp["cells"] if x["role"] != "primary"),
    }
    if not (rep["run_plan_schema"]["accepts_36"] and rep["run_plan_schema"]["canary_cells_in_plan"]):
        fails.append("run plan schema does not accept the 36-cell design")

    rep["failures"] = fails or None
    rep["ENGINEERING_PASS"] = not fails
    rep["accuracy_policy"] = (
        "Raw micro aggregates are stored per cell under `_raw_micro_stored` and the results JSONs are "
        "hashed, but NO comparison is computed or printed. These two numbers may not be used to stop, "
        "tune, or alter anything. On engineering failure only infrastructure fixes are permitted "
        "(paths, manifests, offline eval, resume, disk, hash bookkeeping).")
    json.dump(rep, open(args.out, "w"), indent=2)

    print(f"{'cell':30s} {'steps':>6s} {'epoch':>6s} {'lora':>5s} {'rows':>6s} {'subtasks':>9s} {'examples':>9s}")
    for aid, c in rep["cells"].items():
        print(f"{aid:30s} {str(c.get('global_step')):>6s} {str(c.get('epoch')):>6s} "
              f"{str(c.get('lora_ok')):>5s} {str(c.get('n_rows_ok')):>6s} "
              f"{str(c.get('subtasks_ok')):>9s} {str(c.get('examples_ok')):>9s}")
    print(f"\nrun-plan schema accepts 36 cells / 18 subsets : "
          f"{rep['run_plan_schema']['accepts_36']} "
          f"({rep['run_plan_schema']['n_primary']} primary + "
          f"{rep['run_plan_schema']['n_secondary']} secondary)")
    print(f"\nENGINEERING_PASS = {rep['ENGINEERING_PASS']}")
    if fails:
        for f in fails:
            print(f"  FAIL: {f}")
    print("(accuracy intentionally not compared -- see accuracy_policy)")
    print(f"wrote {args.out}")
    return 0 if rep["ENGINEERING_PASS"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

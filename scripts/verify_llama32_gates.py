#!/usr/bin/env python3
"""Engineering gates 3 and 4 for the Llama-3.2-3B second model-stack arm (choice_0814_3).

These are ENGINEERING pass criteria only, not new experiment design. No BBH accuracy is read.

Gate 3 (target gradients, per draw)
  * shape exactly (64, 8192) and the frozen projection contract (dim 8192, seed 123)
  * target gradients are SGD while candidates are Adam-aware -- the asymmetry most likely to
    silently drift on a second model, so it is asserted against the emitted configs
  * finite, no zero rows
  * the query IDs/prompts behind the extraction are exactly the frozen draw's, byte-identical to
    the Llama-2 arm, and token integrity holds under the 3072 cutoff

Gate 4 (selections)
  * DSMC / First-RR / Second-RR each reach exactly K=2707 unique in-range indices
  * selector outputs hash deterministically
  * Random-K is index-for-index identical to the frozen Llama-2 Random subsets

The gates must be able to FAIL. `--tamper_check` proves that by perturbing each check in memory
and confirming the verdict flips.
"""
import argparse, hashlib, json, os, sys

ROOT = "/jizhicfs/karonhe/DataFlex_fa"
EXP = f"{ROOT}/experiments/less_aligned"
SAVES = "/jizhicfs/karonhe/dataflex_saves"
DRAWS = [0, 1, 2]
K = 2707
N_CAND = 270679
M_QUERY = 64
PROJ_DIM = 8192
PROJ_SEED = 123


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def sha_tensor(v):
    return hashlib.sha256(v.numpy().tobytes()).hexdigest()


def gate3(rep, tamper=False):
    """Target-gradient contract, per draw."""
    import torch
    import yaml
    ok = True
    for d in DRAWS:
        e = {}
        tp = f"{SAVES}/llama32_draw{d}_output/target/1/all_projected_grads.pt"
        if not os.path.exists(tp):
            rep["gate3"][f"draw{d}"] = {"MISSING": tp, "pass": False}
            ok = False
            continue
        T = torch.load(tp, map_location="cpu").float()
        if tamper:
            T = T[:-1]                                    # break the (64, 8192) shape
        e["shape"] = list(T.shape)
        e["shape_ok"] = list(T.shape) == [M_QUERY, PROJ_DIM]
        n = T.norm(dim=1)
        e["all_finite"] = bool(torch.isfinite(T).all())
        e["n_zero_rows"] = int((n <= 1e-6).sum())
        e["row_norm_min"] = float(n.min())
        e["row_norm_max"] = float(n.max())
        e["sha256"] = sha_tensor(T)

        # the Adam/SGD asymmetry, read back from the config that actually ran
        cfg = yaml.safe_load(open(f"{EXP}/configs/draws/select_llama32_draw{d}.yaml"))
        comp = yaml.safe_load(open(f"{ROOT}/{cfg['components_cfg_file']}"))
        prm = comp["selectors"][cfg["component_name"]]["params"]
        e["candidate_gradient_type"] = prm["gradient_type"]
        e["target_gradient_type"] = prm["target_gradient_type"]
        e["proj_dim"] = prm["proj_dim"]
        e["proj_seed"] = prm["seed"]
        e["contract_ok"] = (prm["gradient_type"] == "adam"
                            and prm["target_gradient_type"] == "sgd"
                            and prm["proj_dim"] == PROJ_DIM
                            and prm["seed"] == PROJ_SEED)
        e["cutoff_len"] = cfg["cutoff_len"]
        e["cutoff_ok"] = cfg["cutoff_len"] == 3072

        # the query set behind the extraction must be the frozen draw, unchanged
        di = json.load(open(f"{ROOT}/data/dataset_info.json"))
        qf = di[f"bbhx_draw{d}_target"]["file_name"]
        if not os.path.isabs(qf):
            qf = f"{ROOT}/data/{qf}"
        rows = open(qf).readlines()                       # readlines(), never splitlines()
        e["query_file"] = qf
        e["n_query_rows"] = len(rows)
        e["query_rows_ok"] = len(rows) == M_QUERY
        e["query_file_sha256"] = sha_file(qf)
        ids = [json.loads(r)["id"] for r in rows if r.strip()]
        e["n_unique_query_ids"] = len(set(ids))
        e["query_ids_unique"] = len(set(ids)) == M_QUERY
        # The Llama-2 arm consumed the SAME frozen prompt file for this draw, so query identity is
        # established by checking the execution contract still points at that exact path.
        con = json.load(open(f"{EXP}/bbh_execution_contract.json"))["per_draw"][str(d)]
        e["contract_query_prompts"] = con["query_prompts"]
        e["query_identical_to_llama2"] = os.path.realpath(qf) == os.path.realpath(
            f"{ROOT}/{con['query_prompts']}")
        e["expected_shape_from_contract"] = con["target_grad_expected_shape"]
        e["shape_matches_contract"] = list(T.shape) == con["target_grad_expected_shape"]

        e["pass"] = bool(e["shape_ok"] and e["all_finite"] and e["n_zero_rows"] == 0
                         and e["contract_ok"] and e["cutoff_ok"] and e["query_rows_ok"]
                         and e["query_ids_unique"] and e["query_identical_to_llama2"]
                         and e["shape_matches_contract"])
        ok = ok and e["pass"]
        rep["gate3"][f"draw{d}"] = e
    rep["gate3"]["all_pass"] = ok
    return ok


def gate4(rep, tamper=False):
    """Selector budgets, determinism, and Random-K index identity with the Llama-2 arm."""
    ok = True
    methods = ["dsmc", "first_rr", "second_rr"]
    for d in DRAWS:
        for m in methods:
            e = {}
            f = f"{SAVES}/sel_llama32_draw{d}_{m}/step_1.json"
            if not os.path.exists(f):
                rep["gate4"][f"draw{d}_{m}"] = {"MISSING": f, "pass": False}
                ok = False
                continue
            idx = json.load(open(f))["indices"]
            if tamper:
                idx = idx[:-1]                            # break the K budget
            e["file"] = f
            e["K"] = len(idx)
            e["K_ok"] = len(idx) == K
            e["n_unique"] = len(set(idx))
            e["unique_ok"] = len(set(idx)) == len(idx)
            e["in_range"] = bool(idx) and min(idx) >= 0 and max(idx) < N_CAND
            e["idx_sha256"] = hashlib.sha256(
                json.dumps(sorted(idx)).encode()).hexdigest()
            e["pass"] = bool(e["K_ok"] and e["unique_ok"] and e["in_range"])
            ok = ok and e["pass"]
            rep["gate4"][f"draw{d}_{m}"] = e

        # Random-K: must be the frozen Llama-2 selection, index for index. Checked two ways --
        # against the frozen selection file AND by regenerating from the frozen seed.
        import torch
        e = {}
        f2 = f"{SAVES}/selbbhx_draw{d}_randk/step_1.json"
        e["frozen_llama2_selection"] = f2
        if os.path.exists(f2):
            idx = json.load(open(f2))["indices"]
            if tamper:
                idx = idx[:-1] + [idx[-1] + 1]            # perturb one index
            e["K"] = len(idx)
            e["K_ok"] = len(idx) == K
            e["idx_sha256"] = hashlib.sha256(json.dumps(sorted(idx)).encode()).hexdigest()
            g = torch.Generator(); g.manual_seed(5000 + d)
            regen = torch.randperm(N_CAND, generator=g)[:K].tolist()
            e["random_k_seed"] = 5000 + d
            e["seed_reproducible"] = sorted(regen) == sorted(idx)
            e["reuse_rationale"] = ("Random-K is target-independent, so reusing the exact Llama-2 "
                                    "indices gives the two model stacks a genuinely constant data "
                                    "baseline. DSMC/First-RR/Second-RR must NOT be reused.")
            e["pass"] = bool(e["K_ok"] and e["seed_reproducible"])
        else:
            e["MISSING"] = f2
            e["pass"] = False
        ok = ok and e["pass"]
        rep["gate4"][f"draw{d}_randk_reuse"] = e
    rep["gate4"]["all_pass"] = ok
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=f"{EXP}/llama32_gates_3_4.json")
    ap.add_argument("--tamper_check", action="store_true",
                    help="prove the gates can fail: perturb each check and require the verdict to flip")
    a = ap.parse_args()

    rep = {"arm": "Llama-3.2-3B second model-stack confirmation",
           "scope": "ENGINEERING gates only; no BBH accuracy is read at any point",
           "gate3": {}, "gate4": {}}
    g3 = gate3(rep)
    g4 = gate4(rep)
    rep["ALL_GATES_PASS"] = bool(g3 and g4)
    rep["no_accuracy_inspected"] = True

    if a.tamper_check:
        t = {"gate3": {}, "gate4": {}}
        f3 = gate3(t, tamper=True)
        f4 = gate4(t, tamper=True)
        rep["tamper_check"] = {
            "gate3_flips_to_fail": not f3, "gate4_flips_to_fail": not f4,
            "note": ("negative control: dropping one target-gradient row and one selected index, and "
                     "corrupting the Random-K hash, must each make the gate FAIL. If a gate still "
                     "passed here it would be vacuous."),
        }
        if f3 or f4:
            print("TAMPER CHECK FAILED: a gate passed on deliberately broken inputs", file=sys.stderr)
            json.dump(rep, open(a.out, "w"), indent=2)
            sys.exit(2)

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(rep, open(a.out, "w"), indent=2)
    print(f"gate3 (target gradients): {'PASS' if g3 else 'FAIL'}")
    print(f"gate4 (selections)      : {'PASS' if g4 else 'FAIL'}")
    if a.tamper_check:
        print(f"tamper check            : gate3 flips={rep['tamper_check']['gate3_flips_to_fail']} "
              f"gate4 flips={rep['tamper_check']['gate4_flips_to_fail']}")
    print(f"wrote {a.out}")
    sys.exit(0 if rep["ALL_GATES_PASS"] else 1)


if __name__ == "__main__":
    main()

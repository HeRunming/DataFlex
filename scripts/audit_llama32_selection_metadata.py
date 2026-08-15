#!/usr/bin/env python3
"""Context-drift audit of the 9 Llama-3.2 target-aware selections (code_review_0815, item 1).

Reads the existing `step_1.json` files ONLY. No selection is re-run, no accuracy is read.

Why this is not redundant with Gate 4
-------------------------------------
Gate 4 checks K=2707, uniqueness and index range, then records a hash. Those pass for a
selection produced with the WRONG hyperparameters:

  * `select_round_robin.py --perm_seed` DEFAULTS TO 0, while the frozen BBH contract requires
    `perm_seed = 6000 + draw_id`. A forgotten flag still yields 2707 legal unique indices, so
    Gate 4 passes and even the low Jaccard against the Llama-2 selections looks normal.
  * `select_moment_mmd.py --alpha` must be 0.0 for the frozen DSMC endpoint. Another alpha also
    yields 2707 legal unique indices.

So this audit asserts the recorded provenance fields themselves, which is exactly the failure
mode a compacted context would produce.

Also checks that First-RR and Second-RR within a draw share a byte-identical `query_order`
(the frozen design shares the RR permutation between the two orders), and that every selection
consumed Llama-3.2's OWN candidate/target caches rather than a Llama-2 path.
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
CAND_CACHE = f"{SAVES}/llama32_less_output/train/1/all_projected_grads.pt"


def tgt_cache(d):
    return f"{SAVES}/llama32_draw{d}_output/target/1/all_projected_grads.pt"


def rp(p):
    return os.path.realpath(p) if p else p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=f"{EXP}/llama32_selection_metadata_audit.json")
    ap.add_argument("--tamper_check", action="store_true")
    a = ap.parse_args()

    rep = {"audit": "Llama-3.2 selection metadata / context-drift audit",
           "reads": "existing step_1.json only; no selection re-run, no accuracy read",
           "why": ("Gate 4's K/unique/in-range checks pass even when perm_seed silently defaults "
                   "to 0 or DSMC runs a non-zero alpha, because those still produce 2707 legal "
                   "unique indices. This audit asserts the recorded provenance fields."),
           "cells": {}}
    ok = True

    for d in DRAWS:
        qo = {}
        for m in ("dsmc", "first_rr", "second_rr"):
            e, f = {}, f"{SAVES}/sel_llama32_draw{d}_{m}/step_1.json"
            if not os.path.exists(f):
                rep["cells"][f"draw{d}_{m}"] = {"MISSING": f, "pass": False}
                ok = False
                continue
            j = json.load(open(f))
            mt = j["metric"]
            e["file"] = f
            e["kernel"] = mt.get("kernel")
            e["n_indices"] = len(j["indices"])

            # shared contract
            e["num_select"] = mt.get("num_select")
            e["n_candidates"] = mt.get("n_candidates")
            e["n_target"] = mt.get("n_target")
            e["proj_dim"] = mt.get("proj_dim")
            shared = (mt.get("num_select") == K and mt.get("n_candidates") == N_CAND
                      and mt.get("n_target") == M_QUERY and mt.get("proj_dim") == PROJ_DIM)

            # both caches must be Llama-3.2's OWN
            e["train_grads"] = mt.get("train_grads")
            e["target_grads"] = mt.get("target_grads")
            e["train_grads_is_llama32_cache"] = rp(mt.get("train_grads")) == rp(CAND_CACHE)
            e["target_grads_is_this_draw"] = rp(mt.get("target_grads")) == rp(tgt_cache(d))
            # Reject Llama-2 caches by matching their exact directory NAMES, not substrings.
            # A substring blocklist is wrong here: the correct Llama-3.2 path
            # `.../llama32_less_output/train/...` CONTAINS `less_output/train`, so a naive
            # substring test rejects the right cache. Compare path components instead.
            l2_dirs = {"less_output", "less_sgd_output"} | {f"draw_bbhx_draw{i}_output"
                                                            for i in range(10)}
            parts = {q for k in ("train_grads", "target_grads")
                     for q in str(mt.get(k) or "").split(os.sep)}
            e["llama2_dirs_present"] = sorted(parts & l2_dirs)
            e["no_llama2_path"] = not (parts & l2_dirs)
            caches = (e["train_grads_is_llama32_cache"] and e["target_grads_is_this_draw"]
                      and e["no_llama2_path"])

            if m == "dsmc":
                e["alpha"] = mt.get("alpha")
                if a.tamper_check:
                    e["alpha"] = 0.5                     # negative control
                e["alpha_ok"] = e["alpha"] == 0.0
                e["subsample_indices"] = mt.get("subsample_indices")
                e["no_subsampling"] = mt.get("subsample_indices") in (None, -1, [], "null")
                e["pass"] = bool(shared and caches and e["alpha_ok"] and e["no_subsampling"])
            else:
                want_order = "first" if m == "first_rr" else "second"
                e["order"] = mt.get("order")
                e["order_ok"] = mt.get("order") == want_order
                e["perm_seed"] = mt.get("perm_seed")
                if a.tamper_check:
                    e["perm_seed"] = 0                   # the exact default-value drift we fear
                e["perm_seed_expected"] = 6000 + d
                e["perm_seed_ok"] = e["perm_seed"] == 6000 + d
                e["perm_seed_not_default"] = e["perm_seed"] != 0
                q = mt.get("query_order")
                e["query_order_len"] = len(q) if q is not None else None
                e["query_order_sha256"] = (
                    hashlib.sha256(json.dumps(q).encode()).hexdigest() if q is not None else None)
                qo[m] = e["query_order_sha256"]
                e["pass"] = bool(shared and caches and e["order_ok"] and e["perm_seed_ok"]
                                 and e["perm_seed_not_default"]
                                 and e["query_order_len"] == M_QUERY)
            e["shared_contract_ok"] = shared
            e["caches_ok"] = caches
            ok = ok and e["pass"]
            rep["cells"][f"draw{d}_{m}"] = e

        # First-RR and Second-RR share the RR permutation within a draw
        same = (qo.get("first_rr") is not None and qo["first_rr"] == qo.get("second_rr"))
        rep["cells"][f"draw{d}_rr_query_order_shared"] = {
            "first_rr_sha256": qo.get("first_rr"), "second_rr_sha256": qo.get("second_rr"),
            "identical": same,
            "why": ("the frozen design shares one RR permutation (perm_seed 6000+d) between the "
                    "first- and second-order arms, so the only difference between them is the "
                    "representation order"),
            "pass": same}
        ok = ok and same

    rep["ALL_PASS"] = ok
    rep["no_accuracy_read"] = True
    if a.tamper_check:
        rep["tamper_check"] = {
            "flips_to_fail": not ok,
            "note": ("negative control: DSMC alpha forced to 0.5 and RR perm_seed forced to 0 (the "
                     "library default). Both must make the audit FAIL, else it is vacuous.")}
        if ok:
            print("TAMPER CHECK FAILED: audit passed on deliberately wrong metadata", file=sys.stderr)
            json.dump(rep, open(a.out, "w"), indent=2)
            sys.exit(2)

    json.dump(rep, open(a.out, "w"), indent=2)
    if not a.tamper_check:
        for d in DRAWS:
            for m in ("dsmc", "first_rr", "second_rr"):
                e = rep["cells"][f"draw{d}_{m}"]
                extra = (f"alpha={e.get('alpha')}" if m == "dsmc"
                         else f"order={e.get('order')} perm_seed={e.get('perm_seed')}")
                print(f"draw{d} {m:10s} {extra:32s} caches_ok={e.get('caches_ok')} "
                      f"PASS={e['pass']}")
            print(f"draw{d} RR query_order shared: "
                  f"{rep['cells'][f'draw{d}_rr_query_order_shared']['identical']}")
    print(f"\nALL_PASS: {ok}")
    if a.tamper_check:
        print(f"tamper check flips to fail: {rep['tamper_check']['flips_to_fail']}")
    print(f"wrote {a.out}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

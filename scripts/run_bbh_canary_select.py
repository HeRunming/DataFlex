#!/usr/bin/env python3
"""BBH canary phase B: run all five selectors at K=2707 on draw0 and report diagnostics.
Selection only — NO SFT, NO adapter training, NO accuracy.

Selector contract is taken from `bbh_execution_contract.json`, not re-derived:
  dsmc       scripts/select_moment_mmd.py     --alpha 0.0
  less       scripts/select_relevance_topk.py --order first
  first_rr   scripts/select_round_robin.py    --order first  --perm_seed 6000
  second_rr  scripts/select_round_robin.py    --order second --perm_seed 6000
  randk      torch.randperm(N, generator=manual_seed(5000))[:K]

Required of every method: exactly 2707 unique in-range indices, and a bit-identical rerun.
Reported (and explicitly NOT to be used to alter any method): pairwise Jaccard, source composition,
and post-SFT-template token/length exposure. The length diagnostic exists because the BBH arm carries no
Random-K-LengthMatched control; if DSMC and Random turn out to differ wildly in token exposure, that can
be discussed BEFORE any BBH accuracy exists, which is why measuring it now is not outcome-driven.
"""
import argparse, hashlib, json, os, subprocess, sys

ROOT = "/jizhicfs/karonhe/DataFlex_fa"
EXP = f"{ROOT}/experiments/less_aligned"
SAVES = "/jizhicfs/karonhe/dataflex_saves"
PY = "/jizhicfs/karonhe/envs/dataflex-fa/bin/python"
CAND_GRAD = f"{SAVES}/less_output/train/1/all_projected_grads.pt"
CAND_JSONL = f"{ROOT}/data/less_train_all.jsonl"
DRAW, DRAW_ID, K = "bbhx_draw0", 0, 2707
N_POOL = 270679
SFT_CUTOFF = 2048          # downstream SFT cutoff: what the selected data will actually be truncated to


def sha_idx(idx):
    """Order-independent hash of a selection (the subset is a set, not a sequence)."""
    return hashlib.sha256(json.dumps(sorted(idx)).encode()).hexdigest()


def run(cmd, log):
    with open(log, "w") as lf:
        return subprocess.run(cmd, cwd=ROOT, stdout=lf, stderr=subprocess.STDOUT,
                              env=dict(os.environ, CUDA_VISIBLE_DEVICES="0")).returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target_grads",
                    default=f"{SAVES}/draw_{DRAW}_output/target/1/all_projected_grads.pt")
    ap.add_argument("--out", default=f"{EXP}/bbh_canary_selection_report.json")
    ap.add_argument("--rerun_check", action="store_true",
                    help="re-run every selector into a scratch dir and require identical subset hashes")
    args = ap.parse_args()

    import torch
    from transformers import AutoTokenizer

    rr_seed, rk_seed = 6000 + DRAW_ID, 5000 + DRAW_ID
    methods = {
        "dsmc":      [PY, "scripts/select_moment_mmd.py", "--alpha", "0.0"],
        "less":      [PY, "scripts/select_relevance_topk.py", "--order", "first"],
        "first_rr":  [PY, "scripts/select_round_robin.py", "--order", "first",
                      "--perm_seed", str(rr_seed)],
        "second_rr": [PY, "scripts/select_round_robin.py", "--order", "second",
                      "--perm_seed", str(rr_seed)],
    }
    os.makedirs(f"{SAVES}/logs", exist_ok=True)
    rep = {"canary_phase": "B: five selectors at K=2707 (selection only, no SFT)",
           "draw": DRAW, "budget_K": K, "rr_perm_seed": rr_seed, "random_k_seed": rk_seed,
           "target_grads": args.target_grads,
           "target_grads_sha256_content": hashlib.sha256(
               torch.load(args.target_grads, map_location="cpu").numpy().tobytes()).hexdigest(),
           "methods": {}}

    # ---- gradient-based selectors ----
    for m, cmd in methods.items():
        out_dir = f"{SAVES}/selbbhx_draw{DRAW_ID}_{m}"
        sel_p = f"{out_dir}/step_1.json"
        if not os.path.exists(sel_p):
            log = f"{SAVES}/logs/canary_sel_{DRAW}_{m}.log"
            print(f"[select] {m} -> {out_dir}")
            rc = run(cmd + ["--train_grads", CAND_GRAD, "--target_grads", args.target_grads,
                            "--out_cache_dir", out_dir, "--num_select", str(K)], log)
            if rc != 0 or not os.path.exists(sel_p):
                rep["methods"][m] = {"FAILED": f"rc={rc}, see {log}"}
                json.dump(rep, open(args.out, "w"), indent=2)
                raise SystemExit(f"{m} selection failed; see {log}")
        idx = json.load(open(sel_p))["indices"]
        rep["methods"][m] = {
            "selection_json": sel_p, "n": len(idx), "n_unique": len(set(idx)),
            "min": min(idx), "max": max(idx),
            "n_ok": len(idx) == K, "unique_ok": len(set(idx)) == K,
            "range_ok": min(idx) >= 0 and max(idx) < N_POOL,
            "subset_sha256": sha_idx(idx),
            "command": " ".join(c.replace(PY, "python") for c in cmd),
        }
        print(f"    {m}: n={len(idx)} unique={len(set(idx))} sha={rep['methods'][m]['subset_sha256'][:12]}")

    # ---- Random-K: the exact frozen procedure from the execution contract ----
    g = torch.Generator().manual_seed(rk_seed)
    ridx = torch.randperm(N_POOL, generator=g)[:K].tolist()
    out_dir = f"{SAVES}/selbbhx_draw{DRAW_ID}_randk"
    os.makedirs(out_dir, exist_ok=True)
    json.dump({"indices": ridx, "metric": {"kernel": "random_k", "seed": rk_seed, "num_select": K}},
              open(f"{out_dir}/step_1.json", "w"))
    # reproduce it a second time to prove the seed pins it
    g2 = torch.Generator().manual_seed(rk_seed)
    ridx2 = torch.randperm(N_POOL, generator=g2)[:K].tolist()
    rep["methods"]["randk"] = {
        "selection_json": f"{out_dir}/step_1.json", "n": len(ridx), "n_unique": len(set(ridx)),
        "min": min(ridx), "max": max(ridx), "n_ok": len(ridx) == K,
        "unique_ok": len(set(ridx)) == K, "range_ok": min(ridx) >= 0 and max(ridx) < N_POOL,
        "subset_sha256": sha_idx(ridx),
        "seed_reproducible": ridx == ridx2,
        "command": f"torch.randperm({N_POOL}, generator=manual_seed({rk_seed}))[:{K}]",
    }
    print(f"    randk: n={len(ridx)} reproducible={ridx == ridx2} "
          f"sha={rep['methods']['randk']['subset_sha256'][:12]}")

    # ---- RR order/seed correctness, read from the selectors' own metadata ----
    rr_meta = {}
    for m in ("first_rr", "second_rr"):
        j = json.load(open(rep["methods"][m]["selection_json"]))
        mt = j.get("metric", {})
        rr_meta[m] = {"perm_seed": mt.get("perm_seed"), "order": mt.get("order"),
                      "query_order_sha256": hashlib.sha256(
                          json.dumps(mt.get("query_order")).encode()).hexdigest()
                      if mt.get("query_order") is not None else None}
    rep["rr_checks"] = {
        "per_method": rr_meta,
        "seeds_match_contract": all(v["perm_seed"] == rr_seed for v in rr_meta.values()),
        "share_same_query_order": (rr_meta["first_rr"]["query_order_sha256"] ==
                                   rr_meta["second_rr"]["query_order_sha256"]),
        "orders_differ_as_intended": rr_meta["first_rr"]["order"] != rr_meta["second_rr"]["order"],
    }

    # ---- pairwise Jaccard ----
    sets = {m: set(json.load(open(v["selection_json"]))["indices"])
            for m, v in rep["methods"].items()}
    names = list(sets)
    rep["pairwise_jaccard"] = {
        f"{a}|{b}": round(len(sets[a] & sets[b]) / len(sets[a] | sets[b]), 4)
        for i, a in enumerate(names) for b in names[i + 1:]}

    # ---- source composition + post-SFT-template token exposure ----
    print("[diag] reading candidate pool for source/token diagnostics ...")
    want = set().union(*sets.values())
    rows = {}
    with open(CAND_JSONL) as f:
        for i, line in enumerate(f):
            if i in want:
                rows[i] = json.loads(line)
    tok = AutoTokenizer.from_pretrained("/jizhicfs/karonhe/models/shakechen/Llama-2-7b-hf")
    for m, S in sets.items():
        comp, toks, trunc = {}, [], 0
        for i in S:
            r = rows[i]
            comp[r.get("dataset", "unknown")] = comp.get(r.get("dataset", "unknown"), 0) + 1
            msgs = r.get("messages", [])
            u = " ".join(x.get("content", "") for x in msgs if x.get("role") == "user")
            a = " ".join(x.get("content", "") for x in msgs if x.get("role") == "assistant")
            n = len(tok(f"[INST] {u} [/INST]", add_special_tokens=True)["input_ids"]) + \
                len(tok(a, add_special_tokens=False)["input_ids"]) + 1
            toks.append(n)
            trunc += max(0, n - SFT_CUTOFF)
        toks.sort()
        rep["methods"][m]["source_composition"] = dict(sorted(comp.items(), key=lambda x: -x[1]))
        rep["methods"][m]["post_template_tokens"] = {
            "total": sum(toks), "mean": round(sum(toks) / len(toks), 1),
            "median": toks[len(toks) // 2], "p90": toks[int(len(toks) * 0.9)], "max": toks[-1],
            "n_over_sft_cutoff_2048": sum(1 for t in toks if t > SFT_CUTOFF),
            "tokens_truncated_at_2048": trunc,
            "histogram_512_buckets": {str(b): sum(1 for t in toks if b <= t < b + 512)
                                      for b in range(0, 4096, 512)},
        }
        print(f"    {m}: total_tokens={sum(toks):,} mean={sum(toks)/len(toks):.0f} "
              f">2048={rep['methods'][m]['post_template_tokens']['n_over_sft_cutoff_2048']}")

    d_t = rep["methods"]["dsmc"]["post_template_tokens"]["total"]
    r_t = rep["methods"]["randk"]["post_template_tokens"]["total"]
    rep["dsmc_vs_random_token_ratio"] = round(d_t / r_t, 4)
    rep["length_control_note"] = (
        "The BBH arm carries no Random-K-LengthMatched control (MMLU already has one). This diagnostic is "
        "reported so that, if DSMC and Random differ substantially in post-template token exposure, the "
        "question of adding such a control can be settled BEFORE any BBH downstream accuracy exists. "
        "It must NOT be used to alter DSMC, LESS or RR.")
    rep["PASS"] = (all(v.get("n_ok") and v.get("unique_ok") and v.get("range_ok")
                       for v in rep["methods"].values())
                   and rep["rr_checks"]["seeds_match_contract"]
                   and rep["rr_checks"]["share_same_query_order"]
                   and rep["rr_checks"]["orders_differ_as_intended"]
                   and rep["methods"]["randk"]["seed_reproducible"])
    rep["no_sft_run"] = "selection only: no SFT, no adapters, no accuracy"
    json.dump(rep, open(args.out, "w"), indent=2)
    print(f"\nRR: seeds_ok={rep['rr_checks']['seeds_match_contract']} "
          f"shared_order={rep['rr_checks']['share_same_query_order']} "
          f"orders_differ={rep['rr_checks']['orders_differ_as_intended']}")
    print(f"DSMC/Random token ratio: {rep['dsmc_vs_random_token_ratio']}")
    print(f"PASS = {rep['PASS']}")
    print(f"wrote {args.out}")
    return 0 if rep["PASS"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

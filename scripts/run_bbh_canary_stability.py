#!/usr/bin/env python3
"""Phase-A CLOSURE: dual-cache selection-stability gate (advice_0810). Selection only — no SFT, no eval.

The two draw0 target-gradient extractions are not bit-identical (row cosine >= 0.99992, 1/2707 DSMC
replacements). The question this answers is NOT "which cache is better" — it is "does a ~1e-4 numerical
perturbation in the target gradients move any target-aware selector by more than a boundary amount?"

Only the four TARGET-AWARE selectors are checked. Random-K is target-independent, so its subset cannot
depend on the target cache at all; including it would just pad the table with a trivial Jaccard of 1.

THE RULE IS FIXED HERE, BEFORE ANY OUTPUT IS INSPECTED (advice_0810):

    PASS  every target-aware selector replaces at most 1% of K -- i.e. <= 27 of 2707 examples --
          with no pathological round-robin cascade
    HOLD  any selector exceeds that. Then STOP and report. Do NOT change deterministic settings,
          dropout, eval/train mode, seeds, or the feature recipe in response.

The threshold judges NUMERICAL STABILITY ONLY. It says nothing about method quality and must never be
cited in a downstream claim. RR is the important case: greedy round-robin walks per-query
nearest-neighbour lists, so a tiny perturbation could in principle cascade — that is precisely why the
earlier DSMC-only evidence was insufficient.
"""
import argparse, hashlib, json, os, subprocess

ROOT = "/jizhicfs/karonhe/DataFlex_fa"
EXP = f"{ROOT}/experiments/less_aligned"
SAVES = "/jizhicfs/karonhe/dataflex_saves"
PY = "/jizhicfs/karonhe/envs/dataflex-fa/bin/python"
CAND = f"{SAVES}/less_output/train/1/all_projected_grads.pt"
RUN1 = f"{SAVES}/canary_target_run1.pt"
RUN2 = f"{SAVES}/draw_bbhx_draw0_output/target/1/all_projected_grads.pt"
K, DRAW_ID = 2707, 0
RR_SEED = 6000 + DRAW_ID
MAX_FRAC = 0.01                      # 1% of K == 27 replacements
MAX_REPL = int(K * MAX_FRAC)

# target-aware selectors only, exactly as the execution contract pins them
SELECTORS = {
    "dsmc":      [PY, "scripts/select_moment_mmd.py", "--alpha", "0.0"],
    "less":      [PY, "scripts/select_relevance_topk.py", "--order", "first"],
    "first_rr":  [PY, "scripts/select_round_robin.py", "--order", "first", "--perm_seed", str(RR_SEED)],
    "second_rr": [PY, "scripts/select_round_robin.py", "--order", "second", "--perm_seed", str(RR_SEED)],
}


def sha_idx(idx):
    return hashlib.sha256(json.dumps(sorted(idx)).encode()).hexdigest()


def sel(method, cmd, target, tag):
    out_dir = f"{SAVES}/canary_stab_{tag}_{method}"
    sel_p = f"{out_dir}/step_1.json"
    if not os.path.exists(sel_p):
        log = f"{SAVES}/logs/canary_stab_{tag}_{method}.log"
        print(f"  [{tag}] {method} ...", flush=True)
        with open(log, "w") as lf:
            rc = subprocess.run(cmd + ["--train_grads", CAND, "--target_grads", target,
                                       "--out_cache_dir", out_dir, "--num_select", str(K)],
                                cwd=ROOT, stdout=lf, stderr=subprocess.STDOUT,
                                env=dict(os.environ, CUDA_VISIBLE_DEVICES="0")).returncode
        if rc != 0 or not os.path.exists(sel_p):
            raise SystemExit(f"{tag}/{method} selection failed; see {log}")
    return json.load(open(sel_p))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=f"{EXP}/bbh_canary_stability_gate.json")
    args = ap.parse_args()
    import torch

    def tsha(p):
        return hashlib.sha256(torch.load(p, map_location="cpu").numpy().tobytes()).hexdigest()

    rep = {
        "gate": "phase-A closure: dual-cache selection stability over the four target-aware selectors",
        "why": ("the two draw0 extractions differ by ~1e-4 (row cosine >= 0.99992); this asks whether that "
                "moves any target-aware selector by more than a boundary amount, NOT which cache is better"),
        "rule_fixed_before_inspection": {
            "pass_if": f"every target-aware selector replaces <= {MAX_REPL} of K={K} ({MAX_FRAC:.0%})",
            "hold_if": "any selector exceeds that, or a round-robin cascade appears",
            "scope": ("numerical stability ONLY -- never a statement about method quality, and never to "
                      "be cited in a downstream claim"),
            "on_hold": ("STOP and report. Do NOT change deterministic settings, dropout, eval/train mode, "
                        "seeds, or the feature recipe."),
        },
        "random_k_excluded_because": "Random-K is target-independent; its subset cannot depend on the cache",
        "caches": {"run1": {"path": RUN1, "tensor_sha256": tsha(RUN1)},
                   "run2": {"path": RUN2, "tensor_sha256": tsha(RUN2)}},
        "K": K, "rr_perm_seed": RR_SEED, "methods": {},
    }
    print(f"rule (fixed in advance): PASS iff every selector replaces <= {MAX_REPL}/{K}\n")

    for m, cmd in SELECTORS.items():
        a = set(sel(m, cmd, RUN1, "run1")["indices"])
        b = set(sel(m, cmd, RUN2, "run2")["indices"])
        inter, union = len(a & b), len(a | b)
        repl = K - inter
        rep["methods"][m] = {
            "n_run1": len(a), "n_run2": len(b), "intersection": inter,
            "jaccard": round(inter / union, 6), "n_replaced": repl,
            "frac_replaced": round(repl / K, 6),
            "within_threshold": repl <= MAX_REPL,
            "subset_sha256_run1": sha_idx(a), "subset_sha256_run2": sha_idx(b),
            "identical": a == b,
            "command": " ".join(c.replace(PY, "python") for c in cmd),
        }
        v = rep["methods"][m]
        print(f"  {m:10s} inter={inter:5d}/{K}  J={v['jaccard']:.6f}  replaced={repl:3d}  "
              f"within_1pct={v['within_threshold']}")

    rr = [rep["methods"][m]["n_replaced"] for m in ("first_rr", "second_rr")]
    grad = [rep["methods"][m]["n_replaced"] for m in ("dsmc", "less")]
    # a "cascade" would show as RR being dramatically less stable than the non-sequential selectors
    rep["rr_cascade_check"] = {
        "rr_replacements": rr, "non_rr_replacements": grad,
        "rr_max": max(rr), "non_rr_max": max(grad),
        "ratio_rr_over_non_rr": (round(max(rr) / max(grad), 3) if max(grad) else None),
        "cascade_detected": max(rr) > MAX_REPL,
        "note": ("greedy round-robin advances per-query nearest-neighbour pointers, so a tiny target "
                 "perturbation could in principle cascade. This is why DSMC alone was not sufficient "
                 "evidence."),
    }
    worst = max(v["n_replaced"] for v in rep["methods"].values())
    rep["worst_case_replacements"] = worst
    rep["VERDICT"] = "PASS" if worst <= MAX_REPL else "HOLD"
    rep["verdict_meaning"] = (
        f"worst-case {worst}/{K} replacements ({worst / K:.4%}) across the four target-aware selectors. "
        + ("Within the pre-registered 1% band, so the extraction is stable AT SELECTION LEVEL despite not "
           "being bit-reproducible. This does not make the gradients deterministic; it bounds the "
           "downstream consequence."
           if worst <= MAX_REPL else
           "EXCEEDS the pre-registered band -- STOP and report; do not alter the recipe in response."))
    rep["no_sft_run"] = "selection only: no SFT, no adapters, no accuracy"
    json.dump(rep, open(args.out, "w"), indent=2)
    print(f"\nRR cascade detected : {rep['rr_cascade_check']['cascade_detected']}")
    print(f"worst case          : {worst}/{K} replacements (threshold {MAX_REPL})")
    print(f"VERDICT             : {rep['VERDICT']}")
    print(f"wrote {args.out}")
    return 0 if rep["VERDICT"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())

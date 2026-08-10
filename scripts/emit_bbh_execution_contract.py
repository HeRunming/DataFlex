#!/usr/bin/env python3
"""AUTHORITATIVE BBH execution contract (code_review_0810 P0-1). Artifacts only — nothing is run.

WHY THIS FILE EXISTS
--------------------
The frozen method is NOT "MMD over Adam-preconditioned gradients". That summary is ambiguous in the one
place that matters and would silently produce a DIFFERENT experiment from the completed MMLU main line:

    candidate gradients : Adam-aware
    target/query grads  : SGD            <-- NOT the same as the candidate side
    projection          : dim 8192, seed 123
    candidate cache     : the existing frozen 270,679 x 8192 artifact, reused verbatim

The generic online `MMDSelector` defaults to `gradient_type=adam` with `target_gradient_type=same`, so
re-deriving DSMC from that class would compute Adam target gradients and would not be the DSMC of the
completed experiments. The frozen DSMC endpoint is the OFFLINE script:

    scripts/select_moment_mmd.py --alpha 0.0        # k(u,v) = <u,v>^2, exact marginal greedy

This contract pins that mapping for all five BBH methods, resolves it against the frozen MMLU master
manifest, and verifies the referenced caches/checkpoints still hash as recorded. It is the single source
of truth for the BBH round; nothing should be reconstructed from prose or from memory.
"""
import argparse, hashlib, json, os

ROOT = "/jizhicfs/karonhe/DataFlex_fa"
EXP = f"{ROOT}/experiments/less_aligned"
SAVES = "/jizhicfs/karonhe/dataflex_saves"
MASTER = f"{EXP}/targetdraw_10draw_master_manifest.json"
BASE_MODEL = "/jizhicfs/karonhe/models/shakechen/Llama-2-7b-hf"
WARMUP = f"{SAVES}/sft_results/warmup_seed42/checkpoint-1692"
CAND_GRAD = f"{SAVES}/less_output/train/1/all_projected_grads.pt"

DRAWS = [0, 1, 2]
BUDGET = 2707
RANDOM_SEED_BASE, RR_PERM_SEED_BASE = 5000, 6000

# The frozen selector contract. Derived from the driver of the completed experiments
# (experiments/less_aligned/run_targetdraw_pilot.sh), NOT re-invented here.
SELECTORS = {
    "dsmc": {
        "script": "scripts/select_moment_mmd.py",
        "args": ["--alpha", "0.0"],
        "kernel": "k(u,v) = <u,v>^2 (pure 2nd-order), exact marginal greedy MMD",
        "must_not_use": ("the generic online MMDSelector in src/dataflex/train/selector/mmd_selector.py "
                         "— its target_gradient_type defaults to 'same', which would make target "
                         "gradients Adam-aware and would NOT be the frozen DSMC"),
        "seeded": False,
    },
    "less": {
        "script": "scripts/select_relevance_topk.py",
        "args": ["--order", "first"],
        "kernel": "1st-order relevance top-k, s = <x, mean_t t> (LESS-style)",
        "must_not_use": "official trajectory-LESS; this arm is the relevance-topk reimplementation",
        "seeded": False,
    },
    "first_rr": {
        "script": "scripts/select_round_robin.py",
        "args": ["--order", "first"],
        "kernel": "greedy round-robin over queries, s = <x, t>",
        "seeded": "rr_perm_seed",
    },
    "second_rr": {
        "script": "scripts/select_round_robin.py",
        "args": ["--order", "second"],
        "kernel": "greedy round-robin over queries, s = <x, t>^2",
        "seeded": "rr_perm_seed",
        "note": "SAME script and SAME rr_perm_seed as first_rr; only the representation order differs",
    },
    "randk": {
        "script": "inline torch.randperm (see run_targetdraw_pilot.sh 'randk (uniform fixed-K)')",
        "exact_procedure": ("N = candidate_cache.shape[0]; "
                            "g = torch.Generator().manual_seed(random_k_seed); "
                            "indices = torch.randperm(N, generator=g)[:K].tolist()"),
        "args": ["--seed", "<random_k_seed>"],
        "kernel": "uniform fixed-K sample (no target information)",
        "seeded": "random_k_seed",
        "note": ("plain uniform Random-K on a CPU torch.Generator, bit-reproducible. NOT "
                 "select_randk_lenmatch.py — randk_lenmatch was an MMLU-only 9th arm and is NOT part "
                 "of the 5-method BBH set."),
    },
}


def sha_file(p):
    """sha256 of raw FILE bytes (streamed). Used for checkpoints/safetensors."""
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def sha_tensor(p):
    """sha256 of TENSOR CONTENT, matching `tsha` in build_targetdraw_master_manifest.py.

    The master manifest hashes `v.numpy().tobytes()`, NOT the file bytes — a torch.save container can
    differ byte-wise (pickle framing, storage order) while holding identical numbers, so the content
    hash is the meaningful invariant for a gradient cache. Comparing a file hash against a recorded
    tensor hash produces a spurious mismatch, which is exactly what happened on the first pass here.
    """
    import torch
    v = torch.load(p, map_location="cpu")
    return hashlib.sha256(v.numpy().tobytes()).hexdigest(), tuple(v.shape)


def dir_sha(d, names):
    out = {}
    for n in names:
        p = os.path.join(d, n)
        out[n] = sha_file(p) if os.path.exists(p) else None
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=f"{EXP}/bbh_execution_contract.json")
    ap.add_argument("--verify_candidate_cache", action="store_true",
                    help="re-hash the 8.9GB candidate cache (slow); otherwise record size+mtime only")
    args = ap.parse_args()

    master = json.load(open(MASTER))

    # ---- the frozen feature contract, READ from the MMLU master manifest, not retyped ----
    feature = {
        "proj_dim": master["proj_dim"],
        "proj_seed": master["proj_seed"],
        "gradient_type_candidate": master["gradient_type_candidate"],
        "gradient_type_target": master["gradient_type_target"],
        "source": os.path.relpath(MASTER, ROOT),
        "emphasis": ("candidate and target gradient types DIFFER (adam vs sgd). This asymmetry is part "
                     "of the frozen protocol and must be preserved for the BBH round."),
    }
    for k, want in [("proj_dim", 8192), ("proj_seed", 123),
                    ("gradient_type_candidate", "adam"), ("gradient_type_target", "sgd")]:
        if feature[k] != want:
            raise SystemExit(f"frozen contract mismatch: {k}={feature[k]!r}, expected {want!r}")

    cand = {"path": CAND_GRAD,
            "expected_sha256": master["candidate_cache"]["sha256"],
            "expected_shape": master["candidate_cache"]["shape"],
            "exists": os.path.exists(CAND_GRAD),
            "size_bytes": os.path.getsize(CAND_GRAD) if os.path.exists(CAND_GRAD) else None,
            "reuse_policy": "REUSED VERBATIM. Candidate gradients are target-independent, so the BBH "
                            "round must not regenerate them; only target/query gradients are new.",
            "hash_convention": "sha256 of TENSOR CONTENT (v.numpy().tobytes()), matching `tsha` in "
                               "build_targetdraw_master_manifest.py — NOT the file bytes."}
    if args.verify_candidate_cache and cand["exists"]:
        cand["measured_sha256"], shape = sha_tensor(CAND_GRAD)
        cand["measured_shape"] = list(shape)
        cand["sha256_matches"] = cand["measured_sha256"] == cand["expected_sha256"]
        cand["shape_matches"] = list(shape) == list(cand["expected_shape"])
    else:
        cand["sha256_matches"] = None
        cand["note"] = "re-hash with --verify_candidate_cache before the canary (8.9GB, takes minutes)"

    warm = {"path": WARMUP,
            "expected_adapter_sha256": master["warmup_ckpt"]["adapter_sha256"],
            "expected_optimizer_sha256": master["warmup_ckpt"]["optimizer_sha256"],
            "measured": dir_sha(WARMUP, ["adapter_model.safetensors", "optimizer.pt"])}
    am = warm["measured"].get("adapter_model.safetensors")
    warm["adapter_sha256_matches"] = (am == warm["expected_adapter_sha256"]) if am else None

    per_draw = {str(d): {"random_k_seed": RANDOM_SEED_BASE + d,
                         "rr_perm_seed": RR_PERM_SEED_BASE + d,
                         "query_prompts": f"data/bbh_external/query_prompts/bbh_query_draw{d}_prompts.jsonl",
                         "target_grads_expected": f"{SAVES}/draw_bbhx_draw{d}_output/target/1/all_projected_grads.pt",
                         "target_grad_expected_shape": [64, feature["proj_dim"]]}
                for d in DRAWS}

    contract = {
        "contract": "BBH external-validation execution contract",
        "status": "AUTHORITATIVE. Supersedes any prose or model-memory summary of the method.",
        "why": ("A fresh context summarised the method as 'MMD over Adam-preconditioned gradients', "
                "which omits that TARGET gradients are SGD while CANDIDATE gradients are Adam-aware. "
                "Implementing from that summary would have produced a different experiment."),
        "frozen_feature_contract": feature,
        "candidate_cache": cand,
        "warmup_checkpoint": warm,
        "base_model": BASE_MODEL,
        "budget_K": BUDGET,
        "selector_contract": SELECTORS,
        "n_methods": len(SELECTORS),
        "per_draw": per_draw,
        "target_gradient_extraction": {
            "driver": "dataflex-cli train <select yaml> (target-only phase), as in run_targetdraw_pilot.sh",
            "template": "llama2", "formatting": "sharegpt", "cutoff_len": 2048,
            "candidate_symlink_rule": ("the per-draw cache's train/1/all_projected_grads.pt must be a "
                                       "symlink resolving to the frozen candidate cache; verify with "
                                       "readlink -f before selection"),
            "TRUNCATION_GATE": ("bbh_token_truncation_audit.json MUST report verdict=PASS before any "
                                "target-gradient extraction. It currently reports HOLD."),
        },
        "counts": {"draws": len(DRAWS), "methods": len(SELECTORS),
                   "frozen_subsets": len(DRAWS) * len(SELECTORS),
                   "sft_seeds": 2, "adapters": len(DRAWS) * len(SELECTORS) * 2},
        "no_compute_run": "artifacts only: no gradients, no selection, no SFT",
    }

    json.dump(contract, open(args.out, "w"), indent=2)
    print(f"frozen features : proj_dim={feature['proj_dim']} proj_seed={feature['proj_seed']} "
          f"candidate={feature['gradient_type_candidate']} target={feature['gradient_type_target']}")
    print(f"candidate cache : exists={cand['exists']} sha_verified={cand['sha256_matches']}")
    print(f"warmup adapter  : sha_matches={warm['adapter_sha256_matches']}")
    for m, v in SELECTORS.items():
        print(f"  {m:10s} -> {v['script']} {' '.join(v['args'])}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

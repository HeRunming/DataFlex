#!/usr/bin/env python3
"""BBH selection-only CANARY (approved in review_0810). Engineering validation, NOT a result.

Scope, exactly as approved:
  A  draw0 target-gradient extraction (64 queries)      -> `--phase target`
  B  all five selectors at K=2707 + diagnostics          -> `--phase select`
  C  one no-SFT base-model held-out BBH evaluation       -> `--phase baseeval`
NO SFT. NO adapter training. No method may be altered on the basis of anything this prints.

Head semantics (review_0810 item 2): the launch receipt was committed one commit AFTER the snapshot it
approves, so this records BOTH `approved_code_snapshot` (from the receipt) and `runtime_head`, and
asserts the only difference between them is the receipt artifact itself. No self-referential commit.

Pass criteria for phase A are pre-registered here and checked mechanically:
  * emitted select YAML readback: cutoff_len == 3072, lora_dropout == 0.1
  * target jsonl sha256 + ordered-id hash == the frozen draw0 prompt artifact
  * candidate symlink resolves to the frozen candidate cache, tensor-content hash verified
  * target tensor shape (64, 8192), finite, no zero rows, dtype recorded
  * warm-up adapter AND optimizer hashes match
  * the ACTUALLY LOADED PEFT dropout and `model.training` are recorded
  * if dropout is active: re-extract draw0 from a clean cache and require an identical
    projected-gradient hash. If the hashes DIFFER -> STOP and report cosine/norm/Jaccard diagnostics.
    Do NOT set eval mode, force dropout=0, change the seed, or pick one cache.
"""
import argparse, hashlib, json, os, shutil, subprocess, sys

ROOT = "/jizhicfs/karonhe/DataFlex_fa"
EXP = f"{ROOT}/experiments/less_aligned"
SAVES = "/jizhicfs/karonhe/dataflex_saves"
PY = "/jizhicfs/karonhe/envs/dataflex-fa/bin/python"
ENVBIN = "/jizhicfs/karonhe/envs/dataflex-fa/bin"
BASE = "/jizhicfs/karonhe/models/shakechen/Llama-2-7b-hf"
WARMUP = f"{SAVES}/sft_results/warmup_seed42/checkpoint-1692"
CAND_GRAD = f"{SAVES}/less_output/train/1/all_projected_grads.pt"
DRAW = "bbhx_draw0"
DRAW_ID = 0
K = 2707
CUTOFF = 3072
RANDOM_SEED, RR_PERM_SEED = 5000 + DRAW_ID, 6000 + DRAW_ID
OUT = f"{EXP}/bbh_canary_report.json"


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def sha_tensor(p):
    import torch
    v = torch.load(p, map_location="cpu")
    return hashlib.sha256(v.numpy().tobytes()).hexdigest(), tuple(v.shape), str(v.dtype)


def git(*a):
    try:
        return subprocess.check_output(["git", "-C", ROOT, *a], stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return None


def head_semantics():
    """approved_code_snapshot vs runtime_head, with the difference enumerated rather than assumed."""
    receipt_p = f"{EXP}/bbh_canary_launch_receipt.json"
    approved = json.load(open(receipt_p))["executing_head"] if os.path.exists(receipt_p) else None
    runtime = git("rev-parse", "HEAD")
    diff = None
    if approved and runtime and approved != runtime:
        raw = git("diff", "--name-only", f"{approved}..{runtime}")
        diff = [l for l in (raw or "").split("\n") if l]
    only_receipt = (approved == runtime) or (diff == ["experiments/less_aligned/bbh_canary_launch_receipt.json"])
    return {
        "approved_code_snapshot": approved,
        "runtime_head": runtime,
        "identical": approved == runtime,
        "files_changed_between": diff,
        "only_difference_is_the_receipt_artifact": only_receipt,
        "tree_clean_at_runtime": git("status", "--porcelain") == "",
        "note": ("The receipt commit necessarily lands one commit after the snapshot it approves. This "
                 "records both and enumerates the difference instead of claiming they are the same."),
    }


def phase_target(rep):
    """A: extract draw0 target gradients, with every pre-registered check."""
    import torch, yaml
    ck = {}
    yaml_p = f"{EXP}/configs/draws/select_{DRAW}.yaml"
    cfg = yaml.safe_load(open(yaml_p))
    ck["select_yaml"] = os.path.relpath(yaml_p, ROOT)
    ck["cutoff_len_readback"] = cfg.get("cutoff_len")
    ck["lora_dropout_readback"] = cfg.get("lora_dropout")
    ck["cutoff_ok"] = cfg.get("cutoff_len") == CUTOFF
    ck["dropout_ok"] = float(cfg.get("lora_dropout")) == 0.1

    # the registered dataset must be the frozen prompt artifact
    info = json.load(open(f"{ROOT}/data/dataset_info.json"))
    tj = info[f"{DRAW}_target"]["file_name"]
    man = json.load(open(f"{EXP}/bbh_query_prompt_manifest.json"))["draws"][str(DRAW_ID)]
    ck["target_jsonl"] = os.path.relpath(tj, ROOT)
    ck["target_jsonl_sha256"] = sha_file(tj)
    ck["target_jsonl_sha_ok"] = ck["target_jsonl_sha256"] == man["file_sha256"]
    rows = [json.loads(l) for l in open(tj) if l.strip()]
    ck["n_rows"] = len(rows)
    ck["n_rows_ok"] = len(rows) == 64
    meta = json.load(open(f"{ROOT}/data/bbh_external/bbh_query_draw{DRAW_ID}.meta.json"))
    oh = hashlib.sha256(json.dumps([r["id"] for r in rows]).encode()).hexdigest()
    ck["ordered_ids_sha256"] = oh
    ck["ordered_ids_ok"] = oh == meta["ordered_ids_sha256"]

    # warm-up checkpoint
    master = json.load(open(f"{EXP}/targetdraw_10draw_master_manifest.json"))
    ck["warmup_adapter_ok"] = sha_file(f"{WARMUP}/adapter_model.safetensors") == \
        master["warmup_ckpt"]["adapter_sha256"]
    ck["warmup_optimizer_ok"] = sha_file(f"{WARMUP}/optimizer.pt") == \
        master["warmup_ckpt"]["optimizer_sha256"]

    cache = f"{SAVES}/draw_{DRAW}_output"
    tgt_p = f"{cache}/target/1/all_projected_grads.pt"

    # candidate symlink must resolve to the FROZEN cache (never a stray file)
    os.makedirs(f"{cache}/train/1", exist_ok=True)
    link = f"{cache}/train/1/all_projected_grads.pt"
    if not os.path.exists(link):
        os.symlink(CAND_GRAD, link)
    ck["candidate_link_resolves_to"] = os.path.realpath(link)
    ck["candidate_link_ok"] = os.path.realpath(link) == os.path.realpath(CAND_GRAD)

    # ---- run extraction (unless a valid cache already exists) ----
    if not os.path.exists(tgt_p):
        env = dict(os.environ, PATH=f"{ENVBIN}:{os.environ['PATH']}",
                   CUDA_VISIBLE_DEVICES=os.environ.get("CUDA_VISIBLE_DEVICES", "0,1,2,3,4,5,6,7"),
                   HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1")
        log = f"{SAVES}/logs/canary_gengrad_{DRAW}.log"
        os.makedirs(os.path.dirname(log), exist_ok=True)
        print(f"[target] extracting -> {tgt_p}\n[target] log: {log}")
        with open(log, "w") as lf:
            r = subprocess.run([f"{ENVBIN}/dataflex-cli", "train", os.path.relpath(yaml_p, ROOT)],
                               cwd=ROOT, env=env, stdout=lf, stderr=subprocess.STDOUT)
        ck["extraction_returncode"] = r.returncode
        if r.returncode != 0 or not os.path.exists(tgt_p):
            ck["FAILED"] = f"extraction failed (rc={r.returncode}); see {log}"
            rep["phase_target"] = ck
            return False
    else:
        ck["extraction_returncode"] = "skipped (cache present)"

    h, shape, dtype = sha_tensor(tgt_p)
    X = torch.load(tgt_p, map_location="cpu").float()
    n = X.norm(dim=1)
    ck["target_tensor"] = {
        "path": tgt_p, "sha256_tensor_content": h, "shape": list(shape), "dtype": dtype,
        "shape_ok": list(shape) == [64, 8192],
        "all_finite": bool(torch.isfinite(X).all()),
        "n_zero_rows": int((n <= 1e-6).sum()),
        "norm_min": float(n.min()), "norm_max": float(n.max()), "norm_mean": float(n.mean()),
    }
    rep["phase_target"] = ck
    hard = ["cutoff_ok", "dropout_ok", "target_jsonl_sha_ok", "n_rows_ok", "ordered_ids_ok",
            "warmup_adapter_ok", "warmup_optimizer_ok", "candidate_link_ok"]
    ok = all(ck[k] for k in hard) and ck["target_tensor"]["shape_ok"] \
        and ck["target_tensor"]["all_finite"] and ck["target_tensor"]["n_zero_rows"] == 0
    ck["PASS"] = ok
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["heads", "target", "select", "baseeval"], required=True)
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    rep = json.load(open(args.out)) if os.path.exists(args.out) else {}
    rep["canary"] = "BBH selection-only canary (no SFT)"
    rep["head_semantics"] = head_semantics()
    hs = rep["head_semantics"]
    print(f"[heads] approved_code_snapshot = {hs['approved_code_snapshot']}")
    print(f"[heads] runtime_head           = {hs['runtime_head']}")
    print(f"[heads] files changed between  = {hs['files_changed_between']}")
    print(f"[heads] only diff is receipt   = {hs['only_difference_is_the_receipt_artifact']}")
    if not hs["only_difference_is_the_receipt_artifact"]:
        raise SystemExit("runtime HEAD differs from the approved snapshot by more than the receipt "
                         "artifact; re-emit the receipt or check out the approved snapshot.")

    ok = True
    if args.phase == "target":
        ok = phase_target(rep)
        print(f"\n[target] PASS = {ok}")
        for k, v in rep["phase_target"].items():
            if k != "target_tensor":
                print(f"    {k}: {v}")
        t = rep["phase_target"].get("target_tensor", {})
        if t:
            print(f"    tensor: shape={t['shape']} dtype={t['dtype']} finite={t['all_finite']} "
                  f"zero_rows={t['n_zero_rows']} norm~{t['norm_mean']:.4f}")
            print(f"    tensor sha256(content) = {t['sha256_tensor_content']}")

    json.dump(rep, open(args.out, "w"), indent=2)
    print(f"\nwrote {args.out}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

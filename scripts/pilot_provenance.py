#!/usr/bin/env python3
"""Provenance helpers for the pilot SFT driver (code_review_0801). Subcommands:
  register       --plan P                : validate each unique adapter subset (rows + SHA256 vs plan) and register dataset key
  write_train_manifest --plan P --aid A --adapter_dir D : write train_manifest.json (subset/seed/args/hashes/commit/adapter hash)
  check_train    --plan P --aid A --adapter_dir D       : exit 0 if a valid matching train manifest+adapter exists (resume), else 1
  find_eval      --eval_dir D            : print the single authoritative results file path, or exit 1
  write_eval_manifest --aid A --adapter_dir AD --eval_dir D : write eval_manifest.json (result path/hash, adapter hash, versions, tasks)
  check_eval     --eval_dir D --adapter_dir AD           : exit 0 if valid eval manifest matches current adapter, else 1
"""
import argparse, json, os, glob, hashlib, subprocess

K = 13533
ROOT = "/jizhicfs/karonhe/DataFlex_fa"


def fsha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def git_commit():
    try:
        return subprocess.check_output(["git", "-C", ROOT, "rev-parse", "--short", "HEAD"]).decode().strip()
    except Exception:
        return "unknown"


def cmd_register(a):
    plan = json.load(open(a.plan))
    info = json.load(open(f"{ROOT}/data/dataset_info.json"))
    for aid, ad in plan["adapters"].items():
        j = ad["subset_jsonl"]
        n = sum(1 for _ in open(j))
        assert n == K, f"{aid}: {j} has {n} != {K}"
        got = fsha(j)
        assert got == ad["subset_sha256"], f"{aid}: subset SHA mismatch {got} != {ad['subset_sha256']}"
        info[ad["dataset_key"]] = {"file_name": j, "formatting": "sharegpt",
                                   "columns": {"messages": "messages"},
                                   "tags": {"role_tag": "role", "content_tag": "content",
                                            "user_tag": "user", "assistant_tag": "assistant"}}
    json.dump(info, open(f"{ROOT}/data/dataset_info.json", "w"), indent=2, ensure_ascii=False)
    print(f"registered + hash-validated {len(plan['adapters'])} unique adapter datasets")


def _train_args(ad):
    return {"per_device_train_batch_size": 4, "gradient_accumulation_steps": 4,
            "lora_alpha": 512, "num_train_epochs": 4, "effective_batch": 128,
            "dataset_key": ad["dataset_key"], "seed": ad["train_seed"]}


def cmd_write_train_manifest(a):
    plan = json.load(open(a.plan)); ad = plan["adapters"][a.aid]
    man = {"adapter_id": a.aid, "subset_jsonl": ad["subset_jsonl"],
           "subset_sha256": ad["subset_sha256"], "dataset_key": ad["dataset_key"],
           "train_seed": ad["train_seed"], "base_model": a.base_model,
           "train_args": _train_args(ad), "run_plan_sha256": fsha(a.plan),
           "master_manifest_sha256": fsha(a.master) if a.master and os.path.exists(a.master) else None,
           "git_commit": git_commit(),
           "adapter_sha256": fsha(f"{a.adapter_dir}/adapter_model.safetensors")}
    json.dump(man, open(f"{a.adapter_dir}/train_manifest.json", "w"), indent=2)
    print(f"wrote {a.adapter_dir}/train_manifest.json")


def cmd_check_train(a):
    mf = f"{a.adapter_dir}/train_manifest.json"
    ck = f"{a.adapter_dir}/adapter_model.safetensors"
    if not (os.path.exists(mf) and os.path.exists(ck)):
        raise SystemExit(1)
    m = json.load(open(mf)); plan = json.load(open(a.plan)); ad = plan["adapters"][a.aid]
    ok = (m.get("subset_sha256") == ad["subset_sha256"] and m.get("train_seed") == ad["train_seed"]
          and m.get("dataset_key") == ad["dataset_key"]
          and m.get("adapter_sha256") == fsha(ck))
    raise SystemExit(0 if ok else 1)


def cmd_find_eval(a):
    fs = glob.glob(f"{a.eval_dir}/**/results_*.json", recursive=True)
    if len(fs) != 1:
        raise SystemExit(1)   # require exactly one authoritative file
    print(fs[0])


def cmd_write_eval_manifest(a):
    fs = glob.glob(f"{a.eval_dir}/**/results_*.json", recursive=True)
    assert len(fs) == 1, f"expected exactly 1 results file, found {len(fs)}"
    import torch, transformers
    try:
        import lm_eval; lmev = lm_eval.__version__
    except Exception:
        lmev = "unknown"
    try:
        import peft; peftv = peft.__version__
    except Exception:
        peftv = "unknown"
    try:
        import accelerate; accv = accelerate.__version__
    except Exception:
        accv = "unknown"
    man = {"adapter_id": a.aid, "result_path": fs[0], "result_sha256": fsha(fs[0]),
           "adapter_sha256": fsha(f"{a.adapter_dir}/adapter_model.safetensors"),
           "tasks": ["mmlu_stem", "mmlu_humanities"], "num_fewshot": 5,
           "versions": {"lm_eval": lmev, "transformers": transformers.__version__,
                        "peft": peftv, "accelerate": accv, "torch": torch.__version__}}
    json.dump(man, open(f"{a.eval_dir}/eval_manifest.json", "w"), indent=2)
    print(f"wrote {a.eval_dir}/eval_manifest.json -> {fs[0]}")


def cmd_check_eval(a):
    mf = f"{a.eval_dir}/eval_manifest.json"
    if not os.path.exists(mf):
        raise SystemExit(1)
    m = json.load(open(mf))
    rp = m.get("result_path", "")
    ok = (os.path.exists(rp) and fsha(rp) == m.get("result_sha256")
          and m.get("adapter_sha256") == fsha(f"{a.adapter_dir}/adapter_model.safetensors"))
    raise SystemExit(0 if ok else 1)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ["register", "write_train_manifest", "check_train", "find_eval",
                 "write_eval_manifest", "check_eval"]:
        p = sub.add_parser(name)
        p.add_argument("--plan"); p.add_argument("--aid"); p.add_argument("--adapter_dir")
        p.add_argument("--eval_dir"); p.add_argument("--master"); p.add_argument("--base_model")
    a = ap.parse_args()
    {"register": cmd_register, "write_train_manifest": cmd_write_train_manifest,
     "check_train": cmd_check_train, "find_eval": cmd_find_eval,
     "write_eval_manifest": cmd_write_eval_manifest, "check_eval": cmd_check_eval}[a.cmd](a)


if __name__ == "__main__":
    main()

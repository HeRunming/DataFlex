#!/usr/bin/env python3
"""Resumable driver for the Llama-3.2-3B second model-stack confirmation (choice_0814_3).

24 adapters = 3 draws x 2 SFT seeds x {dsmc, first_rr, second_rr, randk}, plus one shared no-SFT
reference. Launched only after engineering gates 1-4 were green
(`experiments/less_aligned/llama32_gates_3_4.json`, ALL_GATES_PASS true).

Phases: materialize subsets -> loader check -> train -> eval. Every phase is idempotent and the
state file is rewritten after each cell, so an interrupted run resumes without repeating work.

ACCURACY POLICY: identical to the Llama-2 arm. Accuracy is written to disk but NOT summarized or
compared until all 24 cells are done, so no interim number can influence the protocol. All four
pre-registered outcomes are reportable and none may trigger tuning.
"""
import argparse, hashlib, json, os, subprocess, sys, time

ROOT = "/jizhicfs/karonhe/DataFlex_fa"
EXP = f"{ROOT}/experiments/less_aligned"
SAVES = "/jizhicfs/karonhe/dataflex_saves"
SUBSETS = f"{SAVES}/sft_subsets"
PY = "/jizhicfs/karonhe/envs/dataflex-fa/bin/python"
ENVBIN = "/jizhicfs/karonhe/envs/dataflex-fa/bin"
BASE = "/jizhicfs/karonhe/models/modelscope/LLM-Research/Llama-3___2-3B"
CAND_JSONL = f"{ROOT}/data/less_train_all.jsonl"
STATE = f"{EXP}/llama32_full_run_state.json"

METHODS = ["dsmc", "first_rr", "second_rr", "randk"]
DRAWS = [0, 1, 2]
SEEDS = [42, 1]
K = 2707
N_POOL = 270679


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def key_of(d, m):
    return f"l32_draw{d}_{m}"


def sel_path(d, m):
    """Llama-3.2's own selections for the targeted arms; the FROZEN Llama-2 indices for Random-K."""
    if m == "randk":
        return f"{SAVES}/selbbhx_draw{d}_randk/step_1.json"
    return f"{SAVES}/sel_llama32_draw{d}_{m}/step_1.json"


def load_state():
    if os.path.exists(STATE):
        return json.load(open(STATE))
    cells = []
    for d in DRAWS:
        for m in METHODS:
            for s in SEEDS:
                cells.append({"adapter_id": f"l32_draw{d}_{m}_seed{s}", "draw": d, "method": m,
                              "train_seed": s, "dataset_key": key_of(d, m),
                              "sft_out": f"{SAVES}/sft_results/l32_draw{d}_{m}_seed{s}",
                              "trained": False, "evaluated": False})
    return {"arm": "Llama-3.2-3B second model-stack confirmation",
            "base_model": BASE, "n_cells": len(cells), "budget_K": K,
            "accuracy_policy": ("accuracy is written to disk but NOT summarized or compared until all "
                                "24 cells complete, so no interim number can influence the protocol"),
            "materialized": False, "loader_checked": False, "cells": cells, "base_eval": False}


def save_state(st):
    json.dump(st, open(STATE, "w"), indent=2)


def materialize(st):
    """Write each selection as a sharegpt jsonl and register it. Fail-loud on row counts."""
    os.makedirs(SUBSETS, exist_ok=True)
    info_p = f"{ROOT}/data/dataset_info.json"
    info = json.load(open(info_p))
    rows = None
    rec = {}
    for d in DRAWS:
        for m in METHODS:
            sp = sel_path(d, m)
            if not os.path.exists(sp):
                raise SystemExit(f"missing selection {sp}")
            idx = json.load(open(sp))["indices"]
            if len(idx) != K or len(set(idx)) != K:
                raise SystemExit(f"draw{d}/{m}: {len(idx)} indices ({len(set(idx))} unique), want {K}")
            if min(idx) < 0 or max(idx) >= N_POOL:
                raise SystemExit(f"draw{d}/{m}: index out of range")
            if rows is None:
                # readlines(), NOT read().splitlines(): splitlines() also breaks on \x0b \x0c \x1c
                # \x1d \x1e \x85    , which occur inside JSON string values in this pool
                # and yield 274,187 "lines" for 270,679 records -- silently shifting every index.
                rows = open(CAND_JSONL).readlines()
                if len(rows) != N_POOL:
                    raise SystemExit(f"pool has {len(rows)} rows, expected {N_POOL}")
            jl = f"{SUBSETS}/{key_of(d, m)}_sel.jsonl"
            with open(jl, "w") as f:                     # selection order = faithful selector record
                for i in idx:
                    f.write(rows[i].rstrip("\n") + "\n")
            n = sum(1 for l in open(jl) if l.strip())
            if n != K:
                raise SystemExit(f"draw{d}/{m}: wrote {n} rows, expected {K}")
            info[key_of(d, m)] = {"file_name": jl, "formatting": "sharegpt",
                                  "columns": {"messages": "messages"},
                                  "tags": {"role_tag": "role", "content_tag": "content",
                                           "user_tag": "user", "assistant_tag": "assistant"}}
            rec[key_of(d, m)] = {"jsonl": jl, "n": n, "jsonl_sha256": sha_file(jl),
                                 "selection": sp,
                                 "reused_llama2_indices": m == "randk"}
            print(f"[materialize] {key_of(d, m):24s} {n} rows  {rec[key_of(d,m)]['jsonl_sha256'][:12]}")
    json.dump(info, open(info_p, "w"), indent=2, ensure_ascii=False)
    st["subsets"] = rec
    st["materialized"] = True
    save_state(st)


def loader_check(st):
    """Load every key through the REAL LlamaFactory SFT path under the llama3 template."""
    import warnings
    warnings.filterwarnings("ignore")
    from transformers import AutoTokenizer
    from llamafactory.data.loader import get_dataset
    from llamafactory.data.template import get_template_and_fix_tokenizer
    from llamafactory.hparams import get_train_args
    out = {}
    for d in DRAWS:
        for m in METHODS:
            k = key_of(d, m)
            args = {"model_name_or_path": BASE, "stage": "sft", "do_train": True,
                    "finetuning_type": "lora", "dataset": k, "template": "llama3",
                    "cutoff_len": 2048, "overwrite_cache": True, "preprocessing_num_workers": 16,
                    "output_dir": f"/tmp/l32_loadchk_{k}", "report_to": "none"}
            ma, da, ta, _, _ = get_train_args(args)
            tok_mod = {"tokenizer": AutoTokenizer.from_pretrained(BASE)}
            tmpl = get_template_and_fix_tokenizer(tok_mod["tokenizer"], da)
            ds = get_dataset(tmpl, ma, da, ta, "sft", **tok_mod)
            n = len(ds["train_dataset"])
            if n != K:
                raise SystemExit(f"loader check FAILED for {k}: {n} examples, expected {K}")
            out[k] = n
            print(f"[loader] {k:24s} {n} examples OK")
    st["loader_check"] = out
    st["loader_checked"] = True
    save_state(st)


def train_cell(c):
    """Frozen SFT recipe: the SAME base YAML and the SAME CLI overrides as the Llama-2 36-cell arm.
    Only model_name_or_path and template differ (verified when the config was emitted). Nothing is
    re-tuned for the 3B model -- re-tuning would confound the model axis with a tuning axis."""
    env = dict(os.environ, PATH=f"{ENVBIN}:{os.environ['PATH']}",
               HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1",
               CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7")
    log = f"{SAVES}/logs/l32_sft_{c['adapter_id']}.log"
    with open(log, "w") as f:
        rc = subprocess.call(
            [f"{ENVBIN}/dataflex-cli", "train",
             "experiments/less_aligned/configs/train_llama32_lora.yaml",
             f"dataset={c['dataset_key']}", f"output_dir={c['sft_out']}",
             f"seed={c['train_seed']}",
             "per_device_train_batch_size=4", "gradient_accumulation_steps=4",
             "lora_alpha=512", "num_train_epochs=4"],
            cwd=ROOT, env=env, stdout=f, stderr=subprocess.STDOUT)
    ok = rc == 0 and os.path.exists(f"{c['sft_out']}/adapter_model.safetensors")
    meta = {}
    if ok:
        ts = json.load(open(f"{c['sft_out']}/trainer_state.json"))
        meta = {"global_step": ts["global_step"], "epoch": round(ts["epoch"], 3),
                "adapter_sha256": sha_file(f"{c['sft_out']}/adapter_model.safetensors")}
    return ok, log, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="all", choices=["materialize", "loader", "train", "all"])
    a = ap.parse_args()
    st = load_state()

    gates = json.load(open(f"{EXP}/llama32_gates_3_4.json"))
    if not gates.get("ALL_GATES_PASS"):
        raise SystemExit("refusing to run: engineering gates are not green")
    st["gates_verified"] = True

    if not st["materialized"]:
        materialize(st)
    if a.phase == "materialize":
        return
    if not st["loader_checked"]:
        loader_check(st)
    if a.phase == "loader":
        return

    todo = [c for c in st["cells"] if not c["trained"]]
    print(f"[train] {len(todo)} of {len(st['cells'])} cells remaining", flush=True)
    for c in todo:
        t0 = time.time()
        print(f"[train] {c['adapter_id']}", flush=True)
        ok, log, meta = train_cell(c)
        if not ok:
            print(f"[train] FAILED {c['adapter_id']}; see {log}", file=sys.stderr)
            save_state(st)
            sys.exit(1)
        c["trained"] = True
        c.update(meta)
        c["train_minutes"] = round((time.time() - t0) / 60, 1)
        save_state(st)
        print(f"[train] done {c['adapter_id']} in {c['train_minutes']} min", flush=True)
    print(f"[train] all {len(st['cells'])} adapters trained")


if __name__ == "__main__":
    main()

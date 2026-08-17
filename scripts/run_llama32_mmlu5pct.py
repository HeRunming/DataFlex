#!/usr/bin/env python3
"""Driver for the Llama-3.2-3B x MMLU 5% arm (stop-rule amendment #1, advice_0817_2).

Pre-registered in `prereg_llama32_mmlu5pct.md`, which was committed BEFORE any computation.

Phases: select -> materialize -> loader -> train -> eval. Every phase is idempotent and the state
file is rewritten after each unit, so an interruption resumes without repeating work.

Design (frozen, reproducing the historical MMLU design with only the stack changed)
  K = 13,533 (5%)
  ten target draws stem80_draw{0..4} + hum80_draw{0..4}
  draw index -> training seed {0:42, 1:1, 2:2, 3:3, 4:4}   (the ORIGINAL mapping)
  methods DSMC / First-RR / Second-RR / Random-K + one shared no-SFT reference
  40 analysis cells but 35 UNIQUE adapters: Random-K is target-independent, so one Random subset
  is shared between the STEM and HUM directions of each draw index (randk_drawidx{0..4}).

Historical protocol values recovered from the frozen artifacts, NOT carried over from the BBH arm:
  RR perm_seed = 3000 + draw_index   (BBH used 6000 + d)
  target-gradient cutoff_len = 2048  (BBH used 3072)
  DSMC endpoint = select_moment_mmd.py --alpha 0.0

ACCURACY POLICY: results are written to disk but NOT summarized or compared until all 35 adapters
and the base reference are evaluated. All four pre-registered outcomes are reportable and none may
trigger tuning, a 1% follow-up, or a new method.
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
CAND_CACHE = f"{SAVES}/llama32_less_output/train/1/all_projected_grads.pt"
STATE = f"{EXP}/llama32_mmlu5pct_run_state.json"

TARGETED = ["dsmc", "first_rr", "second_rr"]
DIRECTIONS = ["stem80", "hum80"]
IDX = [0, 1, 2, 3, 4]
SEED_OF = {0: 42, 1: 1, 2: 2, 3: 3, 4: 4}
K = 13533
N_POOL = 270679
N_MMLU_SUBTASKS = 57


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def draw(direction, i):
    return f"{direction}80_draw{i}" if not direction.endswith("80") else f"{direction}_draw{i}"


def tgt_cache(d):
    return f"{SAVES}/l32_{d}_output/target/1/all_projected_grads.pt"


def load_state():
    if os.path.exists(STATE):
        return json.load(open(STATE))
    cells, adapters = [], {}
    for i in IDX:
        s = SEED_OF[i]
        for direction in DIRECTIONS:
            d = f"{direction}_draw{i}"
            for m in TARGETED:
                aid = f"l32_{d}_{m}_seed{s}"
                cells.append({"draw": d, "direction": direction, "idx": i, "method": m,
                              "train_seed": s, "adapter_id": aid,
                              "dataset_key": f"l32_{d}_{m}_sel"})
                adapters[aid] = {"dataset_key": f"l32_{d}_{m}_sel", "train_seed": s}
            # Random-K: ONE adapter per draw index, shared by both directions
            aid = f"l32_randk_drawidx{i}_seed{s}"
            cells.append({"draw": d, "direction": direction, "idx": i, "method": "randk",
                          "train_seed": s, "adapter_id": aid,
                          "dataset_key": f"l32_randk_drawidx{i}_sel"})
            adapters[aid] = {"dataset_key": f"l32_randk_drawidx{i}_sel", "train_seed": s}
    for aid, a in adapters.items():
        a.update({"adapter_id": aid, "sft_out": f"{SAVES}/sft_results/{aid}",
                  "eval_out": f"{SAVES}/eval_results/{aid}",
                  "trained": False, "evaluated": False})
    return {"arm": "Llama-3.2-3B x MMLU 5% (stop-rule amendment #1)",
            "prereg": "experiments/less_aligned/prereg_llama32_mmlu5pct.md",
            "base_model": BASE, "budget_K": K,
            "n_analysis_cells": len(cells), "n_unique_adapters": len(adapters),
            "accuracy_policy": ("written to disk but NOT summarized or compared until all 35 adapters "
                                "plus the base reference are evaluated"),
            "selected": False, "materialized": False, "loader_checked": False,
            "cells": cells, "adapters": adapters, "base_eval": False}


def save_state(st):
    json.dump(st, open(STATE, "w"), indent=2)


def select(st):
    """Run the three target-aware selectors on Llama-3.2's own MMLU target gradients."""
    rec = {}
    for i in IDX:
        for direction in DIRECTIONS:
            d = f"{direction}_draw{i}"
            T = tgt_cache(d)
            if not os.path.exists(T):
                raise SystemExit(f"missing target cache {T}")
            for m in TARGETED:
                out = f"{SAVES}/sel_l32_{d}_{m}"
                sp = f"{out}/step_1.json"
                if not os.path.exists(sp):
                    if m == "dsmc":
                        cmd = [PY, "scripts/select_moment_mmd.py", "--alpha", "0.0"]
                    else:
                        cmd = [PY, "scripts/select_round_robin.py",
                               "--order", "first" if m == "first_rr" else "second",
                               "--perm_seed", str(3000 + i)]      # MMLU seed, NOT BBH's 6000+d
                    log = f"{SAVES}/logs/sel_l32_{d}_{m}.log"
                    print(f"[select] {d} {m}", flush=True)
                    with open(log, "w") as lf:
                        rc = subprocess.call(
                            cmd + ["--train_grads", CAND_CACHE, "--target_grads", T,
                                   "--out_cache_dir", out, "--num_select", str(K)],
                            cwd=ROOT, stdout=lf, stderr=subprocess.STDOUT,
                            env=dict(os.environ, CUDA_VISIBLE_DEVICES="0",
                                     HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1"))
                    if rc != 0 or not os.path.exists(sp):
                        raise SystemExit(f"selection FAILED {d}/{m}; see {log}")
                j = json.load(open(sp))
                idx, mt = j["indices"], j["metric"]
                if len(idx) != K or len(set(idx)) != K:
                    raise SystemExit(f"{d}/{m}: {len(idx)} indices ({len(set(idx))} unique), want {K}")
                # assert the recovered historical protocol, not just the budget
                if m == "dsmc" and mt.get("alpha") != 0.0:
                    raise SystemExit(f"{d}/dsmc: alpha={mt.get('alpha')}, want 0.0")
                if m != "dsmc" and mt.get("perm_seed") != 3000 + i:
                    raise SystemExit(f"{d}/{m}: perm_seed={mt.get('perm_seed')}, want {3000+i}")
                if os.path.realpath(mt.get("train_grads", "")) != os.path.realpath(CAND_CACHE):
                    raise SystemExit(f"{d}/{m}: wrong candidate cache {mt.get('train_grads')}")
                if os.path.realpath(mt.get("target_grads", "")) != os.path.realpath(T):
                    raise SystemExit(f"{d}/{m}: wrong target cache {mt.get('target_grads')}")
                rec[f"{d}_{m}"] = {"selection": sp, "K": len(idx),
                                   "alpha": mt.get("alpha"), "perm_seed": mt.get("perm_seed"),
                                   "idx_sha256": hashlib.sha256(
                                       json.dumps(sorted(idx)).encode()).hexdigest()}
    st["selections"] = rec
    st["selected"] = True
    save_state(st)
    print(f"[select] {len(rec)}/30 selections verified at K={K}")


def materialize(st):
    """Write each subset as sharegpt jsonl and register it. Fail-loud on row counts."""
    os.makedirs(SUBSETS, exist_ok=True)
    info_p = f"{ROOT}/data/dataset_info.json"
    info = json.load(open(info_p))
    rows = None
    rec = {}

    def emit(key, idx):
        nonlocal rows
        if rows is None:
            # readlines(), NOT read().splitlines(): splitlines() also breaks on \x0b \x0c \x1c
            # \x1d \x1e \x85 and yields 274,187 "lines" for 270,679 records, shifting every index.
            rows = open(CAND_JSONL).readlines()
            if len(rows) != N_POOL:
                raise SystemExit(f"pool has {len(rows)} rows, expected {N_POOL}")
        jl = f"{SUBSETS}/{key}.jsonl"
        with open(jl, "w") as f:
            for j in idx:
                f.write(rows[j].rstrip("\n") + "\n")
        n = sum(1 for l in open(jl) if l.strip())
        if n != K:
            raise SystemExit(f"{key}: wrote {n} rows, expected {K}")
        info[key] = {"file_name": jl, "formatting": "sharegpt",
                     "columns": {"messages": "messages"},
                     "tags": {"role_tag": "role", "content_tag": "content",
                              "user_tag": "user", "assistant_tag": "assistant"}}
        rec[key] = {"jsonl": jl, "n": n, "jsonl_sha256": sha_file(jl)}
        print(f"[materialize] {key:34s} {n} rows  {rec[key]['jsonl_sha256'][:12]}", flush=True)

    for i in IDX:
        for direction in DIRECTIONS:
            d = f"{direction}_draw{i}"
            for m in TARGETED:
                emit(f"l32_{d}_{m}_sel",
                     json.load(open(f"{SAVES}/sel_l32_{d}_{m}/step_1.json"))["indices"])
        # Random-K reuses the EXACT frozen Llama-2 5% indices for this draw index, shared by
        # both directions -- the one constant data baseline across the two model stacks.
        src = f"{SUBSETS}/stem80_draw{i}_randk_sel.jsonl"
        lines = open(src).readlines()
        if len(lines) != K:
            raise SystemExit(f"frozen Llama-2 Random subset {src} has {len(lines)} rows, want {K}")
        key = f"l32_randk_drawidx{i}_sel"
        jl = f"{SUBSETS}/{key}.jsonl"
        open(jl, "w").writelines(lines)
        h_src, h_new = sha_file(src), sha_file(jl)
        if h_src != h_new:
            raise SystemExit(f"{key}: copy is not byte-identical to the frozen Llama-2 subset")
        info[key] = {"file_name": jl, "formatting": "sharegpt",
                     "columns": {"messages": "messages"},
                     "tags": {"role_tag": "role", "content_tag": "content",
                              "user_tag": "user", "assistant_tag": "assistant"}}
        rec[key] = {"jsonl": jl, "n": len(lines), "jsonl_sha256": h_new,
                    "reused_from": src, "byte_identical_to_llama2": True}
        print(f"[materialize] {key:34s} {len(lines)} rows  {h_new[:12]}  (frozen Llama-2 indices)",
              flush=True)

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
    for key in sorted(st["subsets"]):
        args = {"model_name_or_path": BASE, "stage": "sft", "do_train": True,
                "finetuning_type": "lora", "dataset": key, "template": "llama3",
                "cutoff_len": 2048, "overwrite_cache": True, "preprocessing_num_workers": 16,
                "output_dir": f"/tmp/l32m_{key}", "report_to": "none"}
        ma, da, ta, _, _ = get_train_args(args)
        tok = {"tokenizer": AutoTokenizer.from_pretrained(BASE)}
        tmpl = get_template_and_fix_tokenizer(tok["tokenizer"], da)
        n = len(get_dataset(tmpl, ma, da, ta, "sft", **tok)["train_dataset"])
        if n != K:
            raise SystemExit(f"loader check FAILED for {key}: {n} examples, expected {K}")
        out[key] = n
        print(f"[loader] {key:34s} {n} OK", flush=True)
    st["loader_check"] = out
    st["loader_checked"] = True
    save_state(st)


def train_phase(st):
    """Frozen SFT recipe: the same base YAML and CLI overrides as the Llama-2 MMLU arm."""
    env = dict(os.environ, PATH=f"{ENVBIN}:{os.environ['PATH']}",
               HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1",
               CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7")
    for v in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT",
              "LOCAL_WORLD_SIZE"):
        env.pop(v, None)
    todo = [a for a in st["adapters"].values() if not a["trained"]]
    print(f"[train] {len(todo)} of {len(st['adapters'])} adapters remaining", flush=True)
    for a in todo:
        t0 = time.time()
        print(f"[train] {a['adapter_id']}", flush=True)
        log = f"{SAVES}/logs/l32m_sft_{a['adapter_id']}.log"
        with open(log, "w") as f:
            rc = subprocess.call(
                [f"{ENVBIN}/dataflex-cli", "train",
                 "experiments/less_aligned/configs/train_llama32_lora.yaml",
                 f"dataset={a['dataset_key']}", f"output_dir={a['sft_out']}",
                 f"seed={a['train_seed']}",
                 "per_device_train_batch_size=4", "gradient_accumulation_steps=4",
                 "lora_alpha=512", "num_train_epochs=4"],
                cwd=ROOT, env=env, stdout=f, stderr=subprocess.STDOUT)
        if rc != 0 or not os.path.exists(f"{a['sft_out']}/adapter_model.safetensors"):
            print(f"[train] FAILED {a['adapter_id']}; see {log}", file=sys.stderr)
            save_state(st)
            sys.exit(1)
        ts = json.load(open(f"{a['sft_out']}/trainer_state.json"))
        a.update({"trained": True, "global_step": ts["global_step"],
                  "epoch": round(ts["epoch"], 3),
                  "adapter_sha256": sha_file(f"{a['sft_out']}/adapter_model.safetensors"),
                  "train_minutes": round((time.time() - t0) / 60, 1)})
        save_state(st)
        print(f"[train] done {a['adapter_id']} steps={ts['global_step']} "
              f"{a['train_minutes']} min", flush=True)
    shas = [a["adapter_sha256"] for a in st["adapters"].values() if a.get("adapter_sha256")]
    if len(set(shas)) != len(shas):
        raise SystemExit("adapter hashes are NOT distinct -- a subset or seed failed to vary")
    print(f"[train] all {len(st['adapters'])} adapters trained; hashes distinct")


def eval_result(out_dir):
    import glob
    for f in sorted(glob.glob(f"{out_dir}/**/results_*.json", recursive=True)):
        try:
            r = json.load(open(f))
        except Exception:
            continue
        if "mmlu" in r.get("results", {}):
            return f, r
    return None


def run_eval(aid, out_dir, peft, port, env_base):
    """lm_eval mmlu --num_fewshot 5, the historical MMLU evaluation, data-parallel over 8 GPUs."""
    os.makedirs(out_dir, exist_ok=True)
    ma = f"pretrained={BASE},dtype=bfloat16" + (f",peft={peft}" if peft else "")
    lf = open(f"{SAVES}/logs/l32m_eval_{aid}.log", "w")
    return subprocess.Popen(
        [f"{ENVBIN}/accelerate", "launch", "--num_processes", "8",
         "--main_process_port", str(port), "-m", "lm_eval", "--model", "hf",
         "--model_args", ma, "--tasks", "mmlu", "--num_fewshot", "5",
         "--batch_size", "16", "--output_path", out_dir],
        cwd=ROOT, env=env_base, stdout=lf, stderr=subprocess.STDOUT), lf


def record_eval(aid, out_dir, into):
    got = eval_result(out_dir)
    if got is None:
        raise SystemExit(f"EVAL FAILED {aid}; see {SAVES}/logs/l32m_eval_{aid}.log")
    path, r = got
    sub = {k: v for k, v in r["results"].items()
           if k.startswith("mmlu_") and k not in ("mmlu_stem", "mmlu_humanities",
                                                  "mmlu_social_sciences", "mmlu_other")}
    if len(sub) != N_MMLU_SUBTASKS:
        raise SystemExit(f"EVAL {aid}: {len(sub)} MMLU subtasks, expected {N_MMLU_SUBTASKS}")
    g = r["results"]
    into.update({"evaluated": True, "n_subtasks": len(sub),
                 "results_json": path, "results_sha256": sha_file(path),
                 # sealed for the FINAL analysis only; never compared here
                 "_mmlu_sealed": g["mmlu"]["acc,none"],
                 "_stem_sealed": g["mmlu_stem"]["acc,none"],
                 "_hum_sealed": g["mmlu_humanities"]["acc,none"]})
    print(f"    ok  {aid}  {len(sub)}/{N_MMLU_SUBTASKS} subtasks", flush=True)


def eval_phase(st):
    env_base = dict(os.environ, PATH=f"{ENVBIN}:{os.environ['PATH']}",
                    HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1",
                    CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7")
    for v in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT",
              "LOCAL_WORLD_SIZE"):
        env_base.pop(v, None)

    # the shared no-SFT reference first, so every later delta has its denominator on disk
    st.setdefault("base_cell", {"adapter_id": "l32m_base_noSFT",
                                "eval_out": f"{SAVES}/eval_results/l32m_base_noSFT",
                                "evaluated": False})
    b = st["base_cell"]
    if not b.get("evaluated"):
        if eval_result(b["eval_out"]) is None:
            print(f"[eval] {b['adapter_id']} (shared no-SFT reference)", flush=True)
            pr, lf = run_eval(b["adapter_id"], b["eval_out"], None, 29571, env_base)
            pr.wait()
            lf.close()
        record_eval(b["adapter_id"], b["eval_out"], b)
        st["base_eval"] = True
        save_state(st)

    todo = [a for a in st["adapters"].values() if eval_result(a["eval_out"]) is None]
    print(f"[eval] {len(st['adapters']) - len(todo)}/{len(st['adapters'])} done; "
          f"{len(todo)} to run", flush=True)
    for a in todo:
        print(f"[eval] {a['adapter_id']}", flush=True)
        pr, lf = run_eval(a["adapter_id"], a["eval_out"], a["sft_out"], 29572, env_base)
        pr.wait()
        lf.close()
        record_eval(a["adapter_id"], a["eval_out"], a)
        save_state(st)
    for a in st["adapters"].values():
        if not a.get("evaluated"):
            record_eval(a["adapter_id"], a["eval_out"], a)
            save_state(st)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="all",
                    choices=["select", "materialize", "loader", "train", "eval", "all"])
    a = ap.parse_args()
    st = load_state()

    if not st["selected"]:
        select(st)
    if a.phase == "select":
        return
    if not st["materialized"]:
        materialize(st)
    if a.phase == "materialize":
        return
    if not st["loader_checked"]:
        loader_check(st)
    if a.phase == "loader":
        return
    if a.phase in ("train", "all"):
        train_phase(st)
    if a.phase == "train":
        return
    eval_phase(st)
    ne = sum(1 for x in st["adapters"].values() if x.get("evaluated"))
    st["progress"] = {"trained": sum(1 for x in st["adapters"].values() if x["trained"]),
                      "evaluated": ne, "total": len(st["adapters"]),
                      "base_eval": bool(st.get("base_eval"))}
    save_state(st)
    print(f"\nPROGRESS: {st['progress']}")
    print("(comparative accuracy intentionally sealed until 35/35 + base)")


if __name__ == "__main__":
    main()

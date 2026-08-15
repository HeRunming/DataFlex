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
TASKS = "/jizhicfs/karonhe/DataFlex_fa/experiments/less_aligned/bbh_external_tasks"
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
N_SUBTASKS = 27          # frozen held-out BBH suite
N_EXAMPLES = 5209        # frozen 20/80 split


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


def eval_result(out_dir):
    """Return (path, parsed) for a completed lm-eval run, else None."""
    import glob
    for f in sorted(glob.glob(f"{out_dir}/**/results_*.json", recursive=True)):
        try:
            r = json.load(open(f))
        except Exception:
            continue
        if "bbh_external_heldout" in r.get("results", {}):
            return f, r
    return None


def run_one_eval(aid, out_dir, peft, gpus, env_base):
    """One lm-eval run. `peft=None` is the shared no-SFT reference; everything else -- task,
    include_path, batch size, dtype -- is byte-identical to the adapter evals."""
    os.makedirs(out_dir, exist_ok=True)
    ma = f"pretrained={BASE},dtype=bfloat16" + (f",peft={peft}" if peft else "")
    env = dict(env_base, CUDA_VISIBLE_DEVICES=gpus)
    for v in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT",
              "LOCAL_WORLD_SIZE"):
        env.pop(v, None)
    lf = open(f"{SAVES}/logs/l32_eval_{aid}.log", "w")
    return subprocess.Popen(
        [f"{ENVBIN}/lm_eval", "--model", "hf", "--model_args", ma,
         "--tasks", "bbh_external_heldout", "--include_path", TASKS,
         "--batch_size", "16", "--output_path", out_dir, "--log_samples"],
        cwd=ROOT, env=env, stdout=lf, stderr=subprocess.STDOUT), lf


def record_eval(aid, out_dir, into):
    """Fail-loud: refuse anything that is not exactly 27 subtasks / 5209 effective examples.

    Without these assertions a partially-registered task suite would be recorded as a valid
    result and would silently enter the final comparison."""
    got = eval_result(out_dir)
    if got is None:
        raise SystemExit(f"EVAL FAILED {aid}; see {SAVES}/logs/l32_eval_{aid}.log")
    path, r = got
    sub = {k: v for k, v in r["results"].items() if k.startswith("bbh_external_heldout_")}
    n_ex = sum(r["n-samples"][k]["effective"] for k in sub)
    if len(sub) != N_SUBTASKS:
        raise SystemExit(f"EVAL {aid}: {len(sub)} subtasks, expected exactly {N_SUBTASKS}")
    if n_ex != N_EXAMPLES:
        raise SystemExit(f"EVAL {aid}: {n_ex} effective examples, expected exactly {N_EXAMPLES}")
    into.update({"evaluated": True, "n_subtasks": len(sub), "n_examples": n_ex,
                 "results_json": path, "results_sha256": sha_file(path),
                 "_micro_sealed": r["results"]["bbh_external_heldout"]["exact_match,get-answer"]})
    print(f"    ok  {aid}  {len(sub)}/{N_SUBTASKS} subtasks, {n_ex}/{N_EXAMPLES} examples",
          flush=True)


def base_eval_phase(st, gpus, env_base):
    """The 25th pre-registered evaluation: the SHARED no-SFT reference for this model stack.

    The prereg says "24 adapters + 1 shared no-SFT reference", and every method is reported as a
    delta against the model's OWN base, since cross-model absolute accuracy is not comparable.
    The driver previously evaluated only the 24 adapter cells, so this was a real gap."""
    st.setdefault("base_cell", {"adapter_id": "l32_base_noSFT",
                                "eval_out": f"{SAVES}/eval_results/l32_base_noSFT",
                                "evaluated": False,
                                "note": "no PEFT adapter; identical task/include_path/batch/dtype"})
    b = st["base_cell"]
    if eval_result(b["eval_out"]) is None:
        print(f"[eval] {b['adapter_id']} (shared no-SFT reference) on GPUs {gpus}", flush=True)
        pr, lf = run_one_eval(b["adapter_id"], b["eval_out"], None, gpus, env_base)
        pr.wait()
        lf.close()
    record_eval(b["adapter_id"], b["eval_out"], b)
    st["base_eval"] = True
    save_state(st)


def eval_phase(st, groups):
    """Frozen held-out BBH suite, identical task config and batch size to the Llama-2 arm.

    Per-cell micro aggregates are stored under `_micro_sealed` and deliberately NOT compared or
    printed here -- the comparison happens only after 24/24 train and 24/24 eval."""
    for c in st["cells"]:
        c.setdefault("eval_out", f"{SAVES}/eval_results/{c['adapter_id']}")
    env_base = dict(os.environ, PATH=f"{ENVBIN}:{os.environ['PATH']}",
                    HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1")

    # the shared no-SFT reference first, so every later delta has its denominator on disk
    if not st.get("base_eval"):
        base_eval_phase(st, groups[0], env_base)

    todo = [c for c in st["cells"] if eval_result(c["eval_out"]) is None]
    print(f"[eval] {len(st['cells']) - len(todo)}/{len(st['cells'])} adapter evals complete; "
          f"{len(todo)} to run ({len(groups)} at a time)", flush=True)
    for i in range(0, len(todo), len(groups)):
        procs = []
        for c, g in zip(todo[i:i + len(groups)], groups):
            print(f"[eval] {c['adapter_id']} on GPUs {g}", flush=True)
            procs.append((c,) + run_one_eval(c["adapter_id"], c["eval_out"], c["sft_out"],
                                             g, env_base))
        for c, pr, lf in procs:
            pr.wait()
            lf.close()
            record_eval(c["adapter_id"], c["eval_out"], c)
            save_state(st)
    # cells already on disk from an earlier run still need their assertions enforced
    for c in st["cells"]:
        if not c.get("evaluated"):
            record_eval(c["adapter_id"], c["eval_out"], c)
            save_state(st)


def recipe_readback(st):
    """Verify all 24 adapters from what is ON DISK, not from the CLI strings we passed.

    code_review_0815: the base YAML still says alpha 256 / batch 16 / accum 8 / 3 epochs and the
    driver overrides those on the command line, so the only trustworthy evidence is the written
    trainer_state and adapter_config. Fail-loud before any unseal."""
    import glob
    out, ok = {}, True
    for c in st["cells"]:
        e = {}
        ts_p = f"{c['sft_out']}/trainer_state.json"
        ac_p = f"{c['sft_out']}/adapter_config.json"
        ad_p = f"{c['sft_out']}/adapter_model.safetensors"
        if not (os.path.exists(ts_p) and os.path.exists(ac_p) and os.path.exists(ad_p)):
            out[c["adapter_id"]] = {"MISSING_ARTIFACTS": True, "pass": False}
            ok = False
            continue
        ts, ac = json.load(open(ts_p)), json.load(open(ac_p))
        e["global_step"] = ts["global_step"]
        e["epoch"] = round(ts["epoch"], 3)
        e["steps_ok"] = ts["global_step"] == 84
        # 3.847, NOT 4.0: with 2707 examples and effective batch 128 (4 per_device x 4 accum x 8
        # GPUs), 4 epochs of drop-last batching gives floor(2707/128)*4 = 84 steps, which the
        # trainer reports as 84*128/2707 -> 3.847. This is the SAME value the Llama-2 36-cell arm
        # recorded for its 84-step cells, so asserting 4.0 would fail on correct training. Checked
        # against the Llama-2 figure rather than a guess.
        e["epoch_ok"] = abs(ts["epoch"] - 3.847) < 0.01
        e["r"] = ac.get("r")
        e["lora_alpha"] = ac.get("lora_alpha")
        e["lora_dropout"] = ac.get("lora_dropout")
        e["target_modules"] = sorted(ac.get("target_modules") or [])
        e["recipe_ok"] = (ac.get("r") == 128 and ac.get("lora_alpha") == 512
                          and abs((ac.get("lora_dropout") or 0) - 0.05) < 1e-9
                          and e["target_modules"] == ["k_proj", "o_proj", "q_proj", "v_proj"])
        e["train_seed_expected"] = c["train_seed"]
        e["adapter_sha256"] = sha_file(ad_p)
        e["adapter_sha_present"] = bool(e["adapter_sha256"])
        e["pass"] = bool(e["steps_ok"] and e["epoch_ok"] and e["recipe_ok"])
        ok = ok and e["pass"]
        out[c["adapter_id"]] = e
    # every cell must have a DISTINCT adapter: identical hashes would mean a subset or seed
    # silently failed to vary
    shas = [v.get("adapter_sha256") for v in out.values() if v.get("adapter_sha256")]
    uniq = len(set(shas)) == len(shas)
    st["recipe_readback"] = {"cells": out, "n_cells": len(out),
                             "all_pass": ok, "adapter_hashes_unique": uniq,
                             "expected": {"global_step": 84, "epoch": 3.847, "r": 128,
                                          "lora_alpha": 512, "lora_dropout": 0.05,
                                          "target_modules": ["k_proj", "o_proj", "q_proj",
                                                             "v_proj"]},
                             "why": ("the CLI overrides alpha/batch/accum/epochs, so only the "
                                     "written trainer_state and adapter_config are authoritative")}
    save_state(st)
    if not (ok and uniq):
        raise SystemExit("RECIPE READBACK FAILED; see llama32_full_run_state.json")
    print(f"[readback] {len(out)}/{len(out)} adapters: 84 steps, r128/alpha512/dropout0.05/qkvo, "
          f"all adapter hashes distinct", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="all",
                    choices=["materialize", "loader", "train", "eval", "readback", "all"])
    ap.add_argument("--eval_gpus", default="0,1|2,3|4,5|6,7",
                    help="GPU groups run concurrently, mirroring the Llama-2 arm")
    a = ap.parse_args()
    st = load_state()

    gates = json.load(open(f"{EXP}/llama32_gates_3_4.json"))
    if not gates.get("ALL_GATES_PASS"):
        raise SystemExit("refusing to run: engineering gates are not green")
    st["gates_verified"] = True

    if a.phase == "readback":
        recipe_readback(st)
        return

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
    recipe_readback(st)
    if a.phase == "train":
        return

    eval_phase(st, a.eval_gpus.split("|"))
    nt = sum(1 for c in st["cells"] if c["trained"])
    ne = sum(1 for c in st["cells"] if c.get("evaluated"))
    nb = 1 if st.get("base_eval") else 0
    st["progress"] = {"trained": nt, "evaluated": ne, "total": len(st["cells"])}
    save_state(st)
    print(f"\nPROGRESS: trained {nt}/{len(st['cells'])}  adapter evals {ne}/{len(st['cells'])}"
          f"  base eval {nb}/1  (total evaluations {ne + nb}/{len(st['cells']) + 1})")
    print("(comparative accuracy intentionally sealed until 24/24 and 24/24)")


if __name__ == "__main__":
    main()

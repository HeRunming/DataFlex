#!/usr/bin/env python3
"""Drive the remaining BBH adapters to 36/36 train + 36/36 eval. Resumable and fail-loud.

Reads `bbh_external_run_plan.json` and, for each cell, trains then evaluates unless a VALIDATED artifact
already exists (validated-skip, not existence-skip: a truncated run is retried rather than accepted).

Frozen recipe, taken from the driver overrides that the completed MMLU arm actually executed
(`resolved_run_provenance.json`): per_device 4 x accum 4 x 8 GPUs = eff. batch 128, lora_alpha 512,
4 epochs, cutoff 2048, dropout 0.05, lr 2e-5 linear, warmup 0.03, bf16.

Deliberately does NOT compute or print any accuracy comparison: comparative accuracy stays sealed until
36/36 is complete. Per-cell micro aggregates are recorded in the state file for the final analysis only.
"""
import argparse, glob, hashlib, json, os, subprocess, time

ROOT = "/jizhicfs/karonhe/DataFlex_fa"
EXP = f"{ROOT}/experiments/less_aligned"
SAVES = "/jizhicfs/karonhe/dataflex_saves"
ENVBIN = "/jizhicfs/karonhe/envs/dataflex-fa/bin"
BASE = "/jizhicfs/karonhe/models/shakechen/Llama-2-7b-hf"
TASKS = f"{EXP}/bbh_external_tasks"
STEPS, N_SUB, N_EX = 84, 27, 5209
STATE = f"{EXP}/bbh_full_run_state.json"


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def train_ok(ad):
    """Validated: adapter present AND trainer_state shows the full 84 steps."""
    a, t = f"{ad}/adapter_model.safetensors", f"{ad}/trainer_state.json"
    if not (os.path.exists(a) and os.path.exists(t)):
        return False
    try:
        return json.load(open(t))["global_step"] == STEPS
    except Exception:
        return False


def eval_result(ev):
    """Return the newest COMPLETE results json (27 subtasks, 5209 examples), else None."""
    for p in sorted(glob.glob(f"{ev}/*/results_*.json"), reverse=True):
        try:
            r = json.load(open(p))
            sub = {k: v for k, v in r["results"].items() if k.startswith("bbh_external_heldout_")}
            n = sum(r["n-samples"][k]["effective"] for k in sub)
            if len(sub) == N_SUB and n == N_EX and "bbh_external_heldout" in r["results"]:
                return p, r
        except Exception:
            continue
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["train", "eval", "both"], default="both")
    ap.add_argument("--eval_gpus", default="0,1,2,3|4,5,6,7",
                    help="pipe-separated GPU groups; evals run concurrently, one per group")
    ap.add_argument("--only", default=None, help="comma list of adapter_ids")
    args = ap.parse_args()

    plan = json.load(open(f"{EXP}/bbh_external_run_plan.json"))
    cells = plan["cells"]
    if args.only:
        want = set(args.only.split(","))
        cells = [c for c in cells if c["adapter_id"] in want]
    state = json.load(open(STATE)) if os.path.exists(STATE) else {"cells": {}}
    os.makedirs(f"{SAVES}/logs", exist_ok=True)

    # ---------------- TRAIN ----------------
    if args.phase in ("train", "both"):
        todo = [c for c in cells if not train_ok(c["sft_out"])]
        print(f"[train] {len(cells) - len(todo)}/{len(cells)} already validated; {len(todo)} to train")
        env = dict(os.environ, PATH=f"{ENVBIN}:{os.environ['PATH']}",
                   CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7",
                   HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1")
        for i, c in enumerate(todo, 1):
            aid, out = c["adapter_id"], c["sft_out"]
            log = f"{SAVES}/logs/{aid}_train.log"
            print(f"[train {i}/{len(todo)}] {aid} (dataset={c['dataset_key']} seed={c['train_seed']})",
                  flush=True)
            t0 = time.time()
            with open(log, "w") as lf:
                rc = subprocess.run(
                    [f"{ENVBIN}/dataflex-cli", "train",
                     "experiments/less_aligned/configs/train_llama7b_lora.yaml",
                     f"dataset={c['dataset_key']}", f"output_dir={out}",
                     f"seed={c['train_seed']}",
                     "per_device_train_batch_size=4", "gradient_accumulation_steps=4",
                     "lora_alpha=512", "num_train_epochs=4"],
                    cwd=ROOT, env=env, stdout=lf, stderr=subprocess.STDOUT).returncode
            if rc != 0 or not train_ok(out):
                raise SystemExit(f"TRAIN FAILED {aid} (rc={rc}); see {log}")
            ts = json.load(open(f"{out}/trainer_state.json"))
            state["cells"].setdefault(aid, {}).update({
                "train_ok": True, "global_step": ts["global_step"], "epoch": round(ts["epoch"], 3),
                "adapter_sha256": sha_file(f"{out}/adapter_model.safetensors"),
                "train_minutes": round((time.time() - t0) / 60, 1)})
            json.dump(state, open(STATE, "w"), indent=2)
            print(f"    ok  steps={ts['global_step']}  {state['cells'][aid]['train_minutes']} min",
                  flush=True)

    # ---------------- EVAL ----------------
    if args.phase in ("eval", "both"):
        groups = args.eval_gpus.split("|")
        todo = [c for c in cells if eval_result(c["eval_out"]) is None]
        print(f"[eval] {len(cells) - len(todo)}/{len(cells)} already complete; {len(todo)} to run "
              f"({len(groups)} at a time)")
        for i in range(0, len(todo), len(groups)):
            batch = todo[i:i + len(groups)]
            procs = []
            for c, g in zip(batch, groups):
                aid = c["adapter_id"]
                os.makedirs(c["eval_out"], exist_ok=True)
                log = f"{SAVES}/logs/{aid}_eval.log"
                print(f"[eval] {aid} on GPUs {g}", flush=True)
                env = dict(os.environ, PATH=f"{ENVBIN}:{os.environ['PATH']}",
                           CUDA_VISIBLE_DEVICES=g, HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1")
                lf = open(log, "w")
                procs.append((c, subprocess.Popen(
                    [f"{ENVBIN}/lm_eval", "--model", "hf",
                     "--model_args", f"pretrained={BASE},peft={c['sft_out']},dtype=bfloat16",
                     "--tasks", "bbh_external_heldout", "--include_path", TASKS,
                     "--batch_size", "16", "--output_path", c["eval_out"], "--log_samples"],
                    cwd=ROOT, env=env, stdout=lf, stderr=subprocess.STDOUT), lf))
            for c, p, lf in procs:
                p.wait()
                lf.close()
                aid = c["adapter_id"]
                got = eval_result(c["eval_out"])
                if got is None:
                    raise SystemExit(f"EVAL FAILED {aid}; see {SAVES}/logs/{aid}_eval.log")
                rp, r = got
                sub = {k: v for k, v in r["results"].items() if k.startswith("bbh_external_heldout_")}
                state["cells"].setdefault(aid, {}).update({
                    "eval_ok": True, "n_subtasks": len(sub),
                    "n_examples": sum(r["n-samples"][k]["effective"] for k in sub),
                    "results_json": rp, "results_sha256": sha_file(rp),
                    # recorded for the FINAL analysis only; no comparison is computed here
                    "_micro_sealed": r["results"]["bbh_external_heldout"]["exact_match,get-answer"]})
                json.dump(state, open(STATE, "w"), indent=2)
                print(f"    ok  {aid}  27/27 subtasks, 5209 examples", flush=True)

    done_t = sum(1 for c in plan["cells"] if train_ok(c["sft_out"]))
    done_e = sum(1 for c in plan["cells"] if eval_result(c["eval_out"]) is not None)
    state["progress"] = {"trained": done_t, "evaluated": done_e, "total": len(plan["cells"])}
    state["accuracy_policy"] = ("per-cell micro aggregates are recorded under `_micro_sealed` for the "
                                "final analysis only. No comparison is computed or printed until "
                                "36/36 train and 36/36 eval are complete.")
    json.dump(state, open(STATE, "w"), indent=2)
    print(f"\nPROGRESS: trained {done_t}/{len(plan['cells'])}  evaluated {done_e}/{len(plan['cells'])}")
    print("(comparative accuracy intentionally sealed)")


if __name__ == "__main__":
    main()

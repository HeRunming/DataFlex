#!/usr/bin/env python3
"""Emit the authoritative RESOLVED run provenance (artifact audit items C, D, E).

Why this exists: the repository could not previously be used to rebuild an actual run from
YAML + manifest alone.
  C. `train_llama7b_lora.yaml` carries older nominal values (alpha 256, per-device batch 16,
     accum 8, 3 epochs) while `run_pilot_sft.sh` overrides them (alpha 512, batch 4, accum 4,
     4 epochs). We therefore export the RESOLVED config per run family instead of trusting the YAML.
  D. The equal-step arm injects `max_steps=420` through the shell (`TRAIN_EXTRA`), which the train
     manifest did not record. We recover the ACTUAL `global_step`/`epoch` from each run's
     `trainer_state.json` — the ground truth — and hash that file.
  E. Warm-up `checkpoint-1692` provenance was spread over several configs with differing nominal
     values. We emit one authoritative record (resolved args, world size, global batch, adapter and
     optimizer hashes, environment).

Read-only over existing artifacts; writes one JSON. No training.
"""
import json, os, glob, hashlib, subprocess
import torch, transformers

SAVES = "/jizhicfs/karonhe/dataflex_saves"
ROOT = "/jizhicfs/karonhe/DataFlex_fa"
WARMUP = f"{SAVES}/sft_results/warmup_seed42/checkpoint-1692"
OUT = f"{ROOT}/experiments/less_aligned/resolved_run_provenance.json"

# Resolved SFT recipe actually used by run_pilot_sft.sh (driver overrides win over the YAML).
RESOLVED_SFT = {
    "config_file": "experiments/less_aligned/configs/train_llama7b_lora.yaml",
    "note": "YAML holds older nominal values; the driver CLI overrides are authoritative.",
    "yaml_nominal_superseded": {"lora_alpha": 256, "per_device_train_batch_size": 16,
                                "gradient_accumulation_steps": 8, "num_train_epochs": 3},
    "resolved_overrides": {"lora_alpha": 512, "per_device_train_batch_size": 4,
                           "gradient_accumulation_steps": 4, "num_train_epochs": 4},
    "not_overridden_by_driver": {"lora_dropout": 0.05,
                                 "note": "run_pilot_sft.sh does NOT pass lora_dropout, so the YAML "
                                         "value 0.05 is what the executed runs used."},
    "world_size": 8,
    "effective_batch": 4 * 4 * 8,   # per_device 4 x grad_accum 4 x 8 GPUs = 128
    "effective_batch_note": "per_device 4 x accum 4 x 8 GPUs = 128 examples per optimizer step",
    "lora": {"rank": 128, "alpha": 512, "dropout": 0.05,
             "target": "q_proj,k_proj,v_proj,o_proj"},
    "optimizer": {"lr": 2e-5, "scheduler": "linear", "warmup_ratio": 0.03},
    "precision": "bf16", "cutoff_len": 2048,
    "base_model": "/jizhicfs/karonhe/models/shakechen/Llama-2-7b-hf",
}

FAMILIES = {
    "5pct_fixed_epoch":  {"plan": "experiments/less_aligned/pilot_run_plan.json",
                          "train_extra": None, "expect_steps": 420},
    "1pct_fixed_epoch":  {"plan": "experiments/less_aligned/pilot1pct_run_plan.json",
                          "train_extra": None, "expect_steps": 84},
    "1pct_equal_step":   {"plan": "experiments/less_aligned/pilot1pctES_run_plan.json",
                          "train_extra": "max_steps=420", "expect_steps": 420},
}


def fsha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    out = {
        "resolved_sft_recipe": RESOLVED_SFT,
        "env": {"torch": torch.__version__, "transformers": transformers.__version__,
                "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
                "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"},
        "git_commit": subprocess.check_output(["git", "-C", ROOT, "rev-parse", "--short", "HEAD"]).decode().strip(),
        "warmup_checkpoint": {},
        "run_families": {},
    }

    # ---- E. authoritative warm-up checkpoint record ----
    wc = {"path": WARMUP,
          "adapter_sha256": fsha(f"{WARMUP}/adapter_model.safetensors"),
          "optimizer_sha256": fsha(f"{WARMUP}/optimizer.pt"),
          "note": "Also reachable as sft_results/random_selected/checkpoint-1692 — verified "
                  "byte-identical (adapter + optimizer sha256 match)."}
    ts = f"{WARMUP}/trainer_state.json"
    if os.path.exists(ts):
        s = json.load(open(ts))
        wc.update({"global_step": s.get("global_step"), "epoch": s.get("epoch"),
                   "trainer_state_sha256": fsha(ts)})
    cfgs = sorted(glob.glob(f"{ROOT}/experiments/less_aligned/configs/*warmup*.yaml"))
    wc["candidate_config_files"] = [os.path.relpath(c, ROOT) for c in cfgs]
    wc["config_caveat"] = ("Multiple warm-up configs exist with differing nominal alpha/dropout/batch. "
                           "The authoritative record of what produced this checkpoint is the hashes + "
                           "trainer_state above, not any single YAML.")
    out["warmup_checkpoint"] = wc

    # ---- C + D. per-family resolved config and ACTUAL step counts ----
    for fam, spec in FAMILIES.items():
        planp = f"{ROOT}/{spec['plan']}"
        if not os.path.exists(planp):
            continue
        plan = json.load(open(planp))
        rec = {"plan": spec["plan"], "plan_sha256": fsha(planp),
               "budget_K": plan.get("budget"), "tag": plan.get("tag"),
               "n_cells": plan.get("n_cells"), "n_unique_adapters": plan.get("n_unique_adapters"),
               "train_extra_injected": spec["train_extra"],
               "expected_optimizer_steps": spec["expect_steps"],
               "actual_steps_observed": {}}
        steps_seen = {}
        for aid in plan["adapters"]:
            cell = [c for c in plan["cells"] if c["adapter_id"] == aid][0]
            d = cell.get("sft_out") or f"{SAVES}/sft_results/pilot_{aid}"
            tsf = f"{d}/trainer_state.json"
            if os.path.exists(tsf):
                s = json.load(open(tsf))
                gs, ep = s.get("global_step"), round(s.get("epoch", 0), 2)
                steps_seen.setdefault((gs, ep), []).append(aid)
        rec["actual_steps_observed"] = {f"global_step={k[0]},epoch={k[1]}": len(v)
                                        for k, v in steps_seen.items()}
        rec["all_adapters_same_step_count"] = (len(steps_seen) == 1)
        rec["step_count_matches_expected"] = all(k[0] == spec["expect_steps"] for k in steps_seen)
        # hash one representative trainer_state as the ground-truth record
        if steps_seen:
            rep_aid = list(steps_seen.values())[0][0]
            cell = [c for c in plan["cells"] if c["adapter_id"] == rep_aid][0]
            d = cell.get("sft_out") or f"{SAVES}/sft_results/pilot_{rep_aid}"
            rec["representative_trainer_state"] = {
                "adapter_id": rep_aid, "path": f"{d}/trainer_state.json",
                "sha256": fsha(f"{d}/trainer_state.json")}
        out["run_families"][fam] = rec
        print(f"[{fam}] K={rec['budget_K']} adapters={rec['n_unique_adapters']} "
              f"train_extra={rec['train_extra_injected']} steps={rec['actual_steps_observed']} "
              f"matches_expected={rec['step_count_matches_expected']}")

    json.dump(out, open(OUT, "w"), indent=2)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()

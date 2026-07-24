#!/usr/bin/env python3
"""
Generate per-seed components.yaml + select configs for the multi-seed matrix,
and set up candidate-gradient symlinks so each (method,target) reuses the seed's
shared adam/sgd candidate gradient cache (only per-target grads are recomputed).

For seed s, gradient methods use caches:
  adam methods -> less_output_seed{s}/train/1/all_projected_grads.pt
  sgd  methods -> less_sgd_output_seed{s}/train/1/all_projected_grads.pt
Each (method,target) gets its own cache dir {method}_{target}_seed{s}_output with
train/1 SYMLINKED to the shared cache, so the selector skips the 270k recompute
and only computes the small target grads + greedy.

Usage: python gen_multiseed_configs.py --seed 1
Writes:
  src/dataflex/configs/components_seed{s}.yaml
  experiments/less_aligned/configs/multiseed/select_{method}_{target}_seed{s}.yaml
"""
import argparse
import os
import yaml
from pathlib import Path

ROOT = Path("/jizhicfs/karonhe/DataFlex_fa")
SAVES = Path("/jizhicfs/karonhe/dataflex_saves")
MODEL = "/jizhicfs/karonhe/models/shakechen/Llama-2-7b-hf"

TARGETS = ["bbh", "mmlu", "tydiqa"]
TARGET_DS = {"bbh": "bbh_target_100", "mmlu": "mmlu_target", "tydiqa": "tydiqa_target"}

# method -> (registry, gradient_type, shared_cache_base, kernel_extra)
GRAD_METHODS = {
    "less_sgd":         ("less", "sgd",  "less_sgd_output", {}),
    "less_adam":        ("less", "adam", "less_output",     {}),
    "mmd_grad_rbf_sgd": ("mmd",  "sgd",  "less_sgd_output", {"kernel_type": "grad_rbf", "target_gradient_type": "same"}),
    "mmd_grad_rbf_adam":("mmd",  "adam", "less_output",     {"kernel_type": "grad_rbf", "target_gradient_type": "sgd"}),
    "mmd_grad_cov_sgd": ("mmd",  "sgd",  "less_sgd_output", {"kernel_type": "grad_cov", "target_gradient_type": "same"}),
    "mmd_grad_cov_adam":("mmd",  "adam", "less_output",     {"kernel_type": "grad_cov", "target_gradient_type": "sgd"}),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    args = ap.parse_args()
    s = args.seed
    ckpt = str(SAVES / f"sft_results/warmup_seed{s}/checkpoint-1692")

    comp = {"selectors": {}}
    cfg_dir = ROOT / "experiments/less_aligned/configs/multiseed"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    comp_file = ROOT / f"src/dataflex/configs/components_seed{s}.yaml"

    for method, (reg, gtype, shared_base, kextra) in GRAD_METHODS.items():
        shared_cache = SAVES / f"{shared_base}_seed{s}"
        for t in TARGETS:
            comp_name = f"{method}_{t}_seed{s}"
            cache_dir = SAVES / f"{comp_name}_output"
            # symlink train/1 to shared candidate grads (reuse; skip 270k recompute)
            (cache_dir / "train").mkdir(parents=True, exist_ok=True)
            link = cache_dir / "train" / "1"
            src = shared_cache / "train" / "1"
            if not link.exists() and not link.is_symlink():
                try:
                    os.symlink(src, link)
                except FileExistsError:
                    pass
            params = {
                "cache_dir": str(cache_dir),
                "proj_dim": 8192, "save_interval": 16, "seed": 123,
                "candidate_subsample": -1, "greedy_device": "auto",
                "gradient_type": gtype, "sigma": None,
            }
            params.update(kextra)
            comp["selectors"][comp_name] = {"name": reg, "params": params}

            # select yaml
            select = {
                "model_name_or_path": MODEL, "adapter_name_or_path": ckpt,
                "trust_remote_code": True, "stage": "sft", "do_train": True,
                "finetuning_type": "lora", "lora_rank": 128, "lora_alpha": 512,
                "lora_target": "q_proj,k_proj,v_proj,o_proj", "lora_dropout": 0.1,
                "dataset": "less_train_all", "template": "llama2", "cutoff_len": 2048,
                "overwrite_cache": True, "preprocessing_num_workers": 16,
                "output_dir": str(SAVES / f"less_aligned/{comp_name}"),
                "logging_steps": 10, "overwrite_output_dir": True, "save_steps": 99999,
                "report_to": "none", "per_device_train_batch_size": 1,
                "gradient_accumulation_steps": 1, "learning_rate": 2.0e-5,
                "num_train_epochs": 1.0, "bf16": True, "ddp_timeout": 180000000,
                "train_step": 2, "train_type": "dynamic_select",
                "components_cfg_file": f"src/dataflex/configs/components_seed{s}.yaml",
                "component_name": comp_name, "warmup_step": 1, "update_step": 1,
                "update_times": 1, "selection_ratio": 0.05,
                "optimizer_state_path": ckpt,
                "target_dataset": TARGET_DS[t], "eval_dataset": TARGET_DS[t],
                "eval_strategy": "no",
            }
            with open(cfg_dir / f"select_{comp_name}.yaml", "w") as f:
                yaml.safe_dump(select, f, sort_keys=False)

    with open(comp_file, "w") as f:
        yaml.safe_dump(comp, f, sort_keys=False)
    print(f"[gen] seed {s}: wrote {comp_file} ({len(comp['selectors'])} components) + select configs in {cfg_dir}")
    print(f"[gen] symlinked candidate grads from less_output_seed{s} / less_sgd_output_seed{s}")


if __name__ == "__main__":
    main()

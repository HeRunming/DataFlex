#!/usr/bin/env python3
"""Generate per-(target, method) components.yaml entries + select configs
with candidate-gradient/embedding cache REUSE for new LESS target tasks.

Methods (target-dependent, need new selection):
  Gradient-Adam : less_adam, mmd_grad_rbf_adam, mmd_grad_cov_adam
  Gradient-SGD  : less_sgd,  mmd_grad_rbf_sgd,  mmd_grad_cov_sgd
  Embedding     : mmd_emb_rbf, mmd_emb_rbf_stochastic

random / base are target-independent → handled separately (reuse model, re-eval).

For each (target, method) we:
  * point cache_dir to a per-target dir   (…_<target>_output)
  * symlink candidate train grads from the BBH run (target-independent)
  * recompute only the small target-side grads/embeddings
  * for emb methods, set target_embeddings_path to the new target .npy

Outputs:
  * appends entries to a generated components file (kept separate so we don't
    clobber the main one): src/dataflex/configs/components_targets.yaml
  * writes select configs under experiments/less_aligned/configs/targets/
"""
import os
import yaml
from pathlib import Path

ROOT = Path("/jizhicfs/karonhe/DataFlex")
SAVES = Path("/jizhicfs/karonhe/dataflex_saves")
EMB = SAVES / "embeddings"
WARMUP_CKPT = SAVES / "sft_results/random_selected/checkpoint-1692"
MODEL = "/jizhicfs/karonhe/models/shakechen/Llama-2-7b-hf"

TARGETS = {
    "mmlu":   {"dataset": "mmlu_target",   "target_emb": str(EMB / "target_mmlu.npy")},
    "tydiqa": {"dataset": "tydiqa_target", "target_emb": str(EMB / "target_tydiqa.npy")},
}

# method -> (registry_name, base_cache (BBH run for candidate-grad reuse), kind, extra params)
# kind: "grad_adam" | "grad_sgd" | "emb" | "emb_stoch"
METHODS = {
    "less_adam":            ("less", "less_output",            "grad_adam", {"gradient_type": "adam"}),
    "mmd_grad_rbf_adam":    ("mmd",  "mmd_grad_rbf_output",    "grad_adam", {"kernel_type": "grad_rbf", "gradient_type": "adam", "target_gradient_type": "same"}),
    "mmd_grad_cov_adam":    ("mmd",  "mmd_grad_cov_output",    "grad_adam", {"kernel_type": "grad_cov", "gradient_type": "adam", "target_gradient_type": "same"}),
    "less_sgd":             ("less", "less_sgd_output",        "grad_sgd",  {"gradient_type": "sgd"}),
    "mmd_grad_rbf_sgd":     ("mmd",  "mmd_grad_rbf_sgd_output","grad_sgd",  {"kernel_type": "grad_rbf", "gradient_type": "sgd", "target_gradient_type": "same"}),
    "mmd_grad_cov_sgd":     ("mmd",  "mmd_grad_cov_sgd_output","grad_sgd",  {"kernel_type": "grad_cov", "gradient_type": "sgd", "target_gradient_type": "same"}),
    "mmd_emb_rbf":          ("mmd",  None,                     "emb",       {"kernel_type": "emb_rbf", "stochastic_eps": 0.0}),
    "mmd_emb_rbf_stochastic":("mmd", None,                     "emb_stoch", {"kernel_type": "emb_rbf", "stochastic_eps": 0.01}),
}

COMPONENTS = {"selectors": {}}
CONFIG_DIR = ROOT / "experiments/less_aligned/configs/targets"
CONFIG_DIR.mkdir(parents=True, exist_ok=True)


def make_component(target, method):
    reg, base_cache, kind, extra = METHODS[method]
    comp_name = f"{method}_{target}"
    cache_dir = str(SAVES / f"{method}_{target}_output")
    params = {
        "cache_dir": cache_dir,
        "proj_dim": 8192,
        "save_interval": 16,
        "seed": 123 if "grad" in kind or method.startswith("less") else 42,
        "candidate_subsample": -1,
        "greedy_device": "auto",
    }
    params.update(extra)
    if kind in ("emb", "emb_stoch"):
        params["candidate_embeddings_path"] = str(EMB / "candidate_270k.npy")
        params["target_embeddings_path"] = TARGETS[target]["target_emb"]
        params["sigma"] = None
        params["seed"] = 42
    else:
        params["sigma"] = None
    COMPONENTS["selectors"][comp_name] = {"name": reg, "params": params}
    return comp_name, cache_dir, base_cache, kind


def make_select_config(target, method, comp_name, kind):
    tinfo = TARGETS[target]
    cfg = {
        "model_name_or_path": MODEL,
        "trust_remote_code": True,
        "stage": "sft", "do_train": True, "finetuning_type": "lora",
        "lora_rank": 128, "lora_alpha": 512,
        "lora_target": "q_proj,k_proj,v_proj,o_proj", "lora_dropout": 0.1,
        "dataset": "less_train_all", "template": "llama2", "cutoff_len": 2048,
        "overwrite_cache": True, "preprocessing_num_workers": 16,
        "output_dir": str(SAVES / f"less_aligned/{comp_name}"),
        "logging_steps": 10, "overwrite_output_dir": True, "save_steps": 99999,
        "report_to": "none",
        "per_device_train_batch_size": 1, "gradient_accumulation_steps": 1,
        "learning_rate": 2.0e-5, "num_train_epochs": 1.0, "bf16": True,
        "ddp_timeout": 180000000, "train_step": 2,
        "train_type": "dynamic_select",
        "components_cfg_file": "src/dataflex/configs/components_targets.yaml",
        "component_name": comp_name,
        "warmup_step": 1, "update_step": 1, "update_times": 1, "selection_ratio": 0.05,
        "target_dataset": tinfo["dataset"], "eval_dataset": tinfo["dataset"],
        "eval_strategy": "no",
    }
    if kind == "grad_adam":
        cfg["adapter_name_or_path"] = str(WARMUP_CKPT)
        cfg["optimizer_state_path"] = str(WARMUP_CKPT)
    path = CONFIG_DIR / f"select_{comp_name}.yaml"
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, default_flow_style=False)
    return path


def main():
    manifest = []
    for target in TARGETS:
        for method in METHODS:
            comp_name, cache_dir, base_cache, kind = make_component(target, method)
            cfg_path = make_select_config(target, method, comp_name, kind)
            manifest.append({
                "target": target, "method": method, "comp_name": comp_name,
                "cache_dir": cache_dir, "base_cache": base_cache, "kind": kind,
                "config": str(cfg_path),
            })

    # Write components file
    comp_path = ROOT / "src/dataflex/configs/components_targets.yaml"
    with open(comp_path, "w") as f:
        yaml.safe_dump(COMPONENTS, f, sort_keys=False, default_flow_style=False)
    print(f"Wrote {len(COMPONENTS['selectors'])} component entries to {comp_path}")

    # Write manifest
    man_path = ROOT / "experiments/less_aligned/configs/targets/manifest.yaml"
    with open(man_path, "w") as f:
        yaml.safe_dump(manifest, f, sort_keys=False)
    print(f"Wrote manifest with {len(manifest)} (target,method) combos to {man_path}")
    for m in manifest:
        print(f"  {m['comp_name']:<35} kind={m['kind']:<10} base_cache={m['base_cache']}")


if __name__ == "__main__":
    main()

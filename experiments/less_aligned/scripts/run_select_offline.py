#!/usr/bin/env python3
"""
Phase D — Offline selection driver for LESS-aligned final-table experiment.

Loads the warmup checkpoint (Llama-2-7B + LoRA r=128 + AdamW), then for each of
the 16 selectors runs `selector.select()` over the 270K Tulu-V2 pool exactly
once and writes selected_indices.npy to <save_dir>/<method>/.

Designed to be invoked under `accelerate launch` / `torchrun` so that gradient
extraction is parallelised over all visible GPUs (the selectors call
`self.accelerator.prepare(dataloader)` which handles DDP transparently).

Usage:
    accelerate launch --num_processes 8 \
      experiments/less_aligned/scripts/run_select_offline.py \
      --warmup_ckpt /jizhicfs/karonhe/dataflex_saves/less_aligned/warmup_ckpt \
      --base_model /jizhicfs/karonhe/models/Llama-2-7b-hf \
      --dataset_path /jizhicfs/karonhe/DataFlex/data/tulu2_270k.json \
      --eval_dataset_path /jizhicfs/karonhe/less_data_zip/data/eval/mmlu/dev_alpaca.json \
      --save_dir /jizhicfs/karonhe/dataflex_saves/less_aligned/selections \
      --num_samples 13533 \
      --methods random_s42 less_s42 fisher_sft_s42 ...

Exit codes:
    0  all methods produced selected_indices.npy
    >0 any method failed (other methods still attempted)
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from accelerate import Accelerator
from datasets import Dataset


# Ensure DataFlex is importable
sys.path.insert(0, "/jizhicfs/karonhe/DataFlex/src")

# DataFlex / LLaMA-Factory imports
os.environ.setdefault("DISABLE_VERSION_CHECK", "1")

from dataflex.core.registry import REGISTRY  # noqa: E402

# Force selector module imports so registry is populated
import dataflex.train.selector.less_selector  # noqa: F401, E402
import dataflex.train.selector.spec_gcs_selector  # noqa: F401, E402
import dataflex.train.selector.fisher_sft_selector  # noqa: F401, E402
import dataflex.train.selector.negative_control_selectors  # noqa: F401, E402
import dataflex.train.selector.loss_selector  # noqa: F401, E402
import dataflex.train.selector.random_selector  # noqa: F401, E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_alpaca_dataset_as_hf(path: str, tokenizer, max_length: int = 1024,
                                cache_dir: str = None,
                                accelerator=None) -> Dataset:
    """Load alpaca-format JSON file and tokenize using LLaMA-Factory's alpaca
    template (must match the warmup/final-SFT runs exactly).

    Alpaca template structure:
        {system}### Instruction:\n{user}\n\n### Response:\n{response}<eos>\n\n
    where {system} = "Below is an instruction that describes a task. Write a
    response that appropriately completes the request.\n\n"
    and {user}    = "{instruction}\n{input}" (input dropped if empty).

    Loss is computed only on the response tokens (prompt is masked with -100).

    For multi-rank scenarios: rank 0 tokenizes once, saves to `cache_dir`, and
    other ranks load from that cached arrow dataset. This avoids 8 ranks each
    re-tokenizing 270K examples and the rendezvous timeout that follows.
    """
    SYSTEM = ("Below is an instruction that describes a task. Write a response "
              "that appropriately completes the request.\n\n")
    eos = tokenizer.eos_token or ""

    def _tokenize_all():
        with open(path, "r", encoding="utf-8") as f:
            records = json.load(f)
        input_ids_list, attn_list, labels_list = [], [], []
        for rec in records:
            instr = (rec.get("instruction") or "").strip()
            inp = (rec.get("input") or "").strip()
            out = (rec.get("output") or "").strip()
            user = instr if not inp else f"{instr}\n{inp}"
            prompt_text = f"{SYSTEM}### Instruction:\n{user}\n\n### Response:\n"
            full_text = prompt_text + out + eos + "\n\n"

            prompt_ids = tokenizer(prompt_text, truncation=True, max_length=max_length,
                                    add_special_tokens=True)["input_ids"]
            full_enc = tokenizer(full_text, truncation=True, max_length=max_length,
                                  add_special_tokens=True)
            ids = full_enc["input_ids"]
            attn = full_enc["attention_mask"]
            labels = list(ids)
            for i in range(min(len(prompt_ids), len(labels))):
                labels[i] = -100
            input_ids_list.append(ids)
            attn_list.append(attn)
            labels_list.append(labels)
        return Dataset.from_dict({
            "input_ids": input_ids_list,
            "attention_mask": attn_list,
            "labels": labels_list,
        })

    # Single-rank fast path (no caching needed)
    if accelerator is None or cache_dir is None or accelerator.num_processes == 1:
        return _tokenize_all()

    cache_path = Path(cache_dir)
    is_main = accelerator.is_main_process

    if is_main:
        if cache_path.exists() and (cache_path / "dataset_info.json").exists():
            print(f"[tokenize] cache hit: {cache_path}", flush=True)
        else:
            cache_path.mkdir(parents=True, exist_ok=True)
            print(f"[tokenize] rank 0 tokenizing {path} -> {cache_path}", flush=True)
            ds = _tokenize_all()
            print(f"[tokenize] saving {len(ds)} examples to {cache_path}", flush=True)
            ds.save_to_disk(str(cache_path))
            print(f"[tokenize] done", flush=True)
    accelerator.wait_for_everyone()
    return Dataset.load_from_disk(str(cache_path))


def load_method_configs(yaml_path: str, method_names):
    """Read components.yaml and return {method_name: (selector_name, params)} for
    each requested method."""
    with open(yaml_path, "r") as f:
        cfg = yaml.safe_load(f)
    selectors_cfg = cfg.get("selectors", cfg)  # support either nesting
    out = {}
    for name in method_names:
        if name not in selectors_cfg:
            raise KeyError(f"Method '{name}' not found in {yaml_path}")
        entry = selectors_cfg[name]
        out[name] = (entry["name"], entry.get("params", {}))
    return out


def reconstruct_optimizer_state(model, opt_state_dict):
    """Map serialized AdamW state (int-keyed) back to a {param_tensor: state} dict
    matching what the selectors expect from `optimizer.state`.

    PyTorch's optimizer.state_dict() saves:
        {"state": {0: {"step": ..., "exp_avg": ..., "exp_avg_sq": ...},
                    1: {...}, ...},
         "param_groups": [{"params": [0, 1, ...], ...}]}
    The integer keys correspond to *positions* in param_groups[0]["params"],
    which in turn maps to the order of trainable params in the model.
    """
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    state_int = opt_state_dict.get("state", opt_state_dict)
    out = {}
    # PyTorch saves state keyed by integers matching param order
    for idx, param in enumerate(trainable_params):
        if idx in state_int:
            out[param] = state_int[idx]
        elif str(idx) in state_int:
            out[param] = state_int[str(idx)]
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup_ckpt", required=True,
                        help="Directory containing adapter_model.safetensors + optimizer.pt")
    parser.add_argument("--base_model", required=True)
    parser.add_argument("--dataset_path", required=True,
                        help="alpaca-format JSON of the 270K pool")
    parser.add_argument("--eval_dataset_path", default=None,
                        help="Optional eval set (used by LESS as target tasks)")
    parser.add_argument("--save_dir", required=True)
    parser.add_argument("--components_yaml",
                        default="/jizhicfs/karonhe/DataFlex/src/dataflex/configs/components.yaml")
    parser.add_argument("--num_samples", type=int, default=13533,
                        help="5% of 270K = 13533 (LESS-paper default)")
    parser.add_argument("--methods", nargs="+", required=True)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--shared_grad_method", default="opt_gcs_logdet",
                        help="Method whose gradient computation we reuse across all "
                             "other methods (via symlink). Must be a real, registered method.")
    args = parser.parse_args()

    # Bump the distributed collective timeout to 4h. After gradient extraction,
    # rank 0 runs each per-method selection (logdet/spectral) solo while ranks
    # 1-7 sit at a barrier; the default 10-min NCCL watchdog would SIGABRT the
    # whole job mid-selection. 4h is plenty for all 16 methods.
    from datetime import timedelta
    from accelerate.utils import InitProcessGroupKwargs
    ipg = InitProcessGroupKwargs(timeout=timedelta(hours=4))
    accelerator = Accelerator(kwargs_handlers=[ipg])
    is_main = accelerator.is_main_process

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    if is_main:
        print(f"[start] save_dir={save_dir}", flush=True)
        print(f"[start] methods={args.methods}", flush=True)
        print(f"[start] num_samples={args.num_samples}", flush=True)

    # ---- 1. Load tokenizer + model + warmup adapter ----
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel

    if is_main:
        print(f"[load] tokenizer + base model: {args.base_model}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16, device_map={"": accelerator.device}
    )
    if is_main:
        print(f"[load] LoRA adapter from {args.warmup_ckpt}", flush=True)
    model = PeftModel.from_pretrained(base_model, args.warmup_ckpt, is_trainable=True)
    model.train()  # ensure grad flow

    # ---- 2. Restore optimizer state ----
    opt_pt = Path(args.warmup_ckpt) / "optimizer.pt"
    if not opt_pt.exists():
        if is_main:
            print(f"[error] missing {opt_pt}", flush=True)
        sys.exit(2)
    opt_state_dict = torch.load(opt_pt, map_location="cpu")
    optimizer_state_map = reconstruct_optimizer_state(model, opt_state_dict)
    if is_main:
        print(f"[load] reconstructed optimizer state for {len(optimizer_state_map)} params", flush=True)

    # ---- 3. Load 270K dataset ----
    if is_main:
        print(f"[load] dataset: {args.dataset_path}", flush=True)
    pool_cache = Path(args.save_dir) / "_tokenized_pool"
    dataset = load_alpaca_dataset_as_hf(
        args.dataset_path, tokenizer, max_length=args.max_length,
        cache_dir=str(pool_cache), accelerator=accelerator,
    )
    if is_main:
        print(f"[load] dataset size: {len(dataset)}", flush=True)

    eval_dataset = None
    if args.eval_dataset_path:
        if is_main:
            print(f"[load] eval dataset: {args.eval_dataset_path}", flush=True)
        eval_cache = Path(args.save_dir) / "_tokenized_eval"
        eval_dataset = load_alpaca_dataset_as_hf(
            args.eval_dataset_path, tokenizer, max_length=args.max_length,
            cache_dir=str(eval_cache), accelerator=accelerator,
        )
        if is_main:
            print(f"[load] eval size: {len(eval_dataset)}", flush=True)

    # Simple data collator (pad-to-longest)
    from transformers import DataCollatorForSeq2Seq
    data_collator = DataCollatorForSeq2Seq(tokenizer, padding="longest", label_pad_token_id=-100)

    # ---- 4. Compute shared gradients via the seed selector ----
    method_cfgs = load_method_configs(args.components_yaml, args.methods)
    if args.shared_grad_method not in method_cfgs:
        # Allow shared_grad_method to be a method we don't directly run
        method_cfgs.update(load_method_configs(args.components_yaml, [args.shared_grad_method]))

    shared_dir = save_dir / "_shared_grads"
    shared_dir.mkdir(parents=True, exist_ok=True)

    if is_main:
        print(f"\n[shared] computing Adam-preconditioned gradients via {args.shared_grad_method}", flush=True)
    seed_name, seed_params = method_cfgs[args.shared_grad_method]
    seed_params = dict(seed_params)
    seed_params["cache_dir"] = str(shared_dir)
    # Force adam preconditioning for share-grad pass
    seed_params["gradient_type"] = "adam"
    seed_runtime = dict(
        dataset=dataset, accelerator=accelerator,
        data_collator=data_collator, eval_dataset=eval_dataset,
    )
    seed_selector = REGISTRY.build(
        "selector", seed_name, runtime=seed_runtime, cfg=seed_params,
    )
    # Run select() once to populate gradient cache (we'll discard the chosen indices)
    _ = seed_selector.select(
        model=model, step_id=0, num_samples=args.num_samples,
        optimizer_state=optimizer_state_map,
        current_update_times=1, update_times=1,
    )
    accelerator.wait_for_everyone()

    # The gradient cache should now exist at shared_dir/gradients/step_0_<hash>/all_projected_grads.pt
    # Collect the actual grad dir for symlinking later
    shared_grad_subdirs = list((shared_dir / "gradients").glob("step_0_*"))
    if not shared_grad_subdirs:
        # LessSelector uses a slightly different layout (cache_dir/train/<step_id>/)
        shared_grad_subdirs = list((shared_dir / "train").glob("0*"))
    shared_grad_subdir = shared_grad_subdirs[0] if shared_grad_subdirs else None
    if is_main:
        print(f"[shared] gradient cache dir: {shared_grad_subdir}", flush=True)

    # ---- 5. Per-method selection ----
    failed = []
    for method_name in args.methods:
        method_dir = save_dir / method_name
        method_dir.mkdir(parents=True, exist_ok=True)
        out_path = method_dir / "selected_indices.npy"

        if out_path.exists():
            if is_main:
                print(f"[skip] {method_name}: indices already saved", flush=True)
            continue

        try:
            sel_name, params = method_cfgs[method_name]
            params = dict(params)
            params["cache_dir"] = str(method_dir)
            # Force Adam preconditioning to align with LESS paper
            if "gradient_type" in params:
                params["gradient_type"] = "adam"

            # Negative-control selectors (grad_norm_topk, random_subspace_logdet_*)
            # default to compute_own_grads=True, which would re-run the full 270K
            # gradient extraction (~3h each). Point them at the shared cache instead.
            if sel_name == "grad_norm_topk" or sel_name.startswith("random_subspace_logdet"):
                params["compute_own_grads"] = False
                params["source_grad_dirs"] = [str(shared_dir)]

            # Symlink shared gradient cache into this method's cache_dir so the
            # selector finds gradients on first .select() call
            if shared_grad_subdir is not None and is_main:
                _link_shared_grads(shared_dir, method_dir)
            accelerator.wait_for_everyone()

            if is_main:
                print(f"\n[run] {method_name} (selector={sel_name})", flush=True)

            method_runtime = dict(
                dataset=dataset, accelerator=accelerator,
                data_collator=data_collator, eval_dataset=eval_dataset,
            )
            selector = REGISTRY.build(
                "selector", sel_name, runtime=method_runtime, cfg=params,
            )
            indices = selector.select(
                model=model, step_id=0, num_samples=args.num_samples,
                optimizer_state=optimizer_state_map,
                current_update_times=1, update_times=1,
            )
            accelerator.wait_for_everyone()

            if is_main and indices:
                arr = np.array(sorted(set(int(i) for i in indices)), dtype=np.int64)
                np.save(out_path, arr)
                print(f"[done] {method_name}: saved {len(arr)} indices -> {out_path}",
                      flush=True)
        except Exception as e:
            if is_main:
                import traceback
                print(f"[fail] {method_name}: {e}", flush=True)
                traceback.print_exc()
            failed.append(method_name)
            continue

    if is_main:
        print(f"\n[summary] {len(args.methods) - len(failed)}/{len(args.methods)} OK; "
              f"failed: {failed}", flush=True)
    sys.exit(1 if failed else 0)


def _link_shared_grads(shared_dir: Path, method_dir: Path):
    """Symlink shared gradient cache into a method's cache_dir."""
    for sub in ["gradients", "train", "eval"]:
        src = shared_dir / sub
        if not src.exists():
            continue
        dst = method_dir / sub
        if dst.exists() or dst.is_symlink():
            continue
        try:
            dst.symlink_to(src, target_is_directory=True)
        except OSError:
            pass


if __name__ == "__main__":
    main()

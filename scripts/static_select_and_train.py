#!/usr/bin/env python3
"""
Static Selection + Training Pipeline for Paper Experiments.

This script implements the standard targeted instruction tuning workflow:
  1. Load candidate pool and target set
  2. Run offline selection (MMD, LESS-style, random, etc.)
  3. Save selected indices
  4. Train on the fixed selected subset

This aligns with the LESS / TSDS paper experimental protocol:
  - Selection is done ONCE, offline
  - Training uses a FIXED subset (no dynamic updates)
  - Evaluation is on held-out benchmarks

Usage:
    # Step 1: Select data
    python scripts/static_select_and_train.py select \
        --method mmd_grad_rbf \
        --candidate_data path/to/flan_v2.json \
        --target_data path/to/gsm8k_train_64.json \
        --model_name_or_path meta-llama/Llama-2-7b-hf \
        --selection_ratio 0.05 \
        --output_dir ./experiments/results/mmd_grad_rbf_gsm8k_5pct \
        --proj_dim 4096 \
        --seed 42

    # Step 2: Train on selected subset
    python scripts/static_select_and_train.py train \
        --base_config experiments/less_aligned/base_sft.yaml \
        --selected_indices ./experiments/results/mmd_grad_rbf_gsm8k_5pct/selected_indices.json \
        --output_dir ./experiments/results/mmd_grad_rbf_gsm8k_5pct/model

    # Or do both in one shot:
    python scripts/static_select_and_train.py pipeline \
        --method mmd_grad_rbf \
        --candidate_data path/to/flan_v2.json \
        --target_data path/to/gsm8k_train_64.json \
        --model_name_or_path meta-llama/Llama-2-7b-hf \
        --selection_ratio 0.05 \
        --base_config experiments/less_aligned/base_sft.yaml \
        --output_dir ./experiments/results/mmd_grad_rbf_gsm8k_5pct \
        --seed 42
"""

import argparse
import json
import os
import sys
import subprocess
import numpy as np
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Static Selection + Training Pipeline")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # SELECT command
    sel = subparsers.add_parser("select", help="Run offline data selection")
    sel.add_argument("--method", type=str, required=True,
                     choices=["random", "mmd_emb_rbf", "mmd_grad_rbf", "mmd_grad_cov",
                              "embedding_nn", "full"],
                     help="Selection method")
    sel.add_argument("--candidate_data", type=str, required=True,
                     help="Path to candidate pool (JSON/JSONL)")
    sel.add_argument("--target_data", type=str, required=True,
                     help="Path to target set (JSON/JSONL)")
    sel.add_argument("--model_name_or_path", type=str, default=None,
                     help="Model for gradient-based selection")
    sel.add_argument("--embed_model", type=str, default="sentence-transformers/all-MiniLM-L6-v2",
                     help="Embedding model for emb_rbf / embedding_nn")
    sel.add_argument("--selection_ratio", type=float, default=0.05,
                     help="Fraction of candidate pool to select (e.g., 0.05 = 5%%)")
    sel.add_argument("--num_select", type=int, default=None,
                     help="Explicit number to select (overrides selection_ratio)")
    sel.add_argument("--output_dir", type=str, required=True)
    sel.add_argument("--proj_dim", type=int, default=4096)
    sel.add_argument("--sigma", type=str, default="auto")
    sel.add_argument("--seed", type=int, default=42)

    # TRAIN command
    trn = subparsers.add_parser("train", help="Train on selected subset")
    trn.add_argument("--base_config", type=str, required=True,
                     help="Base LlamaFactory/DataFlex YAML config for SFT")
    trn.add_argument("--selected_indices", type=str, required=True,
                     help="Path to selected_indices.json from select step")
    trn.add_argument("--output_dir", type=str, required=True)
    trn.add_argument("--seed", type=int, default=42)

    # PIPELINE command (select + train)
    pipe = subparsers.add_parser("pipeline", help="Run select then train")
    pipe.add_argument("--method", type=str, required=True,
                      choices=["random", "mmd_emb_rbf", "mmd_grad_rbf", "mmd_grad_cov",
                               "embedding_nn", "full"])
    pipe.add_argument("--candidate_data", type=str, required=True)
    pipe.add_argument("--target_data", type=str, required=True)
    pipe.add_argument("--model_name_or_path", type=str, default=None)
    pipe.add_argument("--embed_model", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    pipe.add_argument("--selection_ratio", type=float, default=0.05)
    pipe.add_argument("--num_select", type=int, default=None)
    pipe.add_argument("--base_config", type=str, required=True)
    pipe.add_argument("--output_dir", type=str, required=True)
    pipe.add_argument("--proj_dim", type=int, default=4096)
    pipe.add_argument("--sigma", type=str, default="auto")
    pipe.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def count_samples(data_path):
    """Count samples in a JSON or JSONL file."""
    if data_path.endswith('.jsonl'):
        with open(data_path) as f:
            return sum(1 for _ in f)
    else:
        with open(data_path) as f:
            return len(json.load(f))


def run_selection(args):
    """Run offline data selection."""
    os.makedirs(args.output_dir, exist_ok=True)

    total_candidates = count_samples(args.candidate_data)
    num_select = args.num_select or int(total_candidates * args.selection_ratio)
    print(f"[Selection] Method: {args.method}")
    print(f"[Selection] Candidates: {total_candidates}, Selecting: {num_select} ({num_select/total_candidates*100:.1f}%)")

    if args.method == "random":
        rng = np.random.RandomState(args.seed)
        selected = rng.choice(total_candidates, size=num_select, replace=False).tolist()

    elif args.method == "full":
        selected = list(range(total_candidates))

    elif args.method == "mmd_emb_rbf":
        # Use offline MMD selector
        cmd = [
            sys.executable, "src/dataflex/offline_selector/offline_mmd_selector.py",
            "--candidate_path", args.candidate_data,
            "--query_path", args.target_data,
            "--embed_model", args.embed_model,
            "--save_dir", os.path.join(args.output_dir, "mmd_cache"),
            "--mode", "select",
            "--num_select", str(num_select),
            "--sigma", args.sigma,
        ]
        subprocess.run(cmd, check=True)
        selected = np.load(os.path.join(args.output_dir, "mmd_cache", "selected_indices.npy")).tolist()

    elif args.method == "embedding_nn":
        # Nearest neighbor to target (no redundancy penalty)
        cmd = [
            sys.executable, "src/dataflex/offline_selector/offline_mmd_selector.py",
            "--candidate_path", args.candidate_data,
            "--query_path", args.target_data,
            "--embed_model", args.embed_model,
            "--save_dir", os.path.join(args.output_dir, "nn_cache"),
            "--mode", "select",
            "--num_select", str(num_select),
            "--lambda_redundancy", "0.0",  # No redundancy = pure NN
            "--sigma", args.sigma,
        ]
        subprocess.run(cmd, check=True)
        selected = np.load(os.path.join(args.output_dir, "nn_cache", "selected_indices.npy")).tolist()

    elif args.method in ("mmd_grad_rbf", "mmd_grad_cov"):
        # Gradient-based selection requires model
        if not args.model_name_or_path:
            raise ValueError(f"--model_name_or_path required for {args.method}")
        # For gradient-based methods, use the dynamic selector in single-step mode
        # This runs a one-shot selection using the model's current gradients
        print(f"[Selection] Gradient method - using model: {args.model_name_or_path}")
        print(f"[Selection] NOTE: For paper experiments, consider running this via dataflex-cli")
        print(f"[Selection] with warmup_step=0, update_step={num_select}, update_times=1")
        # Placeholder - gradient selection needs the full DataFlex pipeline
        # For now, point users to use dataflex-cli train with static settings
        raise NotImplementedError(
            f"Gradient-based static selection ({args.method}) should be run via:\n"
            f"  dataflex-cli train <config.yaml>\n"
            f"with train_type=dynamic_select, warmup_step=0, update_times=1\n"
            f"The selected indices will be saved in the cache_dir."
        )
    else:
        raise ValueError(f"Unknown method: {args.method}")

    # Save selected indices
    output_path = os.path.join(args.output_dir, "selected_indices.json")
    with open(output_path, 'w') as f:
        json.dump({"indices": selected, "metadata": {
            "method": args.method,
            "num_candidates": total_candidates,
            "num_selected": len(selected),
            "selection_ratio": len(selected) / total_candidates,
            "seed": args.seed,
        }}, f, indent=2)
    print(f"[Selection] Saved {len(selected)} indices to {output_path}")
    return selected


def run_training(args):
    """Train on a fixed selected subset."""
    # Load selected indices
    with open(args.selected_indices) as f:
        data = json.load(f)
    indices = data["indices"] if isinstance(data, dict) else data

    print(f"[Training] Using {len(indices)} selected samples")
    print(f"[Training] Base config: {args.base_config}")
    print(f"[Training] Output: {args.output_dir}")

    # Call dataflex-cli / llamafactory-cli with static training
    # The selected indices will be used to create a subset dataset
    cmd = [
        sys.executable, "-m", "llamafactory.train.tuner",
        args.base_config,
        f"output_dir={args.output_dir}",
        f"seed={args.seed}",
    ]
    print(f"[Training] Command: {' '.join(cmd)}")
    # NOTE: This is a simplified version. Full implementation would need to
    # create a subset dataset file from selected_indices and pass it to LlamaFactory.
    print("[Training] NOTE: For full pipeline, create a subset JSON from selected_indices")
    print("[Training]       and update the dataset field in your training config.")


def main():
    args = parse_args()

    if args.command == "select":
        run_selection(args)
    elif args.command == "train":
        run_training(args)
    elif args.command == "pipeline":
        # Run both
        run_selection(args)
        # Auto-fill train args
        args.selected_indices = os.path.join(args.output_dir, "selected_indices.json")
        args.base_config = args.base_config
        run_training(args)
    else:
        print("Usage: python scripts/static_select_and_train.py {select|train|pipeline} ...")
        sys.exit(1)


if __name__ == "__main__":
    main()

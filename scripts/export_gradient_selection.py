#!/usr/bin/env python3
"""
Export dynamic selection results to standard format.

After running gradient-based selection via dataflex-cli (dynamic_select mode),
this script extracts selected indices from the selector cache and produces
the same output format as static_select_and_train.py:
  - selected_indices.json
  - selected_subset.json
  - selection_metadata.json
  - run_manifest.json

Usage:
    python scripts/export_gradient_selection.py \
        --candidate_data data/flan_v2_100k.json \
        --cache_dir ../dataflex_saves/mmd_grad_rbf_sgd_output \
        --output_dir experiments/less_aligned/results/mmd_grad_rbf_sgd/gsm8k/ratio_0.05/seed_42 \
        --method mmd_grad_rbf_sgd \
        --target_data data/gsm8k_train_64.json \
        --selection_ratio 0.05 \
        --seed 42
"""

import argparse
import json
import os
import subprocess
import numpy as np
from pathlib import Path
from glob import glob


def parse_args():
    parser = argparse.ArgumentParser(description="Export gradient selection results")
    parser.add_argument("--candidate_data", type=str, required=True,
                        help="Path to original candidate pool (JSON/JSONL)")
    parser.add_argument("--cache_dir", type=str, required=True,
                        help="Selector cache directory (contains step_*.json)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for exported results")
    parser.add_argument("--method", type=str, default="unknown",
                        help="Method name for metadata")
    parser.add_argument("--target_data", type=str, default="",
                        help="Target data path for metadata")
    parser.add_argument("--selection_ratio", type=float, default=0.0,
                        help="Selection ratio for metadata")
    parser.add_argument("--seed", type=int, default=42,
                        help="Seed for metadata")
    return parser.parse_args()


def load_data(path):
    if path.endswith('.jsonl'):
        data = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
        return data
    else:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)


def find_latest_step_file(cache_dir):
    """Find the latest step_*.json file in cache directory."""
    pattern = os.path.join(cache_dir, "step_*.json")
    files = glob(pattern)
    if not files:
        # Also search subdirectories
        pattern = os.path.join(cache_dir, "**", "step_*.json")
        files = glob(pattern, recursive=True)
    if not files:
        return None

    # Sort by step number (extract from filename)
    def get_step(f):
        name = os.path.basename(f)
        try:
            return int(name.replace("step_", "").replace(".json", ""))
        except ValueError:
            return -1

    files.sort(key=get_step)
    return files[-1]  # Latest step


def get_git_commit():
    """Get current git commit hash."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Step 1: Find and load selection results
    print(f"[Export] Searching for selection results in: {args.cache_dir}")
    step_file = find_latest_step_file(args.cache_dir)

    if step_file is None:
        print(f"[Export] ERROR: No step_*.json found in {args.cache_dir}")
        print(f"[Export] Make sure gradient selection has been run first.")
        return 1

    print(f"[Export] Found: {step_file}")
    with open(step_file, 'r') as f:
        step_data = json.load(f)

    # Extract indices (format: {"indices": [...], "metric": {...}} or just list)
    if isinstance(step_data, dict) and "indices" in step_data:
        indices = step_data["indices"]
    elif isinstance(step_data, list):
        indices = step_data
    else:
        print(f"[Export] ERROR: Unexpected format in {step_file}")
        return 1

    print(f"[Export] Loaded {len(indices)} selected indices")

    # Step 2: Load candidate data and extract subset
    print(f"[Export] Loading candidate data: {args.candidate_data}")
    candidate_data = load_data(args.candidate_data)
    total_candidates = len(candidate_data)

    # Validate indices
    invalid = [i for i in indices if i < 0 or i >= total_candidates]
    if invalid:
        print(f"[Export] WARNING: {len(invalid)} indices out of range [0, {total_candidates})")
        indices = [i for i in indices if 0 <= i < total_candidates]

    subset_data = [candidate_data[i] for i in indices]

    # Step 3: Save outputs
    # selected_indices.json
    indices_path = os.path.join(args.output_dir, "selected_indices.json")
    with open(indices_path, 'w') as f:
        json.dump({
            "indices": indices,
            "metadata": {
                "method": args.method,
                "num_candidates": total_candidates,
                "num_selected": len(indices),
                "selection_ratio": len(indices) / total_candidates if total_candidates > 0 else 0,
                "seed": args.seed,
                "source_step_file": step_file,
            }
        }, f, indent=2)
    print(f"[Export] Saved: {indices_path}")

    # selected_subset.json
    subset_path = os.path.join(args.output_dir, "selected_subset.json")
    with open(subset_path, 'w', encoding='utf-8') as f:
        json.dump(subset_data, f, ensure_ascii=False, indent=2)
    print(f"[Export] Saved: {subset_path} ({len(subset_data)} samples)")

    # selection_metadata.json
    meta_path = os.path.join(args.output_dir, "selection_metadata.json")
    metadata = {
        "method": args.method,
        "candidate_data": args.candidate_data,
        "target_data": args.target_data,
        "num_candidates": total_candidates,
        "num_selected": len(indices),
        "selection_ratio": args.selection_ratio,
        "seed": args.seed,
        "cache_dir": args.cache_dir,
    }
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"[Export] Saved: {meta_path}")

    # run_manifest.json
    manifest_path = os.path.join(args.output_dir, "run_manifest.json")
    manifest = {
        "git_commit": get_git_commit(),
        "method": args.method,
        "candidate_data": os.path.abspath(args.candidate_data),
        "target_data": os.path.abspath(args.target_data) if args.target_data else "",
        "selection_ratio": args.selection_ratio,
        "seed": args.seed,
        "selector_cache_dir": os.path.abspath(args.cache_dir),
        "selected_indices_path": os.path.abspath(indices_path),
        "selected_subset_path": os.path.abspath(subset_path),
        "output_dir": os.path.abspath(args.output_dir),
        "num_selected": len(indices),
        "num_candidates": total_candidates,
    }
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    print(f"[Export] Saved: {manifest_path}")

    # Token stats
    lengths = [len(str(item).split()) for item in subset_data]
    print(f"[Export] Token stats: total≈{sum(lengths)}, mean≈{np.mean(lengths):.0f}, "
          f"median≈{np.median(lengths):.0f}")

    print(f"[Export] Done. All outputs in: {args.output_dir}")
    return 0


if __name__ == "__main__":
    exit(main())

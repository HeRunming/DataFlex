#!/usr/bin/env python3
"""Set up cache-reuse selection for a new target task.

Candidate gradients (train/{step}/all_projected_grads.pt, ~8.3G each) and
candidate embeddings are TARGET-INDEPENDENT. When switching the selection
target (e.g. BBH -> MMLU), we only need to recompute the small target-side
features. This script creates a per-target cache directory that symlinks the
candidate gradients from the original (BBH) run, so the expensive 22-GPU-hour
candidate gradient computation is reused.

For gradient methods (less, mmd_grad_rbf, mmd_grad_cov):
  new_cache/train/1/all_projected_grads.pt  -> symlink to bbh run
  new_cache/train/1/*.pt (subsample etc)    -> symlink
  new_cache/target/  (left empty -> recomputed for new target)
  Also handles the .bak (sanitized) situation.

For emb methods: embeddings are passed via config paths, nothing to symlink.

Usage:
  python scripts/setup_target_cache.py --method less_adam --src_cache .../less_output \
      --dst_cache .../less_output_mmlu
"""
import argparse
import os
from pathlib import Path


def link_candidate_grads(src_cache: str, dst_cache: str):
    src = Path(src_cache)
    dst = Path(dst_cache)
    # train/{step}/ holds candidate grads; symlink the whole step dir contents
    src_train = src / "train"
    if not src_train.exists():
        raise FileNotFoundError(f"No train/ dir in {src_cache}")
    for step_dir in src_train.iterdir():
        if not step_dir.is_dir():
            continue
        dst_step = dst / "train" / step_dir.name
        dst_step.mkdir(parents=True, exist_ok=True)
        for f in step_dir.iterdir():
            link = dst_step / f.name
            if link.exists() or link.is_symlink():
                continue
            os.symlink(f.resolve(), link)
            print(f"  linked {link} -> {f.resolve()}")
    # Do NOT link target/ — that must be recomputed for the new target.
    print(f"[setup] candidate train grads linked into {dst_cache}; target/ left for recompute.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src_cache", required=True, help="source cache_dir (has train/{step}/all_projected_grads.pt)")
    ap.add_argument("--dst_cache", required=True, help="destination cache_dir for the new target")
    args = ap.parse_args()
    link_candidate_grads(args.src_cache, args.dst_cache)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Sanitize NaN in cached LESS/MMD gradient files.

For each .pt file containing projected gradients, replace NaN with 0.
This is needed because some samples produce NaN gradients under Adam
preconditioning (numerical instability when v ≈ 0 for some dims).

After cleanup, selectors will treat NaN rows as 0-vectors:
- LESS dot-product score = 0 (not selected)
- MMD-GradCov polynomial score = 0
- MMD-Grad-RBF distance from 0-vec is well-defined
"""
import sys
import torch
import numpy as np
from pathlib import Path

def sanitize(path: Path, dry_run: bool = False) -> int:
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(obj, dict):
        for k in list(obj.keys()):
            v = obj[k]
            if torch.is_tensor(v) and (v.dtype.is_floating_point):
                nan_mask = torch.isnan(v)
                inf_mask = torch.isinf(v)
                bad = (nan_mask | inf_mask)
                bad_count = int(bad.sum().item())
                if bad_count > 0:
                    print(f"  [{path.name}::{k}] bad: {bad_count}/{v.numel()} -> setting to 0")
                    if not dry_run:
                        v_new = v.clone()
                        v_new[bad] = 0.0
                        obj[k] = v_new
        n_bad_total = sum(int((torch.isnan(v) | torch.isinf(v)).sum().item())
                          for v in obj.values() if torch.is_tensor(v) and v.dtype.is_floating_point)
    elif torch.is_tensor(obj):
        nan_mask = torch.isnan(obj)
        inf_mask = torch.isinf(obj)
        bad = (nan_mask | inf_mask)
        n_bad_total = int(bad.sum().item())
        if n_bad_total > 0:
            print(f"  [{path.name}] bad: {n_bad_total}/{obj.numel()} -> setting to 0")
            if not dry_run:
                obj = obj.clone()
                obj[bad] = 0.0
    else:
        print(f"  [{path.name}] unsupported type {type(obj)}, skipping")
        return 0

    if not dry_run:
        backup = path.with_suffix(path.suffix + ".bak")
        if not backup.exists():
            path.rename(backup)
            print(f"  backup: {backup}")
        torch.save(obj, path)
        print(f"  saved: {path}")
    return n_bad_total


def main():
    targets = [
        "/jizhicfs/karonhe/dataflex_saves/less_output/train/1/all_projected_grads.pt",
        "/jizhicfs/karonhe/dataflex_saves/less_output/eval/1/all_projected_grads.pt",
        "/jizhicfs/karonhe/dataflex_saves/mmd_grad_rbf_output/train/1/all_projected_grads.pt",
        "/jizhicfs/karonhe/dataflex_saves/mmd_grad_rbf_output/target/1/all_projected_grads.pt",
        "/jizhicfs/karonhe/dataflex_saves/mmd_grad_cov_output/train/1/all_projected_grads.pt",
        "/jizhicfs/karonhe/dataflex_saves/mmd_grad_cov_output/target/1/all_projected_grads.pt",
    ]
    dry_run = "--dry" in sys.argv
    if dry_run:
        print("DRY RUN MODE")
    for t in targets:
        p = Path(t)
        if not p.exists():
            print(f"[skip] missing: {p}")
            continue
        print(f"=== {p} ===")
        sanitize(p, dry_run=dry_run)


if __name__ == "__main__":
    main()

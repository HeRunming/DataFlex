#!/usr/bin/env python3
"""
Mechanism diagnostics for selection sets: effective rank + redundancy of the
selected-set gradients. Tests the GradCov theory that distribution/covariance
matching spreads selection across more gradient directions (higher eff-rank,
lower pairwise redundancy) than mean-alignment (LESS) or metric-alignment (NICE).

Uses a pre-computed candidate gradient cache (L2-normalized, [N, proj_dim]) and
each method's selection step_1.json. No training needed.
"""
import argparse, json, glob, os
import numpy as np
import torch


def eff_rank(X):
    """exp(entropy of normalized covariance eigenvalues) via SVD singular values."""
    s = np.linalg.svd(X, compute_uv=False)
    ev = s ** 2
    ev = ev / ev.sum()
    return float(np.exp(-(ev * np.log(ev + 1e-12)).sum()))


def mean_pairwise_cos(X, cap=1000):
    sub = X[:cap]
    sim = sub @ sub.T
    n = sim.shape[0]
    return float((sim.sum() - np.trace(sim)) / (n * (n - 1)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grad_cache", required=True, help="all_projected_grads.pt [N, d]")
    ap.add_argument("--selections", nargs="+", required=True,
                    help="method:cache_dir pairs, e.g. less:/path/less_output")
    ap.add_argument("--subsample", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    G = torch.load(args.grad_cache, map_location="cpu").numpy().astype(np.float32)
    rng = np.random.RandomState(args.seed)
    print(f"candidate grads: {G.shape}")
    print(f"{'method':28s} {'eff_rank':10s} {'pairwise_cos':12s} {'n_sel'}")
    for spec in args.selections:
        name, cdir = spec.split(":", 1)
        step = os.path.join(cdir, "step_1.json")
        idx = json.load(open(step))["indices"]
        ii = np.array(idx)
        sel = ii if len(ii) <= args.subsample else rng.choice(ii, args.subsample, replace=False)
        X = G[sel]
        print(f"{name:28s} {eff_rank(X):<10.1f} {mean_pairwise_cos(X):<12.4f} {len(idx)}")


if __name__ == "__main__":
    main()

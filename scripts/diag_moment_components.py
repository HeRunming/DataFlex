#!/usr/bin/env python3
"""
Component-scale diagnostics for the Moment-MMD alpha sweep (review 0725, P1).
For each alpha's selection, on the shared gradient cache, compute WITHOUT training:
  - D1 = ||mu_S - mu_T||^2                 (first-order / mean-direction error)
  - D2 = ||M_S - M_T||_F^2                 (second-order / directional-moment error)
  - Jaccard overlap vs alpha=0 selection
  - effective rank of selected gradients
  - per-step marginal-score scale of the linear vs quadratic terms (on a subsample),
    to see whether the two components are on comparable numeric scales.
mu_P = E_P[u], M_P = E_P[u u^T]; u = unit projected gradient.
"""
import argparse, json, glob, os
import numpy as np
import torch


def load_idx(cache):
    return json.load(open(os.path.join(cache, "step_1.json")))["indices"]


def eff_rank(X):
    s = np.linalg.svd(X, compute_uv=False)
    ev = s ** 2; ev = ev / ev.sum()
    return float(np.exp(-(ev * np.log(ev + 1e-12)).sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_grads", required=True)
    ap.add_argument("--target_grads", required=True)
    ap.add_argument("--alphas", nargs="+", required=True)
    ap.add_argument("--cache_tmpl", required=True, help="e.g. /.../moment_a{a}_stem80_output")
    ap.add_argument("--subsample", type=int, default=4000)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    X = torch.load(args.train_grads, map_location="cpu").float()
    Tg = torch.load(args.target_grads, map_location="cpu").float()
    X = X / X.norm(dim=1, keepdim=True).clamp_min(1e-12)
    Tg = Tg / Tg.norm(dim=1, keepdim=True).clamp_min(1e-12)
    Xg = X.to(dev)
    mu_T = Tg.mean(0)                                  # (d,)
    M_T = (Tg.T @ Tg) / Tg.shape[0]                    # (d,d)
    D = X.shape[1]

    # per-step marginal scale at the FIRST greedy step (m=0): score = r_T(x).
    # linear term contribution = 0.5*(1+<x,mu_T>) mean/std; quad = <x,mu-of-outer>...
    # simpler: report the raw component target-relevances' spread across candidates.
    XmuT = (Xg @ mu_T.to(dev))                         # <x, mu_T>  (N,)  -> first-order signal
    # second-order relevance mean_t <x,t>^2 = x^T M_T x
    XM = (Xg @ M_T.to(dev))                            # (N,d)
    quad_rel = (XM * Xg).sum(1)                        # x^T M_T x   (N,)
    lin_rel = 0.5 * (1.0 + XmuT)                       # k_lin relevance
    print(f"[diag] candidate first-order rel: mean={lin_rel.mean():.4f} std={lin_rel.std():.4f}")
    print(f"[diag] candidate second-order rel: mean={quad_rel.mean():.4f} std={quad_rel.std():.4f}")
    print(f"[diag] std ratio (2nd/1st) = {float(quad_rel.std()/lin_rel.std()):.3f}  "
          f"(if !=1, components are on different scales -> alpha mixing is miscalibrated)")
    print()

    idx0 = set(load_idx(args.cache_tmpl.format(a="0.0")))
    rng = np.random.RandomState(0)
    Xn = X.numpy()
    muT = mu_T.numpy(); MT = M_T.numpy()
    print(f"{'alpha':7s} {'D1(mean)':10s} {'D2(moment)':11s} {'Jacc_vs_a0':11s} {'eff_rank':9s}")
    for a in args.alphas:
        cache = args.cache_tmpl.format(a=a)
        idx = load_idx(cache)
        sel = np.array(idx)
        U = Xn[sel]
        mu_S = U.mean(0); M_S = (U.T @ U) / len(U)
        D1 = float(((mu_S - muT) ** 2).sum())
        D2 = float(((M_S - MT) ** 2).sum())
        s = set(idx); jac = len(s & idx0) / len(s | idx0)
        sub = sel if len(sel) <= args.subsample else rng.choice(sel, args.subsample, replace=False)
        er = eff_rank(Xn[sub])
        print(f"{a:7s} {D1:<10.5f} {D2:<11.5f} {jac:<11.3f} {er:.1f}")


if __name__ == "__main__":
    main()

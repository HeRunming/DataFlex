#!/usr/bin/env python3
"""Selection-only diagnostic for the lambda-parameterized Moment-MMD sweep
(reviews/choice_0725.md). For each lambda's selection report D1, D2, Jaccard vs
lambda=0 (=GradCov) and vs a linear reference, and effective rank -- to pick the
two Pareto joint candidates to train."""
import argparse, json, os
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
    ap.add_argument("--lams", nargs="+", required=True)
    ap.add_argument("--cache_tmpl", required=True)
    ap.add_argument("--linear_cache", default=None,
                    help="a linear-endpoint selection to measure Jaccard against (e.g. nmoment_b1.0)")
    ap.add_argument("--subsample", type=int, default=4000)
    args = ap.parse_args()

    X = torch.load(args.train_grads, map_location="cpu").float()
    Tg = torch.load(args.target_grads, map_location="cpu").float()
    X = X / X.norm(dim=1, keepdim=True).clamp_min(1e-12)
    Tg = Tg / Tg.norm(dim=1, keepdim=True).clamp_min(1e-12)
    muT = Tg.mean(0).numpy()
    MT = ((Tg.T @ Tg) / Tg.shape[0]).numpy()
    Xn = X.numpy()

    idx0 = set(load_idx(args.cache_tmpl.format(lam="0.0")))
    idxL = set(load_idx(args.linear_cache)) if args.linear_cache else None
    rng = np.random.RandomState(0)
    print(f"{'lambda':7s} {'D1':10s} {'D2':11s} {'Jac_gc':8s} {'Jac_lin':8s} {'eff_rank':8s}")
    for lam in args.lams:
        idx = load_idx(args.cache_tmpl.format(lam=lam))
        sel = np.array(idx); U = Xn[sel]
        mS = U.mean(0); MS = (U.T @ U) / len(U)
        D1 = 0.5 * float(((mS - muT) ** 2).sum())
        D2 = float(((MS - MT) ** 2).sum())
        s = set(idx)
        jgc = len(s & idx0) / len(s | idx0)
        jlin = (len(s & idxL) / len(s | idxL)) if idxL else float("nan")
        sub = sel if len(sel) <= args.subsample else rng.choice(sel, args.subsample, replace=False)
        er = eff_rank(Xn[sub])
        print(f"{lam:7s} {D1:<10.5f} {D2:<11.5f} {jgc:<8.3f} {jlin:<8.3f} {er:<8.1f}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Phase-1 selection-only diagnostic for scale-normalized Moment-MMD (review_0725_2).

For each beta's selection (no training), on the shared gradient cache report:
  - D1 = (1/2)||muS-muT||^2 , D2 = ||M_S-M_T||_F^2 , and normalized D1/s1, D2/s2
  - Jaccard overlap vs beta=0 (=GradCov) and vs beta=1 (=linear-MMD)
  - effective rank of the selected gradients (subsampled)
The KEY calibration check (step 4): the cross-candidate std of the *normalized
greedy marginal* of the linear term c1*(1+<x,muT>)/2 vs the quadratic term
c2*(x^T M_T x), evaluated at the FIRST greedy step (m=0, r_T only). At beta=0.5
these two spreads should be within ~3-5x (not the raw 14x) if random-MMD
normalization aligns with greedy-marginal scale.
"""
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
    ap.add_argument("--betas", nargs="+", required=True)
    ap.add_argument("--cache_tmpl", required=True, help="e.g. /.../nmoment_b{b}_stem80_output")
    ap.add_argument("--n_random", type=int, default=256)
    ap.add_argument("--calib_seed", type=int, default=0)
    ap.add_argument("--num_select", type=int, default=13533)
    ap.add_argument("--subsample", type=int, default=4000)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    X = torch.load(args.train_grads, map_location="cpu").float().to(dev)
    Tg = torch.load(args.target_grads, map_location="cpu").float().to(dev)
    X = X / X.norm(dim=1, keepdim=True).clamp_min(1e-12)
    Tg = Tg / Tg.norm(dim=1, keepdim=True).clamp_min(1e-12)
    muT = Tg.mean(0)
    MT = (Tg.T @ Tg) / Tg.shape[0]
    N, D = X.shape; K = args.num_select

    # --- recompute s1,s2 identically to the selection script ---
    g = torch.Generator(device="cpu"); g.manual_seed(args.calib_seed)
    avail = (X * X).sum(1) > 0
    valid_idx = torch.nonzero(avail).squeeze(1)
    d1s, d2s = [], []
    for _ in range(args.n_random):
        sel_local = torch.randperm(valid_idx.numel(), generator=g)[:K].to(dev)
        perm = valid_idx[sel_local]
        SU = X.index_select(0, perm)
        mS = SU.mean(0); MS = (SU.T @ SU) / SU.shape[0]
        d1s.append(0.5 * float(((mS - muT) ** 2).sum()))
        d2s.append(float(((MS - MT) ** 2).sum()))
    s1 = float(np.mean(d1s)); s2 = float(np.mean(d2s))
    print(f"[diag] s1(mean rand D1)={s1:.6g}  s2(mean rand D2)={s2:.6g}  ratio s1/s2={s1/s2:.3f}")

    # --- greedy-marginal scale check at m=0 (first step): r_T(x) per component ---
    # linear marginal spread (before c1) : (1+<x,muT>)/2
    lin_term = 0.5 * (1.0 + (X @ muT))                 # (N,)
    quad_term = ((X @ MT) * X).sum(1)                  # x^T M_T x  (N,)
    sd_lin = float(lin_term.std()); sd_quad = float(quad_term.std())
    print(f"[diag] RAW marginal std: lin={sd_lin:.6g} quad={sd_quad:.6g} ratio lin/quad={sd_lin/sd_quad:.2f}")
    for b in [0.5]:
        c1 = b / (s1 + 1e-12); c2 = (1.0 - b) / (s2 + 1e-12)
        nsd_lin = c1 * sd_lin; nsd_quad = c2 * sd_quad
        print(f"[diag] beta={b}: NORMALIZED marginal std lin(c1*)={nsd_lin:.6g} "
              f"quad(c2*)={nsd_quad:.6g}  ratio lin/quad={nsd_lin/nsd_quad:.2f}   "
              f"<-- want within ~3-5x, not 14x")

    idx0 = set(load_idx(args.cache_tmpl.format(b="0.0")))
    idx1 = set(load_idx(args.cache_tmpl.format(b="1.0")))
    Xn = X.cpu().numpy()
    muTn = muT.cpu().numpy(); MTn = MT.cpu().numpy()
    rng = np.random.RandomState(0)
    print()
    print(f"{'beta':6s} {'D1':10s} {'D2':11s} {'D1/s1':9s} {'D2/s2':9s} "
          f"{'Jac_b0':8s} {'Jac_b1':8s} {'eff_rank':8s}")
    for b in args.betas:
        idx = load_idx(args.cache_tmpl.format(b=b))
        sel = np.array(idx); U = Xn[sel]
        mS = U.mean(0); MS = (U.T @ U) / len(U)
        D1 = 0.5 * float(((mS - muTn) ** 2).sum())
        D2 = float(((MS - MTn) ** 2).sum())
        s = set(idx)
        j0 = len(s & idx0) / len(s | idx0)
        j1 = len(s & idx1) / len(s | idx1)
        sub = sel if len(sel) <= args.subsample else rng.choice(sel, args.subsample, replace=False)
        er = eff_rank(Xn[sub])
        print(f"{b:6s} {D1:<10.5f} {D2:<11.5f} {D1/s1:<9.3f} {D2/s2:<9.3f} "
              f"{j0:<8.3f} {j1:<8.3f} {er:<8.1f}")


if __name__ == "__main__":
    main()

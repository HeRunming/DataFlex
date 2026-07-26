#!/usr/bin/env python3
"""
Scale-normalized Moment-MMD selection (review_0725_2).

The raw kernel k_a = a*k_lin + (1-a)*k_quad is mis-calibrated: the linear
component's cross-candidate spread is ~14x the quadratic's, so any a>0 lets the
first-order term hijack the ranking. Fix: normalize each component's MMD by its
expected value under same-budget RANDOM selection, so beta expresses true
relative weight:

    Dtilde_beta(S,T) = beta * D1(S,T)/s1 + (1-beta) * D2(S,T)/s2
    D1 = MMD^2_klin(S,T),  D2 = MMD^2_kquad(S,T)
    s_j = E_{random R, |R|=K}[ D_j(R,T) ]   (estimated from B random subsets; no labels)

Equivalent normalized kernel:
    ktilde_beta(u,v) = (beta/s1) * k_lin(u,v) + ((1-beta)/s2) * k_quad(u,v)
    k_lin(u,v)=(1+<u,v>)/2 ,  k_quad(u,v)=<u,v>^2   (u,v unit vectors)

Greedy minimizes the (biased empirical) Dtilde via the standard marginal:
    score(x) = rT(x) - (rS(x) + ktilde(x,x)/2)/(|S|+1)
CAREFUL: ktilde(x,x) = beta/s1 + (1-beta)/s2  (constant but NOT 1 after normalization).

s1,s2 depend only on (candidate pool, target, budget K) — estimated once per
(target,budget), fixed calibration seed, shared across all beta. Not per-step.
"""
import argparse, json, os
import numpy as np
import torch


def mmd2_components(SU, Tg):
    """Return (D1, D2) = biased empirical MMD^2 for k_lin and k_quad between
    selected-set unit grads SU [m,d] and target unit grads Tg [M,d].
    D1 = ||muS-muT||^2 * (1/2)?  -> we use k_lin=(1+<>)/2 directly:
      MMD^2_klin = mean_SS klin - 2 mean_ST klin + mean_TT klin
    Constant 1/2 offsets cancel in the MMD, leaving (1/2)||muS-muT||^2. We compute
    the kernel-form MMD directly so it matches the greedy objective exactly."""
    muS = SU.mean(0); muT = Tg.mean(0)
    # k_lin MMD^2 = (1/2)||muS - muT||^2   (the +1/2 constants cancel)
    D1 = 0.5 * float(((muS - muT) ** 2).sum())
    # k_quad MMD^2 = ||M_S - M_T||_F^2 with M = E[u u^T]
    MS = (SU.T @ SU) / SU.shape[0]
    MT = (Tg.T @ Tg) / Tg.shape[0]
    D2 = float(((MS - MT) ** 2).sum())
    return D1, D2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_grads", required=True)
    ap.add_argument("--target_grads", required=True)
    ap.add_argument("--out_cache_dir", required=True)
    ap.add_argument("--num_select", type=int, required=True)
    ap.add_argument("--beta", type=float, required=True, help="weight on (normalized) 1st-order; (1-beta) on 2nd-order")
    ap.add_argument("--n_random", type=int, default=256, help="random subsets to estimate s1,s2")
    ap.add_argument("--calib_seed", type=int, default=0)
    ap.add_argument("--subsample_indices", default=None)
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    b = float(args.beta)
    if not np.isfinite(b) or not (0.0 <= b <= 1.0):
        raise ValueError(f"--beta must be in [0,1], got {b}")
    if args.num_select <= 0:
        raise ValueError("--num_select must be positive")

    X = torch.load(args.train_grads, map_location="cpu").float().to(dev)
    Tg = torch.load(args.target_grads, map_location="cpu").float().to(dev)
    if X.ndim != 2 or Tg.ndim != 2 or X.shape[1] != Tg.shape[1]:
        raise ValueError("bad cache shapes")
    if not torch.isfinite(X).all() or not torch.isfinite(Tg).all():
        raise ValueError("cache has NaN/Inf")
    if (Tg.norm(dim=1) <= 1e-12).any():
        raise ValueError("target cache has zero rows")
    X = X / X.norm(dim=1, keepdim=True).clamp_min(1e-12)
    Tg = Tg / Tg.norm(dim=1, keepdim=True).clamp_min(1e-12)
    N, D = X.shape; M = Tg.shape[0]
    X_sq = (X * X).sum(1); avail = X_sq > 0
    valid = int(avail.sum().item())
    if args.num_select > valid:
        raise ValueError(f"budget {args.num_select} > valid {valid}")
    K = args.num_select

    # --- estimate reference scales s1,s2 from B random subsets of size K ---
    # Keep everything on-device; index-select the K rows per draw (cheap) instead
    # of copying the full [N,d] cache to CPU each iteration.
    g = torch.Generator(device="cpu"); g.manual_seed(args.calib_seed)
    valid_idx = torch.nonzero(avail).squeeze(1)   # on dev
    d1s, d2s = [], []
    for _bi in range(args.n_random):
        sel_local = torch.randperm(valid_idx.numel(), generator=g)[:K].to(dev)
        perm = valid_idx[sel_local]
        SU = X.index_select(0, perm)
        D1, D2 = mmd2_components(SU, Tg)
        d1s.append(D1); d2s.append(D2)
    s1 = float(np.mean(d1s)); s2 = float(np.mean(d2s))
    eps = 1e-12
    c1 = b / (s1 + eps); c2 = (1.0 - b) / (s2 + eps)
    self_k_val = c1 * 1.0 + c2 * 1.0   # klin(x,x)=(1+1)/2=1 ; kquad(x,x)=1
    print(f"[norm-moment] s1={s1:.6g} s2={s2:.6g} c1={c1:.4g} c2={c2:.4g} "
          f"self_k={self_k_val:.4g} beta={b} K={K}", flush=True)

    def kmix(inner):
        return c1 * (1.0 + inner) / 2.0 + c2 * (inner * inner)

    # NOTE greedy MAXIMIZES  -Dtilde marginal; same form as before with normalized k.
    XT = X @ Tg.T
    target_rel = kmix(XT).mean(dim=1)   # r_T(x) under normalized kernel
    del XT
    self_k = torch.full((N,), self_k_val, device=dev)
    selected, ksum = [], torch.zeros(N, device=dev)
    neg = torch.tensor(float("-inf"), device=dev)
    log_every = max(1, K // 20)
    for step in range(K):
        m = len(selected)
        scores = target_rel.clone() if m == 0 else target_rel - (1.0/(m+1))*(ksum + self_k/2.0)
        scores = torch.where(avail, scores, neg)
        bi = int(torch.argmax(scores).item())
        selected.append(bi); avail[bi] = False
        ksum.add_(kmix(X @ X[bi]))
        if (step+1) % log_every == 0: print(f"[norm-moment] {step+1}/{K}", flush=True)
    assert len(selected) == len(set(selected)) == K

    output = selected
    if args.subsample_indices is not None:
        mp = torch.load(args.subsample_indices, map_location="cpu")
        mp = mp.tolist() if torch.is_tensor(mp) else list(mp)
        if len(mp) != N: raise ValueError("subsample map size mismatch")
        output = [int(mp[i]) for i in selected]

    os.makedirs(args.out_cache_dir, exist_ok=True)
    meta = {"kernel": "moment_normalized", "beta": b, "s1": s1, "s2": s2,
            "c1": c1, "c2": c2, "self_k": self_k_val, "n_random": args.n_random,
            "calib_seed": args.calib_seed, "num_select": K, "n_candidates": N,
            "n_target": M, "proj_dim": D,
            "train_grads": os.path.abspath(args.train_grads),
            "target_grads": os.path.abspath(args.target_grads)}
    json.dump({"indices": output, "metric": meta},
              open(os.path.join(args.out_cache_dir, "step_1.json"), "w"))
    print(f"[norm-moment] wrote {args.out_cache_dir}/step_1.json ({len(output)})", flush=True)


if __name__ == "__main__":
    main()

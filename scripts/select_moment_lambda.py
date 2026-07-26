#!/usr/bin/env python3
"""
Lambda-parameterized Moment-MMD selection (reviews/choice_0725.md).

Insight from the review: any fixed per-component normalization (random-MMD s1,s2,
or marginal-std q1,q2) is just a REPARAMETERIZATION of the same kernel-mixture path
-- a global positive rescale never changes greedy argmax. So instead of an arbitrary
normalizer we use the direct coefficient ratio lambda:

    k_lambda(u,v) = k_quad(u,v) + lambda * k_lin(u,v)
                  = <u,v>^2 + lambda * (1 + <u,v>)/2      (u,v unit vectors)

lambda is the first-order kernel weight RELATIVE to second-order, free of any s1,s2.
lambda=0  == pure GradCov (2nd-order).  lambda->inf == linear-MMD (1st-order).
Marginal-balanced point (equal per-candidate greedy-marginal std) is
    lambda_balanced ~= sd_quad / sd_lin ~= 0.00040/0.00581 ~= 0.069.

Self-kernel: k_lambda(x,x) = 1 + lambda*1 = 1 + lambda  (constant, != 1).

Greedy marginal (minimize biased empirical MMD^2 as before):
    score(x) = rT(x) - (rS(x) + k(x,x)/2)/(|S|+1)
"""
import argparse, json, os
import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_grads", required=True)
    ap.add_argument("--target_grads", required=True)
    ap.add_argument("--out_cache_dir", required=True)
    ap.add_argument("--num_select", type=int, required=True)
    ap.add_argument("--lam", type=float, required=True, help="first-order weight relative to second-order")
    ap.add_argument("--subsample_indices", default=None)
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lam = float(args.lam)
    if not np.isfinite(lam) or lam < 0.0:
        raise ValueError(f"--lam must be >=0 and finite, got {lam}")
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
    avail = (X * X).sum(1) > 0
    valid = int(avail.sum().item())
    if args.num_select > valid:
        raise ValueError(f"budget {args.num_select} > valid {valid}")
    K = args.num_select

    self_k_val = 1.0 + lam   # kquad(x,x)=1 ; klin(x,x)=1
    print(f"[lam-moment] lambda={lam} self_k={self_k_val:.4g} K={K}", flush=True)

    def kmix(inner):
        return (inner * inner) + lam * (1.0 + inner) / 2.0

    XT = X @ Tg.T
    target_rel = kmix(XT).mean(dim=1)
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
        if (step+1) % log_every == 0: print(f"[lam-moment] {step+1}/{K}", flush=True)
    assert len(selected) == len(set(selected)) == K

    output = selected
    if args.subsample_indices is not None:
        mp = torch.load(args.subsample_indices, map_location="cpu")
        mp = mp.tolist() if torch.is_tensor(mp) else list(mp)
        if len(mp) != N: raise ValueError("subsample map size mismatch")
        output = [int(mp[i]) for i in selected]

    os.makedirs(args.out_cache_dir, exist_ok=True)
    meta = {"kernel": "moment_lambda", "lam": lam, "self_k": self_k_val,
            "num_select": K, "n_candidates": N, "n_target": M, "proj_dim": D,
            "train_grads": os.path.abspath(args.train_grads),
            "target_grads": os.path.abspath(args.target_grads)}
    json.dump({"indices": output, "metric": meta},
              open(os.path.join(args.out_cache_dir, "step_1.json"), "w"))
    print(f"[lam-moment] wrote {args.out_cache_dir}/step_1.json ({len(output)})", flush=True)


if __name__ == "__main__":
    main()

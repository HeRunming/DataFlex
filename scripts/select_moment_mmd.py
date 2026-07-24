#!/usr/bin/env python3
"""
Offline Moment-MMD selection over pre-computed gradient features.

Moment kernel unifies LESS (first-order mean-gradient matching) and GradCov
(second-order directional-moment matching) via a convex combination:

    k_moment(u,v) = alpha * k_lin(u,v) + (1-alpha) * k_quad(u,v)
    k_lin(u,v)  = (1 + <u,v>) / 2          # non-negative angular linear kernel
    k_quad(u,v) = <u,v>^2                   # directional second moment (GradCov)

Gradients u,v are L2-normalized projected gradients (unit vectors), so:
  - alpha=1  -> pure first-order (mean-direction) matching  (~LESS-like, but MMD)
  - alpha=0  -> pure second-order directional-moment matching (= GradCov)
  - 0<alpha<1-> joint first+second gradient-moment matching (Moment-MMD)

MMD greedy (exact marginal): at step m select argmax
    r_T(x) - 1/(m+1) * (r_S(x) + k(x,x)/2)
with r_T(x)=mean_t k(x,t), r_S(x)=sum_{s in S} k(x,s). For unit vectors k(x,x)=1.

Reuses a LESS/GradCov gradient cache (all_projected_grads.pt, L2-normalized).
Writes step_1.json compatible with export_gradient_selection.py.
"""
import argparse, json, os
import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_grads", required=True, help="candidate all_projected_grads.pt [N,d] (L2-normed)")
    ap.add_argument("--target_grads", required=True, help="target all_projected_grads.pt [M,d] (L2-normed)")
    ap.add_argument("--out_cache_dir", required=True)
    ap.add_argument("--num_select", type=int, required=True)
    ap.add_argument("--alpha", type=float, required=True, help="weight on linear (1st-order) term; (1-alpha) on quadratic")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    a = float(args.alpha)

    X = torch.load(args.train_grads, map_location="cpu")
    Tg = torch.load(args.target_grads, map_location="cpu")
    X = (X if torch.is_tensor(X) else torch.as_tensor(X)).float().to(dev)
    Tg = (Tg if torch.is_tensor(Tg) else torch.as_tensor(Tg)).float().to(dev)
    # re-normalize defensively (cache is already L2-normed)
    X = X / X.norm(dim=1, keepdim=True).clamp_min(1e-12)
    Tg = Tg / Tg.norm(dim=1, keepdim=True).clamp_min(1e-12)
    N, D = X.shape; M = Tg.shape[0]
    k = min(args.num_select, N)
    print(f"[moment] X={tuple(X.shape)} T={tuple(Tg.shape)} alpha={a} select={k} dev={dev}", flush=True)

    # exclude all-zero candidate rows (sanitized NaN grads)
    X_sq = (X * X).sum(1)
    avail = X_sq > 0
    nz = int((~avail).sum().item())
    if nz: print(f"[moment] excluding {nz} zero rows", flush=True)

    def kmix(inner):  # inner = <x, .> ; returns k_moment elementwise
        return a * (1.0 + inner) / 2.0 + (1.0 - a) * (inner * inner)

    # target relevance r_T(x) = mean_t k_moment(x, t)
    XT = X @ Tg.T                       # (N, M)
    target_rel = kmix(XT).mean(dim=1)   # (N,)
    del XT
    self_k = torch.ones(N, device=dev)  # unit vectors: k(x,x)=a*(1+1)/2+(1-a)*1 = 1

    selected, ksum = [], torch.zeros(N, device=dev)
    neg = torch.tensor(float("-inf"), device=dev)
    log_every = max(1, k // 20)
    for step in range(k):
        m = len(selected)
        scores = target_rel.clone() if m == 0 else target_rel - (1.0/(m+1))*(ksum + self_k/2.0)
        scores = torch.where(avail, scores, neg)
        b = int(torch.argmax(scores).item())
        selected.append(b); avail[b] = False
        inner = X @ X[b]                # (N,)
        ksum.add_(kmix(inner))
        del inner
        if (step+1) % log_every == 0: print(f"[moment] {step+1}/{k}", flush=True)

    os.makedirs(args.out_cache_dir, exist_ok=True)
    json.dump({"indices": selected, "metric": {"kernel": "moment", "alpha": a}},
              open(os.path.join(args.out_cache_dir, "step_1.json"), "w"))
    print(f"[moment] wrote {args.out_cache_dir}/step_1.json ({len(selected)})", flush=True)


if __name__ == "__main__":
    main()

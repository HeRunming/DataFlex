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
    ap.add_argument("--subsample_indices", default=None,
                    help="optional .pt mapping local row -> global candidate index "
                         "(required if the grad cache was built with candidate_subsample)")
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    a = float(args.alpha)
    if not np.isfinite(a) or not (0.0 <= a <= 1.0):
        raise ValueError(f"--alpha must be finite and in [0,1], got {a}")
    if args.num_select <= 0:
        raise ValueError("--num_select must be positive")

    X = torch.load(args.train_grads, map_location="cpu")
    Tg = torch.load(args.target_grads, map_location="cpu")
    X = (X if torch.is_tensor(X) else torch.as_tensor(X)).float().to(dev)
    Tg = (Tg if torch.is_tensor(Tg) else torch.as_tensor(Tg)).float().to(dev)
    # --- input validation (fail loud; paper runs must not silently degrade) ---
    if X.ndim != 2 or Tg.ndim != 2:
        raise ValueError("gradient caches must be rank-2 tensors")
    if X.shape[1] != Tg.shape[1]:
        raise ValueError(f"feature-dim mismatch: candidate {X.shape[1]} vs target {Tg.shape[1]}")
    if X.shape[0] == 0 or Tg.shape[0] == 0:
        raise ValueError("candidate and target caches must be non-empty")
    if not torch.isfinite(X).all():
        raise ValueError("candidate cache contains NaN/Inf")
    if not torch.isfinite(Tg).all():
        raise ValueError("target cache contains NaN/Inf")
    t_norm = Tg.norm(dim=1)
    if (t_norm <= 1e-12).any():
        raise ValueError(f"target cache has {(t_norm <= 1e-12).sum().item()} zero rows "
                         f"(would destroy the MMD target signal)")
    # re-normalize defensively (cache is already L2-normed)
    X = X / X.norm(dim=1, keepdim=True).clamp_min(1e-12)
    Tg = Tg / t_norm.clamp_min(1e-12).unsqueeze(1)
    N, D = X.shape; M = Tg.shape[0]

    # exclude all-zero candidate rows (sanitized NaN grads), then check budget
    X_sq = (X * X).sum(1)
    avail = X_sq > 0
    nz = int((~avail).sum().item())
    if nz: print(f"[moment] excluding {nz} zero candidate rows", flush=True)
    valid_count = int(avail.sum().item())
    if args.num_select > valid_count:
        raise ValueError(f"requested {args.num_select} samples but only "
                         f"{valid_count}/{N} candidates have valid gradients")
    k = args.num_select
    print(f"[moment] X={tuple(X.shape)} T={tuple(Tg.shape)} alpha={a} select={k} dev={dev}", flush=True)

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

    assert len(selected) == len(set(selected)) == k, "greedy produced duplicate/short selection"

    # map local rows -> global candidate indices if a subsample cache was used
    output = selected
    if args.subsample_indices is not None:
        mp = torch.load(args.subsample_indices, map_location="cpu")
        mp = mp.tolist() if torch.is_tensor(mp) else list(mp)
        if len(mp) != N:
            raise ValueError(f"subsample map has {len(mp)} entries, expected {N}")
        output = [int(mp[i]) for i in selected]

    os.makedirs(args.out_cache_dir, exist_ok=True)
    meta = {"kernel": "moment", "alpha": a, "num_select": k, "n_candidates": N,
            "n_target": M, "proj_dim": D, "train_grads": os.path.abspath(args.train_grads),
            "target_grads": os.path.abspath(args.target_grads),
            "subsample_indices": args.subsample_indices}
    json.dump({"indices": output, "metric": meta},
              open(os.path.join(args.out_cache_dir, "step_1.json"), "w"))
    print(f"[moment] wrote {args.out_cache_dir}/step_1.json ({len(output)})", flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Relevance top-k selection (no coreset repulsion) — the selector-axis control for the
representation×selector 2x2 attribution gate (review_0729 / code_review_0729).

MMD-coreset selectors (select_moment_mmd.py) add a repulsion/diversity term via the greedy
marginal  score = r_T(x) - (r_S(x)+k(x,x)/2)/(|S|+1). This script keeps ONLY the target-
relevance term r_T(x) and takes the top-K — i.e. the SAME representation, selector stripped of
diversity. Contrast (DSMC vs Second-TopK) isolates how much of DSMC's effect is the 2nd-order
representation vs the MMD diversity.

  --order first : r_T(x) = mean_t k_lin(x,t) = (1 + <x, mu_T>)/2   (monotone in mean_t <x,t>)
                  -> top-k by mean_t <x,t>  == LESS-like first-order relevance ("First-TopK").
  --order second: r_T(x) = mean_t k_quad(x,t) = mean_t <x,t>^2 = x^T M_T x , M_T = E_t[t t^T]
                  -> top-k by mean_t <x,t>^2  ("Second-TopK").

NOTE (code_review_0729): this is relevance TOP-K, NOT greedy round-robin (RR selects, per query,
the nearest unpicked candidate, cycling over queries). The 2x2 cells are therefore First-TopK /
Second-TopK; a true First-RR/Second-RR is a separate selector for the external-validity phase.
Unit-normalized projected gradients, same caches/guards as select_moment_mmd.py.

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
    ap.add_argument("--order", choices=["first", "second"], required=True)
    ap.add_argument("--subsample_indices", default=None)
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.num_select <= 0:
        raise ValueError("--num_select must be positive")

    X = torch.load(args.train_grads, map_location="cpu").float().to(dev)
    Tg = torch.load(args.target_grads, map_location="cpu").float().to(dev)
    if X.ndim != 2 or Tg.ndim != 2 or X.shape[1] != Tg.shape[1]:
        raise ValueError("bad cache shapes")
    if X.shape[0] == 0 or Tg.shape[0] == 0:
        raise ValueError("candidate and target caches must be non-empty")
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

    # Both orders via the M target inner-products (O(NDM)); avoids materializing the
    # D×D moment matrix and the N×D product (O(ND^2), ~8.9GB at D=8192). Identity used:
    #   x^T (1/M sum_t t t^T) x = (1/M) sum_t (x^T t)^2  = mean_t <x,t>^2
    #   mean_t k_lin relevance is monotone in <x,mu_T> = mean_t <x,t>
    XT = X @ Tg.T                              # (N, M), M small (~80)
    if args.order == "first":
        rel = XT.mean(1)                       # monotone in <x, mu_T>  -> First-TopK
    else:
        rel = XT.square().mean(1)              # x^T M_T x = mean_t <x,t>^2  -> Second-TopK
    del XT
    rel = torch.where(avail, rel, torch.tensor(float("-inf"), device=dev))
    sel = torch.topk(rel, K).indices
    selected = sel.tolist()
    assert len(selected) == len(set(selected)) == K
    print(f"[rel-topk] order={args.order} K={K} rel[min/mean/max]="
          f"{float(rel[avail].min()):.4g}/{float(rel[avail].mean()):.4g}/{float(rel[avail].max()):.4g}",
          flush=True)

    output = selected
    if args.subsample_indices is not None:
        mp = torch.load(args.subsample_indices, map_location="cpu")
        mp = mp.tolist() if torch.is_tensor(mp) else list(mp)
        if len(mp) != N: raise ValueError("subsample map size mismatch")
        output = [int(mp[i]) for i in selected]

    os.makedirs(args.out_cache_dir, exist_ok=True)
    meta = {"kernel": f"relevance_topk_{args.order}", "order": args.order,
            "num_select": K, "n_candidates": N, "n_target": M, "proj_dim": D,
            "train_grads": os.path.abspath(args.train_grads),
            "target_grads": os.path.abspath(args.target_grads)}
    json.dump({"indices": output, "metric": meta},
              open(os.path.join(args.out_cache_dir, "step_1.json"), "w"))
    print(f"[rel-topk] wrote {args.out_cache_dir}/step_1.json ({len(output)})", flush=True)


if __name__ == "__main__":
    main()

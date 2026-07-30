#!/usr/bin/env python3
"""
GIST selection (Gradient Isometric Subspace Transformation), arXiv 2602.18584 v2.
Baseline for the external-validity pilot (review_0729 / choice_0730). Closest conceptual
competitor to DSMC: both recover a target-relevant subspace from gradients.

Paper algorithm (Eqs 14-17), same LESS 270k / LoRA-warmup / MMLU-val setting as ours:
  Step 2 (task subspace): build target gradient matrix G_val = [g_val^1 ... g_val^M] in R^{d x M}
    (one column per target example), SVD G_val = U Σ V^T (Eq 14). Effective rank r = smallest r
    with cumulative explained variance (sum σ_i^2) >= tau (paper: 95%). Target projector
    Π = U_r^T in R^{r x d}, top-r left singular vectors (Eq 15).
  Step 3 (geometric scoring): Sim(z_i, z_val^j) = cos(Π g_i , Π g_val^j)   (Eq 16)
  Aggregate (Eq 17, max-relevance like LESS): FinalScore(z_i) = max_j Sim(z_i, z_val^j); top-k.

ADAPTATION TO OUR CACHES (documented for the numerical review):
  * The paper does SVD in the raw d-dim LoRA gradient space. We only have the shared 8192-dim
    TRAK/Johnson-Lindenstrauss-projected gradients (seed 123) used by LESS and DSMC. SVD of the
    projected target grads recovers the JL image of the same task subspace; using the SAME
    projected caches as every other method is the controlled, apples-to-apples choice here and
    keeps candidate & target in one common projection (a paper requirement).
  * Our cached grads are unit-normalized per example (like DSMC/LESS). We therefore SVD the
    unit-normalized target grads, so every target example contributes equally to the subspace
    (consistent with DSMC). Cosine in Eq 16 makes the candidate scoring scale-invariant anyway.
  * Optimizer protocol (Adam-candidate / SGD-target vs raw/raw) is a CLI choice via the cache
    paths, NOT hardcoded — the protocol-alignment decision is made by the caller. Recorded in meta.
Sanity properties (asserted/reported): Π has orthonormal rows (U_r columns orthonormal); score is
invariant to any orthonormal rotation of the subspace basis; r == M reproduces full-subspace
cosine on the M target directions. Verify before the pilot.
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
    ap.add_argument("--evr", type=float, default=0.95, help="explained-variance ratio for rank r (paper: 0.95)")
    ap.add_argument("--rank", type=int, default=0, help="override: fixed rank r (0 = use --evr rule)")
    ap.add_argument("--subsample_indices", default=None)
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.num_select <= 0:
        raise ValueError("--num_select must be positive")
    if not (0.0 < args.evr <= 1.0):
        raise ValueError("--evr must be in (0,1]")

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
    # unit-normalize (idempotent if already normalized) — same convention as DSMC/LESS here
    X = X / X.norm(dim=1, keepdim=True).clamp_min(1e-12)
    Tg = Tg / Tg.norm(dim=1, keepdim=True).clamp_min(1e-12)
    N, Dp = X.shape; M = Tg.shape[0]
    avail = (X * X).sum(1) > 0
    valid = int(avail.sum().item())
    if args.num_select > valid:
        raise ValueError(f"budget {args.num_select} > valid {valid}")
    K = args.num_select

    # --- Step 2: task subspace via SVD of G_val = Tg^T (Dp x M) ---
    Gval = Tg.t().contiguous()                      # (Dp, M)
    U, S, _ = torch.linalg.svd(Gval, full_matrices=False)   # U:(Dp, M), S:(M,)
    ev = (S ** 2)
    evr = torch.cumsum(ev, 0) / ev.sum().clamp_min(1e-30)
    if args.rank > 0:
        r = min(args.rank, U.shape[1])
    else:
        r = int(torch.searchsorted(evr, torch.tensor(args.evr, device=dev)).item()) + 1
        r = max(1, min(r, U.shape[1]))
    Ur = U[:, :r]                                   # (Dp, r), orthonormal columns
    # sanity: orthonormal basis
    ortho_err = float((Ur.t() @ Ur - torch.eye(r, device=dev)).abs().max())
    print(f"[gist] Dp={Dp} M={M} rank r={r} (evr rule={args.evr}, "
          f"cum_evr@r={float(evr[r-1]):.4f}) ortho_err={ortho_err:.2e}", flush=True)

    # --- Step 3: projected cosine, max over targets ---
    # Project into subspace: coords = Ur^T g  (r-dim). Cosine on projected coords.
    Ct = (Tg @ Ur)                                  # (M, r) projected targets
    Ct = Ct / Ct.norm(dim=1, keepdim=True).clamp_min(1e-12)
    neg = torch.tensor(float("-inf"), device=dev)
    best = torch.full((N,), -1.0, device=dev)
    bs = 8192
    for s in range(0, N, bs):
        Xb = X[s:s+bs]                              # (b, Dp)
        Cb = Xb @ Ur                                # (b, r) projected candidates
        Cb = Cb / Cb.norm(dim=1, keepdim=True).clamp_min(1e-12)
        sim = Cb @ Ct.t()                           # (b, M) cosine (both unit)
        best[s:s+bs] = sim.max(dim=1).values
    best = torch.where(avail, best, neg)
    sel = torch.topk(best, K).indices
    selected = sel.tolist()
    assert len(selected) == len(set(selected)) == K
    print(f"[gist] score[min/mean/max]={float(best[avail].min()):.4f}/"
          f"{float(best[avail].mean()):.4f}/{float(best[avail].max()):.4f}", flush=True)

    output = selected
    if args.subsample_indices is not None:
        mp = torch.load(args.subsample_indices, map_location="cpu")
        mp = mp.tolist() if torch.is_tensor(mp) else list(mp)
        if len(mp) != N: raise ValueError("subsample map size mismatch")
        output = [int(mp[i]) for i in selected]

    os.makedirs(args.out_cache_dir, exist_ok=True)
    meta = {"kernel": "gist", "rank": r, "evr_rule": args.evr,
            "cum_evr_at_r": float(evr[r-1]), "ortho_err": ortho_err,
            "num_select": K, "n_candidates": N, "n_target": M, "proj_dim": Dp,
            "train_grads": os.path.abspath(args.train_grads),
            "target_grads": os.path.abspath(args.target_grads)}
    json.dump({"indices": output, "metric": meta},
              open(os.path.join(args.out_cache_dir, "step_1.json"), "w"))
    print(f"[gist] wrote {args.out_cache_dir}/step_1.json ({len(output)})", flush=True)


if __name__ == "__main__":
    main()

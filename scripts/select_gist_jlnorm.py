#!/usr/bin/env python3
"""
GIST-JL-Norm: an ADAPTATION of GIST (arXiv 2602.18584) to our shared normalized 8192-D caches.
NOT an exact reproduction (see code_review_0730). Official GIST = github.com/GuanghuiMin/GIST.
Differences from official GIST:
  1. operates on unit-normalized 8192-D JL-projected grads (official: raw d-dim LoRA grads);
  2. normalizes target grads BEFORE SVD (official: raw target Gram matrix) -> different subspace;
  3. plain projector Pi=U_r^T (official: whitening/isometric P = G_val^T U_k S_k^{-1});
  4. rank via 95% EVR (official target_dim is a FIXED arg, default 150; EVR is only how they pick it).
Kept as a cheap, clearly-labelled ablation / sensitivity point; meta records the deviations so it
is never mistaken for faithful GIST. A faithful variant that adds the S^{-1} whitening + fixed rank
+ no pre-SVD normalization is `select_gist_faithful.py`.

Algorithm here: SVD of unit-normalized target matrix -> top-r left singular vectors (95% EVR) ->
projected cosine (Eq 16) -> max over targets (Eq 17) -> top-k.
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
    meta = {"kernel": "gist_jlnorm", "raw_or_projected": "JL_projected_8192",
            "normalize_before_svd": True, "aggregation": "max_per_example",
            "rank_rule": "95pct_evr", "rank": r, "evr_rule": args.evr,
            "cum_evr_at_r": float(evr[r-1]), "ortho_err": ortho_err,
            "num_select": K, "n_candidates": N, "n_target": M, "proj_dim": Dp,
            "train_grads": os.path.abspath(args.train_grads),
            "target_grads": os.path.abspath(args.target_grads)}
    json.dump({"indices": output, "metric": meta},
              open(os.path.join(args.out_cache_dir, "step_1.json"), "w"))
    print(f"[gist] wrote {args.out_cache_dir}/step_1.json ({len(output)})", flush=True)


if __name__ == "__main__":
    main()

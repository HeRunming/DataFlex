#!/usr/bin/env python3
"""
GIST-faithful: the official GIST algorithm (github.com/GuanghuiMin/GIST, get_gist_gradients.py),
as opposed to the GIST-JL-Norm adaptation. Implements the ISOMETRIC whitening projection that the
name "Gradient Isometric Subspace Transformation" refers to.

Official pipeline (raw LoRA-parameter gradients g in R^D):
  1. target Gram matrix   K = G_val G_val^T           (G_val: [M, D] rows = raw target grads)
     via eigendecomposition K = U diag(lambda) U^T, S = sqrt(lambda), take top-k (k = target_dim,
     a FIXED rank; paper uses 150 for MMLU, chosen offline by ~95% explained variance).
  2. whitening projection  P = G_val^T (U_k S_k^{-1})  in R^{D x k}    <-- the "isometric" transform
     (this maps a D-dim gradient into k whitened target-subspace coordinates; NOT plain U_k^T g).
  3. low-dim features      z_i = g_i P   (candidate),   z_val^j = g_val^j P
     score  Sim(i,j) = cos(z_i, z_val^j)  (Eq 16),  FinalScore_i = max_j Sim(i,j)  (Eq 17), top-k.

IMPORTANT provenance note (code_review_0730): the OFFICIAL code does the Gram/whitening on RAW
(un-normalized) target gradients, and projects RAW candidate gradients through P. Our on-disk caches
are unit-normalized per example and the raw pre-normalization chunks were deleted, so an *exact*
reproduction requires RE-EXTRACTING raw grads. This script therefore has two modes:
  --assume_raw  : treat the provided caches AS raw (use only if you pass freshly re-extracted raw
                  caches). Faithful.
  (default)     : caches are unit-normalized; we still apply the official Gram+whitening+cosine
                  math, but on normalized inputs. This is "GIST-faithful-math on normalized caches"
                  — closer to official than JL-Norm (correct whitening + fixed rank + no extra
                  pre-SVD renormalization beyond what the cache already has), but NOT byte-faithful.
                  Meta records exact_reproduction=false with the reason.
Because cosine is invariant to per-example candidate scale, the candidate-side normalization is
harmless; the only fidelity gap in default mode is that the target Gram is built from unit-norm
targets. --rank sets k (default 150 per paper); --evr picks k by explained variance if --rank<=0.
"""
import argparse, json, os
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_grads", required=True)
    ap.add_argument("--target_grads", required=True)
    ap.add_argument("--out_cache_dir", required=True)
    ap.add_argument("--num_select", type=int, required=True)
    ap.add_argument("--rank", type=int, default=150, help="fixed subspace rank k (official default 150)")
    ap.add_argument("--evr", type=float, default=0.0, help="if >0 and --rank<=0, pick k by this explained-variance ratio")
    ap.add_argument("--assume_raw", action="store_true",
                    help="treat inputs as RAW (un-normalized) gradients for an exact reproduction")
    ap.add_argument("--subsample_indices", default=None)
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.num_select <= 0:
        raise ValueError("--num_select must be positive")

    X = torch.load(args.train_grads, map_location="cpu").float().to(dev)
    Tg = torch.load(args.target_grads, map_location="cpu").float().to(dev)   # [M, D]
    if X.ndim != 2 or Tg.ndim != 2 or X.shape[1] != Tg.shape[1]:
        raise ValueError("bad cache shapes")
    if X.shape[0] == 0 or Tg.shape[0] == 0:
        raise ValueError("candidate and target caches must be non-empty")
    if not torch.isfinite(X).all() or not torch.isfinite(Tg).all():
        raise ValueError("cache has NaN/Inf")
    if (Tg.norm(dim=1) <= 1e-12).any():
        raise ValueError("target cache has zero rows")
    N, D = X.shape; M = Tg.shape[0]
    avail = (X * X).sum(1) > 0
    valid = int(avail.sum().item())
    if args.num_select > valid:
        raise ValueError(f"budget {args.num_select} > valid {valid}")
    K = args.num_select

    prenormalized = bool(abs(float(Tg.norm(dim=1).mean()) - 1.0) < 1e-3)
    exact = args.assume_raw and not prenormalized

    # --- official: Gram of raw target grads, eigendecomp, top-k, whitening projection P ---
    Gram = Tg @ Tg.t()                                  # [M, M]
    evals, evecs = torch.linalg.eigh(Gram)              # ascending
    order = torch.argsort(evals, descending=True)
    evals = evals[order].clamp_min(0); evecs = evecs[:, order]
    S = torch.sqrt(evals)                               # [M]
    if args.rank > 0:
        k = min(args.rank, M)
    else:
        evr = torch.cumsum(evals, 0) / evals.sum().clamp_min(1e-30)
        thr = args.evr if args.evr > 0 else 0.95
        k = int(torch.searchsorted(evr, torch.tensor(thr, device=dev)).item()) + 1
        k = max(1, min(k, M))
    Uk = evecs[:, :k]                                   # [M, k]
    Sk_inv = 1.0 / (S[:k] + 1e-6)                       # [k]
    # P = G_val^T (Uk Sk^{-1})  in R^{D x k}
    W = Uk * Sk_inv.unsqueeze(0)                        # [M, k]
    P = Tg.t() @ W                                      # [D, k]
    print(f"[gist-faithful] D={D} M={M} k={k} (rank arg={args.rank}, evr={args.evr}) "
          f"exact_reproduction={exact} prenormalized_cache={prenormalized}", flush=True)

    # project targets + candidates through P, cosine, max over targets
    Zt = Tg @ P                                         # [M, k]
    Zt = Zt / Zt.norm(dim=1, keepdim=True).clamp_min(1e-12)
    neg = torch.tensor(float("-inf"), device=dev)
    best = torch.full((N,), -1.0, device=dev)
    bs = 8192
    for s in range(0, N, bs):
        Zc = X[s:s+bs] @ P
        Zc = Zc / Zc.norm(dim=1, keepdim=True).clamp_min(1e-12)
        best[s:s+bs] = (Zc @ Zt.t()).max(dim=1).values
    best = torch.where(avail, best, neg)
    sel = torch.topk(best, K).indices
    selected = sel.tolist()
    assert len(selected) == len(set(selected)) == K
    print(f"[gist-faithful] score[min/mean/max]={float(best[avail].min()):.4f}/"
          f"{float(best[avail].mean()):.4f}/{float(best[avail].max()):.4f}", flush=True)

    output = selected
    if args.subsample_indices is not None:
        mp = torch.load(args.subsample_indices, map_location="cpu")
        mp = mp.tolist() if torch.is_tensor(mp) else list(mp)
        if len(mp) != N: raise ValueError("subsample map size mismatch")
        output = [int(mp[i]) for i in selected]

    os.makedirs(args.out_cache_dir, exist_ok=True)
    meta = {"kernel": "gist_faithful", "rank_k": k, "rank_arg": args.rank, "evr": args.evr,
            "whitening": "P = G_val^T U_k S_k^{-1}", "aggregation": "max_per_example",
            "exact_reproduction": exact, "prenormalized_cache": prenormalized,
            "note": ("exact raw reproduction" if exact else
                     "official Gram/whitening/cosine math on unit-normalized cache "
                     "(raw grads not on disk; target Gram built from normalized targets)"),
            "num_select": K, "n_candidates": N, "n_target": M, "proj_dim": D,
            "train_grads": os.path.abspath(args.train_grads),
            "target_grads": os.path.abspath(args.target_grads)}
    json.dump({"indices": output, "metric": meta},
              open(os.path.join(args.out_cache_dir, "step_1.json"), "w"))
    print(f"[gist-faithful] wrote {args.out_cache_dir}/step_1.json ({len(output)})", flush=True)


if __name__ == "__main__":
    main()

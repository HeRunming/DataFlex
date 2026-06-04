#!/usr/bin/env python3
"""
Diagnose why Adam-preconditioned gradients hurt MMD but help LESS.

Loads cached L2-normalized projected gradients (train pool + target) for both
adam and sgd, and computes statistics that distinguish the two regimes:

  (A) LESS uses MEAN dot-product to target:  score_i = mean_t <g_i, g_t>
      -> only the DIRECTION of the mean target gradient matters.
  (B) MMD uses pairwise kernel among SELECTED set (redundancy term):
      Delta(x) = r_T(x) - 1/(m+1) [ sum_{s in S} k(x,s) + k(x,x)/2 ]
      -> the GEOMETRY / spread of the candidate cloud matters.

Hypotheses to test:
  H1. Adam preconditioning collapses gradient directions (anisotropy up):
      candidates become more mutually similar -> redundancy term dominates,
      MMD's diversity pressure misfires.
  H2. Adam inflates effective dimensionality differently; RBF median sigma
      shifts so the kernel is in a bad regime (all ~1 or all ~0).
  H3. Target relevance ranking (what LESS uses) is preserved/improved by Adam,
      while pairwise candidate structure (what MMD adds) is degraded.
"""
import os
import numpy as np
import torch

SAVES = "/jizhicfs/karonhe/dataflex_saves"

PAIRS = {
    "grad_rbf": ("mmd_grad_rbf_output", "mmd_grad_rbf_sgd_output"),
    "grad_cov": ("mmd_grad_cov_output", "mmd_grad_cov_sgd_output"),
    "less":     ("less_output", "less_sgd_output"),
}

def load(d, split):
    import glob
    # split is 'train' or 'target'; less uses 'eval' for target. Path has a step subdir.
    cands = glob.glob(os.path.join(SAVES, d, split, "*", "all_projected_grads.pt"))
    if not cands and split == "target":
        cands = glob.glob(os.path.join(SAVES, d, "eval", "*", "all_projected_grads.pt"))
    if not cands:
        raise FileNotFoundError(f"no grads under {SAVES}/{d}/{split}/*/")
    return torch.load(sorted(cands)[-1], map_location="cpu").float()

def subsample(X, n, seed=0):
    if X.shape[0] <= n:
        return X
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(X.shape[0], generator=g)[:n]
    return X[idx]

def median_sigma(X, sub=2000, seed=42):
    Xs = subsample(X, sub, seed)
    sq = (Xs*Xs).sum(1, keepdim=True)
    d2 = (sq + sq.T - 2*Xs@Xs.T).clamp_min(0)
    iu = torch.triu_indices(Xs.shape[0], Xs.shape[0], offset=1)
    return d2[iu[0], iu[1]].clamp_min(0).sqrt().median().item()

def stats(tag, adam_dir, sgd_dir):
    print(f"\n{'='*70}\n{tag}\n{'='*70}")
    for label, d in (("ADAM", adam_dir), ("SGD", sgd_dir)):
        Xtr = load(d, "train")          # (N, D) already L2-normalized
        Xtg = load(d, "target")
        # sanitize
        torch.nan_to_num_(Xtr, 0,0,0); torch.nan_to_num_(Xtg, 0,0,0)
        N, D = Xtr.shape
        # zero rows
        nz = int((Xtr.norm(dim=1) == 0).sum())

        Xs = subsample(Xtr, 4000, seed=1)        # candidate cloud sample
        # --- pairwise cosine among candidates (rows are unit-norm) ---
        G = Xs @ Xs.T
        iu = torch.triu_indices(Xs.shape[0], Xs.shape[0], offset=1)
        cos_cand = G[iu[0], iu[1]]
        # --- candidate vs target mean cosine (the LESS signal) ---
        tgt_mean = Xtg.mean(0)
        tgt_mean = tgt_mean / tgt_mean.norm().clamp_min(1e-12)
        less_score = Xs @ tgt_mean               # alignment to mean target dir
        # --- effective rank of candidate cloud (participation ratio) ---
        # eigvals of covariance ~ singular values^2 of centered Xs
        Xc = Xs - Xs.mean(0, keepdim=True)
        sv = torch.linalg.svdvals(Xc)
        ev = sv**2
        erank = (ev.sum()**2 / (ev**2).sum()).item()
        # --- target cloud anisotropy: how concentrated is target direction ---
        Tc = Xtg - Xtg.mean(0, keepdim=True)
        svt = torch.linalg.svdvals(Tc)
        evt = svt**2
        erank_t = (evt.sum()**2 / (evt**2).sum()).item()
        # how much of target energy is in its mean direction (coherence)
        tgt_coherence = (Xtg @ (Xtg.mean(0)/Xtg.mean(0).norm().clamp_min(1e-12))).mean().item()

        sig = median_sigma(Xtr)

        print(f"\n  [{label}]  N={N} D={D} zero_rows={nz}")
        print(f"    pairwise candidate cosine:  mean={cos_cand.mean():+.4f}  std={cos_cand.std():.4f}  "
              f"|mean|={cos_cand.abs().mean():.4f}")
        print(f"    candidate eff.rank (PR):    {erank:7.1f} / {D}   ({100*erank/D:.1f}%)")
        print(f"    target    eff.rank (PR):    {erank_t:7.1f} / {min(Xtg.shape)} ")
        print(f"    target mean-dir coherence:  {tgt_coherence:+.4f}   (how aligned target grads are)")
        print(f"    LESS signal (cos to t̄):     mean={less_score.mean():+.4f}  std={less_score.std():.4f}  "
              f"-> spread/mean ratio={less_score.std()/ (less_score.mean().abs()+1e-9):.2f}")
        print(f"    RBF median sigma:           {sig:.4f}  (RBF k at mean cand dist e^-0.5≈0.61 when ||Δ||=σ)")

if __name__ == "__main__":
    for tag, (a, s) in PAIRS.items():
        try:
            stats(tag, a, s)
        except Exception as e:
            print(f"[{tag}] FAILED: {type(e).__name__}: {e}")

#!/usr/bin/env python3
"""
Offline embedding-MMD selection from pre-computed embeddings -> step_1.json.
Wraps OfflineMMDSelector.greedy_mmd_select (exact marginal greedy) but reads
already-saved candidate/target .npy (seed-independent bge embeddings), and
writes a step_1.json compatible with export_gradient_selection.py.

For the stochastic variant, uses the same exact-greedy core (the offline greedy
is deterministic); stochastic_eps only affects the online GPU path, so for the
offline reproduction we keep exact greedy for both emb_rbf and its 'stochastic'
label — matching how the repo's offline path produced both (the eps distinction
is an online-selector ablation). We record the eps in metadata.
"""
import argparse
import json
import os
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate_emb", required=True)
    ap.add_argument("--target_emb", required=True)
    ap.add_argument("--out_cache_dir", required=True)
    ap.add_argument("--num_select", type=int, required=True)
    ap.add_argument("--stochastic_eps", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    import torch
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    C = np.load(args.candidate_emb).astype(np.float32)
    T = np.load(args.target_emb).astype(np.float32)
    C = torch.from_numpy(C).to(dev)
    T = torch.from_numpy(T).to(dev)
    C = C / C.norm(dim=1, keepdim=True).clamp_min(1e-12)
    T = T / T.norm(dim=1, keepdim=True).clamp_min(1e-12)
    N = C.shape[0]
    k = min(args.num_select, N)

    # median-heuristic sigma on a candidate subsample (GPU)
    g = torch.Generator(device=dev); g.manual_seed(args.seed)
    sub = C[torch.randperm(N, generator=g, device=dev)[:2000]]
    d2 = torch.cdist(sub, sub) ** 2
    tri = d2[torch.triu(torch.ones_like(d2), diagonal=1) > 0]
    sigma = float(torch.sqrt(tri.clamp_min(0)).median()); sigma = max(sigma, 1e-6)
    two_s2 = 2.0 * sigma * sigma
    print(f"[emb-mmd] C={tuple(C.shape)} T={tuple(T.shape)} sigma={sigma:.4f} select={k} dev={dev}", flush=True)

    # target relevance r_T(x) = mean_t exp(-||x-t||^2/2s^2)
    C_sq = (C * C).sum(1)                        # (N,)
    T_sq = (T * T).sum(1)                        # (M,)
    rel = torch.zeros(N, device=dev)
    CT = C @ T.T                                 # (N,M)
    sq = C_sq[:, None] + T_sq[None, :] - 2 * CT
    rel = torch.exp(-sq.clamp_min(0) / two_s2).mean(1)
    del CT, sq

    self_k = 1.0  # RBF
    selected = []
    avail = torch.ones(N, dtype=torch.bool, device=dev)
    ksum = torch.zeros(N, device=dev)
    neg_inf = torch.tensor(float("-inf"), device=dev)
    log_every = max(1, k // 20)
    for step in range(k):
        m = len(selected)
        scores = rel.clone() if m == 0 else rel - (1.0/(m+1))*(ksum + self_k/2.0)
        scores = torch.where(avail, scores, neg_inf)
        b = int(torch.argmax(scores).item())
        selected.append(b); avail[b] = False
        # incremental k(., x_b)
        xb = C[b]
        d2b = C_sq + C_sq[b] - 2.0 * (C @ xb)
        ksum += torch.exp(-d2b.clamp_min(0) / two_s2)
        if (step + 1) % log_every == 0:
            print(f"[emb-mmd] {step+1}/{k}", flush=True)

    os.makedirs(args.out_cache_dir, exist_ok=True)
    json.dump({"indices": selected, "metric": {"sigma": sigma, "stochastic_eps": args.stochastic_eps}},
              open(os.path.join(args.out_cache_dir, "step_1.json"), "w"))
    print(f"[emb-mmd] wrote {args.out_cache_dir}/step_1.json ({len(selected)})", flush=True)


if __name__ == "__main__":
    main()


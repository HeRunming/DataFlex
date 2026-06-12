#!/usr/bin/env python3
"""
Run TSDS (Zhang et al. 2024) selection directly from precomputed bge embeddings,
reusing the OT/KDE solver from dataflex's offline_tsds_selector so the algorithm
matches the official ZifanL/TSDS implementation. One run per target.

We feed the SAME bge-base-en-v1.5 embeddings used by MMD-Emb-RBF so that
TSDS vs MMD-Emb-RBF differ only in the selection algorithm (OT vs MMD), not the
representation. Outputs:
  - tsds_probs_<target>.npy   (global sampling prob over the 270k pool)
  - selected indices via no-replacement sampling of `num_select`

Official TSDS defaults: alpha=0.5, C=5.0, sigma=0.75, max_K=5000, kde_K=1000.
"""
import argparse, heapq, json, os
import numpy as np
import faiss


class FaissIVF:
    def __init__(self, data, nprobe=10):
        data = np.ascontiguousarray(data.astype(np.float32))
        N, D = data.shape
        nlist = max(1, int(np.sqrt(N)) // 2)
        quant = faiss.IndexFlatL2(D)
        idx = faiss.IndexIVFFlat(quant, D, nlist)
        idx.train(data); idx.add(data); idx.nprobe = nprobe
        self.index = idx
    def search(self, q, K):
        return self.index.search(np.ascontiguousarray(q.astype(np.float32)), K)


def tsds_probs(xb, xq, sigma=0.75, alpha=0.5, C=5.0, max_K=5000, kde_K=1000):
    """Faithful port of offline_tsds_selector.selector() (== official tsds.py)."""
    M, N = xq.shape[0], xb.shape[0]
    MAX_K = min(max_K, N // 10)
    KDE_K = min(kde_K, N // 10)

    index = FaissIVF(xb)
    top_d2, top_idx = index.search(xq, MAX_K)
    order = np.argsort(top_d2, axis=-1)
    si = np.indices(top_d2.shape)[0]
    top_d = np.sqrt(np.maximum(top_d2[si, order], 0))
    top_idx = top_idx[si, order].astype(int)

    if sigma == 0:
        top_kde = np.ones_like(top_idx, dtype=float)
    else:
        uniq = list(set(top_idx.reshape(-1)))
        sub = xb[uniq]
        ikde = FaissIVF(sub)
        D2, _ = ikde.search(sub, KDE_K)
        kernel = np.maximum(0.0, 1 - D2 / (sigma ** 2))
        kde = kernel.sum(axis=1)
        kmap = {uniq[i]: kde[i] for i in range(len(uniq))}
        top_kde = np.vectorize(lambda t: kmap[t])(top_idx)

    lastK = [0] * M
    heap = [(1.0 / top_kde[j][0], 0, j) for j in range(M)]
    heapq.heapify(heap)
    cost = np.zeros(M)
    dist_wsum = [top_d[j][0] / top_kde[j][0] for j in range(M)]
    total_cost, s = 0.0, 0.0
    while heap:
        count, k, j = heapq.heappop(heap)
        s = count
        total_cost -= cost[j]
        cost[j] = top_d[j][k + 1] * count - dist_wsum[j]
        total_cost += cost[j]
        if alpha / C * total_cost >= (1 - alpha) * M:
            break
        lastK[j] = k
        if k < MAX_K - 2:
            count += 1.0 / top_kde[j][k + 1]
            heapq.heappush(heap, (count, k + 1, j))
            dist_wsum[j] += top_d[j][k + 1] / top_kde[j][k + 1]

    probs = np.zeros(N, dtype=np.float64)
    inv_M = 1.0 / M
    for j in range(M):
        psum = 0.0
        for k in range(lastK[j] + 1):
            w = inv_M / s / top_kde[j][k]
            probs[top_idx[j][k]] += w
            psum += w
        probs[top_idx[j][lastK[j] + 1]] += max(inv_M - psum, 0)
    probs = np.maximum(probs, 0)
    probs /= probs.sum()
    return probs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate_emb", required=True)
    ap.add_argument("--target_emb", required=True)
    ap.add_argument("--num_select", type=int, required=True)
    ap.add_argument("--probs_out", required=True)
    ap.add_argument("--indices_out", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--sigma", type=float, default=0.75)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--C", type=float, default=5.0)
    args = ap.parse_args()

    xb = np.load(args.candidate_emb).astype(np.float32)
    xq = np.load(args.target_emb).astype(np.float32)
    print(f"[tsds] candidates={xb.shape} target={xq.shape}", flush=True)

    probs = tsds_probs(xb, xq, sigma=args.sigma, alpha=args.alpha, C=args.C)
    np.save(args.probs_out, probs)
    nz = int((probs > 0).sum())
    print(f"[tsds] nonzero probs: {nz} / {len(probs)}", flush=True)

    rng = np.random.default_rng(args.seed)
    k = min(args.num_select, nz)
    sel = rng.choice(len(probs), size=k, replace=False, p=probs)
    np.save(args.indices_out, sel)
    print(f"[tsds] selected {len(sel)} (target {args.num_select}) -> {args.indices_out}", flush=True)


if __name__ == "__main__":
    main()

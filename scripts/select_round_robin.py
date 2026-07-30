#!/usr/bin/env python3
"""
True greedy round-robin (RR) selection, per "A Critical Look at Targeted Instruction Selection"
(arXiv 2602.14696). Distinct from relevance top-k (select_relevance_topk.py): RR cycles over the
target queries and, on each visit to a query, appends that query's highest-similarity candidate
that has NOT yet been selected. This spreads the budget across all target examples (coverage /
anti-collapse) instead of letting a few high-mean-relevance regions dominate. Often strong at low
budget.

similarity(candidate x, target t), unit-normalized projected gradients:
  --order first  : s = <x, t>           (1st-order / signed direction)
  --order second : s = <x, t>^2         (2nd-order / directional, sign-invariant)

Algorithm (deterministic given inputs + --perm_seed):
  order the M target queries by a fixed permutation (recorded in meta);
  maintain, per query j, a descending-sorted candidate list by similarity to t_j;
  round r = 0,1,2,...: for each query j in order, advance its pointer past already-selected
    candidates and select the next unpicked top candidate for j; stop as soon as K are selected.
Ties/duplicates across queries are handled by the "already selected" check, so each candidate is
taken by whichever query reaches it first in the RR order. Exactly K unique indices.
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
    ap.add_argument("--perm_seed", type=int, default=0, help="seed for the fixed target-query visiting order")
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
    avail_mask = (X * X).sum(1) > 0
    valid = int(avail_mask.sum().item())
    if args.num_select > valid:
        raise ValueError(f"budget {args.num_select} > valid {valid}")
    K = args.num_select

    # fixed target-query visiting order (recorded)
    g = torch.Generator(device="cpu"); g.manual_seed(args.perm_seed)
    query_order = torch.randperm(M, generator=g).tolist()

    # per-query candidate ranking (descending similarity). For K<<N we don't need a full sort of
    # all N per query; but M is small (<=80) and a full argsort per query is simple + exact.
    # sim_j = <x, t_j> ; second order squares it. Rank descending.
    # Build an (M, N) ranking lazily per query to bound memory: argsort one query at a time.
    INVALID = ~avail_mask  # candidates to never pick
    ranked = []            # ranked[j] = LongTensor of candidate idx, best-first, for query j
    for j in range(M):
        sim = X @ Tg[j]                       # (N,)
        if args.order == "second":
            sim = sim * sim
        sim = torch.where(avail_mask, sim, torch.tensor(float("-inf"), device=dev))
        ranked.append(torch.argsort(sim, descending=True).tolist())

    selected = []
    selected_set = set()
    ptr = [0] * M                              # pointer into ranked[j]
    # round-robin over query_order until K selected
    stalled_rounds = 0
    while len(selected) < K:
        progressed = False
        for j in query_order:
            if len(selected) >= K:
                break
            lst = ranked[j]; p = ptr[j]
            # advance past already-selected candidates
            while p < N and lst[p] in selected_set:
                p += 1
            ptr[j] = p
            if p < N:
                cand = lst[p]
                ptr[j] = p + 1
                selected.append(cand); selected_set.add(cand)
                progressed = True
        if not progressed:
            stalled_rounds += 1
            if stalled_rounds > 1:
                raise RuntimeError("RR stalled: exhausted candidates before reaching K")
    assert len(selected) == len(set(selected)) == K
    print(f"[rr] order={args.order} M={M} K={K} perm_seed={args.perm_seed} "
          f"query_order[:8]={query_order[:8]}", flush=True)

    output = selected
    if args.subsample_indices is not None:
        mp = torch.load(args.subsample_indices, map_location="cpu")
        mp = mp.tolist() if torch.is_tensor(mp) else list(mp)
        if len(mp) != N: raise ValueError("subsample map size mismatch")
        output = [int(mp[i]) for i in selected]

    os.makedirs(args.out_cache_dir, exist_ok=True)
    meta = {"kernel": f"round_robin_{args.order}", "order": args.order,
            "perm_seed": args.perm_seed, "query_order": query_order,
            "num_select": K, "n_candidates": N, "n_target": M, "proj_dim": D,
            "train_grads": os.path.abspath(args.train_grads),
            "target_grads": os.path.abspath(args.target_grads)}
    json.dump({"indices": output, "metric": meta},
              open(os.path.join(args.out_cache_dir, "step_1.json"), "w"))
    print(f"[rr] wrote {args.out_cache_dir}/step_1.json ({len(output)})", flush=True)


if __name__ == "__main__":
    main()

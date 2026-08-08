#!/usr/bin/env python3
"""Zero-training-cost mechanism forensics for the 1%-vs-5% budget result (advice_0808 step 1).

Central hypothesis to test:
  DSMC faithfully matches the SKEWED observed query Q_d, while Random-K preserves broad candidate-pool
  coverage. At a very small budget (K=2707) target-aware selection may spend limited capacity matching
  the observed skew, ending up FARTHER from the balanced latent P* that we actually evaluate on;
  at K=13533 there is enough capacity that this gap shrinks.

Balanced P* reference = union of all 10 target draws (320 STEM + 320 HUM = 640 examples, globally
disjoint by construction) — a validation-only balanced reference, never trained on.

For each budget (1%, 5%) and each method, per draw d:
  D2(S, Q_d)  = ||M_S - M_{Q_d}||_F^2      (2nd moment distance to that draw's own skewed query)
  D2(S, P*)   = ||M_S - M_{P*}||_F^2       (distance to the balanced latent reference)
  D1 counterparts (mean-direction) reported too.
  M_P = E_{u~P}[u u^T] on unit-normalized projected gradients.
Plus selection-side diversity: Tulu source distribution/entropy, unique sources, gradient-space
effective rank, mean pairwise cosine similarity.
"""
import argparse, json, os
import numpy as np
import torch

SAVES = "/jizhicfs/karonhe/dataflex_saves"
ROOT = "/jizhicfs/karonhe/DataFlex_fa"
DRAWS = [f"stem80_draw{i}" for i in range(5)] + [f"hum80_draw{i}" for i in range(5)]
CAND_GRAD = f"{SAVES}/less_output/train/1/all_projected_grads.pt"
CAND_JSONL = f"{ROOT}/data/less_train_all.jsonl"


def unit(X):
    return X / X.norm(dim=1, keepdim=True).clamp_min(1e-12)


def moments(U):
    """mean direction mu and 2nd moment M = E[u u^T] for unit rows U."""
    return U.mean(0), (U.T @ U) / U.shape[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", nargs="+",
                    default=["dsmc", "randk", "randk_lenmatch", "second_rr", "less", "gist", "nice"])
    ap.add_argument("--subsample_rank", type=int, default=3000, help="rows for effective-rank SVD")
    ap.add_argument("--out", default=f"{ROOT}/experiments/less_aligned/results_summary/forensic_pstar_geometry.json")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    X = unit(torch.load(CAND_GRAD, map_location="cpu").float().to(dev))
    N = X.shape[0]

    # per-draw query moments + balanced P* (union of all 10 draws)
    Q = {}
    allT = []
    for d in DRAWS:
        T = unit(torch.load(f"{SAVES}/draw_{d}_output/target/1/all_projected_grads.pt",
                            map_location="cpu").float().to(dev))
        Q[d] = moments(T)
        allT.append(T)
    Tstar = torch.cat(allT, 0)
    assert Tstar.shape[0] == 640, Tstar.shape
    mu_star, M_star = moments(Tstar)
    print(f"[P*] balanced reference built from {Tstar.shape[0]} target examples "
          f"(union of 10 disjoint draws = 320 STEM + 320 HUM)")

    # candidate source labels (Tulu subsets)
    srcs = []
    with open(CAND_JSONL) as f:
        for line in f:
            if line.strip():
                srcs.append(json.loads(line).get("dataset", "unknown"))
    srcs = np.array(srcs)
    assert len(srcs) == N, (len(srcs), N)

    def sel_stats(idx):
        idx_t = torch.tensor(idx, device=dev)
        U = X.index_select(0, idx_t)
        mu, M = moments(U)
        s = srcs[np.array(idx)]
        vals, cnts = np.unique(s, return_counts=True)
        p = cnts / cnts.sum()
        ent = float(-(p * np.log(p)).sum())
        n = min(args.subsample_rank, U.shape[0])
        g = torch.Generator(device="cpu").manual_seed(0)
        sub = U.index_select(0, torch.randperm(U.shape[0], generator=g)[:n].to(dev))
        sv = torch.linalg.svdvals(sub.float())
        ev = (sv ** 2); ev = ev / ev.sum()
        eff_rank = float(torch.exp(-(ev * torch.log(ev.clamp_min(1e-12))).sum()))
        Gm = sub @ sub.T
        off = (Gm.sum() - Gm.diagonal().sum()) / (n * (n - 1))
        return {"mu": mu, "M": M, "source_dist": dict(zip(vals.tolist(), cnts.tolist())),
                "source_entropy": ent, "n_unique_sources": int(len(vals)),
                "eff_rank": eff_rank, "mean_pairwise_cos": float(off)}

    out = {"P_star": {"n": int(Tstar.shape[0]),
                      "note": "union of 10 disjoint target draws, balanced 320 STEM / 320 HUM"},
           "budgets": {}}
    for budget, prefix in [("1pct", "sel1pct"), ("5pct", "sel")]:
        out["budgets"][budget] = {}
        for m in args.methods:
            rows = []
            for d in DRAWS:
                p = f"{SAVES}/{prefix}_{d}_{m}/step_1.json"
                if not os.path.exists(p):
                    continue
                idx = json.load(open(p))["indices"]
                st = sel_stats(idx)
                mu_q, M_q = Q[d]
                rows.append({
                    "draw": d, "K": len(idx),
                    "D2_to_Q": float(((st["M"] - M_q) ** 2).sum()),
                    "D2_to_Pstar": float(((st["M"] - M_star) ** 2).sum()),
                    "D1_to_Q": float(((st["mu"] - mu_q) ** 2).sum()),
                    "D1_to_Pstar": float(((st["mu"] - mu_star) ** 2).sum()),
                    "source_entropy": st["source_entropy"],
                    "n_unique_sources": st["n_unique_sources"],
                    "eff_rank": st["eff_rank"],
                    "mean_pairwise_cos": st["mean_pairwise_cos"],
                    "source_dist": st["source_dist"]})
            if rows:
                agg = {k: float(np.mean([r[k] for r in rows])) for k in
                       ("D2_to_Q", "D2_to_Pstar", "D1_to_Q", "D1_to_Pstar",
                        "source_entropy", "eff_rank", "mean_pairwise_cos")}
                agg["n_draws"] = len(rows)
                out["budgets"][budget][m] = {"mean": agg, "per_draw": rows}
                print(f"[{budget}] {m:15s} D2->Q {agg['D2_to_Q']:.5f}  D2->P* {agg['D2_to_Pstar']:.5f}  "
                      f"srcH {agg['source_entropy']:.3f}  effrank {agg['eff_rank']:.0f}  "
                      f"pcos {agg['mean_pairwise_cos']:.4f}")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()

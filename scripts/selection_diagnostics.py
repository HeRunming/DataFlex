#!/usr/bin/env python3
"""
Selection Diagnostics: compute MMD-related analysis metrics for each selected subset.

Outputs per-method:
  - selected-target relevance (mean RBF kernel to target)
  - selected-selected redundancy (mean pairwise kernel within selected)
  - MMD²(S, T)
  - effective rank of selected embeddings
  - pairwise cosine similarity statistics
  - token count and length distribution
  - overlap between different methods' selected sets

Usage:
    python scripts/selection_diagnostics.py \
        --candidate_data data/flan_v2_100k.json \
        --target_data data/gsm8k_train_64.json \
        --results_dir experiments/less_aligned/results \
        --embed_model /jizhicfs/karonhe/models/sentence-transformers/all-MiniLM-L6-v2 \
        --output diagnostics_report.json
"""

import argparse
import json
import os
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional


def parse_args():
    parser = argparse.ArgumentParser(description="Selection Diagnostics")
    parser.add_argument("--candidate_data", type=str, required=True)
    parser.add_argument("--target_data", type=str, required=True)
    parser.add_argument("--results_dir", type=str, required=True,
                        help="Directory containing method subdirs with selected_indices.json")
    parser.add_argument("--embed_model", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--output", type=str, default="diagnostics_report.json")
    parser.add_argument("--sigma", type=str, default="auto",
                        help="RBF bandwidth ('auto' = median heuristic)")
    parser.add_argument("--max_subset_for_diagnostics", type=int, default=2000,
                        help="Subsample selected set for expensive O(n^2) metrics")
    return parser.parse_args()


def load_data(path):
    if path.endswith('.jsonl'):
        data = []
        with open(path, 'r') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
        return data
    else:
        with open(path, 'r') as f:
            return json.load(f)


def texts_from_data(data):
    texts = []
    for item in data:
        if isinstance(item, str):
            texts.append(item)
        elif isinstance(item, dict):
            parts = []
            for key in ['instruction', 'input', 'output', 'text', 'question', 'answer']:
                if key in item and item[key]:
                    parts.append(str(item[key]))
            texts.append("\n".join(parts) if parts else str(item))
        else:
            texts.append(str(item))
    return texts


def compute_embeddings(texts, embed_model, batch_size=64):
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(embed_model)
    return np.array(model.encode(texts, batch_size=batch_size, show_progress_bar=True,
                                  normalize_embeddings=True), dtype=np.float32)


def median_heuristic(X, subsample=2000):
    N = X.shape[0]
    rng = np.random.RandomState(42)
    idx = rng.choice(N, min(subsample, N), replace=False)
    X_sub = X[idx]
    sq_norms = np.sum(X_sub ** 2, axis=1)
    sq_dists = sq_norms[:, None] + sq_norms[None, :] - 2.0 * (X_sub @ X_sub.T)
    sq_dists = np.maximum(sq_dists, 0.0)
    triu_idx = np.triu_indices(len(X_sub), k=1)
    return max(float(np.median(np.sqrt(sq_dists[triu_idx]))), 1e-6)


def rbf_kernel(X, Y, sigma):
    X_sq = np.sum(X ** 2, axis=1, keepdims=True)
    Y_sq = np.sum(Y ** 2, axis=1, keepdims=True)
    sq_dists = X_sq + Y_sq.T - 2.0 * (X @ Y.T)
    sq_dists = np.maximum(sq_dists, 0.0)
    return np.exp(-sq_dists / (2.0 * sigma ** 2))


def compute_mmd_squared(X, Y, sigma):
    """Unbiased MMD² estimator."""
    K_XX = rbf_kernel(X, X, sigma)
    K_YY = rbf_kernel(Y, Y, sigma)
    K_XY = rbf_kernel(X, Y, sigma)
    n, m = K_XX.shape[0], K_YY.shape[0]
    np.fill_diagonal(K_XX, 0)
    np.fill_diagonal(K_YY, 0)
    return float(K_XX.sum() / (n * (n - 1)) + K_YY.sum() / (m * (m - 1)) - 2 * K_XY.mean())


def compute_effective_rank(X):
    """Effective rank via singular value entropy."""
    _, s, _ = np.linalg.svd(X, full_matrices=False)
    s = s[s > 1e-10]
    p = s / s.sum()
    H = -np.sum(p * np.log(p + 1e-12))
    return float(np.exp(H))


def compute_target_relevance(selected_embs, target_embs, sigma):
    """Mean RBF kernel value between selected and target."""
    K = rbf_kernel(selected_embs, target_embs, sigma)
    return float(K.mean())


def compute_redundancy(selected_embs, sigma):
    """Mean pairwise RBF kernel within selected set (excluding diagonal)."""
    K = rbf_kernel(selected_embs, selected_embs, sigma)
    n = K.shape[0]
    np.fill_diagonal(K, 0)
    return float(K.sum() / (n * (n - 1)))


def compute_pairwise_cosine_stats(selected_embs):
    """Pairwise cosine similarity stats (embeddings are already normalized)."""
    sim = selected_embs @ selected_embs.T
    n = sim.shape[0]
    triu_idx = np.triu_indices(n, k=1)
    pairwise = sim[triu_idx]
    return {
        "mean": float(np.mean(pairwise)),
        "std": float(np.std(pairwise)),
        "max": float(np.max(pairwise)),
        "min": float(np.min(pairwise)),
        "median": float(np.median(pairwise)),
    }


def compute_token_stats(selected_texts):
    """Basic token count statistics (word-level approximation)."""
    lengths = [len(t.split()) for t in selected_texts]
    return {
        "num_samples": len(lengths),
        "total_tokens_approx": int(sum(lengths)),
        "mean_length": float(np.mean(lengths)),
        "std_length": float(np.std(lengths)),
        "min_length": int(min(lengths)),
        "max_length": int(max(lengths)),
        "median_length": float(np.median(lengths)),
    }


def find_selected_indices(results_dir):
    """Find all methods with selected_indices.json."""
    methods = {}
    results_path = Path(results_dir)
    if not results_path.exists():
        return methods

    for method_dir in sorted(results_path.iterdir()):
        if not method_dir.is_dir():
            continue
        # Check directly
        idx_file = method_dir / "selected_indices.json"
        if idx_file.exists():
            methods[method_dir.name] = idx_file
            continue
        # Check subdirs (task/ratio/seed structure)
        for f in method_dir.rglob("selected_indices.json"):
            key = str(f.parent.relative_to(results_path))
            methods[key] = f
    return methods


def main():
    args = parse_args()

    print("=" * 70)
    print(" Selection Diagnostics")
    print("=" * 70)

    # Load and embed data
    print("\n[1/4] Loading data...")
    candidate_data = load_data(args.candidate_data)
    target_data = load_data(args.target_data)
    cand_texts = texts_from_data(candidate_data)
    target_texts = texts_from_data(target_data)
    print(f"  Candidates: {len(cand_texts)}, Targets: {len(target_texts)}")

    print("\n[2/4] Computing embeddings...")
    cand_embs = compute_embeddings(cand_texts, args.embed_model)
    target_embs = compute_embeddings(target_texts, args.embed_model)

    # Determine sigma
    if args.sigma == "auto":
        sigma = median_heuristic(cand_embs)
    else:
        sigma = float(args.sigma)
    print(f"  RBF sigma: {sigma:.6f}")

    # Find selected indices
    print("\n[3/4] Loading selection results...")
    methods = find_selected_indices(args.results_dir)
    print(f"  Found {len(methods)} method results")

    if not methods:
        print("  WARNING: No results found. Run selection first.")
        return

    # Compute diagnostics
    print("\n[4/4] Computing diagnostics...")
    all_diagnostics = {}
    all_indices = {}  # for overlap computation

    for method_key, idx_file in sorted(methods.items()):
        print(f"\n  --- {method_key} ---")
        with open(idx_file) as f:
            data = json.load(f)
        indices = data["indices"] if isinstance(data, dict) else data

        # Subsample for expensive metrics
        max_n = min(len(indices), args.max_subset_for_diagnostics)
        sub_indices = indices[:max_n]

        selected_embs = cand_embs[sub_indices]
        selected_texts = [cand_texts[i] for i in sub_indices]

        diag = {}
        diag["num_selected"] = len(indices)
        diag["selection_ratio"] = len(indices) / len(cand_texts)

        # Target relevance
        diag["target_relevance_rbf"] = compute_target_relevance(selected_embs, target_embs, sigma)

        # Redundancy
        if max_n > 1:
            diag["redundancy_rbf"] = compute_redundancy(selected_embs, sigma)

        # MMD²
        if max_n > 1:
            diag["mmd_squared"] = compute_mmd_squared(selected_embs, target_embs, sigma)

        # Effective rank
        if max_n > 1:
            diag["effective_rank"] = compute_effective_rank(selected_embs)

        # Pairwise cosine
        if max_n > 1:
            diag["pairwise_cosine"] = compute_pairwise_cosine_stats(selected_embs)

        # Token stats
        diag["token_stats"] = compute_token_stats(selected_texts)

        all_diagnostics[method_key] = diag
        all_indices[method_key] = set(indices)

        print(f"    relevance={diag['target_relevance_rbf']:.4f}, "
              f"redundancy={diag.get('redundancy_rbf', 'N/A')}, "
              f"MMD²={diag.get('mmd_squared', 'N/A')}, "
              f"erank={diag.get('effective_rank', 'N/A')}")

    # Compute pairwise overlaps
    print("\n  --- Method Overlaps ---")
    method_names = sorted(all_indices.keys())
    overlaps = {}
    for i, m1 in enumerate(method_names):
        for m2 in method_names[i + 1:]:
            s1, s2 = all_indices[m1], all_indices[m2]
            intersection = len(s1 & s2)
            union = len(s1 | s2)
            jaccard = intersection / union if union > 0 else 0
            overlap_key = f"{m1} ∩ {m2}"
            overlaps[overlap_key] = {
                "intersection": intersection,
                "jaccard": jaccard,
                "overlap_ratio_1": intersection / len(s1) if s1 else 0,
                "overlap_ratio_2": intersection / len(s2) if s2 else 0,
            }
            print(f"    {m1} ∩ {m2}: {intersection} ({jaccard:.3f} Jaccard)")

    # Save report
    report = {
        "sigma": sigma,
        "num_candidates": len(cand_texts),
        "num_targets": len(target_texts),
        "methods": all_diagnostics,
        "overlaps": overlaps,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\n  Report saved to: {output_path}")

    # Print summary table
    print("\n" + "=" * 90)
    print(f"{'Method':<30} {'N':>6} {'Relevance':>10} {'Redundancy':>11} "
          f"{'MMD²':>10} {'Erank':>8} {'Tokens':>8}")
    print("-" * 90)
    for method, diag in sorted(all_diagnostics.items()):
        n = diag["num_selected"]
        rel = f"{diag['target_relevance_rbf']:.4f}"
        red = f"{diag.get('redundancy_rbf', 0):.4f}" if 'redundancy_rbf' in diag else "N/A"
        mmd = f"{diag.get('mmd_squared', 0):.6f}" if 'mmd_squared' in diag else "N/A"
        erk = f"{diag.get('effective_rank', 0):.1f}" if 'effective_rank' in diag else "N/A"
        tok = str(diag["token_stats"]["total_tokens_approx"])
        print(f"{method:<30} {n:>6} {rel:>10} {red:>11} {mmd:>10} {erk:>8} {tok:>8}")
    print("=" * 90)


if __name__ == "__main__":
    main()

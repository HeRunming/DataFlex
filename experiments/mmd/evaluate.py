#!/usr/bin/env python3
"""
MMD Evaluation Script for DataFlex experiments.

Loads selected indices from experiment outputs and computes MMD diagnostics:
- MMD^2(S, T): Maximum Mean Discrepancy between selected subset and target
- Effective rank of the selected subset embeddings
- Target coverage: fraction of target points with a neighbor in the subset
- Redundancy: average pairwise similarity within the selected subset

Usage:
    python experiments/mmd/evaluate.py \
        --results_dir experiments/mmd/outputs \
        --embeddings_dir experiments/mmd/embeddings \
        --output experiments/mmd/results/mmd_evaluation_results.json
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


# ─── Kernel Utilities ────────────────────────────────────────────────────────


def median_heuristic(X: np.ndarray, Y: Optional[np.ndarray] = None) -> float:
    """Compute the median heuristic bandwidth for RBF kernel.

    Uses the median of pairwise distances between all points in X (and Y if given).
    """
    if Y is not None:
        Z = np.vstack([X, Y])
    else:
        Z = X

    # Subsample if too large to avoid memory issues
    max_samples = 2000
    if len(Z) > max_samples:
        indices = np.random.choice(len(Z), max_samples, replace=False)
        Z = Z[indices]

    # Compute pairwise squared distances
    norms_sq = np.sum(Z ** 2, axis=1)
    dists_sq = norms_sq[:, None] + norms_sq[None, :] - 2.0 * Z @ Z.T

    # Take upper triangle (exclude diagonal)
    triu_indices = np.triu_indices(len(Z), k=1)
    pairwise_dists = np.sqrt(np.maximum(dists_sq[triu_indices], 0.0))

    median_dist = np.median(pairwise_dists)
    # Bandwidth sigma = median_dist (so sigma^2 in the kernel)
    return float(median_dist) if median_dist > 0 else 1.0


def rbf_kernel(X: np.ndarray, Y: np.ndarray, sigma: float) -> np.ndarray:
    """Compute RBF (Gaussian) kernel matrix between X and Y.

    K(x, y) = exp(-||x - y||^2 / (2 * sigma^2))
    """
    X_norms = np.sum(X ** 2, axis=1)
    Y_norms = np.sum(Y ** 2, axis=1)
    dists_sq = X_norms[:, None] + Y_norms[None, :] - 2.0 * X @ Y.T
    dists_sq = np.maximum(dists_sq, 0.0)
    return np.exp(-dists_sq / (2.0 * sigma ** 2))


# ─── MMD Diagnostics ────────────────────────────────────────────────────────


def compute_mmd_squared(
    X: np.ndarray, Y: np.ndarray, sigma: Optional[float] = None
) -> float:
    """Compute unbiased MMD^2 estimate between samples X and Y using RBF kernel.

    MMD^2 = E[k(x,x')] - 2*E[k(x,y)] + E[k(y,y')]
    """
    if sigma is None:
        sigma = median_heuristic(X, Y)

    m = len(X)
    n = len(Y)

    K_XX = rbf_kernel(X, X, sigma)
    K_YY = rbf_kernel(Y, Y, sigma)
    K_XY = rbf_kernel(X, Y, sigma)

    # Unbiased estimator: exclude diagonal for K_XX and K_YY
    np.fill_diagonal(K_XX, 0.0)
    np.fill_diagonal(K_YY, 0.0)

    mmd_sq = (
        K_XX.sum() / (m * (m - 1))
        - 2.0 * K_XY.sum() / (m * n)
        + K_YY.sum() / (n * (n - 1))
    )
    return float(mmd_sq)


def compute_effective_rank(X: np.ndarray) -> float:
    """Compute the effective rank of a matrix via Shannon entropy of singular values.

    effective_rank = exp(H(p)) where p_i = sigma_i / sum(sigma_j)
    and H(p) = -sum(p_i * log(p_i))
    """
    # Center the data
    X_centered = X - X.mean(axis=0)
    _, singular_values, _ = np.linalg.svd(X_centered, full_matrices=False)

    # Normalize singular values to form a distribution
    sv_sum = singular_values.sum()
    if sv_sum < 1e-12:
        return 1.0

    p = singular_values / sv_sum
    # Remove zeros for numerical stability
    p = p[p > 1e-12]

    # Shannon entropy
    entropy = -np.sum(p * np.log(p))
    return float(np.exp(entropy))


def compute_target_coverage(
    selected_embeddings: np.ndarray,
    target_embeddings: np.ndarray,
    k: int = 5,
) -> float:
    """Compute target coverage: fraction of target points with at least one
    neighbor in the selected subset within the k-th nearest neighbor distance.

    Uses cosine similarity for neighbor detection.
    """
    # Normalize embeddings
    S_norm = selected_embeddings / (
        np.linalg.norm(selected_embeddings, axis=1, keepdims=True) + 1e-12
    )
    T_norm = target_embeddings / (
        np.linalg.norm(target_embeddings, axis=1, keepdims=True) + 1e-12
    )

    # Compute cosine similarity between target and selected
    similarities = T_norm @ S_norm.T  # (n_target, n_selected)

    # For each target point, find max similarity to any selected point
    max_sims = similarities.max(axis=1)

    # Compute threshold: median of k-th nearest neighbor distance within target
    target_self_sim = T_norm @ T_norm.T
    np.fill_diagonal(target_self_sim, -1.0)  # exclude self
    # Sort similarities descending and take k-th
    sorted_sims = np.sort(target_self_sim, axis=1)[:, ::-1]
    k_actual = min(k, sorted_sims.shape[1] - 1)
    knn_thresholds = sorted_sims[:, k_actual]
    threshold = np.median(knn_thresholds)

    # Coverage: fraction of target points with a selected neighbor above threshold
    coverage = float(np.mean(max_sims >= threshold))
    return coverage


def compute_redundancy(selected_embeddings: np.ndarray) -> float:
    """Compute redundancy as average pairwise cosine similarity in the selected set.

    Lower redundancy means more diverse selection.
    """
    # Normalize
    norms = np.linalg.norm(selected_embeddings, axis=1, keepdims=True) + 1e-12
    S_norm = selected_embeddings / norms

    # Pairwise cosine similarity
    sim_matrix = S_norm @ S_norm.T
    n = len(sim_matrix)
    if n < 2:
        return 0.0

    # Exclude diagonal and compute mean
    np.fill_diagonal(sim_matrix, 0.0)
    redundancy = sim_matrix.sum() / (n * (n - 1))
    return float(redundancy)


# ─── Experiment Loading ──────────────────────────────────────────────────────


def find_selected_indices(experiment_dir: str) -> Optional[np.ndarray]:
    """Find and load selected indices from an experiment output directory."""
    exp_path = Path(experiment_dir)

    # Look for common patterns of saved indices
    possible_files = [
        exp_path / "selected_indices.npy",
        exp_path / "selected_indices.json",
        exp_path / "trainer_state.json",
    ]

    for fpath in possible_files:
        if fpath.exists():
            if fpath.suffix == ".npy":
                return np.load(str(fpath))
            elif fpath.suffix == ".json":
                with open(fpath) as f:
                    data = json.load(f)
                if isinstance(data, list):
                    return np.array(data)
                elif "selected_indices" in data:
                    return np.array(data["selected_indices"])

    # Search recursively for any selected_indices files
    for npy_file in exp_path.rglob("selected_indices*.npy"):
        return np.load(str(npy_file))

    for json_file in exp_path.rglob("selected_indices*.json"):
        with open(json_file) as f:
            data = json.load(f)
        if isinstance(data, list):
            return np.array(data)
        elif "selected_indices" in data:
            return np.array(data["selected_indices"])

    return None


def discover_experiments(results_dir: str) -> Dict[str, List[str]]:
    """Discover all experiment directories organized by method and seed."""
    results_path = Path(results_dir)
    experiments = {}

    if not results_path.exists():
        return experiments

    for method_dir in sorted(results_path.iterdir()):
        if not method_dir.is_dir():
            continue
        method_name = method_dir.name
        seed_dirs = []
        for seed_dir in sorted(method_dir.iterdir()):
            if seed_dir.is_dir() and seed_dir.name.startswith("seed_"):
                seed_dirs.append(str(seed_dir))
            elif seed_dir.is_dir() and seed_dir.name.startswith("lambda_"):
                seed_dirs.append(str(seed_dir))
        if seed_dirs:
            experiments[method_name] = seed_dirs

    return experiments


# ─── Main Evaluation ─────────────────────────────────────────────────────────


def evaluate_experiment(
    experiment_dir: str,
    candidate_embeddings: np.ndarray,
    target_embeddings: np.ndarray,
    sigma: float,
) -> Optional[Dict[str, float]]:
    """Evaluate a single experiment run."""
    indices = find_selected_indices(experiment_dir)
    if indices is None:
        return None

    # Ensure indices are valid
    indices = indices.astype(int)
    valid_mask = (indices >= 0) & (indices < len(candidate_embeddings))
    indices = indices[valid_mask]

    if len(indices) == 0:
        return None

    selected = candidate_embeddings[indices]

    metrics = {
        "num_selected": int(len(indices)),
        "mmd_squared": compute_mmd_squared(selected, target_embeddings, sigma),
        "effective_rank": compute_effective_rank(selected),
        "target_coverage": compute_target_coverage(selected, target_embeddings),
        "redundancy": compute_redundancy(selected),
    }
    return metrics


def print_comparison_table(results: Dict[str, Dict[str, float]]) -> None:
    """Print a formatted comparison table of all methods."""
    if not results:
        print("No results to display.")
        return

    # Column headers
    headers = ["Method", "N_sel", "MMD^2(S,T)", "Eff.Rank", "Coverage", "Redundancy"]
    col_widths = [20, 8, 12, 10, 10, 12]

    # Print header
    header_line = " | ".join(
        h.ljust(w) for h, w in zip(headers, col_widths)
    )
    print("\n" + "=" * len(header_line))
    print(" MMD Evaluation Results")
    print("=" * len(header_line))
    print(header_line)
    print("-" * len(header_line))

    # Print rows sorted by MMD^2 (lower is better)
    sorted_methods = sorted(results.items(), key=lambda x: x[1].get("mmd_squared", 999))

    for method, metrics in sorted_methods:
        row = [
            method[:20].ljust(col_widths[0]),
            str(int(metrics.get("num_selected", 0))).ljust(col_widths[1]),
            f"{metrics.get('mmd_squared', 0):.6f}".ljust(col_widths[2]),
            f"{metrics.get('effective_rank', 0):.2f}".ljust(col_widths[3]),
            f"{metrics.get('target_coverage', 0):.4f}".ljust(col_widths[4]),
            f"{metrics.get('redundancy', 0):.6f}".ljust(col_widths[5]),
        ]
        print(" | ".join(row))

    print("=" * len(header_line))
    print("\nLegend:")
    print("  MMD^2(S,T)  - Lower is better (distribution alignment)")
    print("  Eff.Rank    - Higher is better (diversity)")
    print("  Coverage    - Higher is better (target representation)")
    print("  Redundancy  - Lower is better (less duplication)")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate MMD experiment results with diagnostics."
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        required=True,
        help="Directory containing experiment outputs organized by method/seed.",
    )
    parser.add_argument(
        "--embeddings_dir",
        type=str,
        required=True,
        help="Directory containing precomputed embeddings (candidate/target).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="mmd_evaluation_results.json",
        help="Path to save evaluation results as JSON.",
    )
    args = parser.parse_args()

    # Load embeddings
    candidate_path = os.path.join(args.embeddings_dir, "candidate_embeddings.npy")
    target_path = os.path.join(args.embeddings_dir, "target_embeddings.npy")

    if not os.path.exists(candidate_path):
        print(f"ERROR: Candidate embeddings not found at {candidate_path}")
        print("Run Step 0 (offline embedding computation) first.")
        return

    if not os.path.exists(target_path):
        print(f"ERROR: Target embeddings not found at {target_path}")
        print("Run Step 0 (offline embedding computation) first.")
        return

    print("Loading embeddings...")
    candidate_embeddings = np.load(candidate_path)
    target_embeddings = np.load(target_path)
    print(f"  Candidate embeddings shape: {candidate_embeddings.shape}")
    print(f"  Target embeddings shape:    {target_embeddings.shape}")

    # Compute bandwidth via median heuristic
    print("Computing bandwidth via median heuristic...")
    sigma = median_heuristic(candidate_embeddings, target_embeddings)
    print(f"  Sigma (median heuristic): {sigma:.4f}")

    # Discover experiments
    experiments = discover_experiments(args.results_dir)
    if not experiments:
        print(f"\nNo experiments found in {args.results_dir}")
        print("Make sure experiment outputs exist with selected_indices files.")
        return

    print(f"\nDiscovered {len(experiments)} methods:")
    for method, dirs in experiments.items():
        print(f"  {method}: {len(dirs)} run(s)")

    # Evaluate each method
    all_results = {}

    for method, exp_dirs in experiments.items():
        method_metrics = []

        for exp_dir in exp_dirs:
            metrics = evaluate_experiment(
                exp_dir, candidate_embeddings, target_embeddings, sigma
            )
            if metrics is not None:
                method_metrics.append(metrics)

        if method_metrics:
            # Average metrics across seeds
            avg_metrics = {}
            for key in method_metrics[0]:
                values = [m[key] for m in method_metrics]
                avg_metrics[key] = float(np.mean(values))
                if len(values) > 1:
                    avg_metrics[f"{key}_std"] = float(np.std(values))

            avg_metrics["num_runs"] = len(method_metrics)
            all_results[method] = avg_metrics
        else:
            print(f"  WARNING: No valid results for method '{method}'")

    # Print comparison table
    print_comparison_table(all_results)

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_data = {
        "config": {
            "results_dir": args.results_dir,
            "embeddings_dir": args.embeddings_dir,
            "sigma": sigma,
            "candidate_shape": list(candidate_embeddings.shape),
            "target_shape": list(target_embeddings.shape),
        },
        "results": all_results,
    }

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()

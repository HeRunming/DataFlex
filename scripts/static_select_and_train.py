#!/usr/bin/env python3
"""
Static Selection + Training Pipeline for Paper Experiments.

Implements the standard targeted instruction tuning workflow:
  1. Load candidate pool and target set
  2. Run offline selection (MMD, mean-target-similarity, random, etc.)
  3. Generate selected_subset.json (directly consumable by LlamaFactory)
  4. Train on the fixed selected subset

Aligned with LESS / TSDS paper experimental protocol:
  - Selection is done ONCE, offline
  - Training uses a FIXED subset (no dynamic updates)
  - Evaluation is on held-out benchmarks

Usage:
    python scripts/static_select_and_train.py select \
        --method mmd_emb_rbf \
        --candidate_data data/flan_v2_100k.json \
        --target_data data/gsm8k_train_64.json \
        --selection_ratio 0.05 \
        --output_dir ./results/mmd_emb_rbf_gsm8k_5pct

    python scripts/static_select_and_train.py train \
        --base_config experiments/less_aligned/configs/train_llama7b_lora.yaml \
        --selected_indices ./results/mmd_emb_rbf_gsm8k_5pct/selected_indices.json \
        --candidate_data data/flan_v2_100k.json \
        --output_dir ./results/mmd_emb_rbf_gsm8k_5pct/model
"""

import argparse
import json
import os
import sys
import subprocess
import numpy as np
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Static Selection + Training Pipeline")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # SELECT command
    sel = subparsers.add_parser("select", help="Run offline data selection")
    sel.add_argument("--method", type=str, required=True,
                     choices=["random", "mmd_emb_rbf", "mean_target_sim",
                              "max_target_sim", "mean_target_rbf", "full"],
                     help="Selection method")
    sel.add_argument("--candidate_data", type=str, required=True,
                     help="Path to candidate pool (JSON/JSONL)")
    sel.add_argument("--target_data", type=str, required=True,
                     help="Path to target set (JSON/JSONL)")
    sel.add_argument("--embed_model", type=str, default="sentence-transformers/all-MiniLM-L6-v2",
                     help="Embedding model for embedding-based methods")
    sel.add_argument("--selection_ratio", type=float, default=0.05,
                     help="Fraction of candidate pool to select")
    sel.add_argument("--num_select", type=int, default=None,
                     help="Explicit number to select (overrides selection_ratio)")
    sel.add_argument("--output_dir", type=str, required=True)
    sel.add_argument("--sigma", type=str, default="auto")
    sel.add_argument("--seed", type=int, default=42)

    # TRAIN command
    trn = subparsers.add_parser("train", help="Train on selected subset")
    trn.add_argument("--base_config", type=str, required=True,
                     help="Base LlamaFactory/DataFlex YAML config for SFT")
    trn.add_argument("--selected_indices", type=str, required=True,
                     help="Path to selected_indices.json from select step")
    trn.add_argument("--candidate_data", type=str, required=True,
                     help="Path to original candidate pool (to extract subset)")
    trn.add_argument("--output_dir", type=str, required=True)
    trn.add_argument("--seed", type=int, default=42)

    # PIPELINE command (select + train)
    pipe = subparsers.add_parser("pipeline", help="Run select then train")
    pipe.add_argument("--method", type=str, required=True,
                      choices=["random", "mmd_emb_rbf", "mean_target_sim",
                               "max_target_sim", "mean_target_rbf", "full"])
    pipe.add_argument("--candidate_data", type=str, required=True)
    pipe.add_argument("--target_data", type=str, required=True)
    pipe.add_argument("--embed_model", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    pipe.add_argument("--selection_ratio", type=float, default=0.05)
    pipe.add_argument("--num_select", type=int, default=None)
    pipe.add_argument("--base_config", type=str, required=True)
    pipe.add_argument("--output_dir", type=str, required=True)
    pipe.add_argument("--sigma", type=str, default="auto")
    pipe.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def load_data(data_path):
    """Load data from JSON (alpaca format) or JSONL."""
    if data_path.endswith('.jsonl'):
        data = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
        return data
    else:
        with open(data_path, 'r', encoding='utf-8') as f:
            return json.load(f)


def count_samples(data_path):
    """Count samples in a JSON or JSONL file."""
    return len(load_data(data_path))


def compute_embeddings(texts, embed_model, batch_size=64):
    """Compute embeddings using sentence-transformers."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise ImportError("pip install sentence-transformers")

    model = SentenceTransformer(embed_model)
    embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=True,
                              normalize_embeddings=True)
    return np.array(embeddings, dtype=np.float32)


def texts_from_data(data):
    """Extract text representation from data items for embedding."""
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


def median_heuristic(X, subsample=2000):
    """Compute median heuristic bandwidth for RBF kernel."""
    N = X.shape[0]
    rng = np.random.RandomState(42)
    if N > subsample:
        idx = rng.choice(N, subsample, replace=False)
        X_sub = X[idx]
    else:
        X_sub = X
    sq_norms = np.sum(X_sub ** 2, axis=1)
    sq_dists = sq_norms[:, None] + sq_norms[None, :] - 2.0 * (X_sub @ X_sub.T)
    sq_dists = np.maximum(sq_dists, 0.0)
    triu_idx = np.triu_indices(len(X_sub), k=1)
    return max(float(np.median(np.sqrt(sq_dists[triu_idx]))), 1e-6)


def run_selection(args):
    """Run offline data selection."""
    os.makedirs(args.output_dir, exist_ok=True)

    total_candidates = count_samples(args.candidate_data)
    num_select = args.num_select or int(total_candidates * args.selection_ratio)
    print(f"[Selection] Method: {args.method}")
    print(f"[Selection] Candidates: {total_candidates}, Selecting: {num_select} "
          f"({num_select/total_candidates*100:.1f}%)")

    if args.method == "random":
        rng = np.random.RandomState(args.seed)
        selected = rng.choice(total_candidates, size=num_select, replace=False).tolist()

    elif args.method == "full":
        selected = list(range(total_candidates))

    elif args.method == "mmd_emb_rbf":
        # Use offline MMD selector (exact marginal greedy)
        cmd = [
            sys.executable, "src/dataflex/offline_selector/offline_mmd_selector.py",
            "--candidate_path", args.candidate_data,
            "--query_path", args.target_data,
            "--embed_model", args.embed_model,
            "--save_dir", os.path.join(args.output_dir, "mmd_cache"),
            "--mode", "select",
            "--num_select", str(num_select),
            "--sigma", args.sigma,
        ]
        subprocess.run(cmd, check=True)
        selected = np.load(os.path.join(args.output_dir, "mmd_cache", "selected_indices.npy")).tolist()

    elif args.method == "mean_target_sim":
        # Mean target similarity: r_T(x) = (1/|T|) Σ_t k(x, t), then top-k
        # This is "MMD without redundancy" / "target relevance only" baseline
        print("[Selection] Computing embeddings...")
        candidate_data = load_data(args.candidate_data)
        target_data = load_data(args.target_data)
        cand_texts = texts_from_data(candidate_data)
        target_texts = texts_from_data(target_data)

        cand_embs = compute_embeddings(cand_texts, args.embed_model)
        target_embs = compute_embeddings(target_texts, args.embed_model)

        # Compute mean target similarity (cosine sim since embeddings are normalized)
        # For normalized vectors: cosine_sim = dot product
        print("[Selection] Computing mean target similarity...")
        sim_matrix = cand_embs @ target_embs.T  # (N_cand, N_target)
        mean_sim = sim_matrix.mean(axis=1)  # (N_cand,)

        # Select top-k by mean similarity
        top_indices = np.argsort(mean_sim)[::-1][:num_select]
        selected = top_indices.tolist()

    elif args.method == "max_target_sim":
        # Max target similarity: max_t sim(x, t), then top-k
        # This is strict "nearest neighbor to any target example"
        print("[Selection] Computing embeddings...")
        candidate_data = load_data(args.candidate_data)
        target_data = load_data(args.target_data)
        cand_texts = texts_from_data(candidate_data)
        target_texts = texts_from_data(target_data)

        cand_embs = compute_embeddings(cand_texts, args.embed_model)
        target_embs = compute_embeddings(target_texts, args.embed_model)

        print("[Selection] Computing max target similarity...")
        sim_matrix = cand_embs @ target_embs.T
        max_sim = sim_matrix.max(axis=1)

        top_indices = np.argsort(max_sim)[::-1][:num_select]
        selected = top_indices.tolist()

    elif args.method == "mean_target_rbf":
        # Mean target RBF kernel: r_T(x) = (1/|T|) Σ_t k_RBF(x, t), then top-k
        # This is "MMD without redundancy penalty" — the exact ablation baseline.
        # Uses same RBF kernel as mmd_emb_rbf for fair comparison.
        print("[Selection] Computing embeddings...")
        candidate_data = load_data(args.candidate_data)
        target_data = load_data(args.target_data)
        cand_texts = texts_from_data(candidate_data)
        target_texts = texts_from_data(target_data)

        cand_embs = compute_embeddings(cand_texts, args.embed_model)
        target_embs = compute_embeddings(target_texts, args.embed_model)

        # Compute sigma (same median heuristic as MMD)
        sigma = float(args.sigma) if args.sigma != "auto" else median_heuristic(cand_embs)
        print(f"[Selection] RBF sigma (median heuristic): {sigma:.6f}")

        # Compute RBF target relevance
        print("[Selection] Computing RBF target relevance...")
        N_cand = cand_embs.shape[0]
        N_target = target_embs.shape[0]
        relevance = np.zeros(N_cand, dtype=np.float64)

        chunk_size = 5000
        cand_sq = np.sum(cand_embs ** 2, axis=1, keepdims=True)
        for t_start in range(0, N_target, chunk_size):
            t_end = min(t_start + chunk_size, N_target)
            tgt_chunk = target_embs[t_start:t_end]
            tgt_sq = np.sum(tgt_chunk ** 2, axis=1, keepdims=True)
            sq_dists = cand_sq + tgt_sq.T - 2.0 * (cand_embs @ tgt_chunk.T)
            sq_dists = np.maximum(sq_dists, 0.0)
            K_chunk = np.exp(-sq_dists / (2.0 * sigma ** 2))
            relevance += K_chunk.sum(axis=1)
        relevance /= N_target

        # Select top-k by RBF relevance (no redundancy penalty)
        top_indices = np.argsort(relevance)[::-1][:num_select]
        selected = top_indices.tolist()

    else:
        raise ValueError(f"Unknown method: {args.method}")

    # Save selected indices
    indices_path = os.path.join(args.output_dir, "selected_indices.json")
    with open(indices_path, 'w') as f:
        json.dump({"indices": selected, "metadata": {
            "method": args.method,
            "num_candidates": total_candidates,
            "num_selected": len(selected),
            "selection_ratio": len(selected) / total_candidates,
            "seed": args.seed,
            "candidate_data": args.candidate_data,
            "target_data": args.target_data,
        }}, f, indent=2)
    print(f"[Selection] Saved {len(selected)} indices to {indices_path}")

    # Generate selected_subset.json (directly consumable by LlamaFactory)
    print("[Selection] Generating selected_subset.json...")
    candidate_data = load_data(args.candidate_data)
    subset_data = [candidate_data[i] for i in selected]
    subset_path = os.path.join(args.output_dir, "selected_subset.json")
    with open(subset_path, 'w', encoding='utf-8') as f:
        json.dump(subset_data, f, ensure_ascii=False, indent=2)
    print(f"[Selection] Saved selected subset ({len(subset_data)} samples) to {subset_path}")

    return selected


def run_training(args):
    """Train on a fixed selected subset by generating subset JSON and calling LlamaFactory."""
    # Load selected indices
    with open(args.selected_indices) as f:
        data = json.load(f)
    indices = data["indices"] if isinstance(data, dict) else data

    # Generate subset JSON if not already present
    subset_path = os.path.join(os.path.dirname(args.selected_indices), "selected_subset.json")
    if not os.path.exists(subset_path):
        print(f"[Training] Generating selected_subset.json from {args.candidate_data}...")
        candidate_data = load_data(args.candidate_data)
        subset_data = [candidate_data[i] for i in indices]
        with open(subset_path, 'w', encoding='utf-8') as f:
            json.dump(subset_data, f, ensure_ascii=False, indent=2)
        print(f"[Training] Saved {len(subset_data)} samples to {subset_path}")

    print(f"[Training] Using {len(indices)} selected samples")
    print(f"[Training] Subset: {subset_path}")
    print(f"[Training] Base config: {args.base_config}")
    print(f"[Training] Output: {args.output_dir}")

    # Register subset in a temporary dataset_info.json
    dataset_info_path = os.path.join(args.output_dir, "dataset_info.json")
    os.makedirs(args.output_dir, exist_ok=True)
    dataset_info = {
        "selected_subset": {
            "file_name": os.path.abspath(subset_path),
            "formatting": "alpaca",
        }
    }
    with open(dataset_info_path, 'w') as f:
        json.dump(dataset_info, f, indent=2)

    # Call dataflex-cli train with the subset dataset
    cmd = [
        "dataflex-cli", "train", args.base_config,
        f"dataset=selected_subset",
        f"dataset_dir={args.output_dir}",
        f"output_dir={args.output_dir}",
        f"seed={args.seed}",
    ]
    print(f"[Training] Running: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"[Training] WARNING: Training exited with code {result.returncode}")
    else:
        print(f"[Training] Training complete. Model saved to {args.output_dir}")


def main():
    args = parse_args()

    if args.command == "select":
        run_selection(args)
    elif args.command == "train":
        run_training(args)
    elif args.command == "pipeline":
        run_selection(args)
        args.selected_indices = os.path.join(args.output_dir, "selected_indices.json")
        args.candidate_data = args.candidate_data
        run_training(args)
    else:
        print("Usage: python scripts/static_select_and_train.py {select|train|pipeline} ...")
        sys.exit(1)


if __name__ == "__main__":
    main()

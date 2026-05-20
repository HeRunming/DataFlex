import os
import json
import argparse
import numpy as np

# ===== auto optional embedding backends =====
try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
except Exception:
    VLLM_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    ST_AVAILABLE = True
except Exception:
    ST_AVAILABLE = False

from dataflex.utils.logging import logger


class OfflineMMDSelector:
    """
    Offline preprocessing script for MMD-based data selection.
    Computes embeddings for candidate pool and target set, then performs
    greedy MMD selection to output selected indices.
    """

    def __init__(self,
                 candidate_path: str = None,
                 query_path: str = None,
                 embed_model: str = "Qwen/Qwen3-Embedding-0.6B",
                 embed_method: str = "auto",
                 batch_size: int = 32,
                 save_dir: str = "./mmd_outputs",
                 kernel: str = "rbf",
                 sigma = "auto",
                 lambda_redundancy: float = 0.5,
                 num_select: int = 5000,
                 mode: str = "embed"):
        """
        Args:
            candidate_path: Path to candidate pool data (alpaca JSON or JSONL).
            query_path: Path to target set data (alpaca JSON or JSONL).
            embed_model: Model name/path for embedding.
            embed_method: Embedding backend - "auto", "vllm", or "sentence-transformer".
            batch_size: Batch size for embedding computation.
            save_dir: Directory to save embeddings and results.
            kernel: Kernel type for MMD ("rbf").
            sigma: Kernel bandwidth - "auto" for median heuristic, or a float value.
            lambda_redundancy: Trade-off between target relevance and redundancy (0 to 1).
            num_select: Number of samples to select from candidate pool.
            mode: Operation mode - "embed" (compute embeddings only) or "select" (full pipeline).
        """
        self.candidate_path = candidate_path
        self.query_path = query_path
        self.embed_model = embed_model
        self.embed_method = embed_method
        self.batch_size = batch_size
        self.save_dir = save_dir
        self.kernel = kernel
        self.sigma = sigma
        self.lambda_redundancy = lambda_redundancy
        self.num_select = num_select
        self.mode = mode

        os.makedirs(self.save_dir, exist_ok=True)

    # ---------- Data Loading Methods ----------
    def _load_alpaca_json(self, path):
        """
        Load data from alpaca JSON format (instruction/input/output) or JSONL format.
        Returns a list of concatenated text strings.
        """
        if path.endswith('.jsonl'):
            data = []
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        data.append(json.loads(line))
        else:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)

        texts = [
            "\n".join([
                f"Instruction: {item.get('instruction', '')}",
                f"Input: {item.get('input', '')}",
                f"Output: {item.get('output', '')}",
                f"Prediction:{item.get('prediction', '')}"
            ])
            for item in data
        ]
        return texts

    # ---------- Embedding Method ----------
    def _embed_texts(self, texts):
        """
        Auto mode tries embedding backends in order:
        1) vLLM (preferred)
        2) sentence-transformers (fallback)
        3) Raises error if neither is available
        """

        # -------- 1. Prefer vLLM --------
        if (VLLM_AVAILABLE and self.embed_method == "auto") or self.embed_method == "vllm":
            try:
                logger.info(f"[EMBED] Using vLLM model: {self.embed_model}")
                llm = LLM(model=self.embed_model, trust_remote_code=True, task="embed")

                outputs = llm.embed(texts)  # [N, D]
                embs = [o.outputs.embedding for o in outputs]
                embs = np.array(embs, dtype=np.float32)

                # normalize
                norms = np.linalg.norm(embs, axis=1, keepdims=True)
                norms = np.maximum(norms, 1e-12)
                embs = embs / norms

                return np.ascontiguousarray(embs)

            except Exception as e:
                logger.warning(f"[EMBED] vLLM available but embedding failed: {e}")

        # -------- 2. Fallback: sentence-transformers --------
        if (ST_AVAILABLE and self.embed_method == "auto") or self.embed_method == "sentence-transformer":
            try:
                logger.info(f"[EMBED] Using SentenceTransformer: {self.embed_model}")
                model = SentenceTransformer(self.embed_model)
                embs = model.encode(
                    texts,
                    batch_size=self.batch_size,
                    show_progress_bar=True
                ).astype(np.float32)

                norms = np.linalg.norm(embs, axis=1, keepdims=True)
                norms = np.maximum(norms, 1e-12)
                embs = embs / norms

                return np.ascontiguousarray(embs)

            except Exception as e:
                raise RuntimeError(
                    f"SentenceTransformer available but embedding failed: {e}"
                )

        # -------- 3. Neither backend available --------
        raise RuntimeError(
            "No available embedding backend!\n"
            "Please install at least one of the following:\n"
            "  - vLLM: pip install vllm\n"
            "  - sentence-transformers: pip install sentence-transformers"
        )

    # ---------- Median Heuristic ----------
    def _median_heuristic(self, X, subsample=2000):
        """
        Compute median pairwise distance for RBF kernel bandwidth selection.
        Subsamples data for efficiency if dataset is large.

        Args:
            X: numpy array of shape (N, D)
            subsample: max number of samples to use for computation

        Returns:
            sigma: median pairwise distance (float)
        """
        N = X.shape[0]
        if N > subsample:
            indices = np.random.choice(N, subsample, replace=False)
            X_sub = X[indices]
        else:
            X_sub = X

        # Compute pairwise squared distances
        # ||x - y||^2 = ||x||^2 + ||y||^2 - 2 * x^T y
        norms_sq = np.sum(X_sub ** 2, axis=1)
        dists_sq = norms_sq[:, None] + norms_sq[None, :] - 2.0 * X_sub @ X_sub.T

        # Extract upper triangle (exclude diagonal)
        triu_indices = np.triu_indices(X_sub.shape[0], k=1)
        pairwise_dists = np.sqrt(np.maximum(dists_sq[triu_indices], 0.0))

        sigma = float(np.median(pairwise_dists))
        logger.info(f"[MMD] Median heuristic sigma = {sigma:.6f}")
        return sigma

    # ---------- Target Relevance ----------
    def _compute_target_relevance(self, candidates, targets, sigma):
        """
        Compute target relevance score r_T(x) = (1/|T|) * sum_{t in T} k(x, t)
        for all candidate points x.

        Uses RBF kernel: k(x, y) = exp(-||x - y||^2 / (2 * sigma^2))

        Args:
            candidates: numpy array (N_c, D) - candidate embeddings
            targets: numpy array (N_t, D) - target embeddings
            sigma: kernel bandwidth

        Returns:
            relevance: numpy array (N_c,) - target relevance for each candidate
        """
        N_c = candidates.shape[0]
        N_t = targets.shape[0]
        two_sigma_sq = 2.0 * sigma * sigma

        # Compute in batches to avoid memory issues
        batch_size = 512
        relevance = np.zeros(N_c, dtype=np.float64)

        for i in range(0, N_c, batch_size):
            batch = candidates[i:i + batch_size]  # (B, D)
            # Pairwise squared distances between batch and all targets
            # (B, D) vs (N_t, D) -> (B, N_t)
            batch_norms = np.sum(batch ** 2, axis=1, keepdims=True)  # (B, 1)
            target_norms = np.sum(targets ** 2, axis=1, keepdims=True).T  # (1, N_t)
            dists_sq = batch_norms + target_norms - 2.0 * batch @ targets.T  # (B, N_t)
            dists_sq = np.maximum(dists_sq, 0.0)

            # RBF kernel values
            kernel_vals = np.exp(-dists_sq / two_sigma_sq)  # (B, N_t)
            relevance[i:i + batch_size] = kernel_vals.mean(axis=1)

        return relevance

    # ---------- Compute Embeddings ----------
    def compute_embeddings(self):
        """
        Compute and save embeddings for both candidate pool and target set.
        Saves as .npy files with a metadata JSON.
        """
        logger.info("[MMD] Computing candidate embeddings...")
        candidate_texts = self._load_alpaca_json(self.candidate_path)
        logger.info(f"[MMD] Loaded {len(candidate_texts)} candidates from {self.candidate_path}")
        candidate_embs = self._embed_texts(candidate_texts)

        candidate_emb_path = os.path.join(self.save_dir, "candidate_embeddings.npy")
        np.save(candidate_emb_path, candidate_embs)
        logger.info(f"[MMD] Candidate embeddings saved to {candidate_emb_path}, shape={candidate_embs.shape}")

        logger.info("[MMD] Computing target embeddings...")
        if self.query_path and os.path.exists(self.query_path):
            target_texts = self._load_alpaca_json(self.query_path)
            logger.info(f"[MMD] Loaded {len(target_texts)} targets from {self.query_path}")
        else:
            logger.info("[MMD] No target set provided — using first 100 candidates as targets")
            target_texts = candidate_texts[:100]

        target_embs = self._embed_texts(target_texts)

        target_emb_path = os.path.join(self.save_dir, "target_embeddings.npy")
        np.save(target_emb_path, target_embs)
        logger.info(f"[MMD] Target embeddings saved to {target_emb_path}, shape={target_embs.shape}")

        # Save metadata
        metadata = {
            "candidate_path": self.candidate_path,
            "query_path": self.query_path,
            "embed_model": self.embed_model,
            "embed_method": self.embed_method,
            "num_candidates": candidate_embs.shape[0],
            "num_targets": target_embs.shape[0],
            "embedding_dim": int(candidate_embs.shape[1]),
        }
        metadata_path = os.path.join(self.save_dir, "embedding_metadata.json")
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"[MMD] Metadata saved to {metadata_path}")

        return candidate_embs, target_embs

    # ---------- Greedy MMD Selection ----------
    def greedy_mmd_select(self):
        """
        Run exact marginal greedy MMD selection on saved embeddings.

        At each step, selects the candidate that minimizes MMD²(S∪{x}, T):
            Δ(x) = r_T(x) - (1/(m+1)) * [r_S(x) + k(x,x)/2]

        For RBF kernel, k(x,x) = 1 for all x, so the self-kernel term
        doesn't affect the argmax and we get:
            Δ(x) = r_T(x) - (1/(m+1)) * [r_S(x) + 0.5]

        This is the SAME algorithm as MMDSelector._greedy_mmd_exact to ensure
        offline and online selection produce identical results.

        Returns:
            selected_indices: numpy array of selected candidate indices
        """
        # Load embeddings
        candidate_emb_path = os.path.join(self.save_dir, "candidate_embeddings.npy")
        target_emb_path = os.path.join(self.save_dir, "target_embeddings.npy")

        if not os.path.exists(candidate_emb_path) or not os.path.exists(target_emb_path):
            raise FileNotFoundError(
                f"Embeddings not found in {self.save_dir}. "
                "Run compute_embeddings() first or set mode='embed'."
            )

        candidates = np.load(candidate_emb_path).astype(np.float64)
        targets = np.load(target_emb_path).astype(np.float64)

        N_c = candidates.shape[0]
        num_select = min(self.num_select, N_c)
        logger.info(f"[MMD] Exact marginal greedy selection: choosing {num_select} from {N_c} candidates")

        # Determine sigma
        if self.sigma == "auto":
            sigma = self._median_heuristic(candidates, subsample=2000)
        else:
            sigma = float(self.sigma)
        logger.info(f"[MMD] RBF bandwidth sigma = {sigma:.6f}")

        # Precompute target relevance: r_T(x_i) = (1/|T|) Σ_t k(x_i, t)
        logger.info("[MMD] Computing target relevance scores...")
        target_relevance = self._compute_target_relevance(candidates, targets, sigma)

        # For RBF kernel, k(x,x) = 1 for all x (self-kernel is constant)
        self_kernel = 1.0  # scalar for RBF

        # Greedy exact marginal MMD selection
        selected_indices = []
        available_mask = np.ones(N_c, dtype=bool)
        # Running sum: Σ_{s∈S} k(x_i, s) for all candidates i
        selected_kernel_sum = np.zeros(N_c, dtype=np.float64)

        logger.info("[MMD] Starting exact marginal greedy MMD selection...")
        for step in range(num_select):
            if step > 0 and step % 500 == 0:
                logger.info(f"[MMD] Selected {step}/{num_select} samples...")

            m = len(selected_indices)  # current |S|

            # Exact marginal: Δ(x) = r_T(x) - (1/(m+1)) * [r_S(x) + k(x,x)/2]
            if m == 0:
                # First selection: pick highest target relevance
                scores = target_relevance.copy()
            else:
                scores = (
                    target_relevance
                    - (1.0 / (m + 1)) * (selected_kernel_sum + self_kernel / 2.0)
                )

            scores[~available_mask] = -np.inf

            # Select best candidate
            best_idx = int(np.argmax(scores))
            selected_indices.append(best_idx)
            available_mask[best_idx] = False

            # Incremental update: add k(x_i, x_best) for all i
            new_point = candidates[best_idx:best_idx + 1]  # (1, D)
            dists_sq = np.sum((candidates - new_point) ** 2, axis=1)  # (N_c,)
            kernel_vals = np.exp(-dists_sq / (2.0 * sigma * sigma))  # (N_c,)
            selected_kernel_sum += kernel_vals

        selected_indices = np.array(selected_indices, dtype=np.int64)

        # Save results
        indices_path = os.path.join(self.save_dir, "selected_indices.npy")
        np.save(indices_path, selected_indices)
        logger.info(f"[MMD] Selected {len(selected_indices)} indices saved to {indices_path}")

        # Save selection metadata
        select_metadata = {
            "algorithm": "exact_marginal_greedy_mmd",
            "num_select": num_select,
            "num_candidates": N_c,
            "num_targets": targets.shape[0],
            "kernel": self.kernel,
            "sigma": float(sigma),
            "lambda_redundancy": "N/A (exact marginal does not use lambda)",
        }
        select_meta_path = os.path.join(self.save_dir, "selection_metadata.json")
        with open(select_meta_path, 'w', encoding='utf-8') as f:
            json.dump(select_metadata, f, indent=2)
        logger.info(f"[MMD] Selection metadata saved to {select_meta_path}")

        return selected_indices

    # ---------- Main Entry Point ----------
    def selector(self):
        """
        Main entry point: compute embeddings then run greedy MMD selection.
        If mode is 'embed', only embeddings are computed and saved.
        If mode is 'select', embeddings are computed and greedy selection is performed.
        """
        logger.info("[MMD] === Starting Offline MMD Selection Pipeline ===")

        # Step 1: Compute embeddings
        logger.info("[MMD] Step 1: Computing embeddings...")
        self.compute_embeddings()

        if self.mode == "embed":
            logger.info("[MMD] Mode is 'embed' — skipping selection step.")
            return None

        # Step 2: Greedy MMD selection (mode == "select")
        logger.info("[MMD] Step 2: Running greedy MMD selection...")
        selected_indices = self.greedy_mmd_select()
        logger.info(f"[MMD] === Pipeline complete. Selected {len(selected_indices)} samples. ===")
        return selected_indices


def parse_args():
    parser = argparse.ArgumentParser(
        description="Offline MMD-based Data Selection"
    )
    parser.add_argument(
        "--candidate_path", type=str, required=True,
        help="Path to candidate pool data (alpaca JSON or JSONL)"
    )
    parser.add_argument(
        "--query_path", type=str, default=None,
        help="Path to target set data (alpaca JSON or JSONL)"
    )
    parser.add_argument(
        "--embed_model", type=str, default="Qwen/Qwen3-Embedding-0.6B",
        help="Model name/path for embedding"
    )
    parser.add_argument(
        "--embed_method", type=str, default="auto",
        choices=["auto", "vllm", "sentence-transformer"],
        help="Embedding backend to use"
    )
    parser.add_argument(
        "--batch_size", type=int, default=32,
        help="Batch size for embedding computation"
    )
    parser.add_argument(
        "--save_dir", type=str, default="./mmd_outputs",
        help="Directory to save embeddings and results"
    )
    parser.add_argument(
        "--kernel", type=str, default="rbf",
        help="Kernel type for MMD"
    )
    parser.add_argument(
        "--sigma", type=str, default="auto",
        help="Kernel bandwidth ('auto' for median heuristic or a float value)"
    )
    parser.add_argument(
        "--lambda_redundancy", type=float, default=0.5,
        help="Trade-off between target relevance and redundancy (0 to 1)"
    )
    parser.add_argument(
        "--num_select", type=int, default=5000,
        help="Number of samples to select from candidate pool"
    )
    parser.add_argument(
        "--mode", type=str, default="select",
        choices=["embed", "select"],
        help="Operation mode: 'embed' for embeddings only, 'select' for full pipeline"
    )
    return parser.parse_args()


if __name__ == "__main__":
    # === Usage Example (programmatic) ===
    # mmd = OfflineMMDSelector(
    #     candidate_path="path/to/candidates.json",
    #     query_path="path/to/targets.json",
    #     embed_model="Qwen/Qwen3-Embedding-0.6B",
    #     embed_method="auto",
    #     batch_size=32,
    #     save_dir="./mmd_outputs",
    #     num_select=5000,
    #     lambda_redundancy=0.5,
    #     sigma="auto",
    # )
    # mmd.selector()

    args = parse_args()
    sigma_val = args.sigma if args.sigma == "auto" else float(args.sigma)

    mmd = OfflineMMDSelector(
        candidate_path=args.candidate_path,
        query_path=args.query_path,
        embed_model=args.embed_model,
        embed_method=args.embed_method,
        batch_size=args.batch_size,
        save_dir=args.save_dir,
        kernel=args.kernel,
        sigma=sigma_val,
        lambda_redundancy=args.lambda_redundancy,
        num_select=args.num_select,
        mode=args.mode,
    )
    mmd.selector()

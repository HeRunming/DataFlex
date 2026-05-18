"""
MMD-based Functional Coreset Selector for Targeted Instruction Tuning.

Implements greedy coreset selection using Maximum Mean Discrepancy (MMD)
with three kernel variants:
  - emb_rbf: RBF kernel over pre-computed embeddings (fast, offline-dependent)
  - grad_rbf: RBF kernel over gradient features (online, model-aware)
  - grad_cov: Degree-2 polynomial kernel over gradients (online, model-aware)

Reference:
  Functional Coreset Selection for Targeted Instruction Tuning
  (see functional_coreset_mmd_targeted_sft_proposal.md)

The core idea: select a subset S from candidate pool C to minimize
MMD_k(S, T) where T is a target set, which naturally decomposes into:
  - Target relevance: r_T(x) = (1/|T|) sum_z k(x, z)
  - Selected-set redundancy: r_S(x) = (1/|S|) sum_s k(x, s)

Greedy score: score(x | S) = r_T(x) - lambda * r_S(x)
"""

import os
import glob
import numpy as np
import torch
import torch.distributed as dist
from typing import List, Optional, Dict
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset

from dataflex.core.registry import register_selector
from dataflex.utils.selector_io import load_cached_selection, save_selection
from dataflex.utils.logging import logger
from .base_selector import Selector


class IndexedDataset(Dataset):
    """Wraps a dataset to return (index, data) pairs for gradient tracking."""

    def __init__(self, original_dataset):
        self.original_dataset = original_dataset

    def __len__(self):
        return len(self.original_dataset)

    def __getitem__(self, index):
        return index, self.original_dataset[index]


@register_selector("mmd")
class MMDSelector(Selector):
    """
    MMD-based coreset selector with three kernel variants for targeted instruction tuning.

    Kernel types:
      - emb_rbf: RBF kernel on pre-computed embeddings. Fast selection (no model forward),
                 requires offline embedding computation.
      - grad_rbf: RBF kernel on projected gradient features. Matches target gradient
                  distribution rather than just mean gradient direction.
      - grad_cov: Degree-2 polynomial kernel on gradients (k(x,y) = <g(x),g(y)>^2).
                  Preserves target gradient covariance / update subspace.

    Parameters (from components.yaml):
        kernel_type: str - "emb_rbf", "grad_rbf", or "grad_cov"
        lambda_redundancy: float - weight for selected-set redundancy penalty (default: 0.5)
        sigma: float or None - RBF bandwidth; None = median heuristic (auto)
        candidate_embeddings_path: str - .npy path for candidate embeddings (emb_rbf only)
        target_embeddings_path: str - .npy path for target embeddings (emb_rbf only)
        proj_dim: int - gradient projection dimension (default: 4096)
        gradient_type: str - "sgd" or "adam" (default: "sgd")
        save_interval: int - gradient chunk save interval (default: 16)
        seed: int - random seed for projections (default: 42)
        candidate_subsample: int - subsample candidates for gradient kernels (-1 = all)
    """

    def __init__(
        self,
        dataset,
        accelerator,
        data_collator,
        cache_dir,
        eval_dataset=None,
        kernel_type: str = "emb_rbf",
        lambda_redundancy: float = 0.5,
        sigma: float = None,
        candidate_embeddings_path: str = None,
        target_embeddings_path: str = None,
        proj_dim: int = 4096,
        gradient_type: str = "sgd",
        save_interval: int = 16,
        seed: int = 42,
        candidate_subsample: int = -1,
    ):
        super().__init__(dataset, accelerator, data_collator, cache_dir)

        self.eval_dataset = eval_dataset
        self.kernel_type = kernel_type
        self.lambda_redundancy = lambda_redundancy
        self.sigma = sigma
        self.candidate_embeddings_path = candidate_embeddings_path
        self.target_embeddings_path = target_embeddings_path
        self.proj_dim = proj_dim
        self.gradient_type = gradient_type
        self.save_interval = save_interval
        self.seed = seed
        self.candidate_subsample = candidate_subsample

        self.device = self.accelerator.device
        self.dtype = torch.float16

        # Pre-loaded embeddings (emb_rbf mode)
        self._candidate_embs = None
        self._target_embs = None
        self._target_relevance_cache = None

        if self.kernel_type == "emb_rbf":
            self._init_embedding_mode()

        os.makedirs(self.cache_dir, exist_ok=True)
        logger.info(
            f"[MMDSelector] Initialized: kernel={kernel_type}, lambda={lambda_redundancy}, "
            f"sigma={sigma}, proj_dim={proj_dim}"
        )

    # ═══════════════════════════════════════════════════════════════════════
    # INITIALIZATION
    # ═══════════════════════════════════════════════════════════════════════

    def _init_embedding_mode(self):
        """Load pre-computed embeddings for embedding-RBF mode."""
        if self.accelerator.is_main_process:
            if not self.candidate_embeddings_path or not os.path.exists(self.candidate_embeddings_path):
                raise FileNotFoundError(
                    f"[MMDSelector] candidate_embeddings_path not found: {self.candidate_embeddings_path}"
                )
            if not self.target_embeddings_path or not os.path.exists(self.target_embeddings_path):
                raise FileNotFoundError(
                    f"[MMDSelector] target_embeddings_path not found: {self.target_embeddings_path}"
                )

            self._candidate_embs = np.load(self.candidate_embeddings_path).astype(np.float32)
            self._target_embs = np.load(self.target_embeddings_path).astype(np.float32)

            logger.info(
                f"[MMDSelector] Loaded embeddings: candidates={self._candidate_embs.shape}, "
                f"targets={self._target_embs.shape}"
            )

            # Auto-compute sigma via median heuristic if not specified
            if self.sigma is None:
                self.sigma = self._median_heuristic(self._candidate_embs)
                logger.info(f"[MMDSelector] Median heuristic sigma: {self.sigma:.6f}")

            # Pre-compute target relevance: r_T(x_i) for all candidates
            self._target_relevance_cache = self._compute_target_relevance_rbf(
                self._candidate_embs, self._target_embs, self.sigma
            )
            logger.info("[MMDSelector] Pre-computed target relevance scores.")

    # ═══════════════════════════════════════════════════════════════════════
    # MAIN SELECT INTERFACE
    # ═══════════════════════════════════════════════════════════════════════

    def select(self, model, step_id: int, num_samples: int, **kwargs) -> List[int]:
        """
        Select samples using MMD-based greedy coreset selection.

        The greedy algorithm iteratively picks samples that maximize:
            score(x | S) = r_T(x) - lambda * r_S(x)
        where:
            r_T(x) = (1/|T|) * sum_{z in T} k(x, z)  [target relevance]
            r_S(x) = (1/|S|) * sum_{s in S} k(x, s)  [redundancy penalty]

        Args:
            model: Current model (used for gradient kernel computation)
            step_id: Current training step
            num_samples: Number of samples to select
            **kwargs: May include optimizer_state, tokenizer, etc.

        Returns:
            List of selected sample indices from the training dataset.
        """
        # Check cache
        os.makedirs(self.cache_dir, exist_ok=True)
        save_path = os.path.join(self.cache_dir, f"step_{step_id}.json")

        if os.path.exists(save_path):
            if self.accelerator.is_main_process:
                cached_indices, _ = load_cached_selection(save_path)
            else:
                cached_indices = None
            cached_indices_list = [cached_indices]
            if dist.is_available() and dist.is_initialized():
                dist.broadcast_object_list(cached_indices_list, src=0)
                cached_indices = cached_indices_list[0]
            else:
                cached_indices = cached_indices or []
            return cached_indices

        # Dispatch based on kernel type
        if self.kernel_type == "emb_rbf":
            selected = self._select_emb_rbf(num_samples, step_id)
        elif self.kernel_type in ("grad_rbf", "grad_cov"):
            selected = self._select_gradient_kernel(model, step_id, num_samples, **kwargs)
        else:
            raise ValueError(f"[MMDSelector] Unknown kernel_type: {self.kernel_type}")

        # Save selection
        if self.accelerator.is_main_process and selected is not None:
            metric_payload = {
                "kernel_type": self.kernel_type,
                "lambda_redundancy": self.lambda_redundancy,
                "sigma": self.sigma,
            }
            save_selection(save_path, selected, metric_payload, self.accelerator)

        # Broadcast result
        obj_list = [selected if self.accelerator.is_main_process else None]
        if dist.is_available() and dist.is_initialized():
            dist.broadcast_object_list(obj_list, src=0)
        selected = obj_list[0]

        return selected

    # ═══════════════════════════════════════════════════════════════════════
    # EMBEDDING-RBF SELECTION PATH
    # ═══════════════════════════════════════════════════════════════════════

    def _select_emb_rbf(self, num_samples: int, step_id: int) -> Optional[List[int]]:
        """Select using pre-computed embeddings with RBF kernel + greedy MMD."""
        if self.accelerator.is_main_process:
            logger.info(f"[MMDSelector] Running greedy EMB-RBF selection for {num_samples} samples...")

            selected = self._greedy_mmd_select(
                candidate_features=self._candidate_embs,
                target_relevance=self._target_relevance_cache,
                num_samples=num_samples,
                sigma=self.sigma,
                kernel_type="rbf",
            )

            logger.info(f"[MMDSelector] EMB-RBF selection done: {len(selected)} samples selected.")
            return selected
        return None

    # ═══════════════════════════════════════════════════════════════════════
    # GRADIENT KERNEL SELECTION PATH (grad_rbf, grad_cov)
    # ═══════════════════════════════════════════════════════════════════════

    def _select_gradient_kernel(self, model, step_id: int, num_samples: int, **kwargs) -> Optional[List[int]]:
        """Select using gradient features with RBF or covariance kernel."""
        train_grads_dir = os.path.join(self.cache_dir, "train", str(step_id))
        eval_grads_dir = os.path.join(self.cache_dir, "eval", str(step_id))
        train_final_path = os.path.join(train_grads_dir, "all_projected_grads.pt")
        eval_final_path = os.path.join(eval_grads_dir, "all_projected_grads.pt")

        # Step 1: Compute training set gradients (possibly subsampled)
        subsample_indices = None
        if not os.path.exists(train_final_path):
            os.makedirs(train_grads_dir, exist_ok=True)
            optimizer_state = kwargs.get("optimizer_state", None)

            # Subsample candidate pool for efficiency
            if 0 < self.candidate_subsample < len(self.dataset):
                gen = torch.Generator()
                gen.manual_seed(self.seed + step_id)
                subsample_indices = torch.randperm(len(self.dataset), generator=gen)[
                    : self.candidate_subsample
                ].tolist()
                if self.accelerator.is_main_process:
                    torch.save(subsample_indices, os.path.join(train_grads_dir, "subsample_indices.pt"))
                dataset_to_use = torch.utils.data.Subset(self.dataset, subsample_indices)
            else:
                dataset_to_use = self.dataset

            self._collect_and_save_projected_gradients(
                model, train_grads_dir, dataset_to_use, self.gradient_type, optimizer_state
            )
            self._merge_and_normalize(train_grads_dir, len(dataset_to_use))

        self.accelerator.wait_for_everyone()

        # Step 2: Compute eval/target set gradients
        if self.eval_dataset is not None and not os.path.exists(eval_final_path):
            os.makedirs(eval_grads_dir, exist_ok=True)
            self._collect_and_save_projected_gradients(
                model, eval_grads_dir, self.eval_dataset, "sgd", None
            )
            self._merge_and_normalize(eval_grads_dir, len(self.eval_dataset))

        self.accelerator.wait_for_everyone()

        # Step 3: Main process runs greedy MMD selection
        if self.accelerator.is_main_process:
            train_grads = torch.load(train_final_path, map_location="cpu").numpy()
            eval_grads = torch.load(eval_final_path, map_location="cpu").numpy()

            logger.info(
                f"[MMDSelector] Loaded gradients: train={train_grads.shape}, eval={eval_grads.shape}"
            )

            # Determine kernel-specific parameters
            kernel_type_for_select = "rbf" if self.kernel_type == "grad_rbf" else "cov"

            if kernel_type_for_select == "rbf":
                sigma = self.sigma if self.sigma is not None else self._median_heuristic(train_grads)
                logger.info(f"[MMDSelector] Grad-RBF sigma: {sigma:.6f}")
            else:
                sigma = None

            # Compute target relevance
            target_relevance = self._compute_target_relevance_generic(
                train_grads, eval_grads, sigma=sigma, kernel_type=kernel_type_for_select
            )

            # Greedy MMD selection
            logger.info(
                f"[MMDSelector] Running greedy {self.kernel_type.upper()} selection "
                f"for {num_samples} samples..."
            )
            local_selected = self._greedy_mmd_select(
                candidate_features=train_grads,
                target_relevance=target_relevance,
                num_samples=num_samples,
                sigma=sigma,
                kernel_type=kernel_type_for_select,
            )

            # Map back to global dataset indices if subsampled
            subsample_idx_path = os.path.join(train_grads_dir, "subsample_indices.pt")
            if os.path.exists(subsample_idx_path):
                subsample_indices_loaded = torch.load(subsample_idx_path, map_location="cpu")
                if isinstance(subsample_indices_loaded, list):
                    selected = [subsample_indices_loaded[i] for i in local_selected]
                else:
                    selected = [int(subsample_indices_loaded[i]) for i in local_selected]
            else:
                selected = local_selected

            logger.info(f"[MMDSelector] {self.kernel_type.upper()} selection done: {len(selected)} samples.")
            return selected

        return None

    # ═══════════════════════════════════════════════════════════════════════
    # GREEDY MMD SELECTION (CORE ALGORITHM)
    # ═══════════════════════════════════════════════════════════════════════

    def _greedy_mmd_select(
        self,
        candidate_features: np.ndarray,
        target_relevance: np.ndarray,
        num_samples: int,
        sigma: Optional[float] = None,
        kernel_type: str = "rbf",
    ) -> List[int]:
        """
        Greedy forward selection minimizing MMD(S, T).

        At each step t, selects:
            x* = argmax_{x not in S} [ r_T(x) - lambda * r_S(x) ]

        where:
            r_T(x) = (1/|T|) sum_{z in T} k(x, z)  [pre-computed]
            r_S(x) = (1/|S|) sum_{s in S} k(x, s)   [maintained incrementally]

        The redundancy term is maintained efficiently:
            redundancy_sum[i] += k(x_i, x_new) at each step
            r_S(x_i) = redundancy_sum[i] / |S|

        Complexity: O(num_samples * N * D) where D is feature dimension.
        """
        N = candidate_features.shape[0]
        num_samples = min(num_samples, N)

        selected = []
        # Running sum: sum_{s in S} k(x_i, s) for all candidates i
        redundancy_sum = np.zeros(N, dtype=np.float64)
        available_mask = np.ones(N, dtype=bool)

        for t in tqdm(
            range(num_samples),
            desc="[MMD Greedy Select]",
            disable=(num_samples < 50),
        ):
            # Compute greedy scores
            if len(selected) == 0:
                scores = target_relevance.astype(np.float64).copy()
            else:
                redundancy = redundancy_sum / len(selected)
                scores = target_relevance.astype(np.float64) - self.lambda_redundancy * redundancy

            # Mask out already selected
            scores[~available_mask] = -np.inf

            # Pick the best candidate
            best_idx = int(np.argmax(scores))
            selected.append(best_idx)
            available_mask[best_idx] = False

            # Incremental update: add k(x_i, x_best) for all i
            k_col = self._compute_kernel_column(
                candidate_features, candidate_features[best_idx], sigma, kernel_type
            )
            redundancy_sum += k_col

        return selected

    # ═══════════════════════════════════════════════════════════════════════
    # KERNEL FUNCTIONS
    # ═══════════════════════════════════════════════════════════════════════

    @staticmethod
    def _rbf_kernel_matrix(X: np.ndarray, Y: np.ndarray, sigma: float) -> np.ndarray:
        """
        RBF (Gaussian) kernel: k(x,y) = exp(-||x-y||^2 / (2*sigma^2))

        Args:
            X: (N, D) array
            Y: (M, D) array
            sigma: bandwidth parameter

        Returns:
            (N, M) kernel matrix
        """
        X_sqnorm = np.sum(X ** 2, axis=1, keepdims=True)  # (N, 1)
        Y_sqnorm = np.sum(Y ** 2, axis=1, keepdims=True)  # (M, 1)
        sq_dists = X_sqnorm + Y_sqnorm.T - 2.0 * (X @ Y.T)  # (N, M)
        sq_dists = np.maximum(sq_dists, 0.0)
        return np.exp(-sq_dists / (2.0 * sigma ** 2))

    @staticmethod
    def _grad_cov_kernel_matrix(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """
        Gradient covariance kernel (degree-2 polynomial): k(x,y) = <x,y>^2

        This matches the target gradient covariance structure:
            E_S[g g^T] ≈ E_T[g g^T]

        Args:
            X: (N, D) gradient features
            Y: (M, D) gradient features

        Returns:
            (N, M) kernel matrix
        """
        inner = X @ Y.T  # (N, M)
        return inner ** 2

    def _compute_kernel_column(
        self, X: np.ndarray, x_new: np.ndarray, sigma: Optional[float], kernel_type: str
    ) -> np.ndarray:
        """
        Compute k(x_i, x_new) for all candidates i.

        This is the core incremental update used in greedy selection.
        Returns shape (N,).
        """
        x_new_2d = x_new.reshape(1, -1)
        if kernel_type == "rbf":
            # k(x_i, x_new) = exp(-||x_i - x_new||^2 / (2*sigma^2))
            diffs = X - x_new_2d  # (N, D)
            sq_dists = np.sum(diffs ** 2, axis=1)  # (N,)
            return np.exp(-sq_dists / (2.0 * sigma ** 2))
        elif kernel_type == "cov":
            # k(x_i, x_new) = <x_i, x_new>^2
            inner = X @ x_new_2d.T  # (N, 1)
            return (inner.squeeze(axis=1)) ** 2
        else:
            raise ValueError(f"Unknown kernel_type: {kernel_type}")

    @staticmethod
    def _median_heuristic(X: np.ndarray, subsample: int = 2000) -> float:
        """
        Median heuristic for RBF bandwidth selection.

        sigma = median(||x_i - x_j||) computed over a random subsample of pairs.
        This is the standard parameter-free bandwidth selection method.
        """
        N = X.shape[0]
        if N > subsample:
            rng = np.random.RandomState(42)
            idx = rng.choice(N, subsample, replace=False)
            X_sub = X[idx]
        else:
            X_sub = X

        # Pairwise squared distances via expansion
        sq_norms = np.sum(X_sub ** 2, axis=1)
        sq_dists = sq_norms[:, None] + sq_norms[None, :] - 2.0 * (X_sub @ X_sub.T)
        sq_dists = np.maximum(sq_dists, 0.0)

        # Upper triangle (exclude diagonal and duplicates)
        triu_idx = np.triu_indices(len(X_sub), k=1)
        pairwise_dists = np.sqrt(sq_dists[triu_idx])

        median_dist = float(np.median(pairwise_dists))
        return max(median_dist, 1e-6)

    # ═══════════════════════════════════════════════════════════════════════
    # TARGET RELEVANCE COMPUTATION
    # ═══════════════════════════════════════════════════════════════════════

    @staticmethod
    def _compute_target_relevance_rbf(
        candidates: np.ndarray, targets: np.ndarray, sigma: float
    ) -> np.ndarray:
        """
        Compute r_T(x_i) = (1/|T|) * sum_{z in T} k(x_i, z) using RBF kernel.

        Processes in chunks to avoid OOM on large target sets.
        Returns shape (N_candidates,).
        """
        N_cand = candidates.shape[0]
        N_target = targets.shape[0]
        chunk_size = 5000
        relevance = np.zeros(N_cand, dtype=np.float64)

        for t_start in range(0, N_target, chunk_size):
            t_end = min(t_start + chunk_size, N_target)
            target_chunk = targets[t_start:t_end]

            cand_sq = np.sum(candidates ** 2, axis=1, keepdims=True)
            tgt_sq = np.sum(target_chunk ** 2, axis=1, keepdims=True)
            sq_dists = cand_sq + tgt_sq.T - 2.0 * (candidates @ target_chunk.T)
            sq_dists = np.maximum(sq_dists, 0.0)
            K_chunk = np.exp(-sq_dists / (2.0 * sigma ** 2))
            relevance += K_chunk.sum(axis=1)

        return relevance / N_target

    def _compute_target_relevance_generic(
        self, candidates: np.ndarray, targets: np.ndarray,
        sigma: Optional[float], kernel_type: str
    ) -> np.ndarray:
        """Compute target relevance for any kernel type."""
        if kernel_type == "rbf":
            return self._compute_target_relevance_rbf(candidates, targets, sigma)
        elif kernel_type == "cov":
            # k(x,y) = <x,y>^2
            N_cand = candidates.shape[0]
            N_target = targets.shape[0]
            chunk_size = 5000
            relevance = np.zeros(N_cand, dtype=np.float64)

            for t_start in range(0, N_target, chunk_size):
                t_end = min(t_start + chunk_size, N_target)
                target_chunk = targets[t_start:t_end]
                inner = candidates @ target_chunk.T  # (N_cand, chunk)
                K_chunk = inner ** 2
                relevance += K_chunk.sum(axis=1)

            return relevance / N_target
        else:
            raise ValueError(f"Unknown kernel_type: {kernel_type}")

    # ═══════════════════════════════════════════════════════════════════════
    # GRADIENT COMPUTATION (REUSES LESS INFRASTRUCTURE)
    # ═══════════════════════════════════════════════════════════════════════

    def _get_number_of_params(self, model) -> int:
        """Count parameters requiring gradients (ZeRO-3 compatible)."""
        num_params = 0
        for p in model.parameters():
            if p.requires_grad:
                if hasattr(p, "ds_numel"):
                    num_params += p.ds_numel
                else:
                    num_params += p.numel()
        if self.accelerator.is_main_process:
            logger.info(f"[MMDSelector] Trainable params: {num_params:,}")
        return num_params

    def _prepare_optimizer_state(self, model, optimizer_state=None):
        """Prepare Adam first/second moment estimates (ZeRO-3 compatible)."""
        avg_list, avg_sq_list = [], []

        if self.accelerator.state.deepspeed_plugin is not None:
            from deepspeed.utils import safe_get_full_optimizer_state
            for param in model.parameters():
                if param.requires_grad:
                    exp_avg = safe_get_full_optimizer_state(param, "exp_avg")
                    exp_avg_sq = safe_get_full_optimizer_state(param, "exp_avg_sq")
                    if exp_avg is not None and exp_avg_sq is not None:
                        avg_list.append(exp_avg.view(-1))
                        avg_sq_list.append(exp_avg_sq.view(-1))
        else:
            if optimizer_state is None:
                raise ValueError("optimizer_state required for non-DeepSpeed 'adam' gradient type.")
            for param in model.parameters():
                if param.requires_grad:
                    avg_list.append(optimizer_state[param]["exp_avg"].view(-1))
                    avg_sq_list.append(optimizer_state[param]["exp_avg_sq"].view(-1))

        avg = torch.cat(avg_list).to(self.device)
        avg_sq = torch.cat(avg_sq_list).to(self.device)
        return avg, avg_sq

    def _obtain_gradients(self, model, batch, gradient_type, m=None, v=None) -> torch.Tensor:
        """Compute gradient vector for a single sample."""
        if self.accelerator.state.deepspeed_plugin is not None:
            loss = model(**batch).loss
            model.backward(loss)
            from deepspeed.utils import safe_get_full_grad
            grads = []
            for name, p in model.named_parameters():
                g = safe_get_full_grad(p)
                if g is not None:
                    grads.append(g.contiguous().view(-1))
            vectorized_grads = torch.cat(grads) if grads else None
        else:
            with self.accelerator.no_sync(model):
                loss = model(**batch).loss
                self.accelerator.backward(loss)
            vectorized_grads = torch.cat(
                [p.grad.view(-1) for p in model.parameters() if p.grad is not None]
            )

        if gradient_type == "adam":
            if m is None or v is None:
                raise ValueError("Adam states (m, v) required for 'adam' gradient type.")
            beta1, beta2, eps = 0.9, 0.999, 1e-08
            denom = v.mul(beta2)
            denom.addcmul_(vectorized_grads, vectorized_grads, value=(1 - beta2))
            denom.sqrt_().add_(eps)
            vectorized_grads.mul_(1 - beta1).add_(m, alpha=beta1)
            vectorized_grads.div_(denom)
            del denom

        model.zero_grad()
        return vectorized_grads

    def _get_trak_projector(self):
        """Get TRAK projector (CUDA if available, else Basic)."""
        from trak.projectors import BasicProjector, CudaProjector, ProjectionType
        try:
            import fast_jl
            num_sms = torch.cuda.get_device_properties(self.device.index).multi_processor_count
            fast_jl.project_rademacher_8(torch.zeros(8, 1_000, device=self.device), 512, 0, num_sms)
            projector = CudaProjector
            if self.accelerator.is_main_process:
                logger.info("[MMDSelector] Using CudaProjector.")
        except (ImportError, RuntimeError):
            projector = BasicProjector
            if self.accelerator.is_main_process:
                logger.info("[MMDSelector] Using BasicProjector (fallback).")
        return projector

    def _collect_and_save_projected_gradients(
        self, model, save_dir, dataset_to_use, gradient_type, optimizer_state=None
    ):
        """
        Compute per-example projected gradients and save in chunks.
        Reuses the LESS gradient computation infrastructure.
        """
        from trak.projectors import BasicProjector, CudaProjector, ProjectionType

        num_params = self._get_number_of_params(model)
        projector_class = self._get_trak_projector()
        projector = projector_class(
            grad_dim=num_params,
            proj_dim=self.proj_dim,
            seed=self.seed,
            proj_type=ProjectionType.rademacher,
            max_batch_size=8,
            block_size=128,
            device=self.device,
            dtype=self.dtype,
        )

        # Prepare optimizer state for Adam preconditioning
        m, v = None, None
        if gradient_type == "adam":
            if self.accelerator.state.deepspeed_plugin is None and optimizer_state is None:
                raise ValueError("optimizer_state required for non-DeepSpeed 'adam' gradient type.")
            m, v = self._prepare_optimizer_state(model, optimizer_state)

        # Build indexed dataloader
        indexed_dataset = IndexedDataset(dataset_to_use)

        def indexed_collator(features):
            indices = [f[0] for f in features]
            original_data = [f[1] for f in features]
            collated_batch = self.data_collator(original_data)
            return {"indices": torch.tensor(indices), "batch": collated_batch}

        dataloader = DataLoader(
            indexed_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=2,
            collate_fn=indexed_collator,
        )
        dataloader = self.accelerator.prepare(dataloader)

        total_samples = len(dataloader)
        model_device = next(model.parameters()).device
        grad_buffer = torch.zeros(
            self.save_interval, num_params, device=model_device, dtype=self.dtype
        )
        idx_buffer = torch.zeros(self.save_interval, dtype=torch.long)
        buf_pos = 0

        for batch_idx, data in enumerate(
            tqdm(
                dataloader,
                desc=f"[MMD Proc {self.accelerator.process_index}] Computing Gradients",
                disable=not self.accelerator.is_local_main_process,
                dynamic_ncols=True,
                position=self.accelerator.process_index,
            ),
            1,
        ):
            indices = data["indices"]
            batch = data["batch"]

            vectorized_grads = self._obtain_gradients(model, batch, gradient_type, m, v)
            grad_buffer[buf_pos].copy_(vectorized_grads)
            del vectorized_grads
            idx_buffer[buf_pos] = indices[0]
            buf_pos += 1

            if buf_pos == self.save_interval or batch_idx == total_samples:
                projected = projector.project(grad_buffer[:buf_pos], model_id=0).cpu()
                save_path = os.path.join(
                    save_dir,
                    f"grads-{idx_buffer[:buf_pos].max().item()}-rank{self.accelerator.process_index}.pt",
                )
                torch.save({"grads": projected, "indices": idx_buffer[:buf_pos].clone()}, save_path)
                del projected
                buf_pos = 0

        del grad_buffer, idx_buffer
        self.accelerator.wait_for_everyone()

    def _merge_and_normalize(self, save_dir, total_samples):
        """Merge gradient chunks from all ranks and L2-normalize."""
        if self.accelerator.is_main_process:
            logger.info(f"[MMDSelector] Merging gradients from {save_dir}")
            files = glob.glob(os.path.join(save_dir, "grads-*-rank*.pt"))
            if not files:
                logger.warning("[MMDSelector] No gradient files found to merge.")
                return

            final_grads = torch.zeros(total_samples, self.proj_dim, dtype=torch.float32)

            for file_path in tqdm(files, desc="[MMD] Merging gradient files"):
                chunk = torch.load(file_path, map_location="cpu")
                grads_chunk = chunk["grads"].to(torch.float32)
                indices_chunk = chunk["indices"]
                final_grads[indices_chunk] = grads_chunk

            # L2 normalize
            norms = final_grads.norm(dim=1, keepdim=True).clamp_(min=1e-12)
            final_grads.div_(norms)
            del norms

            output_file = os.path.join(save_dir, "all_projected_grads.pt")
            torch.save(final_grads, output_file)
            logger.info(
                f"[MMDSelector] Saved merged gradients: {final_grads.shape} -> {output_file}"
            )

            # Clean up chunk files
            for file_path in files:
                os.remove(file_path)
            logger.info(f"[MMDSelector] Cleaned up {len(files)} chunk files.")

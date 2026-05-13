"""
Negative Control Selectors for Opt-GCS Ablation.

These selectors serve as critical negative controls to verify that
Opt-GCS's performance comes from the SPECTRAL EIGENSPACE, not just from:
1. Any random low-dimensional projection + logdet diversity
2. Gradient norm magnitude alone
3. LogDet diversity in any subspace

Controls:
- random_subspace_logdet: random orthogonal projection + logdet (tests if eigenspace matters)
- grad_norm_topk: simple gradient norm top-k (tests if GCS is just picking high-norm samples)
- shuffled_eigen_logdet: uses OptGCS eigenspace but shuffles eigenvalue ordering (tests whitening)
"""

import os
import glob
from typing import List, Optional

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataflex.core.registry import register_selector
from dataflex.utils.logging import logger
from dataflex.utils.selector_io import load_cached_selection, save_selection
from .base_selector import Selector


@register_selector('random_subspace_logdet')
class RandomSubspaceLogDetSelector(Selector):
    """
    Negative Control: LogDet selection in a RANDOM subspace.

    Instead of using the top eigenvectors of gradient covariance,
    projects gradients onto a random orthogonal subspace then does logdet.

    If this performs similarly to OptGCS, it means the eigenspace discovery
    is not contributing — just the diversity-in-any-subspace is sufficient.
    """

    def __init__(
        self,
        dataset,
        accelerator,
        data_collator,
        cache_dir: str,
        proj_dim: int = 4096,
        subspace_dim: int = 50,  # dimension of random subspace
        seed: int = 42,
        logdet_eps: float = 1e-3,
        prefilter_ratio: float = 5.0,
        length_norm_alpha: float = 0.5,
        eval_dataset=None,
        **kwargs,
    ):
        super().__init__(dataset, accelerator, data_collator, cache_dir)
        self.proj_dim = proj_dim
        self.subspace_dim = subspace_dim
        self.seed = seed
        self.logdet_eps = logdet_eps
        self.prefilter_ratio = prefilter_ratio
        self.length_norm_alpha = length_norm_alpha
        self.device = self.accelerator.device
        os.makedirs(self.cache_dir, exist_ok=True)
        logger.info(f"[RandomSubspaceLogDet] Negative control: random {subspace_dim}-dim subspace + logdet")

    def select(self, model, step_id: int, num_samples: int, **kwargs) -> List[int]:
        """Project gradients to random subspace, then logdet select."""
        os.makedirs(self.cache_dir, exist_ok=True)
        save_path = os.path.join(self.cache_dir, f"step_{step_id}.json")

        if os.path.exists(save_path):
            if self.accelerator.is_main_process:
                cached_indices, _ = load_cached_selection(save_path)
            else:
                cached_indices = None
            obj = [cached_indices]
            if dist.is_available() and dist.is_initialized():
                dist.broadcast_object_list(obj, src=0)
            return obj[0] or []

        # Try to load cached gradients from OptGCS (reuse if available)
        # Look for any existing gradient cache
        possible_grad_paths = [
            os.path.join(self.cache_dir, "gradients", str(step_id), "all_projected_grads.pt"),
        ]
        # Also check opt_gcs caches
        parent = os.path.dirname(self.cache_dir)
        for sibling in os.listdir(parent) if os.path.exists(parent) else []:
            p = os.path.join(parent, sibling, "gradients", str(step_id), "all_projected_grads.pt")
            if os.path.exists(p):
                possible_grad_paths.insert(0, p)

        grads_path = None
        lengths_path = None
        for p in possible_grad_paths:
            if os.path.exists(p):
                grads_path = p
                lengths_path = p.replace("all_projected_grads.pt", "all_token_lengths.pt")
                break

        if grads_path is None or not self.accelerator.is_main_process:
            # If no cached gradients, we need to compute them
            # For simplicity, reuse OptGCS's gradient computation
            logger.warning("[RandomSubspaceLogDet] No cached gradients found. Run OptGCS first to generate gradient cache.")
            # Fallback to random selection
            import random
            random.seed(self.seed + step_id)
            selected_indices = random.sample(range(len(self.dataset)), min(num_samples, len(self.dataset)))
            obj = [selected_indices]
            if dist.is_available() and dist.is_initialized():
                dist.broadcast_object_list(obj, src=0)
            return obj[0]

        if self.accelerator.is_main_process:
            grads = torch.load(grads_path, map_location="cpu")
            lengths = torch.load(lengths_path, map_location="cpu") if os.path.exists(lengths_path) else torch.ones(len(grads))

            n, d = grads.shape

            # Length normalize + L2 normalize
            alpha = self.length_norm_alpha
            if alpha > 0:
                h = grads / lengths.float().pow(alpha).unsqueeze(1)
            else:
                h = grads
            norms = h.norm(dim=1, keepdim=True).clamp(min=1e-12)
            h = h / norms

            # Generate RANDOM orthogonal subspace (not data-driven!)
            torch.manual_seed(self.seed + 9999)  # different seed to avoid correlation
            r = min(self.subspace_dim, d)
            random_matrix = torch.randn(d, r)
            Q, _ = torch.linalg.qr(random_matrix)  # orthogonal [d, r]

            # Project to random subspace
            projections = h @ Q  # [n, r]
            scores = (projections ** 2).sum(dim=1)

            # LogDet greedy
            k = min(num_samples, n)
            eps = self.logdet_eps
            prefilter_k = min(int(self.prefilter_ratio * k), n)

            if prefilter_k < n:
                topk = torch.topk(scores, k=prefilter_k, largest=True)
                candidate_indices = topk.indices
                X = projections[candidate_indices].clone()
            else:
                candidate_indices = torch.arange(n)
                X = projections.clone()

            num_candidates = len(X)
            A_inv = torch.eye(r, dtype=X.dtype) / eps
            selected_local = []
            available = torch.ones(num_candidates, dtype=torch.bool)

            for t in range(k):
                gains = (X @ A_inv * X).sum(dim=1)
                gains[~available] = -float('inf')
                best = gains.argmax().item()
                selected_local.append(best)
                available[best] = False
                x = X[best]
                Ax = A_inv @ x
                A_inv -= torch.outer(Ax, Ax) / (1.0 + x @ Ax)

            selected_indices = candidate_indices[torch.tensor(selected_local)].tolist()

            metric_payload = {
                "selection_method": "random_subspace_logdet",
                "subspace_dim": r,
                "negative_control": True,
            }
            save_selection(save_path, selected_indices, metric_payload, self.accelerator)
            logger.info(f"[RandomSubspaceLogDet] Selected {len(selected_indices)} samples (negative control)")
        else:
            selected_indices = None

        obj = [selected_indices]
        if dist.is_available() and dist.is_initialized():
            dist.broadcast_object_list(obj, src=0)
        return obj[0] or []


@register_selector('grad_norm_topk')
class GradNormTopKSelector(Selector):
    """
    Negative Control: Simple gradient norm top-k selection.

    Selects samples with the largest gradient norms. If OptGCS ≈ grad_norm_topk,
    it means the spectral analysis is just finding high-norm samples.
    """

    def __init__(
        self,
        dataset,
        accelerator,
        data_collator,
        cache_dir: str,
        seed: int = 42,
        length_norm_alpha: float = 0.5,
        eval_dataset=None,
        **kwargs,
    ):
        super().__init__(dataset, accelerator, data_collator, cache_dir)
        self.seed = seed
        self.length_norm_alpha = length_norm_alpha
        self.device = self.accelerator.device
        os.makedirs(self.cache_dir, exist_ok=True)
        logger.info("[GradNormTopK] Negative control: gradient norm top-k")

    def select(self, model, step_id: int, num_samples: int, **kwargs) -> List[int]:
        """Select samples with highest gradient norms."""
        os.makedirs(self.cache_dir, exist_ok=True)
        save_path = os.path.join(self.cache_dir, f"step_{step_id}.json")

        if os.path.exists(save_path):
            if self.accelerator.is_main_process:
                cached_indices, _ = load_cached_selection(save_path)
            else:
                cached_indices = None
            obj = [cached_indices]
            if dist.is_available() and dist.is_initialized():
                dist.broadcast_object_list(obj, src=0)
            return obj[0] or []

        # Look for cached gradients
        parent = os.path.dirname(self.cache_dir)
        grads_path = None
        lengths_path = None
        for sibling in os.listdir(parent) if os.path.exists(parent) else []:
            p = os.path.join(parent, sibling, "gradients", str(step_id), "all_projected_grads.pt")
            if os.path.exists(p):
                grads_path = p
                lengths_path = p.replace("all_projected_grads.pt", "all_token_lengths.pt")
                break

        if self.accelerator.is_main_process and grads_path and os.path.exists(grads_path):
            grads = torch.load(grads_path, map_location="cpu")
            lengths = torch.load(lengths_path, map_location="cpu") if lengths_path and os.path.exists(lengths_path) else torch.ones(len(grads))

            # Length normalize
            alpha = self.length_norm_alpha
            if alpha > 0:
                h = grads / lengths.float().pow(alpha).unsqueeze(1)
            else:
                h = grads

            # Compute norms
            norms = h.norm(dim=1)

            # Top-k by norm
            k = min(num_samples, len(norms))
            topk = torch.topk(norms, k=k, largest=True)
            selected_indices = topk.indices.tolist()

            metric_payload = {
                "selection_method": "grad_norm_topk",
                "negative_control": True,
                "mean_norm_selected": float(norms[topk.indices].mean()),
                "mean_norm_all": float(norms.mean()),
            }
            save_selection(save_path, selected_indices, metric_payload, self.accelerator)
            logger.info(f"[GradNormTopK] Selected {len(selected_indices)} samples by gradient norm")
        else:
            if self.accelerator.is_main_process:
                logger.warning("[GradNormTopK] No cached gradients. Falling back to random.")
            import random
            random.seed(self.seed + step_id)
            selected_indices = random.sample(range(len(self.dataset)), min(num_samples, len(self.dataset)))

        obj = [selected_indices if self.accelerator.is_main_process else None]
        if dist.is_available() and dist.is_initialized():
            dist.broadcast_object_list(obj, src=0)
        return obj[0] or []

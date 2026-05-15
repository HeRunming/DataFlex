"""
Negative Control Selectors for Opt-GCS Ablation.

These selectors serve as critical negative controls to verify that
Opt-GCS's performance comes from the SPECTRAL EIGENSPACE, not just from:
1. Any random low-dimensional projection + logdet diversity
2. Gradient norm magnitude alone

Controls:
- random_subspace_logdet: random orthogonal projection + logdet (tests if eigenspace matters)
- grad_norm_topk: simple gradient norm top-k (tests if GCS is just picking high-norm samples)
"""

import os
import glob
import hashlib
import json as _json
from typing import List, Optional

import torch
import torch.distributed as dist

from dataflex.core.registry import register_selector
from dataflex.utils.logging import logger
from dataflex.utils.selector_io import load_cached_selection, save_selection
from .base_selector import Selector


def _find_gradient_cache(parent_dir: str, step_id: int) -> Optional[str]:
    """Search sibling cache directories for existing gradient files."""
    if not os.path.exists(parent_dir):
        return None
    for sibling in os.listdir(parent_dir):
        grad_dir = os.path.join(parent_dir, sibling, "gradients")
        if not os.path.isdir(grad_dir):
            continue
        for sub in os.listdir(grad_dir):
            if sub.startswith(f"step_{step_id}"):
                candidate = os.path.join(grad_dir, sub, "all_projected_grads.pt")
                if os.path.exists(candidate):
                    return candidate
    return None


@register_selector('random_subspace_logdet')
class RandomSubspaceLogDetSelector(Selector):
    """
    Negative Control: LogDet selection in a RANDOM subspace.
    If this performs similarly to OptGCS, the eigenspace discovery is not contributing.
    """

    def __init__(
        self,
        dataset,
        accelerator,
        data_collator,
        cache_dir: str,
        proj_dim: int = 4096,
        subspace_dim: int = 50,
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

    def select(self, model, step_id: int, num_samples: int, **kwargs) -> List[int]:
        os.makedirs(self.cache_dir, exist_ok=True)

        # Config-aware cache key
        cfg = dict(step_id=step_id, num_samples=num_samples, subspace_dim=self.subspace_dim,
                   seed=self.seed, logdet_eps=self.logdet_eps, prefilter_ratio=self.prefilter_ratio,
                   length_norm_alpha=self.length_norm_alpha, dataset_size=len(self.dataset))
        cfg_hash = hashlib.md5(_json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:10]
        save_path = os.path.join(self.cache_dir, f"step_{step_id}_k{num_samples}_{cfg_hash}.json")

        if os.path.exists(save_path):
            if self.accelerator.is_main_process:
                cached_indices, _ = load_cached_selection(save_path)
            else:
                cached_indices = None
            obj = [cached_indices]
            if dist.is_available() and dist.is_initialized():
                dist.broadcast_object_list(obj, src=0)
            return obj[0] or []

        # Find gradient cache from OptGCS or other selectors
        parent = os.path.dirname(self.cache_dir)
        grads_path = _find_gradient_cache(parent, step_id)

        if grads_path is None:
            logger.warning("[RandomSubspaceLogDet] No cached gradients found. Falling back to random.")
            if self.accelerator.is_main_process:
                import random
                random.seed(self.seed + step_id)
                selected_indices = random.sample(range(len(self.dataset)), min(num_samples, len(self.dataset)))
            else:
                selected_indices = None
            obj = [selected_indices]
            if dist.is_available() and dist.is_initialized():
                dist.broadcast_object_list(obj, src=0)
            return obj[0] or []

        if self.accelerator.is_main_process:
            grads = torch.load(grads_path, map_location="cpu")
            lengths_path = grads_path.replace("all_projected_grads.pt", "all_token_lengths.pt")
            lengths = torch.load(lengths_path, map_location="cpu") if os.path.exists(lengths_path) else torch.ones(len(grads))

            n, d = grads.shape

            # Preprocess same as OptGCS
            h = grads.clone()
            h[~torch.isfinite(h)] = 0.0
            alpha = self.length_norm_alpha
            if alpha > 0:
                h = h / lengths.float().pow(alpha).unsqueeze(1).clamp(min=1.0)
            norms = h.norm(dim=1, keepdim=True).clamp(min=1e-12)
            h = h / norms

            # Random orthogonal subspace (NOT data-driven)
            torch.manual_seed(self.seed + 9999)
            r = min(self.subspace_dim, d)
            Q, _ = torch.linalg.qr(torch.randn(d, r))

            projections = h @ Q
            scores = (projections ** 2).sum(dim=1)

            # LogDet greedy
            k = min(num_samples, n)
            eps = self.logdet_eps
            prefilter_k = min(int(self.prefilter_ratio * k), n)

            if prefilter_k < n:
                topk = torch.topk(scores, k=prefilter_k, largest=True)
                cand_idx = topk.indices
                X = projections[cand_idx].clone()
            else:
                cand_idx = torch.arange(n)
                X = projections.clone()

            A_inv = torch.eye(r, dtype=X.dtype) / eps
            selected_local = []
            available = torch.ones(len(X), dtype=torch.bool)

            for t in range(k):
                gains = (X @ A_inv * X).sum(dim=1)
                gains[~available] = -float('inf')
                best = gains.argmax().item()
                selected_local.append(best)
                available[best] = False
                x = X[best]
                Ax = A_inv @ x
                A_inv -= torch.outer(Ax, Ax) / (1.0 + x @ Ax)

            selected_indices = cand_idx[torch.tensor(selected_local)].tolist()

            save_selection(save_path, selected_indices,
                           {"selection_method": "random_subspace_logdet", "subspace_dim": r, "negative_control": True},
                           self.accelerator)
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
    If OptGCS ≈ grad_norm_topk, it means the spectral analysis is just finding high-norm samples.
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

    def select(self, model, step_id: int, num_samples: int, **kwargs) -> List[int]:
        os.makedirs(self.cache_dir, exist_ok=True)

        cfg = dict(step_id=step_id, num_samples=num_samples, seed=self.seed,
                   length_norm_alpha=self.length_norm_alpha, dataset_size=len(self.dataset))
        cfg_hash = hashlib.md5(_json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:10]
        save_path = os.path.join(self.cache_dir, f"step_{step_id}_k{num_samples}_{cfg_hash}.json")

        if os.path.exists(save_path):
            if self.accelerator.is_main_process:
                cached_indices, _ = load_cached_selection(save_path)
            else:
                cached_indices = None
            obj = [cached_indices]
            if dist.is_available() and dist.is_initialized():
                dist.broadcast_object_list(obj, src=0)
            return obj[0] or []

        parent = os.path.dirname(self.cache_dir)
        grads_path = _find_gradient_cache(parent, step_id)

        if self.accelerator.is_main_process and grads_path and os.path.exists(grads_path):
            grads = torch.load(grads_path, map_location="cpu")
            lengths_path = grads_path.replace("all_projected_grads.pt", "all_token_lengths.pt")
            lengths = torch.load(lengths_path, map_location="cpu") if os.path.exists(lengths_path) else torch.ones(len(grads))

            h = grads.clone()
            h[~torch.isfinite(h)] = 0.0
            alpha = self.length_norm_alpha
            if alpha > 0:
                h = h / lengths.float().pow(alpha).unsqueeze(1).clamp(min=1.0)

            norms = h.norm(dim=1)
            k = min(num_samples, len(norms))
            topk = torch.topk(norms, k=k, largest=True)
            selected_indices = topk.indices.tolist()

            save_selection(save_path, selected_indices,
                           {"selection_method": "grad_norm_topk", "negative_control": True,
                            "mean_norm_selected": float(norms[topk.indices].mean()),
                            "mean_norm_all": float(norms.mean())},
                           self.accelerator)
        else:
            if self.accelerator.is_main_process:
                logger.warning("[GradNormTopK] No cached gradients. Falling back to random.")
                import random
                random.seed(self.seed + step_id)
                selected_indices = random.sample(range(len(self.dataset)), min(num_samples, len(self.dataset)))
            else:
                selected_indices = None

        obj = [selected_indices if self.accelerator.is_main_process else None]
        if dist.is_available() and dist.is_initialized():
            dist.broadcast_object_list(obj, src=0)
        return obj[0] or []

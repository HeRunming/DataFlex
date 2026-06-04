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


def _find_gradient_cache(search_dirs: List[str], step_id: int) -> Optional[str]:
    """Search specified directories for existing gradient files.

    Args:
        search_dirs: Explicit list of directories to search for gradients.
                     Each should be a selector cache_dir that may contain a gradients/ subfolder.
        step_id: Training step to look for.

    Returns:
        Path to all_projected_grads.pt if found, else None.
    """
    for search_dir in search_dirs:
        grad_dir = os.path.join(search_dir, "gradients")
        if not os.path.isdir(grad_dir):
            continue
        for sub in os.listdir(grad_dir):
            if sub.startswith(f"step_{step_id}"):
                candidate = os.path.join(grad_dir, sub, "all_projected_grads.pt")
                if os.path.exists(candidate):
                    logger.info(f"[NegativeControl] Found gradient cache: {candidate}")
                    return candidate
    return None


@register_selector('random_subspace_logdet')
class RandomSubspaceLogDetSelector(Selector):
    """
    Negative Control: LogDet selection in a RANDOM subspace.

    Computes its OWN gradient features from the current model state
    (same infrastructure as OptGCS), then projects into a random orthogonal
    subspace instead of the learned eigenspace. This provides a fair
    same-checkpoint comparison.

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
        projector_seed: int = 42,
        logdet_eps: float = 1e-3,
        prefilter_ratio: float = 5.0,
        length_norm_alpha: float = 0.5,
        clipping_method: str = "adaptive",
        gradient_type: str = "adam_diag",
        save_interval: int = 16,
        source_grad_dirs: Optional[List[str]] = None,
        compute_own_grads: bool = True,
        eval_dataset=None,
        **kwargs,
    ):
        super().__init__(dataset, accelerator, data_collator, cache_dir)
        self.proj_dim = proj_dim
        self.subspace_dim = subspace_dim
        self.seed = seed  # controls random subspace generation
        self.projector_seed = projector_seed  # controls TRAK projection (fixed across seeds)
        self.logdet_eps = logdet_eps
        self.prefilter_ratio = prefilter_ratio
        self.length_norm_alpha = length_norm_alpha
        self.clipping_method = clipping_method
        self.gradient_type = gradient_type
        self.save_interval = save_interval
        self.source_grad_dirs = source_grad_dirs or []
        self.compute_own_grads = compute_own_grads
        self.device = self.accelerator.device
        os.makedirs(self.cache_dir, exist_ok=True)

    def select(self, model, step_id: int, num_samples: int, **kwargs) -> List[int]:
        os.makedirs(self.cache_dir, exist_ok=True)

        # Config-aware cache key
        cfg = dict(step_id=step_id, num_samples=num_samples, subspace_dim=self.subspace_dim,
                   seed=self.seed, logdet_eps=self.logdet_eps, prefilter_ratio=self.prefilter_ratio,
                   length_norm_alpha=self.length_norm_alpha, dataset_size=len(self.dataset),
                   compute_own_grads=self.compute_own_grads, gradient_type=self.gradient_type)
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

        # Get gradient features
        grads_path = None
        if self.compute_own_grads:
            # Compute our OWN gradients from the current model state
            grads_path = self._compute_own_gradients(model, step_id, **kwargs)

        if grads_path is None:
            # Fall back to searching for cached gradients from other selectors
            search_dirs = list(self.source_grad_dirs)
            parent = os.path.dirname(self.cache_dir)
            if os.path.exists(parent):
                for sibling in os.listdir(parent):
                    sib_path = os.path.join(parent, sibling)
                    if os.path.isdir(sib_path) and sib_path not in search_dirs:
                        search_dirs.append(sib_path)
            grads_path = _find_gradient_cache(search_dirs, step_id)

        if grads_path is None:
            logger.warning("[RandomSubspaceLogDet] No gradients available. Falling back to random.")
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

            selected_indices = self._do_random_subspace_logdet(grads, lengths, num_samples)

            save_selection(save_path, selected_indices,
                           {"selection_method": "random_subspace_logdet", "subspace_dim": self.subspace_dim,
                            "negative_control": True, "compute_own_grads": self.compute_own_grads},
                           self.accelerator)
        else:
            selected_indices = None

        obj = [selected_indices]
        if dist.is_available() and dist.is_initialized():
            dist.broadcast_object_list(obj, src=0)
        return obj[0] or []

    def _compute_own_gradients(self, model, step_id: int, **kwargs) -> Optional[str]:
        """Compute gradient features using OptGCS infrastructure.
        Uses fixed projector_seed (not subspace seed) so gradient sketches
        are identical across random_subspace seeds."""
        from .spec_gcs_selector import OptGCSSelector

        # Use projector_seed for gradient computation (fixed across subspace seeds)
        grad_cfg = dict(step_id=step_id, gradient_type=self.gradient_type,
                        proj_dim=self.proj_dim, projector_seed=self.projector_seed,
                        dataset_size=len(self.dataset))
        grad_hash = hashlib.md5(_json.dumps(grad_cfg, sort_keys=True).encode()).hexdigest()[:10]
        grad_save_dir = os.path.join(self.cache_dir, "gradients", f"step_{step_id}_{grad_hash}")
        grads_path = os.path.join(grad_save_dir, "all_projected_grads.pt")

        if os.path.exists(grads_path):
            return grads_path

        os.makedirs(grad_save_dir, exist_ok=True)
        # Use proper constructor for safety (avoids missing attribute issues)
        temp_selector = OptGCSSelector(
            dataset=self.dataset,
            accelerator=self.accelerator,
            data_collator=self.data_collator,
            cache_dir=self.cache_dir,
            gradient_type=self.gradient_type,
            proj_dim=self.proj_dim,
            save_interval=self.save_interval,
            seed=self.projector_seed,  # fixed projector seed
        )

        optimizer_state = kwargs.get('optimizer_state', None)
        temp_selector._collect_and_save_projected_gradients(
            model, grad_save_dir, self.dataset,
            self.gradient_type, optimizer_state
        )
        temp_selector._merge_gradients(grad_save_dir, len(self.dataset))

        self.accelerator.wait_for_everyone()
        return grads_path if os.path.exists(grads_path) else None

    def _do_random_subspace_logdet(self, grads: torch.Tensor, lengths: torch.Tensor,
                                    num_samples: int) -> List[int]:
        """Perform random subspace projection + logdet selection.
        Preprocessing matches OptGCS exactly: NaN->0, length_norm, clip, L2_norm."""
        n, d = grads.shape

        # GPU-accelerate (CPU greedy loop overruns watchdog on 13K+ selections)
        dev = self.device if torch.cuda.is_available() else torch.device("cpu")

        # Preprocess: same pipeline as OptGCS._preprocess_grads()
        h = grads.clone().to(dev)
        lengths = lengths.to(dev)
        h[~torch.isfinite(h)] = 0.0
        alpha = self.length_norm_alpha
        if alpha > 0:
            h = h / lengths.float().pow(alpha).unsqueeze(1).clamp(min=1.0)

        # Adaptive clipping (matching OptGCS)
        if self.clipping_method == "adaptive":
            norms_pre = h.norm(dim=1)
            positive_norms = norms_pre[norms_pre > 1e-12]
            if len(positive_norms) > 0:
                tau = float(torch.quantile(positive_norms, 0.95).item())
                scale = torch.clamp(tau / norms_pre.clamp(min=1e-12), max=1.0)
                h = h * scale.unsqueeze(1)

        # L2 normalize
        norms = h.norm(dim=1, keepdim=True).clamp(min=1e-12)
        h = h / norms

        # Random orthogonal subspace (only this uses self.seed, not projector_seed)
        gen = torch.Generator(device=dev).manual_seed(self.seed + 9999)
        r = min(self.subspace_dim, d)
        Q, _ = torch.linalg.qr(torch.randn(d, r, generator=gen, device=dev))

        projections = h @ Q
        scores = (projections ** 2).sum(dim=1)

        # LogDet greedy
        k = min(num_samples, n)
        eps = self.logdet_eps
        prefilter_k = min(int(self.prefilter_ratio * k), n) if self.prefilter_ratio > 0 else n

        if prefilter_k < n:
            topk = torch.topk(scores, k=prefilter_k, largest=True)
            cand_idx = topk.indices
            X = projections[cand_idx].clone()
        else:
            cand_idx = torch.arange(n, device=dev)
            X = projections.clone()

        A_inv = torch.eye(r, dtype=X.dtype, device=dev) / eps
        selected_local = []
        available = torch.ones(len(X), dtype=torch.bool, device=dev)

        neg_inf = torch.tensor(-float('inf'), device=dev, dtype=X.dtype)
        for t in range(k):
            gains = (X @ A_inv * X).sum(dim=1)
            gains = torch.where(available, gains, neg_inf)
            best = gains.argmax().item()
            selected_local.append(best)
            available[best] = False
            x = X[best]
            Ax = A_inv @ x
            A_inv -= torch.outer(Ax, Ax) / (1.0 + x @ Ax)

        return cand_idx[torch.tensor(selected_local, device=dev)].tolist()


@register_selector('grad_norm_topk')
class GradNormTopKSelector(Selector):
    """
    Negative Control: Simple gradient norm top-k selection.
    Computes its own gradient features from current model state for fair comparison.
    If OptGCS ~ grad_norm_topk, it means spectral analysis is just finding high-norm samples.
    """

    def __init__(
        self,
        dataset,
        accelerator,
        data_collator,
        cache_dir: str,
        seed: int = 42,
        length_norm_alpha: float = 0.5,
        gradient_type: str = "adam_diag",
        proj_dim: int = 4096,
        save_interval: int = 16,
        compute_own_grads: bool = True,
        source_grad_dirs: Optional[List[str]] = None,
        eval_dataset=None,
        **kwargs,
    ):
        super().__init__(dataset, accelerator, data_collator, cache_dir)
        self.seed = seed
        self.length_norm_alpha = length_norm_alpha
        self.gradient_type = gradient_type
        self.proj_dim = proj_dim
        self.save_interval = save_interval
        self.compute_own_grads = compute_own_grads
        self.source_grad_dirs = source_grad_dirs or []
        self.device = self.accelerator.device
        os.makedirs(self.cache_dir, exist_ok=True)

    def _compute_own_gradients(self, model, step_id: int, **kwargs) -> Optional[str]:
        """Compute gradient features using OptGCS infrastructure."""
        from .spec_gcs_selector import OptGCSSelector

        grad_cfg = dict(step_id=step_id, gradient_type=self.gradient_type,
                        proj_dim=self.proj_dim, projector_seed=self.seed,
                        dataset_size=len(self.dataset))
        grad_hash = hashlib.md5(_json.dumps(grad_cfg, sort_keys=True).encode()).hexdigest()[:10]
        grad_save_dir = os.path.join(self.cache_dir, "gradients", f"step_{step_id}_{grad_hash}")
        grads_path = os.path.join(grad_save_dir, "all_projected_grads.pt")

        if os.path.exists(grads_path):
            return grads_path

        os.makedirs(grad_save_dir, exist_ok=True)
        temp_selector = OptGCSSelector(
            dataset=self.dataset,
            accelerator=self.accelerator,
            data_collator=self.data_collator,
            cache_dir=self.cache_dir,
            gradient_type=self.gradient_type,
            proj_dim=self.proj_dim,
            save_interval=self.save_interval,
            seed=self.seed,
        )

        optimizer_state = kwargs.get('optimizer_state', None)
        temp_selector._collect_and_save_projected_gradients(
            model, grad_save_dir, self.dataset,
            self.gradient_type, optimizer_state
        )
        temp_selector._merge_gradients(grad_save_dir, len(self.dataset))

        self.accelerator.wait_for_everyone()
        return grads_path if os.path.exists(grads_path) else None

    def select(self, model, step_id: int, num_samples: int, **kwargs) -> List[int]:
        os.makedirs(self.cache_dir, exist_ok=True)

        cfg = dict(step_id=step_id, num_samples=num_samples, seed=self.seed,
                   length_norm_alpha=self.length_norm_alpha, dataset_size=len(self.dataset),
                   compute_own_grads=self.compute_own_grads, gradient_type=self.gradient_type)
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

        # Get gradient features
        grads_path = None
        if self.compute_own_grads:
            grads_path = self._compute_own_gradients(model, step_id, **kwargs)

        if grads_path is None:
            # Fallback: search sibling caches
            search_dirs = list(self.source_grad_dirs)
            parent = os.path.dirname(self.cache_dir)
            if os.path.exists(parent):
                for sibling in os.listdir(parent):
                    sib_path = os.path.join(parent, sibling)
                    if os.path.isdir(sib_path) and sib_path not in search_dirs:
                        search_dirs.append(sib_path)
            grads_path = _find_gradient_cache(search_dirs, step_id)

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
                            "compute_own_grads": self.compute_own_grads,
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

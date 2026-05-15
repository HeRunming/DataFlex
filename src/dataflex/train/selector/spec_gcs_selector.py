"""
Opt-GCS: Rank-Truncated Spectral Coreset Selection in Optimizer-Induced Update Space.

Core idea: formulate unsupervised SFT data selection as spectral coreset construction
in optimizer-induced update space. Each training example is represented by its
frozen-state optimizer-induced local update feature. The method recovers the dominant
update covariance subspace, applies optional spectral whitening, and selects a
logdet-diverse subset that maximally covers the training geometry.

Key differentiators vs related work:
- vs FisherSFT: full multi-layer gradient (not last-layer Fisher proxy)
- vs SAGE: coverage-oriented (not agreement/alignment scoring)
- vs SPICE: spectral whitening + rank truncation (not plain logdet)
- vs TAGCOS: explicit spectral eigenspace (not clustering-based coreset)
- vs LESS: unsupervised / target-free (not requiring eval examples)

Selection variants:
- opt_gcs_score: top-k by whitened projection magnitude
- opt_gcs_diverse: top-qk by score, then k-center for diversity
- opt_gcs_logdet: greedy whitened log-det maximization (main method)

Gradient representation variants:
- "sgd": raw gradient g_i (no optimizer preconditioning)
- "adam_diag": diagonal AdamW-preconditioned D_t * g_i
- "adam": full frozen-state AdamW surrogate (LESS-style)
"""

import os
import glob
import math
from typing import List, Dict, Optional, Tuple

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from dataflex.core.registry import register_selector
from dataflex.utils.logging import logger
from dataflex.utils.selector_io import load_cached_selection, save_selection
from .base_selector import Selector

try:
    from trak.projectors import BasicProjector, CudaProjector, ProjectionType
except ImportError:
    BasicProjector = None
    CudaProjector = None
    ProjectionType = None


class IndexedDataset(Dataset):
    """Wraps a dataset to return (index, sample) pairs."""
    def __init__(self, original_dataset):
        self.original_dataset = original_dataset

    def __len__(self):
        return len(self.original_dataset)

    def __getitem__(self, index):
        return index, self.original_dataset[index]


# Register all variants under the same class
@register_selector('opt_gcs_logdet')
@register_selector('opt_gcs_whitened')
@register_selector('opt_gcs_diverse')
@register_selector('opt_gcs_score')
@register_selector('opt_gcs')
# Ablation variants (same class, different params via components.yaml)
@register_selector('opt_gcs_unwhitened')
@register_selector('opt_gcs_raw_sgd')
@register_selector('opt_gcs_adam_full')
# Keep backward-compatible names
@register_selector('spec_gcs_logdet')
@register_selector('spec_gcs_diverse')
@register_selector('spec_gcs_score')
@register_selector('spec_gcs')
class OptGCSSelector(Selector):
    """
    Opt-GCS: Rank-Truncated Spectral Coreset in Optimizer-Induced Update Space.

    Selects training samples that best cover the principal update subspace
    of the SFT data distribution, without requiring any target/validation set.
    """

    def __init__(
        self,
        dataset,
        accelerator,
        data_collator,
        cache_dir: str,
        # === Gradient representation ===
        gradient_type: str = "adam_diag",  # "sgd" | "adam_diag" | "adam"
        proj_dim: int = 4096,
        save_interval: int = 16,
        seed: int = 42,
        # === Spectral analysis ===
        rank_method: str = "effective",  # "effective" | "eigengap" | "entropy" | "fixed"
        fixed_rank: int = 50,
        eigengap_threshold: float = 2.0,
        # === Whitening (KEY DIFFERENTIATOR) ===
        whitening_beta: float = 0.5,  # 0=unwhitened, 1=fully whitened, (0,1)=partial
        # === Length normalization ===
        length_norm_alpha: float = 0.5,
        # === Clipping ===
        clipping_method: str = "adaptive",  # "none" | "fixed" | "adaptive"
        clipping_threshold: float = 0.0,  # for "fixed"; "adaptive" uses percentile_95
        # === Selection strategy ===
        selection_method: str = "logdet",  # "score" | "diverse" | "logdet"
        logdet_eps: float = 1e-3,
        prefilter_ratio: float = 5.0,
        # === Optional (not used for selection, may be passed by trainer) ===
        eval_dataset=None,
        **kwargs,
    ):
        super().__init__(dataset, accelerator, data_collator, cache_dir)
        self.gradient_type = gradient_type
        self.proj_dim = proj_dim
        self.save_interval = save_interval
        self.seed = seed
        self.rank_method = rank_method
        self.fixed_rank = fixed_rank
        self.eigengap_threshold = eigengap_threshold
        self.whitening_beta = whitening_beta
        self.length_norm_alpha = length_norm_alpha
        self.clipping_method = clipping_method
        self.clipping_threshold = clipping_threshold
        self.selection_method = selection_method
        self.logdet_eps = logdet_eps
        self.prefilter_ratio = prefilter_ratio

        self.device = self.accelerator.device
        self.dtype = torch.float16

        os.makedirs(self.cache_dir, exist_ok=True)
        logger.info(
            f"[OptGCS] Initialized: method={selection_method}, gradient={gradient_type}, "
            f"rank={rank_method}, β={whitening_beta}, α={length_norm_alpha}, "
            f"clip={clipping_method}, proj_dim={proj_dim}"
        )

    # ========================================================================
    # Gradient Computation Infrastructure
    # ========================================================================

    def _get_number_of_params(self, model) -> int:
        """Count parameters requiring gradients (handles DeepSpeed ZeRO-3)."""
        num_params = 0
        for p in model.parameters():
            if p.requires_grad:
                if hasattr(p, 'ds_numel'):
                    num_params += p.ds_numel
                else:
                    num_params += p.numel()
        if self.accelerator.is_main_process:
            logger.info(f"[OptGCS] Trainable parameters: {num_params:,}")
        return num_params

    def _prepare_optimizer_state(self, model, optimizer_state: Optional[Dict] = None) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Prepare Adam first/second moment estimates for optimizer-induced update."""
        if self.gradient_type == "sgd":
            return None, None

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
                logger.warning("[OptGCS] No optimizer_state provided, falling back to SGD gradients.")
                return None, None
            for param in model.parameters():
                if param.requires_grad:
                    if param in optimizer_state and "exp_avg" in optimizer_state[param]:
                        avg_list.append(optimizer_state[param]["exp_avg"].view(-1))
                        avg_sq_list.append(optimizer_state[param]["exp_avg_sq"].view(-1))

        if not avg_list:
            return None, None

        avg = torch.cat(avg_list).to(self.device)
        avg_sq = torch.cat(avg_sq_list).to(self.device)
        return avg, avg_sq

    def _obtain_gradients(self, model, batch, gradient_type: str,
                          m: Optional[torch.Tensor] = None,
                          v: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute per-sample update feature based on gradient_type:
        - "sgd": raw gradient g_i
        - "adam_diag": D_t * g_i where D_t = diag(1/(sqrt(v_t)+eps))
        - "adam": full frozen-state AdamW surrogate (β₁·m + (1-β₁)·g) / (sqrt(v)+eps)
        """
        # Forward + backward to get raw gradient
        if self.accelerator.state.deepspeed_plugin is not None:
            from deepspeed.utils import safe_get_full_grad
            loss = model(**batch).loss
            model.backward(loss)
            grads = []
            for p in model.parameters():
                if p.requires_grad:
                    g = safe_get_full_grad(p)
                    if g is not None:
                        grads.append(g.contiguous().view(-1))
            vectorized_grads = torch.cat(grads) if grads else torch.zeros(1, device=self.device)
        else:
            with self.accelerator.no_sync(model):
                loss = model(**batch).loss
                self.accelerator.backward(loss)
            vectorized_grads = torch.cat(
                [p.grad.view(-1) for p in model.parameters() if p.grad is not None]
            )

        # Apply optimizer-induced transformation
        if gradient_type == "sgd" or m is None or v is None:
            # Raw gradient — no transformation
            pass
        elif gradient_type == "adam_diag":
            # Diagonal AdamW preconditioning: u_i = g_i / (sqrt(v_t) + eps)
            # This is the theoretically recommended "frozen-state local update feature"
            # Clamp v to avoid division by near-zero (especially during early warmup)
            eps = 1e-08
            denom = v.clamp(min=1e-16).sqrt().add_(eps)
            vectorized_grads = vectorized_grads / denom
            # Clamp result to avoid extreme values from immature optimizer state
            vectorized_grads = vectorized_grads.clamp(-1e4, 1e4)
        elif gradient_type == "adam":
            # Full frozen-state AdamW surrogate (LESS-style):
            # u_i = (β₁·m + (1-β₁)·g_i) / (sqrt(β₂·v + (1-β₂)·g_i²) + eps)
            beta1, beta2, eps = 0.9, 0.999, 1e-08
            denom = v.mul(beta2)
            denom.addcmul_(vectorized_grads, vectorized_grads, value=(1 - beta2))
            denom.clamp_(min=1e-16).sqrt_().add_(eps)
            vectorized_grads.mul_(1 - beta1).add_(m, alpha=beta1)
            vectorized_grads.div_(denom)
            vectorized_grads = vectorized_grads.clamp(-1e4, 1e4)
            del denom
        else:
            raise ValueError(f"Unknown gradient_type: {gradient_type}")

        model.zero_grad()
        return vectorized_grads

    def _get_trak_projector(self):
        """Get TRAK projector (CUDA preferred, falls back to Basic)."""
        try:
            import fast_jl
            num_sms = torch.cuda.get_device_properties(self.device.index).multi_processor_count
            fast_jl.project_rademacher_8(
                torch.zeros(8, 1_000, device=self.device), 512, 0, num_sms
            )
            projector = CudaProjector
            if self.accelerator.is_main_process:
                logger.info("[OptGCS] Using CudaProjector.")
        except (ImportError, RuntimeError, Exception):
            projector = BasicProjector
            if self.accelerator.is_main_process:
                logger.info("[OptGCS] Using BasicProjector (CudaProjector unavailable).")
        return projector

    def _get_max_saved_index(self, save_dir: str) -> int:
        """Find max saved sample index for resume."""
        if not os.path.exists(save_dir) or not self.accelerator.is_main_process:
            return -1
        files = [f for f in os.listdir(save_dir) if f.startswith("grads") and f.endswith(".pt")]
        if not files:
            return -1
        indices = [int(f.split('.')[0].split('-')[1]) for f in files]
        return max(indices) if indices else -1

    # ========================================================================
    # Core: Gradient Collection
    # ========================================================================

    def _collect_and_save_projected_gradients(self, model, save_dir: str,
                                               dataset_to_use,
                                               gradient_type: str,
                                               optimizer_state: Optional[Dict] = None):
        """
        Compute per-sample optimizer-induced update features, project via TRAK, save in chunks.
        Also records token lengths for length normalization.
        """
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

        m, v = self._prepare_optimizer_state(model, optimizer_state)

        indexed_dataset = IndexedDataset(dataset_to_use)

        def indexed_collator_wrapper(features):
            indices = [f[0] for f in features]
            original_data = [f[1] for f in features]
            collated_batch = self.data_collator(original_data)
            return {'indices': torch.tensor(indices), 'batch': collated_batch}

        dataloader = DataLoader(
            indexed_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=2,
            collate_fn=indexed_collator_wrapper,
        )
        dataloader = self.accelerator.prepare(dataloader)

        # Resume support
        max_index = self._get_max_saved_index(save_dir)
        if self.accelerator.is_main_process and max_index > 0:
            logger.info(f"[OptGCS] Resuming gradient computation from index {max_index + 1}")
        self.accelerator.wait_for_everyone()

        total_samples_in_loader = len(dataloader)
        model_device = next(model.parameters()).device
        save_interval = self.save_interval

        grad_buffer = torch.zeros(save_interval, num_params, device=model_device, dtype=self.dtype)
        idx_buffer = torch.zeros(save_interval, dtype=torch.long)
        length_buffer = torch.zeros(save_interval, dtype=torch.long)
        buf_pos = 0

        for batch_idx, data in enumerate(tqdm(
            dataloader,
            desc=f"[Rank {self.accelerator.process_index}] OptGCS Gradients",
            disable=not self.accelerator.is_local_main_process,
            dynamic_ncols=True,
            position=self.accelerator.process_index,
        ), 1):
            indices = data['indices']
            batch = data['batch']

            # Compute token length
            labels = batch.get('labels', None)
            if labels is not None:
                token_len = (labels != -100).sum().item()
            else:
                input_ids = batch.get('input_ids', None)
                token_len = input_ids.numel() if input_ids is not None else 1
            token_len = max(token_len, 1)

            vectorized_grads = self._obtain_gradients(model, batch, gradient_type, m, v)
            grad_buffer[buf_pos].copy_(vectorized_grads)
            del vectorized_grads
            idx_buffer[buf_pos] = indices[0]
            length_buffer[buf_pos] = token_len
            buf_pos += 1

            if buf_pos == save_interval or batch_idx == total_samples_in_loader:
                projected = projector.project(grad_buffer[:buf_pos], model_id=0).cpu()
                save_path = os.path.join(
                    save_dir,
                    f"grads-{idx_buffer[:buf_pos].max().item()}-rank{self.accelerator.process_index}.pt",
                )
                torch.save({
                    'grads': projected,
                    'indices': idx_buffer[:buf_pos].clone(),
                    'lengths': length_buffer[:buf_pos].clone(),
                }, save_path)
                del projected
                buf_pos = 0

        del grad_buffer, idx_buffer, length_buffer
        self.accelerator.wait_for_everyone()

    def _merge_gradients(self, save_dir: str, total_samples: int):
        """Merge chunk files from all ranks, producing ordered gradient matrix + lengths."""
        if not self.accelerator.is_main_process:
            return

        logger.info(f"[OptGCS] Merging projected gradients from {save_dir}")
        files = glob.glob(os.path.join(save_dir, "grads-*-rank*.pt"))
        if not files:
            logger.warning("[OptGCS] No gradient files found to merge.")
            return

        final_grads = torch.zeros(total_samples, self.proj_dim, dtype=torch.float32)
        final_lengths = torch.ones(total_samples, dtype=torch.long)

        for file_path in tqdm(files, desc="[OptGCS] Merging"):
            chunk = torch.load(file_path, map_location="cpu")
            grads_chunk = chunk['grads'].to(torch.float32)
            indices_chunk = chunk['indices']
            lengths_chunk = chunk.get('lengths', torch.ones(len(indices_chunk), dtype=torch.long))
            final_grads[indices_chunk] = grads_chunk
            final_lengths[indices_chunk] = lengths_chunk

        output_grads = os.path.join(save_dir, "all_projected_grads.pt")
        output_lengths = os.path.join(save_dir, "all_token_lengths.pt")
        torch.save(final_grads, output_grads)
        torch.save(final_lengths, output_lengths)
        logger.info(f"[OptGCS] Saved merged gradients: {final_grads.shape} to {output_grads}")

        # Clean up chunk files
        for f in files:
            os.remove(f)

    # ========================================================================
    # Core: Spectral Analysis with Whitening
    # ========================================================================

    def _auto_rank(self, eigenvalues: torch.Tensor) -> int:
        """Determine the intrinsic rank from the eigenvalue spectrum."""
        eigenvalues = eigenvalues.float()
        eigenvalues = eigenvalues[eigenvalues > 1e-10]
        if len(eigenvalues) == 0:
            return 1

        if self.rank_method == "fixed":
            return min(self.fixed_rank, len(eigenvalues))

        elif self.rank_method == "effective":
            r_eff = eigenvalues.sum() / eigenvalues[0]
            r = max(1, int(r_eff.item()))
            return min(r, len(eigenvalues))

        elif self.rank_method == "eigengap":
            for j in range(len(eigenvalues) - 1):
                ratio = eigenvalues[j] / eigenvalues[j + 1]
                if ratio > self.eigengap_threshold:
                    return j + 1
            return min(50, len(eigenvalues))

        elif self.rank_method == "entropy":
            p = eigenvalues / eigenvalues.sum()
            entropy = -(p * torch.log(p + 1e-12)).sum()
            r_ent = torch.exp(entropy)
            r = max(1, int(r_ent.item()))
            return min(r, len(eigenvalues))

        else:
            raise ValueError(f"Unknown rank_method: {self.rank_method}")

    def _apply_clipping(self, h: torch.Tensor) -> torch.Tensor:
        """Apply gradient clipping based on clipping_method."""
        if self.clipping_method == "none":
            return h

        norms = h.norm(dim=1)  # [n]

        if self.clipping_method == "fixed":
            tau = self.clipping_threshold if self.clipping_threshold > 0 else norms.median().item() * 3
        elif self.clipping_method == "adaptive":
            # Use 95th percentile as threshold
            tau = float(torch.quantile(norms, 0.95).item())
        else:
            return h

        # Clip: h_i = h_i * min(1, tau / ||h_i||)
        scale = torch.clamp(tau / norms.clamp(min=1e-12), max=1.0)  # [n]
        h = h * scale.unsqueeze(1)
        if self.accelerator.is_main_process:
            n_clipped = (norms > tau).sum().item()
            logger.info(f"[OptGCS] Clipping: τ={tau:.4f}, clipped {n_clipped}/{len(norms)} samples")
        return h

    def _estimate_eigenspace(self, grads: torch.Tensor, lengths: torch.Tensor
                             ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        Estimate principal update subspace with length normalization and clipping.

        Returns:
            U_r: [proj_dim, r] top-r eigenvectors
            eigenvalues: [r] corresponding eigenvalues (for whitening)
            rank: effective rank used
        """
        n, d = grads.shape
        logger.info(f"[OptGCS] Estimating eigenspace from {n} samples, dim={d}")

        # Step 1: Replace NaN/Inf in raw grads BEFORE normalization
        nan_inf_mask = ~torch.isfinite(grads)
        if nan_inf_mask.any():
            n_bad_elements = nan_inf_mask.sum().item()
            logger.warning(f"[OptGCS] Found {n_bad_elements} NaN/Inf elements in gradients, replacing with 0")
            grads = grads.clone()
            grads[nan_inf_mask] = 0.0

        # Step 2: Length normalization
        alpha = self.length_norm_alpha
        if alpha > 0:
            length_factors = lengths.float().pow(alpha).unsqueeze(1)
            h = grads / length_factors
        else:
            h = grads.clone()

        # Step 3: L2 normalize (direction matters)
        norms = h.norm(dim=1, keepdim=True)
        # Remove zero-norm rows (samples with all-zero or all-Inf gradients)
        valid_mask = (norms.squeeze() > 1e-8) & torch.isfinite(norms.squeeze())
        if not valid_mask.all():
            n_invalid = (~valid_mask).sum().item()
            logger.warning(f"[OptGCS] Removing {n_invalid}/{n} samples with zero/invalid norms")
            h = h[valid_mask]
            norms = norms[valid_mask]
            n = h.shape[0]
            if n == 0:
                logger.error("[OptGCS] No valid samples remaining! Returning trivial eigenspace.")
                return torch.eye(d)[:, :1], torch.ones(1), 1

        h = h / norms.clamp(min=1e-12)

        # Step 4: Apply clipping
        h = self._apply_clipping(h)

        # Step 5: Eigendecomposition via randomized SVD (faster + more stable than eigh)
        # svd_lowrank computes top-q singular vectors efficiently
        max_rank = min(200, n - 1, d)  # cap at 200 for efficiency
        logger.info(f"[OptGCS] Computing randomized SVD (n={n}, d={d}, q={max_rank})...")
        try:
            U_svd, S_svd, V_svd = torch.svd_lowrank(h, q=max_rank, niter=5)
            # eigenvalues of covariance = S² / n
            eigenvalues = (S_svd ** 2) / n
            eigenvectors = V_svd  # [d, q]
        except Exception as e:
            logger.warning(f"[OptGCS] svd_lowrank failed: {e}. Falling back to full SVD on smaller matrix.")
            # Fallback: subsample to make it tractable
            if n > 10000:
                perm = torch.randperm(n)[:10000]
                h_sub = h[perm]
            else:
                h_sub = h
            U_svd, S_svd, Vt_svd = torch.linalg.svd(h_sub, full_matrices=False)
            eigenvalues = (S_svd ** 2) / len(h_sub)
            eigenvectors = Vt_svd.T

        # Step 6: Determine rank
        pos_eigenvalues = eigenvalues[eigenvalues > 1e-10]
        rank = self._auto_rank(pos_eigenvalues)

        logger.info(
            f"[OptGCS] Spectrum: top-5={eigenvalues[:5].tolist()}, "
            f"rank={rank}, eff_rank={pos_eigenvalues.sum()/pos_eigenvalues[0]:.1f}, "
            f"top-{rank} explains {pos_eigenvalues[:rank].sum()/pos_eigenvalues.sum()*100:.1f}%"
        )

        U_r = eigenvectors[:, :rank]
        top_eigenvalues = eigenvalues[:rank]

        return U_r, top_eigenvalues, rank

    # ========================================================================
    # Core: Whitened Projection (KEY CONTRIBUTION)
    # ========================================================================

    def _project_to_eigenspace(self, grads: torch.Tensor, lengths: torch.Tensor,
                                U_r: torch.Tensor, eigenvalues: torch.Tensor
                                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Project candidates into the whitened eigenspace.

        x_i^(β) = Λ_r^{-β/2} · U_r^T · h_i

        β=0: unwhitened (preserves original variance, biases toward dominant directions)
        β=1: fully whitened (equal weight to all directions, maximum coverage)
        β∈(0,1): partial whitening (coverage-agreement tradeoff)

        Returns:
            projections: [n, r] whitened projections
            scores: [n] = ||x_i^(β)||²
        """
        alpha = self.length_norm_alpha
        beta = self.whitening_beta

        # Length normalization
        if alpha > 0:
            length_factors = lengths.float().pow(alpha).unsqueeze(1)
            h = grads / length_factors
        else:
            h = grads.clone()

        # L2 normalize
        norms = h.norm(dim=1, keepdim=True).clamp(min=1e-12)
        h = h / norms

        # Handle NaN/Inf
        invalid = ~torch.isfinite(h).all(dim=1)
        if invalid.any():
            h[invalid] = 0.0

        # Project to eigenspace: [n, r]
        projections = h @ U_r

        # Apply whitening: x_i^(β) = Λ^{-β/2} * (U_r^T h_i)
        if beta > 0:
            # Whitening weights: λ_j^{-β/2}
            whiten_weights = eigenvalues.clamp(min=1e-10).pow(-beta / 2)  # [r]
            projections = projections * whiten_weights.unsqueeze(0)  # [n, r]

        scores = (projections ** 2).sum(dim=1)  # [n]
        return projections, scores

    # ========================================================================
    # Selection Strategies
    # ========================================================================

    def _select_by_score(self, scores: torch.Tensor, num_samples: int) -> List[int]:
        """Simple top-k by whitened projection score."""
        k = min(num_samples, len(scores))
        topk = torch.topk(scores, k=k, largest=True)
        return topk.indices.tolist()

    def _select_by_diverse(self, projections: torch.Tensor, scores: torch.Tensor,
                            num_samples: int) -> List[int]:
        """Top-qk by score, then k-center greedy in whitened projection space."""
        n = len(scores)
        k = min(num_samples, n)
        q = self.prefilter_ratio
        prefilter_k = min(int(q * k), n)

        topk = torch.topk(scores, k=prefilter_k, largest=True)
        candidate_indices = topk.indices
        candidate_projections = projections[candidate_indices]

        # K-center greedy
        selected_local = [0]
        if k == 1:
            return [candidate_indices[0].item()]

        min_distances = torch.cdist(
            candidate_projections, candidate_projections[0:1]
        ).squeeze(1)
        min_distances[0] = -1

        for _ in range(1, k):
            next_idx = min_distances.argmax().item()
            selected_local.append(next_idx)
            min_distances[next_idx] = -1
            new_distances = torch.cdist(
                candidate_projections, candidate_projections[next_idx:next_idx+1]
            ).squeeze(1)
            min_distances = torch.minimum(min_distances, new_distances)
            for s in selected_local:
                min_distances[s] = -1

        selected_original = candidate_indices[torch.tensor(selected_local)].tolist()
        return selected_original

    def _select_by_logdet(self, projections: torch.Tensor, scores: torch.Tensor,
                           num_samples: int) -> List[int]:
        """
        Greedy whitened log-det maximization:
        max_{|S|=k} log det(εI + Σ_{i∈S} x_i^(β) (x_i^(β))^T)

        The whitening β is already applied in projections.
        Uses Sherman-Morrison for efficient r×r A^{-1} updates.
        """
        n, r = projections.shape
        k = min(num_samples, n)
        eps = self.logdet_eps

        # Prefilter to top-scoring candidates
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
            # Marginal gain: x_i^T A^{-1} x_i
            gains = (X @ A_inv * X).sum(dim=1)
            gains[~available] = -float('inf')

            best_local = gains.argmax().item()
            selected_local.append(best_local)
            available[best_local] = False

            # Sherman-Morrison update
            x = X[best_local]
            Ax = A_inv @ x
            denom = 1.0 + x @ Ax
            A_inv -= torch.outer(Ax, Ax) / denom

            if (t + 1) % 500 == 0:
                logger.info(f"[OptGCS-LogDet] Selected {t+1}/{k}, gain={gains[best_local]:.4f}")

        selected_original = candidate_indices[torch.tensor(selected_local)].tolist()
        return selected_original

    # ========================================================================
    # Main Entry Point
    # ========================================================================

    def select(self, model, step_id: int, num_samples: int, **kwargs) -> List[int]:
        """
        Select num_samples training examples using spectral update-space analysis.

        Full pipeline:
        1. Compute optimizer-induced update features for all data → TRAK projection
        2. Length normalize + clip → estimate covariance eigenspace
        3. Apply spectral whitening → project all candidates
        4. Select via score/diverse/logdet
        """
        os.makedirs(self.cache_dir, exist_ok=True)

        # Check cache
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

        # Paths
        grad_save_dir = os.path.join(self.cache_dir, "gradients", str(step_id))
        grads_path = os.path.join(grad_save_dir, "all_projected_grads.pt")
        lengths_path = os.path.join(grad_save_dir, "all_token_lengths.pt")

        # Step 1: Compute gradients (all ranks participate)
        if not os.path.exists(grads_path):
            os.makedirs(grad_save_dir, exist_ok=True)
            optimizer_state = kwargs.get('optimizer_state', None)
            self._collect_and_save_projected_gradients(
                model, grad_save_dir, self.dataset,
                self.gradient_type, optimizer_state
            )
            self._merge_gradients(grad_save_dir, len(self.dataset))

        self.accelerator.wait_for_everyone()

        # Steps 2-4: Spectral analysis and selection (main process only)
        if self.accelerator.is_main_process:
            logger.info(f"[OptGCS] Loading gradients from {grads_path}")
            grads = torch.load(grads_path, map_location="cpu")
            lengths = torch.load(lengths_path, map_location="cpu")

            # Estimate eigenspace
            U_r, eigenvalues, rank = self._estimate_eigenspace(grads, lengths)

            # Whitened projection
            projections, scores = self._project_to_eigenspace(grads, lengths, U_r, eigenvalues)

            # Selection
            logger.info(
                f"[OptGCS] Selecting {num_samples} samples via '{self.selection_method}' "
                f"(β={self.whitening_beta}, rank={rank})"
            )
            if self.selection_method == "score":
                selected_indices = self._select_by_score(scores, num_samples)
            elif self.selection_method == "diverse":
                selected_indices = self._select_by_diverse(projections, scores, num_samples)
            elif self.selection_method == "logdet":
                selected_indices = self._select_by_logdet(projections, scores, num_samples)
            else:
                raise ValueError(f"Unknown selection_method: {self.selection_method}")

            # Save with full diagnostic metadata
            metric_payload = {
                "selection_method": self.selection_method,
                "gradient_type": self.gradient_type,
                "whitening_beta": self.whitening_beta,
                "rank_used": rank,
                "rank_method": self.rank_method,
                "eigenvalues_top10": eigenvalues[:10].tolist(),
                "effective_rank": float(eigenvalues[eigenvalues > 1e-10].sum() / eigenvalues[0]) if eigenvalues[0] > 0 else 0,
                "scores_selected_mean": float(scores[torch.tensor(selected_indices)].mean()),
                "scores_all_mean": float(scores.mean()),
                "length_norm_alpha": self.length_norm_alpha,
                "clipping_method": self.clipping_method,
            }
            save_selection(save_path, selected_indices, metric_payload, self.accelerator)
            logger.info(
                f"[OptGCS] Selected {len(selected_indices)} samples. "
                f"Rank={rank}, β={self.whitening_beta}, "
                f"score_selected={metric_payload['scores_selected_mean']:.4f}, "
                f"score_all={metric_payload['scores_all_mean']:.4f}"
            )
        else:
            selected_indices = None

        # Broadcast to all ranks
        obj = [selected_indices]
        if dist.is_available() and dist.is_initialized():
            dist.broadcast_object_list(obj, src=0)
        selected_indices = obj[0] or []

        return selected_indices

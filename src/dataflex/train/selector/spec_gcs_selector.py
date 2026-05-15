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
import hashlib
import json as _json
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
@register_selector('opt_gcs_rank50')
@register_selector('opt_gcs_rank100')
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
        # === Whitening safety ===
        whitening_eigen_floor: float = 1e-6,
        whitening_max_weight: float = 100.0,
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
        self.whitening_eigen_floor = whitening_eigen_floor
        self.whitening_max_weight = whitening_max_weight

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
                num_params += p.ds_numel if hasattr(p, 'ds_numel') else p.numel()
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
        Compute per-sample update feature. All requires_grad params are included;
        None grads are filled with zeros to keep alignment with optimizer state.
        """
        if self.accelerator.state.deepspeed_plugin is not None:
            from deepspeed.utils import safe_get_full_grad
            loss = model(**batch).loss
            model.backward(loss)
            grads = []
            for p in model.parameters():
                if p.requires_grad:
                    g = safe_get_full_grad(p)
                    numel = p.ds_numel if hasattr(p, 'ds_numel') else p.numel()
                    if g is not None:
                        grads.append(g.contiguous().view(-1))
                    else:
                        grads.append(torch.zeros(numel, device=self.device, dtype=self.dtype))
            vectorized_grads = torch.cat(grads)
        else:
            with self.accelerator.no_sync(model):
                loss = model(**batch).loss
                self.accelerator.backward(loss)
            grads_list = []
            for p in model.parameters():
                if p.requires_grad:
                    if p.grad is not None:
                        grads_list.append(p.grad.contiguous().view(-1))
                    else:
                        grads_list.append(torch.zeros(p.numel(), device=p.device, dtype=p.dtype))
            vectorized_grads = torch.cat(grads_list)

        # Apply optimizer-induced transformation
        if gradient_type == "sgd" or m is None or v is None:
            pass
        elif gradient_type == "adam_diag":
            eps = 1e-08
            denom = v.clamp(min=1e-16).sqrt().add_(eps)
            vectorized_grads = vectorized_grads / denom
            vectorized_grads = vectorized_grads.clamp(-1e4, 1e4)
        elif gradient_type == "adam":
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

        # Log optimizer state diagnostics + save to file
        if self.accelerator.is_main_process:
            opt_found = m is not None and v is not None
            diag = {
                "optimizer_state_found": opt_found,
                "gradient_type": gradient_type,
            }
            if opt_found:
                diag.update({
                    "m_norm": float(m.norm().item()),
                    "v_mean": float(v.mean().item()),
                    "v_min": float(v.min().item()),
                    "v_max": float(v.max().item()),
                })
            logger.info(
                f"[OptGCS] Optimizer state: found={opt_found}, gradient_type={gradient_type}"
                + (f", m_norm={diag['m_norm']:.4f}, v_mean={diag['v_mean']:.6f}, v_min={diag['v_min']:.6f}"
                   if opt_found else ", FALLING BACK TO RAW SGD")
            )
            diag_path = os.path.join(save_dir, "optimizer_state_diagnostics.json")
            with open(diag_path, "w") as f:
                _json.dump(diag, f, indent=2)

        # Clean old chunks before starting fresh (no fake resume)
        if self.accelerator.is_main_process:
            old_chunks = glob.glob(os.path.join(save_dir, "grads-*-rank*.pt"))
            if old_chunks:
                logger.info(f"[OptGCS] Cleaning {len(old_chunks)} old chunk files for fresh computation")
                for f in old_chunks:
                    os.remove(f)
        self.accelerator.wait_for_everyone()

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

        for f in files:
            os.remove(f)

    # ========================================================================
    # Unified Preprocessing (shared by eigenspace estimation & projection)
    # ========================================================================

    def _preprocess_grads(self, grads: torch.Tensor, lengths: torch.Tensor
                          ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Unified preprocessing pipeline: NaN/Inf→0, length norm, clip, L2 norm.
        Both _estimate_eigenspace and _project_to_eigenspace use this to ensure
        the eigenspace and the candidate projections are in the SAME geometry.

        Returns:
            h: [n, d] preprocessed (L2-normalized) gradient features
            valid_mask: [n] boolean mask of valid (non-zero) samples
        """
        n, d = grads.shape

        # Step 1: Replace NaN/Inf
        h = grads.clone()
        bad = ~torch.isfinite(h)
        if bad.any():
            h[bad] = 0.0

        # Step 2: Length normalization
        alpha = self.length_norm_alpha
        if alpha > 0:
            h = h / lengths.float().pow(alpha).unsqueeze(1).clamp(min=1.0)

        # Step 3: Clipping BEFORE L2 normalization
        h = self._apply_clipping(h)

        # Step 4: Compute norms → valid mask → L2 normalize
        norms = h.norm(dim=1, keepdim=True)
        valid_mask = (norms.squeeze() > 1e-8) & torch.isfinite(norms.squeeze())
        # Zero out invalid rows (they'll have score=0 and won't be selected)
        h_out = torch.zeros_like(h)
        h_out[valid_mask] = h[valid_mask] / norms[valid_mask].clamp(min=1e-12)

        return h_out, valid_mask

    # ========================================================================
    # Core: Spectral Analysis with Whitening
    # ========================================================================

    def _auto_rank(self, eigenvalues: torch.Tensor) -> int:
        """Determine the intrinsic rank from the eigenvalue spectrum."""
        eigenvalues = eigenvalues.float()
        eigenvalues = eigenvalues[eigenvalues > 1e-10]
        if len(eigenvalues) == 0:
            return 1

        MIN_RANK = 10

        if self.rank_method == "fixed":
            return min(self.fixed_rank, len(eigenvalues))
        elif self.rank_method == "effective":
            r_eff = eigenvalues.sum() / eigenvalues[0]
            return min(max(MIN_RANK, math.ceil(r_eff.item())), len(eigenvalues))
        elif self.rank_method == "eigengap":
            for j in range(len(eigenvalues) - 1):
                if eigenvalues[j] / eigenvalues[j + 1] > self.eigengap_threshold:
                    return max(MIN_RANK, j + 1)
            return min(50, len(eigenvalues))
        elif self.rank_method == "entropy":
            p = eigenvalues / eigenvalues.sum()
            r_ent = torch.exp(-(p * torch.log(p + 1e-12)).sum())
            return min(max(MIN_RANK, math.ceil(r_ent.item())), len(eigenvalues))
        else:
            raise ValueError(f"Unknown rank_method: {self.rank_method}")

    def _apply_clipping(self, h: torch.Tensor) -> torch.Tensor:
        """Apply gradient clipping based on clipping_method."""
        if self.clipping_method == "none":
            return h

        norms = h.norm(dim=1)

        if self.clipping_method == "fixed":
            tau = self.clipping_threshold if self.clipping_threshold > 0 else norms.median().item() * 3
        elif self.clipping_method == "adaptive":
            positive_norms = norms[norms > 1e-12]
            if len(positive_norms) == 0:
                logger.warning("[OptGCS] All gradient norms are zero; skip clipping.")
                return h
            tau = float(torch.quantile(positive_norms, 0.95).item())
        else:
            return h

        scale = torch.clamp(tau / norms.clamp(min=1e-12), max=1.0)
        h = h * scale.unsqueeze(1)
        if self.accelerator.is_main_process:
            n_clipped = (norms > tau).sum().item()
            logger.info(f"[OptGCS] Clipping: τ={tau:.4f}, clipped {n_clipped}/{len(norms)} samples")
        return h

    def _estimate_eigenspace(self, grads: torch.Tensor, lengths: torch.Tensor
                             ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        Estimate principal update subspace using unified preprocessing.

        Returns:
            U_r: [proj_dim, r] top-r eigenvectors
            eigenvalues: [r] corresponding eigenvalues (for whitening)
            rank: effective rank used
        """
        n, d = grads.shape
        logger.info(f"[OptGCS] Estimating eigenspace from {n} samples, dim={d}")

        # Unified preprocessing
        h, valid_mask = self._preprocess_grads(grads, lengths)
        h_valid = h[valid_mask]
        n_valid = h_valid.shape[0]

        if n_valid == 0:
            logger.error("[OptGCS] No valid samples! Returning trivial eigenspace.")
            return torch.eye(d)[:, :1], torch.ones(1), 1

        n_invalid = n - n_valid
        if n_invalid > 0:
            logger.warning(f"[OptGCS] Using {n_valid}/{n} valid samples for eigenspace estimation")

        # Eigendecomposition via randomized SVD
        max_rank = min(200, n_valid - 1, d)
        logger.info(f"[OptGCS] Computing randomized SVD (n={n_valid}, d={d}, q={max_rank})...")
        try:
            U_svd, S_svd, V_svd = torch.svd_lowrank(h_valid, q=max_rank, niter=5)
            eigenvalues = (S_svd ** 2) / n_valid
            eigenvectors = V_svd
        except Exception as e:
            logger.warning(f"[OptGCS] svd_lowrank failed: {e}. Falling back.")
            if n_valid > 10000:
                h_sub = h_valid[torch.randperm(n_valid)[:10000]]
            else:
                h_sub = h_valid
            _, S_svd, Vt_svd = torch.linalg.svd(h_sub, full_matrices=False)
            eigenvalues = (S_svd ** 2) / len(h_sub)
            eigenvectors = Vt_svd.T

        pos_eigenvalues = eigenvalues[eigenvalues > 1e-10]
        rank = self._auto_rank(pos_eigenvalues)

        logger.info(
            f"[OptGCS] Spectrum: top-5={eigenvalues[:5].tolist()}, "
            f"rank={rank}, eff_rank={pos_eigenvalues.sum()/pos_eigenvalues[0]:.1f}, "
            f"top-{rank} explains {pos_eigenvalues[:rank].sum()/pos_eigenvalues.sum()*100:.1f}%"
        )

        return eigenvectors[:, :rank], eigenvalues[:rank], rank

    # ========================================================================
    # Core: Whitened Projection (KEY CONTRIBUTION)
    # ========================================================================

    def _project_to_eigenspace(self, grads: torch.Tensor, lengths: torch.Tensor,
                                U_r: torch.Tensor, eigenvalues: torch.Tensor
                                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Project all candidates into the whitened eigenspace using the SAME
        preprocessing as _estimate_eigenspace.

        Returns:
            projections: [n, r] whitened projections
            scores: [n] = ||x_i^(β)||²
        """
        beta = self.whitening_beta

        # Use same unified preprocessing
        h, _ = self._preprocess_grads(grads, lengths)

        # Project to eigenspace
        projections = h @ U_r  # [n, r]

        # Apply whitening with eigenvalue floor and weight cap
        if beta > 0:
            whiten_weights = eigenvalues.clamp(min=self.whitening_eigen_floor).pow(-beta / 2)
            whiten_weights = whiten_weights.clamp(max=self.whitening_max_weight)
            projections = projections * whiten_weights.unsqueeze(0)

        scores = (projections ** 2).sum(dim=1)
        return projections, scores

    # ========================================================================
    # Selection Strategies
    # ========================================================================

    def _select_by_score(self, scores: torch.Tensor, num_samples: int) -> List[int]:
        """Simple top-k by whitened projection score."""
        k = min(num_samples, len(scores))
        return torch.topk(scores, k=k, largest=True).indices.tolist()

    def _select_by_diverse(self, projections: torch.Tensor, scores: torch.Tensor,
                            num_samples: int) -> List[int]:
        """Top-qk by score, then k-center greedy in whitened projection space."""
        n = len(scores)
        k = min(num_samples, n)
        prefilter_k = min(int(self.prefilter_ratio * k), n)

        topk = torch.topk(scores, k=prefilter_k, largest=True)
        candidate_indices = topk.indices
        cand_proj = projections[candidate_indices]

        selected_local = [0]
        if k == 1:
            return [candidate_indices[0].item()]

        min_distances = torch.cdist(cand_proj, cand_proj[0:1]).squeeze(1)
        min_distances[0] = -1

        for _ in range(1, k):
            next_idx = min_distances.argmax().item()
            selected_local.append(next_idx)
            min_distances[next_idx] = -1
            new_d = torch.cdist(cand_proj, cand_proj[next_idx:next_idx+1]).squeeze(1)
            min_distances = torch.minimum(min_distances, new_d)
            for s in selected_local:
                min_distances[s] = -1

        return candidate_indices[torch.tensor(selected_local)].tolist()

    def _select_by_logdet(self, projections: torch.Tensor, scores: torch.Tensor,
                           num_samples: int) -> List[int]:
        """
        Greedy whitened log-det maximization with Sherman-Morrison updates.
        """
        n, r = projections.shape
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

            best_local = gains.argmax().item()
            selected_local.append(best_local)
            available[best_local] = False

            x = X[best_local]
            Ax = A_inv @ x
            A_inv -= torch.outer(Ax, Ax) / (1.0 + x @ Ax)

            if (t + 1) % 500 == 0:
                logger.info(f"[OptGCS-LogDet] Selected {t+1}/{k}, gain={gains[best_local]:.4f}")

        return candidate_indices[torch.tensor(selected_local)].tolist()

    # ========================================================================
    # Main Entry Point
    # ========================================================================

    def _make_grad_cache_path(self, step_id: int) -> str:
        """Build gradient cache path with hash to avoid stale cache reuse."""
        grad_cfg = dict(
            step_id=step_id,
            gradient_type=self.gradient_type,
            proj_dim=self.proj_dim,
            projector_seed=self.seed,
            dataset_size=len(self.dataset),
        )
        grad_hash = hashlib.md5(_json.dumps(grad_cfg, sort_keys=True).encode()).hexdigest()[:10]
        return os.path.join(self.cache_dir, "gradients", f"step_{step_id}_{grad_hash}")

    def _make_selection_cache_path(self, step_id: int, num_samples: int) -> str:
        """Build selection cache path with full config hash."""
        cfg = dict(
            step_id=step_id,
            num_samples=num_samples,
            gradient_type=self.gradient_type,
            proj_dim=self.proj_dim,
            rank_method=self.rank_method,
            fixed_rank=self.fixed_rank,
            whitening_beta=self.whitening_beta,
            length_norm_alpha=self.length_norm_alpha,
            clipping_method=self.clipping_method,
            selection_method=self.selection_method,
            logdet_eps=self.logdet_eps,
            prefilter_ratio=self.prefilter_ratio,
            dataset_size=len(self.dataset),
        )
        cfg_hash = hashlib.md5(_json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:10]
        return os.path.join(self.cache_dir, f"step_{step_id}_k{num_samples}_{cfg_hash}.json")

    def select(self, model, step_id: int, num_samples: int, **kwargs) -> List[int]:
        """
        Select num_samples training examples using spectral update-space analysis.
        """
        os.makedirs(self.cache_dir, exist_ok=True)

        # Check selection cache
        save_path = self._make_selection_cache_path(step_id, num_samples)
        if os.path.exists(save_path):
            if self.accelerator.is_main_process:
                cached_indices, _ = load_cached_selection(save_path)
            else:
                cached_indices = None
            obj = [cached_indices]
            if dist.is_available() and dist.is_initialized():
                dist.broadcast_object_list(obj, src=0)
            return obj[0] or []

        # Gradient cache paths (with hash)
        grad_save_dir = self._make_grad_cache_path(step_id)
        grads_path = os.path.join(grad_save_dir, "all_projected_grads.pt")
        lengths_path = os.path.join(grad_save_dir, "all_token_lengths.pt")

        # Step 1: Compute gradients (all ranks)
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

            U_r, eigenvalues, rank = self._estimate_eigenspace(grads, lengths)
            projections, scores = self._project_to_eigenspace(grads, lengths, U_r, eigenvalues)

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

            # Compute final logdet diagnostic (slogdet for numerical stability)
            A_diag = self.logdet_eps * torch.eye(rank)
            for idx in selected_indices:
                x = projections[idx]
                A_diag += torch.outer(x, x)
            sign, logabsdet = torch.linalg.slogdet(A_diag.float())
            final_logdet = float(logabsdet.item()) if sign > 0 else float("nan")

            metric_payload = {
                "selection_method": self.selection_method,
                "gradient_type": self.gradient_type,
                "whitening_beta": self.whitening_beta,
                "rank_used": rank,
                "rank_method": self.rank_method,
                "num_samples_requested": num_samples,
                "num_samples_selected": len(selected_indices),
                "dataset_size": len(self.dataset),
                "eigenvalues_top10": eigenvalues[:10].tolist(),
                "effective_rank": float(eigenvalues[eigenvalues > 1e-10].sum() / eigenvalues[0]) if eigenvalues[0] > 0 else 0,
                "scores_selected_mean": float(scores[torch.tensor(selected_indices)].mean()),
                "scores_all_mean": float(scores.mean()),
                "final_logdet": final_logdet,
                "length_norm_alpha": self.length_norm_alpha,
                "clipping_method": self.clipping_method,
                "prefilter_ratio": self.prefilter_ratio,
                "logdet_eps": self.logdet_eps,
            }
            save_selection(save_path, selected_indices, metric_payload, self.accelerator)
            logger.info(
                f"[OptGCS] Selected {len(selected_indices)} samples. "
                f"Rank={rank}, β={self.whitening_beta}, "
                f"logdet={final_logdet:.2f}, "
                f"score_selected={metric_payload['scores_selected_mean']:.4f}, "
                f"score_all={metric_payload['scores_all_mean']:.4f}"
            )
        else:
            selected_indices = None

        obj = [selected_indices]
        if dist.is_available() and dist.is_initialized():
            dist.broadcast_object_list(obj, src=0)
        return obj[0] or []

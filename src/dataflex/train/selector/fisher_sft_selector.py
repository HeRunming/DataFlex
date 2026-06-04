"""
FisherSFT Baseline: Last-Layer Fisher Information LogDet Selection.

This implements the FisherSFT approach (Deb et al., 2025) as a critical baseline.
It uses only the last-layer pre-logit embeddings to approximate Fisher Information,
then selects samples via logdet greedy maximization.

Key difference from OptGCS:
- Only uses last-layer embeddings (forward-only, no backward pass needed for selection)
- Does NOT use optimizer state or multi-layer gradient information
- Much cheaper to compute but potentially less informative

Reference: Deb et al., "FisherSFT: Data-Efficient Supervised Fine-Tuning via
Fisher Information", PMLR 267, 2025.
"""

import os
import glob
from typing import List, Optional

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from dataflex.core.registry import register_selector
from dataflex.utils.logging import logger
from dataflex.utils.selector_io import load_cached_selection, save_selection
from .base_selector import Selector


class IndexedDatasetForward(Dataset):
    """Wraps a dataset to return (index, sample) pairs."""
    def __init__(self, original_dataset):
        self.original_dataset = original_dataset

    def __len__(self):
        return len(self.original_dataset)

    def __getitem__(self, index):
        return index, self.original_dataset[index]


@register_selector('fisher_sft')
class FisherSFTSelector(Selector):
    """
    FisherSFT: Last-Layer Fisher Information LogDet Selection.

    Approximates SFT Fisher Information using last-layer pre-logit embeddings,
    then selects a logdet-diverse subset. Forward-only — no backward pass needed.
    """

    def __init__(
        self,
        dataset,
        accelerator,
        data_collator,
        cache_dir: str,
        seed: int = 42,
        logdet_eps: float = 1e-3,
        prefilter_ratio: float = 5.0,
        embedding_batch_size: int = 4,
        # Not used but may be passed by trainer
        eval_dataset=None,
        **kwargs,
    ):
        super().__init__(dataset, accelerator, data_collator, cache_dir)
        self.seed = seed
        self.logdet_eps = logdet_eps
        self.prefilter_ratio = prefilter_ratio
        self.embedding_batch_size = embedding_batch_size

        self.device = self.accelerator.device
        os.makedirs(self.cache_dir, exist_ok=True)
        logger.info(f"[FisherSFT] Initialized: eps={logdet_eps}, prefilter={prefilter_ratio}")

    def _extract_last_layer_embeddings(self, model, save_dir: str):
        """
        Extract last hidden state embeddings (averaged over completion tokens).
        These serve as the Fisher Information features for logdet selection.
        """
        indexed_dataset = IndexedDatasetForward(self.dataset)

        def collator(features):
            indices = [f[0] for f in features]
            original_data = [f[1] for f in features]
            collated_batch = self.data_collator(original_data)
            return {'indices': torch.tensor(indices), 'batch': collated_batch}

        dataloader = DataLoader(
            indexed_dataset,
            batch_size=self.embedding_batch_size,
            shuffle=False,
            num_workers=2,
            collate_fn=collator,
        )
        dataloader = self.accelerator.prepare(dataloader)

        was_training = model.training
        model.eval()

        all_embeddings = []
        all_indices = []

        with torch.no_grad():
            for data in tqdm(
                dataloader,
                desc=f"[Rank {self.accelerator.process_index}] FisherSFT Embeddings",
                disable=not self.accelerator.is_local_main_process,
                dynamic_ncols=True,
            ):
                indices = data['indices']
                batch = data['batch']

                # Forward pass to get hidden states
                outputs = model(**batch, output_hidden_states=True, return_dict=True)
                hidden = outputs.hidden_states[-1]  # [B, seq_len, hidden_dim]

                # Average over completion tokens (where labels != -100)
                labels = batch.get('labels', None)
                if labels is not None:
                    mask = (labels != -100).float().unsqueeze(-1)  # [B, seq_len, 1]
                else:
                    attention_mask = batch.get('attention_mask', None)
                    mask = attention_mask.float().unsqueeze(-1) if attention_mask is not None else torch.ones_like(hidden[:, :, :1])

                # Masked mean pooling
                pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)  # [B, hidden_dim]

                all_embeddings.append(pooled.float().cpu())
                all_indices.append(indices.cpu())

        if was_training:
            model.train()

        all_embeddings = torch.cat(all_embeddings, dim=0)
        all_indices = torch.cat(all_indices, dim=0)

        # Reorder by index
        ordered = torch.zeros(len(self.dataset), all_embeddings.shape[1], dtype=torch.float32)
        ordered[all_indices] = all_embeddings

        # L2 normalize
        norms = ordered.norm(dim=1, keepdim=True).clamp(min=1e-12)
        ordered = ordered / norms

        # Save
        output_path = os.path.join(save_dir, "fisher_embeddings.pt")
        torch.save(ordered, output_path)
        logger.info(f"[FisherSFT] Saved embeddings: {ordered.shape} to {output_path}")

        self.accelerator.wait_for_everyone()

    def _select_by_logdet(self, embeddings: torch.Tensor, num_samples: int) -> List[int]:
        """Greedy logdet selection in embedding space (GPU-accelerated)."""
        n, d = embeddings.shape
        k = min(num_samples, n)
        eps = self.logdet_eps

        # Run the greedy loop on GPU when available — the CPU version is ~100x
        # slower and overruns the distributed watchdog on 13K+ selections.
        dev = self.device if torch.cuda.is_available() else torch.device("cpu")
        embeddings = embeddings.to(dev)

        # For large d, project to lower dimension first
        if d > 512:
            # Random projection to 512 dims
            gen = torch.Generator(device=dev).manual_seed(self.seed)
            R = torch.randn(d, 512, generator=gen, device=dev) / (512 ** 0.5)
            X = embeddings @ R
        else:
            X = embeddings

        r = X.shape[1]
        A_inv = torch.eye(r, dtype=X.dtype, device=dev) / eps

        selected = []
        available = torch.ones(n, dtype=torch.bool, device=dev)

        # Prefilter by norm (proxy for Fisher information magnitude)
        scores = (X ** 2).sum(dim=1)
        prefilter_k = min(int(self.prefilter_ratio * k), n)
        if prefilter_k < n:
            topk = torch.topk(scores, k=prefilter_k, largest=True)
            candidate_mask = torch.zeros(n, dtype=torch.bool, device=dev)
            candidate_mask[topk.indices] = True
            available = available & candidate_mask

        neg_inf = torch.tensor(-float('inf'), device=dev, dtype=X.dtype)
        for t in range(k):
            gains = (X @ A_inv * X).sum(dim=1)
            gains = torch.where(available, gains, neg_inf)

            best = gains.argmax().item()
            selected.append(best)
            available[best] = False

            x = X[best]
            Ax = A_inv @ x
            denom = 1.0 + x @ Ax
            A_inv -= torch.outer(Ax, Ax) / denom

            if (t + 1) % 1000 == 0:
                logger.info(f"[FisherSFT] Selected {t+1}/{k}")

        return selected

    def select(self, model, step_id: int, num_samples: int, **kwargs) -> List[int]:
        """Select via last-layer Fisher Information logdet maximization."""
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

        # Extract embeddings
        embed_dir = os.path.join(self.cache_dir, "embeddings", str(step_id))
        embed_path = os.path.join(embed_dir, "fisher_embeddings.pt")

        if not os.path.exists(embed_path):
            os.makedirs(embed_dir, exist_ok=True)
            self._extract_last_layer_embeddings(model, embed_dir)

        self.accelerator.wait_for_everyone()

        # Selection (main process)
        if self.accelerator.is_main_process:
            embeddings = torch.load(embed_path, map_location="cpu")
            logger.info(f"[FisherSFT] Loaded embeddings: {embeddings.shape}")

            selected_indices = self._select_by_logdet(embeddings, num_samples)

            metric_payload = {
                "selection_method": "fisher_sft_logdet",
                "embedding_dim": embeddings.shape[1],
                "num_selected": len(selected_indices),
            }
            save_selection(save_path, selected_indices, metric_payload, self.accelerator)
            logger.info(f"[FisherSFT] Selected {len(selected_indices)} samples.")
        else:
            selected_indices = None

        obj = [selected_indices]
        if dist.is_available() and dist.is_initialized():
            dist.broadcast_object_list(obj, src=0)
        selected_indices = obj[0] or []

        return selected_indices

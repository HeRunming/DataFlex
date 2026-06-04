from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Dict
import torch
from torch import distributed as dist


class Selector(ABC):
    def __init__(self, dataset, accelerator, data_collator, cache_dir):
        self.dataset = dataset
        self.accelerator = accelerator
        self.data_collator = data_collator
        self.cache_dir = cache_dir
        self.seed = 42

    def warmup(self, num_samples: int, replacement: bool) -> List[List[int]]:
        if self.accelerator.is_main_process:
            dataset_size = len(self.dataset)
            gen = torch.Generator()
            gen.manual_seed(self.seed)

            if replacement:
                full_indices = torch.randint(
                    low=0, high=dataset_size, size=(num_samples,), generator=gen
                ).tolist()
            else:
                if num_samples > dataset_size:
                    raise ValueError(
                        f"Cannot sample {num_samples} without replacement from {dataset_size} samples"
                    )
                full_indices = torch.randperm(dataset_size, generator=gen)[:num_samples].tolist()
        else:
            full_indices = None

        obj = [full_indices]
        if dist.is_available() and dist.is_initialized():
            dist.broadcast_object_list(obj, src=0)
            full_indices = obj[0]
        else:
            full_indices = full_indices or []

        return full_indices

    # =========================================================================
    # Shared Adam-preconditioning utility (LESS-paper aligned)
    # =========================================================================
    # Used by every gradient-based selector (LessSelector, OptGCSSelector,
    # RandomSubspaceLogDetSelector via OptGCS, GradNormTopKSelector via OptGCS)
    # so that cross-method comparisons run on identical preconditioned gradients.
    # Math: g' = ((1-β₁)·g + β₁·m) / sqrt(β₂·v + (1-β₂)·g² + ε)
    # =========================================================================

    @staticmethod
    def gather_optimizer_state(model, accelerator,
                                optimizer_state: Optional[Dict] = None
                                ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Concatenate Adam (exp_avg, exp_avg_sq) across trainable params into 1D tensors.

        Compatible with both DeepSpeed ZeRO-3 (uses safe_get_full_optimizer_state)
        and the standard PyTorch optimizer.state dict path.

        Returns (None, None) if no optimizer state can be recovered (caller should
        decide whether to fall back to raw gradients or raise).
        """
        avg_list, avg_sq_list = [], []
        device = accelerator.device

        if accelerator.state.deepspeed_plugin is not None:
            from deepspeed.utils import safe_get_full_optimizer_state
            for param in model.parameters():
                if not param.requires_grad:
                    continue
                exp_avg = safe_get_full_optimizer_state(param, "exp_avg")
                exp_avg_sq = safe_get_full_optimizer_state(param, "exp_avg_sq")
                if exp_avg is None or exp_avg_sq is None:
                    return None, None  # warmup not far enough to have state
                avg_list.append(exp_avg.view(-1))
                avg_sq_list.append(exp_avg_sq.view(-1))
        else:
            if optimizer_state is None:
                return None, None
            for param in model.parameters():
                if not param.requires_grad:
                    continue
                state = optimizer_state.get(param)
                if state is None or "exp_avg" not in state or "exp_avg_sq" not in state:
                    return None, None
                avg_list.append(state["exp_avg"].view(-1))
                avg_sq_list.append(state["exp_avg_sq"].view(-1))

        if not avg_list:
            return None, None
        avg = torch.cat(avg_list).to(device)
        avg_sq = torch.cat(avg_sq_list).to(device)
        return avg, avg_sq

    @staticmethod
    def adam_precondition_grads(vectorized_grads: torch.Tensor,
                                  m: torch.Tensor, v: torch.Tensor,
                                  beta1: float = 0.9, beta2: float = 0.999,
                                  eps: float = 1e-8,
                                  clip_value: Optional[float] = 1e4
                                  ) -> torch.Tensor:
        """Apply Adam preconditioning to a 1D gradient vector.

        Matches the LESS-paper formula:
            g' = ((1-β₁)·g + β₁·m) / sqrt(β₂·v + (1-β₂)·g² + ε)

        Args:
            vectorized_grads: shape [num_params], the raw gradient (mutated in-place).
            m, v: shape [num_params], Adam first/second moment estimates.
            clip_value: if not None, clamp output to ±clip_value (used by OptGCS to
                avoid blow-ups when v is tiny; LESS skips clamping but the difference
                is numerically negligible — we apply it consistently for stability).

        Returns the preconditioned gradient (same tensor, modified in-place).
        """
        denom = v.mul(beta2)
        denom.addcmul_(vectorized_grads, vectorized_grads, value=(1 - beta2))
        denom.clamp_(min=1e-16).sqrt_().add_(eps)
        vectorized_grads.mul_(1 - beta1).add_(m, alpha=beta1)
        vectorized_grads.div_(denom)
        if clip_value is not None:
            vectorized_grads = vectorized_grads.clamp(-clip_value, clip_value)
        del denom
        return vectorized_grads

    @abstractmethod
    def select(self, model, step_id: int, num_samples: int, **kwargs):
        """
        Select samples from the dataset for the model in 'step_id'.

        Args:
            model: The model object used in the selection process.
            step_id (int): The ID of the current training step or stage.
            num_samples (int): The number of samples to select.
            **kwargs: Additional keyword arguments, allowing for flexible expansion by subclasses.

        Returns:
            List[int]: A list of the selected sample indices.
        """
        pass

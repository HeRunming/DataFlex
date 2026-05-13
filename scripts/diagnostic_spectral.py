#!/usr/bin/env python3
"""
Spec-GCS Diagnostic Script: Verify Spectral Structure of SFT Gradients.

This script runs a quick diagnostic to verify that the core assumption of
Spec-GCS holds: SFT per-sample gradients exhibit low effective-rank structure
(spiked covariance / power-law eigenvalue decay).

Usage:
    python scripts/diagnostic_spectral.py \
        --model_path /jizhicfs/karonhe/models/LLM-Research/Meta-Llama-3___1-8B \
        --data_path /jizhicfs/karonhe/DataFlex/data/Openhermes_train.json \
        --num_samples 5000 \
        --proj_dim 4096 \
        --output_dir /jizhicfs/karonhe/dataflex_saves/diagnostic

Outputs:
    - Eigenvalue decay plot (PNG)
    - Effective rank statistics
    - Eigenspace stability analysis
    - Score vs token-length correlation
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType
from torch.utils.data import DataLoader, Dataset


class SimpleAlpacaDataset(Dataset):
    """Minimal dataset for diagnostic purposes."""
    def __init__(self, data, tokenizer, max_length=2048):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        instruction = item.get('instruction', '')
        input_text = item.get('input', '')
        output_text = item.get('output', '')

        if input_text:
            prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n"
        else:
            prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"

        full_text = prompt + output_text
        encoding = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt',
        )

        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)

        # Create labels (mask prompt tokens)
        prompt_encoding = self.tokenizer(prompt, truncation=True, max_length=self.max_length)
        prompt_len = len(prompt_encoding['input_ids'])
        labels = input_ids.clone()
        labels[:prompt_len] = -100
        labels[attention_mask == 0] = -100

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'token_length': (labels != -100).sum().item(),
        }


def compute_gradients(model, dataset, num_samples, proj_dim=4096, device='cuda'):
    """Compute per-sample gradients with random projection."""
    from trak.projectors import BasicProjector, ProjectionType

    model.eval()  # We still compute gradients, but no dropout
    model.to(device)

    # Count trainable parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable parameters: {num_params:,}")

    # Setup projector
    projector = BasicProjector(
        grad_dim=num_params,
        proj_dim=proj_dim,
        seed=42,
        proj_type=ProjectionType.rademacher,
        max_batch_size=8,
        block_size=128,
        device=device,
        dtype=torch.float16,
    )

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    projected_grads = []
    token_lengths = []
    count = 0

    print(f"  Computing gradients for {num_samples} samples...")
    for batch in dataloader:
        if count >= num_samples:
            break

        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        token_len = batch['token_length'].item()

        model.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        outputs.loss.backward()

        # Vectorize gradients
        grad_vec = torch.cat([
            p.grad.view(-1) for p in model.parameters() if p.grad is not None
        ]).unsqueeze(0).to(torch.float16)

        # Project
        projected = projector.project(grad_vec, model_id=0).cpu().float()
        projected_grads.append(projected.squeeze(0))
        token_lengths.append(token_len)

        model.zero_grad()
        count += 1

        if count % 500 == 0:
            print(f"    Processed {count}/{num_samples}")

    grads = torch.stack(projected_grads)  # [n, proj_dim]
    lengths = torch.tensor(token_lengths)
    return grads, lengths


def analyze_spectrum(grads, lengths, alpha=0.5, output_dir='.'):
    """Perform spectral analysis and generate diagnostic plots."""
    n, d = grads.shape
    print(f"\n=== Spectral Analysis ===")
    print(f"  Data matrix: {n} samples x {d} dimensions")

    # Length normalization
    length_factors = lengths.float().pow(alpha).unsqueeze(1)
    h = grads / length_factors
    norms = h.norm(dim=1, keepdim=True).clamp(min=1e-12)
    h = h / norms

    # Remove any samples with NaN/Inf
    valid_mask = torch.isfinite(h).all(dim=1)
    if not valid_mask.all():
        n_invalid = (~valid_mask).sum().item()
        print(f"  WARNING: Removing {n_invalid} samples with NaN/Inf gradients")
        h = h[valid_mask]
        lengths = lengths[valid_mask]
        n = h.shape[0]
        print(f"  Remaining: {n} samples")

    # Covariance estimation & Eigendecomposition
    print("  Computing eigendecomposition...")
    if n >= d:
        # Standard: compute d×d covariance then eigendecompose
        cov = (h.T @ h) / n  # [d, d]
        eigenvalues, eigenvectors = torch.linalg.eigh(cov)
        eigenvalues = eigenvalues.flip(0).numpy()
        eigenvectors = eigenvectors.flip(1)
    else:
        # n < d: use SVD of h (more numerically stable, avoids rank-deficient cov)
        U_svd, S_svd, Vt_svd = torch.linalg.svd(h, full_matrices=False)
        eigenvalues = (S_svd ** 2 / n).numpy()
        eigenvectors = Vt_svd.T  # [d, min(n,d)]

    # Only keep positive eigenvalues
    pos_mask = eigenvalues > 1e-10
    eigenvalues = eigenvalues[pos_mask]

    if len(eigenvalues) == 0:
        print("  ERROR: No positive eigenvalues found!")
        return {}

    # Statistics
    effective_rank = eigenvalues.sum() / eigenvalues[0]
    entropy_p = eigenvalues / eigenvalues.sum()
    entropy = -np.sum(entropy_p * np.log(entropy_p + 1e-12))
    entropy_rank = np.exp(entropy)

    # Eigengap analysis
    ratios = eigenvalues[:-1] / eigenvalues[1:]
    max_gap_idx = np.argmax(ratios[:100])  # look within top-100
    max_gap_ratio = ratios[max_gap_idx]

    print(f"\n  Results:")
    print(f"    Top-10 eigenvalues: {eigenvalues[:10]}")
    print(f"    Effective rank (trace/max): {effective_rank:.1f}")
    print(f"    Entropy rank: {entropy_rank:.1f}")
    print(f"    Max eigengap at index {max_gap_idx}: ratio = {max_gap_ratio:.2f}")
    print(f"    Top-1 eigenvalue explains: {eigenvalues[0]/eigenvalues.sum()*100:.1f}%")
    print(f"    Top-10 explain: {eigenvalues[:10].sum()/eigenvalues.sum()*100:.1f}%")
    print(f"    Top-50 explain: {eigenvalues[:50].sum()/eigenvalues.sum()*100:.1f}%")
    print(f"    Top-100 explain: {eigenvalues[:100].sum()/eigenvalues.sum()*100:.1f}%")

    # === Plot 1: Eigenvalue Decay ===
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Log-log plot
    axes[0, 0].loglog(range(1, len(eigenvalues)+1), eigenvalues)
    axes[0, 0].set_xlabel('Index')
    axes[0, 0].set_ylabel('Eigenvalue')
    axes[0, 0].set_title(f'Eigenvalue Decay (log-log)\nEffective Rank = {effective_rank:.1f}')
    axes[0, 0].axvline(x=effective_rank, color='r', linestyle='--', label=f'Eff. Rank={effective_rank:.0f}')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Cumulative explained variance
    cumvar = np.cumsum(eigenvalues) / eigenvalues.sum()
    axes[0, 1].plot(range(1, len(cumvar)+1), cumvar)
    axes[0, 1].set_xlabel('Number of Components')
    axes[0, 1].set_ylabel('Cumulative Explained Variance')
    axes[0, 1].set_title('Cumulative Variance Explained')
    axes[0, 1].axhline(y=0.9, color='r', linestyle='--', label='90%')
    axes[0, 1].axhline(y=0.95, color='orange', linestyle='--', label='95%')
    # Find 90% and 95% points
    r90 = np.searchsorted(cumvar, 0.9) + 1
    r95 = np.searchsorted(cumvar, 0.95) + 1
    axes[0, 1].axvline(x=r90, color='r', linestyle=':', alpha=0.5)
    axes[0, 1].axvline(x=r95, color='orange', linestyle=':', alpha=0.5)
    axes[0, 1].set_xlim(0, min(500, len(cumvar)))
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Eigengap ratio
    axes[1, 0].plot(range(1, min(101, len(ratios)+1)), ratios[:100])
    axes[1, 0].set_xlabel('Index')
    axes[1, 0].set_ylabel('λ_j / λ_{j+1}')
    axes[1, 0].set_title(f'Eigengap Ratios (top-100)\nMax gap at index {max_gap_idx}: {max_gap_ratio:.2f}')
    axes[1, 0].axhline(y=2.0, color='r', linestyle='--', label='Threshold=2.0')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Score vs token length
    scores = (h @ eigenvectors[:, :int(max(10, effective_rank))]).pow(2).sum(dim=1).numpy()
    token_lens = lengths.numpy()
    axes[1, 1].scatter(token_lens, scores, alpha=0.3, s=5)
    axes[1, 1].set_xlabel('Token Length')
    axes[1, 1].set_ylabel('Projection Score')
    axes[1, 1].set_title(f'Score vs Token Length (α={alpha})\nCorr={np.corrcoef(token_lens, scores)[0,1]:.3f}')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'spectral_diagnostic.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Saved diagnostic plot to: {plot_path}")

    # Save numeric results
    results = {
        'effective_rank': float(effective_rank),
        'entropy_rank': float(entropy_rank),
        'max_eigengap_idx': int(max_gap_idx),
        'max_eigengap_ratio': float(max_gap_ratio),
        'top10_eigenvalues': eigenvalues[:10].tolist(),
        'top10_explained_pct': float(eigenvalues[:10].sum()/eigenvalues.sum()*100),
        'top50_explained_pct': float(eigenvalues[:50].sum()/eigenvalues.sum()*100),
        'top100_explained_pct': float(eigenvalues[:100].sum()/eigenvalues.sum()*100),
        'r90': int(r90),
        'r95': int(r95),
        'n_samples': int(n),
        'proj_dim': int(d),
        'length_norm_alpha': float(alpha),
        'score_length_correlation': float(np.corrcoef(token_lens, scores)[0, 1]),
    }
    results_path = os.path.join(output_dir, 'spectral_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  Saved results to: {results_path}")

    # Verdict
    print(f"\n=== VERDICT ===")
    if effective_rank < d * 0.1 and eigenvalues[:50].sum()/eigenvalues.sum() > 0.5:
        print("  ✅ STRONG spectral structure detected!")
        print("  The gradient covariance is clearly low-rank. Spec-GCS should work well.")
    elif effective_rank < d * 0.3:
        print("  ⚠️  MODERATE spectral structure detected.")
        print("  Some low-rank structure exists. Spec-GCS may work with tuning.")
    else:
        print("  ❌ WEAK spectral structure.")
        print("  Gradients appear nearly isotropic. Consider different gradient representation.")

    return results


def main():
    parser = argparse.ArgumentParser(description='Spec-GCS Spectral Diagnostic')
    parser.add_argument('--model_path', type=str,
                        default='/jizhicfs/karonhe/models/LLM-Research/Meta-Llama-3___1-8B')
    parser.add_argument('--data_path', type=str,
                        default='/jizhicfs/karonhe/DataFlex/data/Openhermes_train.json')
    parser.add_argument('--num_samples', type=int, default=5000)
    parser.add_argument('--proj_dim', type=int, default=4096)
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--lora_rank', type=int, default=16)
    parser.add_argument('--max_length', type=int, default=2048)
    parser.add_argument('--output_dir', type=str,
                        default='/jizhicfs/karonhe/dataflex_saves/diagnostic')
    parser.add_argument('--device', type=str, default='cuda:0')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("Spec-GCS Spectral Diagnostic")
    print("=" * 60)
    print(f"Model: {args.model_path}")
    print(f"Data: {args.data_path}")
    print(f"Samples: {args.num_samples}")
    print(f"Proj dim: {args.proj_dim}")
    print(f"Alpha: {args.alpha}")
    print(f"Device: {args.device}")
    print()

    # Load model
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    # Apply LoRA (match DataFlex experiment setup)
    # Detect target modules based on model architecture
    from peft.utils import TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING
    model_type = getattr(model.config, 'model_type', 'llama')
    if model_type in TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING:
        target_modules = TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING[model_type]
    else:
        # Fallback: find all linear layers
        target_modules = []
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                short_name = name.split('.')[-1]
                if short_name not in target_modules:
                    target_modules.append(short_name)
        target_modules = list(set(target_modules))[:10]  # limit

    print(f"  Model type: {model_type}, LoRA targets: {target_modules}")

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_rank,
        lora_alpha=args.lora_rank // 2,
        target_modules=target_modules,
        lora_dropout=0.0,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Load data
    print(f"\nLoading data from {args.data_path}...")
    with open(args.data_path, 'r') as f:
        all_data = json.load(f)

    # Subsample
    import random
    random.seed(42)
    if args.num_samples < len(all_data):
        sampled_data = random.sample(all_data, args.num_samples)
    else:
        sampled_data = all_data
    print(f"  Using {len(sampled_data)} samples")

    dataset = SimpleAlpacaDataset(sampled_data, tokenizer, max_length=args.max_length)

    # Compute gradients
    print("\nComputing projected gradients...")
    t0 = time.time()
    grads, lengths = compute_gradients(
        model, dataset, args.num_samples,
        proj_dim=args.proj_dim, device=args.device
    )
    t1 = time.time()
    print(f"  Gradient computation took {t1-t0:.1f}s ({(t1-t0)/args.num_samples*1000:.1f}ms/sample)")

    # Save raw gradients for later analysis
    torch.save({'grads': grads, 'lengths': lengths},
               os.path.join(args.output_dir, 'diagnostic_grads.pt'))

    # Spectral analysis
    results = analyze_spectrum(grads, lengths, alpha=args.alpha, output_dir=args.output_dir)

    print(f"\n{'='*60}")
    print("Diagnostic complete!")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()

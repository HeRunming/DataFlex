# LESS Selector - Quick Reference & Key Equations

## Core Mathematics

### 1. Gradient Extraction & Vectorization
```
For each sample s with loss L(s):
  g_s ∈ ℝ^m  where m = num_params
  
For LoRA models: m_LoRA ≈ 0.5-2% of m_full
```

### 2. Adam Preconditioning
```
Given:
  g_t: gradient at step t
  m_t: first moment estimate (exp_avg)
  v_t: second moment estimate (exp_avg_sq)
  β₁ = 0.9, β₂ = 0.999, ε = 1e-8

Apply:
  denominator = √(β₂·v_t + (1-β₂)·g_t²) + ε
  g_precond = g_t / denominator
  
Alternative form:
  g_precond ≈ g_raw·(1-β₁) + m_t·β₁ / √(v_t + ε)
```

**Effect**: Gradient components with high variance (large v_t) get smaller; 
reliable directions get emphasized.

### 3. Random Projection (TRAK Rademacher)
```
R ∈ ℝ^(m × d)  where:
  - Each entry is uniformly ±1
  - Generated from seed=42 (deterministic)
  - d = proj_dim (default 8192)

Projected gradient:
  sketch_s = g_s @ R ∈ ℝ^d
  
Effect: Dimension reduction m → d
  - m = 4.2M (LoRA-7B) → d = 8192
  - Compression factor: ~500×
  - Johnson-Lindenstrauss guarantee: preserves distances
```

### 4. L2 Normalization
```
For each sample's sketch:
  norm_s = ||sketch_s||₂
  
  sketch_s_normalized = sketch_s / max(norm_s, 1e-12)
  
Result:
  All sketches lie on unit hypersphere in ℝ^d
  ∀s: ||sketch_s_normalized||₂ = 1
```

### 5. Train-Eval Similarity Scoring
```
Train sketches: T ∈ ℝ^(n_train × d)  (all normalized)
Eval sketches:  E ∈ ℝ^(n_eval × d)   (all normalized)

Similarity matrix:
  S = T @ E^T ∈ ℝ^(n_train × n_eval)
  S[i,j] = T[i] · E[j] = cos(angle) ∈ [-1, 1]
  
Per-sample score:
  score_i = mean(S[i, :]) = (1/n_eval) Σ_j S[i,j]
  
Interpretation:
  score_i ∈ [-1, 1]
  High score → training sample i aligns with eval
  Low score  → training sample i misaligned with eval
```

### 6. Top-K Selection
```
scores ∈ ℝ^(n_train)

selected_indices = arg_topk(scores, k=num_samples, largest=True)
                 = indices of k largest scores
                 
Result: List of k sample indices to include in next training round
```

---

## Implementation Details

### File Locations & Key Functions

| Task | File | Function | Lines |
|------|------|----------|-------|
| Gradient extraction | `less_selector.py` | `_obtain_gradients()` | 100-141 |
| Adam state prep | `less_selector.py` | `_prepare_optimizer_state()` | 71-98 |
| Projector selection | `less_selector.py` | `_get_trak_projector()` | 143-156 |
| Gradient collection | `less_selector.py` | `_collect_and_save_projected_gradients()` | 179-273 |
| Merging & normalization | `less_selector.py` | `_merge_and_normalize_info()` | 277-313 |
| Scoring & selection | `less_selector.py` | `select()` | 315-388 |
| Trainer integration | `select_trainer.py` | `_inner_training_loop()` | 330-940 |
| Caching | `selector_io.py` | `save_selection()`, `load_cached_selection()` | 12-43 |

### Parameter Tuning Guide

| Parameter | Default | Range | Effect |
|-----------|---------|-------|--------|
| `proj_dim` | 8192 | 1024-16384 | Higher = more accuracy but slower |
| `save_interval` | 16 | 4-64 | Higher = fewer disk I/O, more memory |
| `gradient_type` | "adam" | {"adam", "sgd"} | adam = preconditioning applied |
| `seed` | 42 | any int | Ensures deterministic projection |
| `warmup_step` | - | 0-∞ | Steps before first selection |
| `update_step` | - | 1-∞ | Steps between selections |
| `update_times` | 1 | 1-∞ | Total selection rounds |

### Memory Usage Estimation

**Per-GPU during gradient computation** (LLaMA-7B + LoRA-8):
```
Component                    Memory
──────────────────────────
grad_buffer                  268 MB    (16 samples × 4.2M params)
m, v (Adam states)            34 MB    (2 × 4.2M params)
vectorized_grads              17 MB    (4.2M params)
Rademacher matrix (implicit)   0 MB    (seeded, not stored)
──────────────────────────
Peak GPU:                    ~350 MB
```

**Disk storage per selection round** (10K training samples):
```
File                    Size
─────────────────────────────
grads-*-rank*.pt      ~320 MB    (10K samples × 8192 dims × 4 bytes)
all_projected_grads.pt ~320 MB    (merged & normalized)
step_*.json            < 1 MB    (selected indices + scores)
─────────────────────────────
Total:                ~650 MB
```

**Per-sample gradient computation**:
```
Operation         Time      Notes
──────────────────────────────────
Forward/Backward  10-50 ms  Model-dependent
Adam preconditioning < 1 ms  Vectorized ops
Projection         1-5 ms   GPU-accelerated
Save               < 1 ms   Buffered I/O
──────────────────────────────
Total per sample: ~15-60 ms
```

For 10,000 samples: **2.5-10 minutes** per selection round

---

## Distributed Training Details

### Synchronization Points

```python
# Phase 1: After local gradient computation
self.accelerator.wait_for_everyone()  # All ranks finish saving

# Phase 2: Before main process merges files
# (implicitly safe because only rank-0 accesses files)

# Phase 3: After selection computation
dist.broadcast_object_list([selected_indices], src=0)
```

### Data Distribution Across Ranks

```
Total dataset: 10,000 samples
4 GPUs:

Rank 0: samples [0-2500)    → computes gradients, saves grads-2500-rank0.pt
Rank 1: samples [2500-5000) → computes gradients, saves grads-5000-rank1.pt
Rank 2: samples [5000-7500) → computes gradients, saves grads-7500-rank2.pt
Rank 3: samples [7500-10000)→ computes gradients, saves grads-9999-rank3.pt

Main process merges by index:
  final_grads[0-2500] ← grads from rank0 file
  final_grads[2500-5000] ← grads from rank1 file
  ... (same order preserved)
```

### Broadcasting Pattern

```python
# Main process (rank 0) computes selection
selected_indices = [453, 127, 8932, ...]  # On rank 0 only

# Broadcast to all ranks
obj_list = [selected_indices]
dist.broadcast_object_list(obj_list, src=0)

# Now all ranks have identical selected_indices
# Each constructs same Subset(dataset, selected_indices)
```

---

## Cache & Resumption

### Cache File Organization
```
cache_dir/
├── step_1000.json
│   {
│     "indices": [i1, i2, ...],
│     "metric": {"train_eval_similarity": [s1, s2, ...]}
│   }
├── step_2000.json
├── train/
│   ├── 1000/
│   │   ├── grads-2500-rank0.pt
│   │   ├── grads-5000-rank1.pt
│   │   ├── grads-7500-rank2.pt
│   │   ├── grads-9999-rank3.pt
│   │   └── all_projected_grads.pt  [merged & normalized]
│   └── 2000/
│       └── all_projected_grads.pt
└── eval/
    ├── 1000/
    │   └── all_projected_grads.pt
    └── 2000/
        └── all_projected_grads.pt
```

### Resumption Check
```python
# At start of selector.select():
save_path = os.path.join(cache_dir, f"step_{step_id}.json")

if os.path.exists(save_path):
    # Skip all computation, load from cache
    cached_indices, _ = load_cached_selection(save_path)
    # Broadcast to all ranks
    obj_list = [cached_indices]
    dist.broadcast_object_list(obj_list, src=0)
    return cached_indices  # ← Early return, no gradients computed!
```

**Benefit**: If training is interrupted and resumed at same step, 
selection is instant (no re-computation).

---

## Common Issues & Solutions

### Issue 1: Out of Memory (GPU)
**Symptom**: CUDA OOM during gradient computation

**Causes**:
- Too many params in LoRA (large rank)
- Model too large
- save_interval too large

**Solutions**:
```python
# Reduce save_interval (more disk I/O, less GPU memory)
save_interval: 8  # from default 16

# Or reduce LoRA rank (if applicable)
lora_rank: 4  # from 8

# Or enable gradient checkpointing in trainer
gradient_checkpointing: true
```

### Issue 2: Selection Slow
**Symptom**: Selection takes >30 minutes for 10K samples

**Causes**:
- Dataset too large
- proj_dim too large
- CudaProjector not available (fallback to BasicProjector)

**Solutions**:
```python
# Reduce projection dimension
proj_dim: 4096  # from 8192 (2× faster, slight accuracy loss)

# Ensure CudaProjector is used (install fast_jl)
pip install fast_jl

# Profile to find bottleneck
import cProfile
# ... then analyze
```

### Issue 3: Cache Misses
**Symptom**: "Resuming from sample index X" appears in logs constantly

**Cause**: save_interval is small, many disk writes

**Solution**:
```python
save_interval: 32  # Increase (if GPU memory allows)
```

### Issue 4: Selection Indices Different Across Runs
**Symptom**: Different selection each time with same seed

**Causes**:
- seed not set globally
- Non-deterministic projector

**Solutions**:
```python
# In training script
import torch
import numpy as np
import random

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# In selector config
seed: 42
```

---

## Algorithm Pseudocode

```python
def less_selector(model, train_dataset, eval_dataset, num_select):
    """
    End-to-end LESS selection algorithm
    """
    # Phase 1: Compute projections for all training samples (distributed)
    train_sketches = []
    for rank in range(num_ranks):
        samples_for_rank = train_dataset.split(num_ranks)[rank]
        
        projector = get_trak_projector(grad_dim=num_params, proj_dim=8192)
        
        for sample in samples_for_rank:
            # Forward/backward
            loss = model(sample)
            grad = compute_gradient(loss, model.parameters())
            
            # Adam preconditioning
            if use_adam:
                grad = grad / (sqrt(v) + eps)
            
            # Random projection
            sketch = projector.project(grad)
            
            train_sketches.append(sketch)
    
    # Synchronize across ranks
    train_sketches = gather_all_ranks(train_sketches)  # Only on rank 0
    
    # Phase 2: Merge and normalize (rank 0 only)
    train_sketches = torch.stack(train_sketches)  # [n_train, 8192]
    train_sketches = train_sketches / ||train_sketches||_2  # Normalize rows
    
    # Phase 3: Compute eval sketches same way
    eval_sketches = compute_sketches(model, eval_dataset)  # [n_eval, 8192]
    eval_sketches = eval_sketches / ||eval_sketches||_2
    
    # Phase 4: Score and select
    similarities = train_sketches @ eval_sketches.T  # [n_train, n_eval]
    scores = similarities.mean(dim=1)               # [n_train]
    
    selected_indices = topk(scores, k=num_select)
    
    # Broadcast to all ranks
    broadcast(selected_indices)
    
    return selected_indices
```

---

## Performance Profiling

To profile selector performance:

```python
import time
import torch.profiler as profiler

# In selector.select():
with profiler.profile(
    activities=[profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.CUDA],
    record_shapes=True,
) as prof:
    # Gradient computation
    self._collect_and_save_projected_gradients(...)
    
    # Merge & normalize
    self._merge_and_normalize_info(...)
    
    # Scoring
    scores = train_grads @ eval_grads.T

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
```

---

## References & Dependencies

### Key Libraries
- `trak.projectors.BasicProjector / CudaProjector` - Random projection
- `torch.distributed` - Multi-GPU communication
- `accelerate` - Training acceleration
- `deepspeed.utils.safe_get_full_*` - DeepSpeed compatibility

### Papers
- TRAK: Attributing Model Behavior at Scale (Pruthi et al., 2023)
- Influence Functions & Gradient-based Selection
- Johnson-Lindenstrauss Lemma (random projection theory)

### Key Hyperparameters to Track
```yaml
# Logging for monitoring
log_level: "debug"  # See gradient computation details
logging_steps: 10   # Log training metrics

# Selector-specific
warmup_step: 100      # Samples before selection starts
update_step: 200      # Samples between selections
update_times: 5       # Total selections per epoch
```

---

## Troubleshooting Checklist

- [ ] Seed set globally (torch, numpy, random)
- [ ] GPU memory sufficient for proj_dim=8192
- [ ] Eval dataset not empty
- [ ] Optimizer state dict accessible
- [ ] Cache directory writable (each rank)
- [ ] Distributed training initialized (if multi-GPU)
- [ ] CudaProjector available (or BasicProjector fallback OK)
- [ ] Selection triggers at correct step intervals
- [ ] Broadcasted selection identical across ranks


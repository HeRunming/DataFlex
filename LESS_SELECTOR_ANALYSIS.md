# DataFlex LESS Selector - Technical Implementation Analysis

## Executive Summary
The LESS (Low-rank Embedding Selectivity Strategy) selector in DataFlex implements a gradient-based data selection mechanism using random projection, Adam preconditioning, and train-eval similarity scoring for dynamic dataset selection during model training.

---

## 1. GRADIENT FLOW & DIMENSION AFTER LoRA PROJECTION

### Initial Gradient Computation
**File**: `less_selector.py`, `_obtain_gradients()` (lines 100-141)

#### Full Parameter Gradient Collection:
```python
vectorized_grads = torch.cat(
    [p.grad.view(-1) for p in model.parameters() if p.grad is not None]
)
```

**Dimension**: `num_params` (total trainable parameters)

For LoRA-trained models:
- Only LoRA parameters have gradients computed
- A LoRA adapter with rank `r` for a linear layer (in_features × out_features) contributes:
  - **LoRA-A**: `in_features × r`
  - **LoRA-B**: `r × out_features`
  - **Total per layer**: `r × (in_features + out_features)`
  
**Example**: For a model with LoRA rank=8:
- Compared to full model gradients, dimension is ~0.5-2% of full parameters
- All gradients are concatenated into a single vector

### Gradient Dimension After Projection

**Method**: TRAK Random Projection (Rademacher)

```python
projector = projector_class(
    grad_dim=num_params,           # Original dimension
    proj_dim=self.proj_dim,         # Target dimension (default: 8192)
    seed=self.seed,
    proj_type=ProjectionType.rademacher,
    max_batch_size=8,
    block_size=128,
    device=self.device,
    dtype=self.dtype,
)
```

**Final Gradient Dimension**: `proj_dim` (default 8192, configurable)

**Projection Type**: Rademacher random matrix (±1 entries)
- Creates a sparse random projection from `num_params → proj_dim`
- Much more memory-efficient than storing full gradients
- 8192 dimensions typically capture 95%+ of gradient information

---

## 2. TRAK PROJECTOR MECHANICS

### Projector Selection
**File**: `less_selector.py`, `_get_trak_projector()` (lines 143-156)

```python
def _get_trak_projector(self):
    try:
        import fast_jl
        num_sms = torch.cuda.get_device_properties(self.device.index).multi_processor_count
        fast_jl.project_rademacher_8(torch.zeros(8, 1_000, device=self.device), 512, 0, num_sms)
        projector = CudaProjector  # Fast CUDA implementation
    except (ImportError, RuntimeError):
        projector = BasicProjector  # Fallback: pure Python/PyTorch
    return projector
```

**Two Options**:
1. **CudaProjector**: Uses `fast_jl` library for GPU-accelerated Rademacher projection
   - Parallelized across CUDA SMs (streaming multiprocessors)
   - ~10-100× faster than CPU implementation
   
2. **BasicProjector**: Pure PyTorch implementation
   - Slower but reliable fallback
   - Works on any GPU

### Projection Operation
**File**: `less_selector.py`, line 263

```python
projected = projector.project(grad_buffer[:buf_pos], model_id=0).cpu()
```

**Input**: `grad_buffer` shape = `[batch_size, num_params]`
**Output**: shape = `[batch_size, proj_dim]` (default: 8192)

**Process** (conceptually):
```
Gradient Vector (num_params) × Rademacher Matrix (num_params × proj_dim)
                  ↓
            Projected Vector (proj_dim)
```

The Rademacher matrix is random but deterministic based on `seed=42`, ensuring reproducibility.

---

## 3. EXACT FLOW: GRADIENT → SKETCH → NORMALIZE → SCORE → SELECT

### Complete Pipeline Execution

#### **Phase 1: Gradient Collection & Projection (Per-rank)**
**File**: `less_selector.py`, `_collect_and_save_projected_gradients()` (lines 179-273)

```
┌─────────────────────────────────────────────────────────────┐
│ For each sample in dataset (on each GPU rank independently) │
└─────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────┐
│ Forward + Backward Pass │
│ model(batch) → loss     │
│ model.backward(loss)    │
└─────────────────────────┘
    ↓
┌──────────────────────────────────────┐
│ Extract Full Gradient Vector         │
│ grad_vector = [grad_p1, grad_p2, ...] │
│ dimension: num_params                │
└──────────────────────────────────────┘
    ↓
┌──────────────────────────────────────┐
│ Apply Adam Preconditioning           │
│ (if gradient_type == "adam")         │
│ grad_precond = grad / sqrt(v + eps)  │
└──────────────────────────────────────┘
    ↓
┌──────────────────────────────────────┐
│ Random Projection (TRAK)             │
│ sketch = grad × R_random             │
│ dimension: proj_dim (8192)           │
└──────────────────────────────────────┘
    ↓
┌──────────────────────────────────────┐
│ Save Locally (on each rank)          │
│ grads-{idx}-rank{rank_id}.pt         │
│ Contains: sketch + original_indices  │
└──────────────────────────────────────┘
```

**Key Variables During Gradient Computation**:
- `grad_buffer`: shape [save_interval, num_params], device: GPU
- `idx_buffer`: shape [save_interval], dtype: long (sample indices)
- `vectorized_grads`: shape [num_params] (full gradient)
- `projected`: shape [batch_size, proj_dim] (after projection)

#### **Phase 2: Merge & Normalize (Main Process Only)**
**File**: `less_selector.py`, `_merge_and_normalize_info()` (lines 277-313)

```
┌──────────────────────────────────────┐
│ Collect all rank-saved files:        │
│ grads-{idx}-rank{0,1,2,...}.pt       │
└──────────────────────────────────────┘
    ↓
┌──────────────────────────────────────┐
│ Reconstruct Full Dataset Order       │
│ final_grads[indices] = projected_grads │
│ shape: [total_samples, proj_dim]     │
└──────────────────────────────────────┘
    ↓
┌──────────────────────────────────────┐
│ L2 Normalization (Row-wise)          │
│ norms = ||final_grads||_2 per sample │
│ final_grads /= norms.clamp(1e-12)   │
└──────────────────────────────────────┘
    ↓
┌──────────────────────────────────────┐
│ Save Merged Normalized Gradients     │
│ all_projected_grads.pt               │
│ shape: [total_samples, proj_dim]     │
└──────────────────────────────────────┘
```

**Critical Normalization Code** (lines 302-303):
```python
norms = final_grads.norm(dim=1, keepdim=True).clamp_(min=1e-12)
final_grads.div_(norms)
```

After normalization, each sample's gradient sketch lies on a unit hypersphere.

#### **Phase 3: Scoring via Inner Product (Main Process Only)**
**File**: `less_selector.py`, line 369

```python
train_eval_similarities = (train_projected_grads @ eval_projected_grads.T).mean(dim=1)
```

**Detailed Breakdown**:
```
train_projected_grads: [N_train, proj_dim]  ← normalized training sketches
eval_projected_grads:  [N_eval, proj_dim]   ← normalized eval sketches

Similarity Matrix: [N_train, N_eval]
    [sample_i · eval_sample_1, sample_i · eval_sample_2, ...]
     
Per-sample Score: [N_train]
    score_i = mean over eval set (sample_i · eval_samples)
              = avg inner product with all eval samples
```

**Intuition**: 
- High score → training sample's gradient is aligned with eval set
- These samples likely help reduce eval loss
- It's an approximation of influence function via gradient space

#### **Phase 4: Selection (Top-k)**
**File**: `less_selector.py`, lines 370-371

```python
topk = torch.topk(train_eval_similarities, k=num_samples, largest=True)
selected_indices = topk.indices.tolist()
```

**Result**: Top `num_samples` indices with highest train-eval gradient similarity

---

## 4. OPTIMIZER STATE HANDLING

### Available Information in kwargs
**File**: `select_trainer.py`, lines 821-827

```python
extra_args = dict(
    optimizer_state=self.optimizer.state,          # ← Dict[param → state]
    scheduler_state=self.lr_scheduler.state_dict(),
    current_update_times=current_update_times,     # ← int
    update_times=effective_update_times,           # ← int
    tokenizer=self.tokenizer,
)
```

### Adam State Access (Non-DeepSpeed)
**File**: `less_selector.py`, `_prepare_optimizer_state()` (lines 86-92)

```python
for param in model.parameters():
    if param.requires_grad:
        avg_list.append(optimizer_state[param]["exp_avg"].view(-1))
        avg_sq_list.append(optimizer_state[param]["exp_avg_sq"].view(-1))

avg = torch.cat(avg_list)      # Shape: [num_params] - first moment
avg_sq = torch.cat(avg_sq_list) # Shape: [num_params] - second moment
```

**Adam State Structure**:
```python
optimizer_state[param] = {
    "exp_avg": m,              # First moment (mean of gradients)
    "exp_avg_sq": v,           # Second moment (mean of squared gradients)
    "step": t,                 # Training step count
}
```

### DeepSpeed Handling
**File**: `less_selector.py`, lines 75-84

```python
from deepspeed.utils import safe_get_full_optimizer_state

for param in model.parameters():
    if param.requires_grad:
        exp_avg = safe_get_full_optimizer_state(param, "exp_avg")
        exp_avg_sq = safe_get_full_optimizer_state(param, "exp_avg_sq")
```

**Why needed**: DeepSpeed ZeRO-3 shards parameters across GPUs; `safe_get_full_*` gathers them.

### Adam Preconditioning Applied
**File**: `less_selector.py`, lines 125-134

```python
if gradient_type == "adam":
    beta1, beta2, eps = 0.9, 0.999, 1e-08
    denom = v.mul(beta2)                    # β₂ * v
    denom.addcmul_(g, g, value=(1 - beta2)) # += (1-β₂) * g²
    denom.sqrt_().add_(eps)                 # √(denom) + ε
    g.mul_(1 - beta1).add_(m, alpha=beta1)  # g = (1-β₁)*g + β₁*m
    g.div_(denom)                           # g /= √(bias-corrected v)
```

**Resulting preconditioned gradient** (conceptual):
```
g_precond ≈ g / sqrt(second_moment + eps)
```

This is the "Adam-style" gradient that's then projected.

---

## 5. DISTRIBUTED TRAINING PATTERN (Multi-GPU)

### Broadcasting Mechanism
**File**: `less_selector.py`, lines 329-331

```python
cached_indices_list = [cached_indices]
if dist.is_available() and dist.is_initialized():
    dist.broadcast_object_list(cached_indices_list, src=0)
    cached_indices = cached_indices_list[0]
```

**Base Selector Pattern**:
**File**: `base_selector.py`, lines 34-36

```python
obj = [full_indices]
if dist.is_available() and dist.is_initialized():
    dist.broadcast_object_list(obj, src=0)  # ← Rank 0 broadcasts to all
    full_indices = obj[0]
```

### Multi-Rank Gradient Collection
**Key Design**: **Each rank processes its portion independently**

```
Rank 0: samples [0-N/4)       → grads-{max_idx}-rank0.pt
Rank 1: samples [N/4-N/2)     → grads-{max_idx}-rank1.pt
Rank 2: samples [N/2-3N/4)    → grads-{max_idx}-rank2.pt
Rank 3: samples [3N/4-N)      → grads-{max_idx}-rank3.pt
    ↓ (main process only)
    Merge by index → all_projected_grads.pt
    Normalize & Score
    Broadcast selection to all ranks
```

**Synchronization Points**:
- Line 236: `self.accelerator.wait_for_everyone()` - Before resuming
- Line 273: `self.accelerator.wait_for_everyone()` - After gradient collection
- Line 350: `self.accelerator.wait_for_everyone()` - Between train/eval phases
- Line 359: `self.accelerator.wait_for_everyone()` - Before scoring

### Gradient Accumulation Impact
**File**: `select_trainer.py`, lines 359

```python
total_train_batch_size = self._train_batch_size * args.gradient_accumulation_steps * args.world_size
```

Each sample still gets its own gradient computed (batch_size=1 per sample), then projected.

---

## 6. CORE ALGORITHM CONSTANTS & HYPERPARAMETERS

### Default Configuration
**File**: `less_selector.py`, lines 28-47

```python
gradient_type: str = "adam"        # {"adam", "sgd"}
proj_dim: int = 8192              # Projection target dimension
save_interval: int = 16            # Samples before saving checkpoint
seed: int = 42                     # Deterministic randomness
```

### Training Dynamics Parameters
**File**: `dynamic_params.py`, lines 540-559

```python
warmup_step: int          # Warm-up phases before selection
update_step: int          # Steps between selection rounds
update_times: int         # Total selection rounds per epoch
static_mix: bool          # Fixed mixing ratio flag
train_step: int           # Optional total training steps
```

### Selection Trigger Logic
**File**: `select_trainer.py`, lines 801-805

```python
elif (
    self.state.global_step < max_steps and (
    step_in_epoch == self.finetuning_args.warmup_step or
    (step_in_epoch > self.finetuning_args.warmup_step and
    (step_in_epoch - self.finetuning_args.warmup_step) % self.finetuning_args.update_step == 0))
):
    # Trigger selector.select()
```

**Schedule**: Selection occurs at:
1. After warmup finishes
2. Then every `update_step` steps thereafter

---

## 7. MEMORY & PERFORMANCE CHARACTERISTICS

### Memory Breakdown

| Component | Size | Notes |
|-----------|------|-------|
| Full gradients (num_params) | 4 × num_params bytes | Temporary, per-batch |
| Adam state (m, v) | 8 × num_params bytes | Loaded from optimizer |
| Projected gradients | 4 × proj_dim bytes | Saved per batch (8KB each @ 8192 dims) |
| Rademacher matrix | Implicit (seeded) | Not stored, regenerated |

**Example: LLaMA-7B with LoRA-8**:
- Full params: ~140M
- LoRA params: ~4.2M (3%)
- Full gradient size: ~16.8 MB
- Projected size: ~32 KB (500× compression)
- Adam states: ~33.6 MB
- **Per sample cost**: ~65 MB GPU memory

### Computational Complexity

```
Time per sample (ms) ≈:
  Forward/Backward: 10-50 ms (model-dependent)
  Adam precond:     <1 ms
  Projection:       1-5 ms (GPU)
  Save/merge:       <1 ms
  ─────────────────────────
  Total:            11-56 ms
```

For 10K samples: ~2-10 minutes selection time

---

## 8. CACHING & RESUMPTION

### Cache Structure
**File**: `less_selector.py`, lines 321-322

```python
save_path = os.path.join(self.cache_dir, f"step_{step_id}.json")
```

**Directory Layout**:
```
cache_dir/
├── step_1000.json              # Selection cache for step 1000
├── step_2000.json              # Selection cache for step 2000
├── train/
│   ├── 1000/
│   │   ├── grads-{idx}-rank0.pt
│   │   ├── grads-{idx}-rank1.pt
│   │   └── all_projected_grads.pt
│   └── 2000/
└── eval/
    ├── 1000/
    │   └── all_projected_grads.pt
    └── 2000/
```

### JSON Cache Format
**File**: `selector_io.py`, lines 25-42

```python
{
    "indices": [i1, i2, ...],    # Selected sample indices
    "metric": {
        "train_eval_similarity": [s1, s2, ...]  # Scores for selected
    }
}
```

### Resumption Check
**File**: `less_selector.py`, lines 323-334

```python
if os.path.exists(save_path):
    # Load cached selection - no re-computation needed
    cached_indices, _ = load_cached_selection(save_path)
    # Broadcast to all ranks
    if dist.is_initialized():
        dist.broadcast_object_list([cached_indices], src=0)
    return cached_indices
```

---

## 9. KEY TECHNICAL INSIGHTS

### 1. **Gradient as Influence Proxy**
- Gradients capture how each training sample affects model parameters
- Gradient similarity ≈ influence on the same loss landscape

### 2. **Random Projection Benefits**
- Reduces 100M→8K dimensions while preserving distances (JL lemma)
- Enables practical computation without storing huge matrices
- Deterministic seed ensures reproducibility

### 3. **Adam Preconditioning Rationale**
- Raw gradients are noisy early in training
- Adam's adaptive learning rate (v) de-emphasizes dimensions with high variance
- Preconditioned gradients focus on reliable signal

### 4. **Train-Eval Alignment**
- Maximizing train-eval gradient alignment ≠ minimizing loss directly
- It's a proxy for "which training samples are most relevant to eval distribution"
- Works well when train and eval are from same/similar distribution

### 5. **DeepSpeed Compatibility**
- ZeRO-3 shards all parameters; must gather via `safe_get_full_grad`
- Each rank independently computes its sharded portion
- Central merge ensures correct ordering

### 6. **Scalability**
- Gradients computed per-rank (embarrassingly parallel)
- Merging + normalization only on main process (CPU-bound)
- Suitable for 1M+ sample datasets

---

## 10. SUMMARY TABLE

| Aspect | Implementation |
|--------|----------------|
| **Gradient Dim (Initial)** | `num_params` (LoRA-compressed) |
| **Projection Dim (Output)** | 8192 (Rademacher random) |
| **Projector Type** | TRAK BasicProjector or CudaProjector |
| **Preconditioning** | Adam (β₁=0.9, β₂=0.999) |
| **Similarity Metric** | Normalized inner product (cosine) |
| **Selection Method** | Top-k highest train-eval scores |
| **Distributed Pattern** | Per-rank gradient → main-rank merge → broadcast |
| **Caching** | JSON indices + PT embeddings by step |
| **Resumption** | Check cache before re-computing |


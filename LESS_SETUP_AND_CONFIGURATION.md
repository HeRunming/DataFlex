# LESS Selector Setup and Configuration Guide

## Overview

This document provides a complete guide to understanding and running experiments with the LESS (Low-rank Embedding Selectivity Strategy) selector in DataFlex. LESS is a gradient-based data selection mechanism that dynamically selects training samples based on their estimated influence on model performance.

---

## 1. LESS Selector Configuration

### 1.1 Component Configuration (components.yaml)

The LESS selector is defined in `src/dataflex/configs/components.yaml` (lines 34-41):

```yaml
less:
  name: less
  params:
    cache_dir: ../dataflex_saves/less_output
    gradient_type: adam                    # Gradient preconditioning type
    proj_dim: 4096                         # Projection dimension (was 8192 in original paper)
    seed: 123                              # Random seed for reproducibility
    save_interval: 16                      # Samples per buffer before projection
```

**Key Parameters:**
- **cache_dir**: Where intermediate gradients and selections are cached (per-GPU and merged)
- **gradient_type**: `adam` for Adam preconditioning or `sgd` for raw gradients
- **proj_dim**: Output dimension after random projection (4096 in this config, 8192 in paper)
- **seed**: For deterministic random projection matrices
- **save_interval**: Batch size for projection buffer (16 samples batched before projection)

### 1.2 Training Config Example (examples/train_lora/selectors/less.yaml)

A complete training configuration for LESS:

```yaml
### Model
model_name_or_path: meta-llama/Llama-3.1-8B
finetuning_type: lora
lora_target: all
lora_rank: 16
lora_alpha: 8

### Dataset
dataset: alpaca_en_demo                    # Training set
eval_dataset: alpaca_zh_demo               # Evaluation set for selection
template: llama3
cutoff_len: 4096

### Training
per_device_train_batch_size: 1             # Process one sample at a time for gradient
gradient_accumulation_steps: 1
learning_rate: 1.0e-4
num_train_epochs: 1.0
bf16: true

### DataFlex Dynamic Selection
train_type: dynamic_select                 # Enables selector framework
components_cfg_file: src/dataflex/configs/components.yaml
component_name: less                       # References "less" in components.yaml
warmup_step: 10                            # Samples before first selection
update_step: 10                            # Samples between selections
update_times: 2                            # Selection rounds per epoch

### Evaluation
eval_dataset: alpaca_zh_demo               # Used for evaluation-based selection scoring
eval_strategy: steps
eval_steps: 10
per_device_eval_batch_size: 1
```

**Critical Parameters for LESS:**
- `train_type: dynamic_select` - Activates the selector training mode
- `warmup_step: 10` - Collects 10 samples without selection
- `update_step: 10` - Every 10 samples, performs selection
- `update_times: 2` - Total 2 selection rounds per epoch
- `eval_dataset` - Used to compute evaluation gradients for similarity scoring

---

## 2. LESS Selection Pipeline

### 2.1 High-Level Flow

The LESS selection process has three main phases:

```
Phase 1: Gradient Collection (Per-GPU, Parallel)
  For each training batch:
    1. Forward pass: compute loss
    2. Backward pass: collect gradients
    3. Apply Adam preconditioning (optional)
    4. Project to 4096 dimensions using TRAK
    5. Buffer results locally

Phase 2: Merge and Normalize (Main Process Only)
  1. Collect all per-GPU buffers
  2. Reconstruct original dataset order using indices
  3. Apply L2 normalization (project to unit hypersphere)
  4. Save merged "all_projected_grads.pt"

Phase 3: Score and Select (Main Process Only)
  1. Load training projected gradients
  2. Load evaluation projected gradients
  3. Compute similarity: train_grads @ eval_grads.T
  4. Average scores per training sample
  5. Top-k selection using torch.topk()
  6. Broadcast selected indices to all GPUs
```

### 2.2 Directory Structure During Selection

```
../dataflex_saves/less_output/
├── step_<id>.json                         # Cached selections (accelerator.main_process_only)
├── train/
│   └── <step_id>/
│       ├── grads-{idx}-rank0.pt           # Per-rank gradient chunks
│       ├── grads-{idx}-rank1.pt
│       └── all_projected_grads.pt         # Merged and normalized
└── eval/
    └── <step_id>/
        ├── grads-{idx}-rank0.pt
        ├── grads-{idx}-rank1.pt
        └── all_projected_grads.pt
```

---

## 3. Distributed Training Mechanics

### 3.1 Multi-GPU Synchronization

The LESS selector handles multi-GPU training through:

1. **Per-GPU Parallel Gradient Collection**
   - Each GPU computes gradients independently
   - No communication needed during collection phase
   - Each GPU saves local files: `grads-{max_idx}-rank{rank_id}.pt`

2. **Main-Process Merging** 
   - Only rank 0 loads all per-rank files
   - Reconstructs dataset order using saved indices
   - Applies L2 normalization
   - Saves merged `all_projected_grads.pt`

3. **Broadcast Selection**
   - Uses `torch.distributed.broadcast_object_list()`
   - Main process broadcasts selected indices to all ranks
   - All GPUs receive same selection before next training epoch

### 3.2 Code Locations for Distributed Patterns

**Per-rank gradient collection:**
- File: `src/dataflex/train/selector/less_selector.py`
- Function: `_collect_and_save_projected_gradients()` (lines 179-273)
- Uses `self.accelerator.process_index` to identify rank

**Main-process merging:**
- File: `src/dataflex/train/selector/less_selector.py`
- Function: `_merge_and_normalize_info()` (lines 277-313)
- Guard: `if self.accelerator.is_main_process:`

**Broadcast:**
- File: `src/dataflex/train/selector/less_selector.py`
- Lines: 384-387
- Code: `dist.broadcast_object_list(obj_list, src=0)`

---

## 4. Gradient Processing Details

### 4.1 Gradient Computation

For each training sample:

```python
# Step 1: Forward + Backward
loss = model(**batch).loss
self.accelerator.backward(loss)

# Step 2: Extract full gradient vector
vectorized_grads = torch.cat([
    p.grad.view(-1) for p in model.parameters() if p.grad is not None
])
# Dimension = num_trainable_parameters
# For Llama-3.1-8B with LoRA-16: ~4.2M parameters
```

### 4.2 Adam Preconditioning

When `gradient_type: adam`:

```python
# From optimizer state: exp_avg (m) and exp_avg_sq (v)
beta1, beta2, eps = 0.9, 0.999, 1e-8

# Preconditioning formula
denom = v * beta2 + (grad ** 2) * (1 - beta2)
denom = sqrt(denom) + eps
grad_precond = grad / denom  # or equivalently
grad_precond = grad * m / denom
```

This emphasizes dimensions with low variance in gradient history.

### 4.3 TRAK Projection

Random projection to 4096 dimensions:

```python
# Uses Rademacher random matrix R: 4096 x num_params
# Deterministic via seed=123
grad_sketch = R @ vectorized_grads  # Result: 4096-dimensional vector

# Projector selection:
# - CudaProjector if fast_jl is available (GPU-accelerated)
# - BasicProjector fallback (CPU-based)

# Johnson-Lindenstrauss: preserves pairwise distances
# Compression: ~4.2M -> 4096 (≈1000x)
```

### 4.4 L2 Normalization

After merging all gradients:

```python
# Normalize each gradient sketch to unit length
norms = grads.norm(dim=1, keepdim=True)
grads_normalized = grads / norms.clamp(min=1e-12)

# Effect: Distances become cosine similarities
# Score = train_normalized_grads @ eval_normalized_grads.T
```

---

## 5. Scoring and Selection

### 5.1 Selection Score Computation

```python
# Load merged, normalized gradients
train_grads: shape (num_train, 4096)      # Training set gradients
eval_grads: shape (num_eval, 4096)        # Evaluation set gradients

# Compute similarity matrix
similarities = train_grads @ eval_grads.T  # shape (num_train, num_eval)

# Average across evaluation set
scores_per_train = similarities.mean(dim=1)  # shape (num_train,)

# Top-k selection
topk = torch.topk(scores_per_train, k=num_samples, largest=True)
selected_indices = topk.indices.tolist()
```

### 5.2 Caching and Resume

Selections are cached to avoid recomputation:

```json
// File: step_0.json
{
    "indices": [i1, i2, i3, ...],           // Selected indices
    "metric": {
        "train_eval_similarity": [s1, s2, s3, ...]  // Scores
    }
}
```

If `step_0.json` exists, it's loaded directly without recomputing.

---

## 6. Integration with Training Loop

### 6.1 Trainer Integration

The LESS selector is invoked from `src/dataflex/train/trainer/select_trainer.py`:

**Selection Triggering** (lines 801-805):
```python
if (step_count - warmup_completed) % update_step == 0:
    selected_indices = selector.select(
        model=model, 
        step_id=step_id,
        num_samples=samples_per_update,
        **extra_args
    )
```

**Kwargs Passed to Selector** (lines 821-827):
```python
extra_args = dict(
    optimizer_state=self.optimizer.state,           # Adam state (exp_avg, exp_avg_sq)
    scheduler_state=self.lr_scheduler.state_dict(), # LR scheduler state
    current_update_times=current_update_times,      # Current selection round
    update_times=effective_update_times,            # Total rounds per epoch
    tokenizer=self.tokenizer,                       # Tokenizer
)
```

### 6.2 Training Timeline

For config with `warmup_step=10, update_step=10, update_times=2`:

```
Epoch 1:
  Steps 0-9:    Warmup phase (no selection, random sampling)
  Steps 10-19:  First selection round (select every 10th step)
  Steps 20-29:  Second selection round (select every 10th step)
  
  Total samples processed: 30
  Total selections: 2 (per epoch)
```

---

## 7. Data Configuration

### 7.1 Demo Datasets

DataFlex includes demo datasets in `data/dataset_info.json`:

**Training Dataset:**
```json
{
  "alpaca_en_demo": {
    "file_name": "alpaca_en_demo.json"
  }
}
```

**Evaluation Dataset:**
```json
{
  "alpaca_zh_demo": {
    "file_name": "alpaca_zh_demo.json"
  }
}
```

Both are small subsets for testing. For actual experiments, use full datasets or replace with custom ones.

### 7.2 Custom Dataset Registration

To add a custom dataset:

1. Add entry to `data/dataset_info.json`:
```json
{
  "my_dataset": {
    "file_name": "my_data.json"
  }
}
```

2. Place data file in `data/` directory or reference via HuggingFace Hub:
```json
{
  "my_dataset": {
    "hf_hub_url": "username/my_dataset"
  }
}
```

3. Reference in training config:
```yaml
dataset: my_dataset
eval_dataset: my_eval_dataset
```

---

## 8. Running LESS Experiments

### 8.1 Basic Command

```bash
FORCE_TORCHRUN=1 DISABLE_VERSION_CHECK=1 \
  dataflex-cli train examples/train_lora/selectors/less.yaml
```

### 8.2 Multi-GPU Execution

```bash
# Using 4 GPUs
FORCE_TORCHRUN=1 DISABLE_VERSION_CHECK=1 \
  torchrun --nproc_per_node=4 \
  -m dataflex.train examples/train_lora/selectors/less.yaml
```

### 8.3 Custom Configuration

Modify `examples/train_lora/selectors/less.yaml`:

```yaml
# Change selection parameters
warmup_step: 50        # Longer warmup
update_step: 100       # More samples between selections
update_times: 5        # More selection rounds per epoch

# Change projection dimension
# Edit src/dataflex/configs/components.yaml:
#   less.params.proj_dim: 8192  # Larger projection
```

### 8.4 Resuming Training

If training is interrupted:
- Checkpoints are saved in `output_dir`
- Selections are cached in `cache_dir`
- Training automatically resumes from checkpoint
- LESS skips recomputing cached selections

```bash
# Resume from checkpoint (handled automatically)
FORCE_TORCHRUN=1 DISABLE_VERSION_CHECK=1 \
  dataflex-cli train examples/train_lora/selectors/less.yaml
```

---

## 9. Monitoring and Debugging

### 9.1 Log Files

Training logs are saved to `output_dir/training_logs/`:
- Look for selection scores in logs
- Check GPU memory usage during gradient collection
- Monitor selection cache hits (indicates resumption)

### 9.2 Cache Directory Inspection

```bash
# Check what's been cached
ls -lh ../dataflex_saves/less_output/

# View cached selections
cat ../dataflex_saves/less_output/step_0.json | jq .

# Check gradient files
ls -lh ../dataflex_saves/less_output/train/0/
```

### 9.3 Memory Considerations

**Peak Memory During Selection:**
- Training gradients: `(num_train, 4096) * 4 bytes = num_train * 16KB`
- Evaluation gradients: `(num_eval, 4096) * 4 bytes = num_eval * 16KB`
- Similarity matrix: `(num_train, num_eval) * 4 bytes`

For typical datasets (10K-100K samples):
- Training grads: 160MB - 1.6GB
- Evaluation grads: 160MB - 1.6GB
- Similarity matrix: 400MB - 40GB (watch this!)

**Gradient Collection Memory:**
- Per-GPU buffer: `save_interval * num_params * 2 bytes`
- For save_interval=16, num_params=4.2M: ~134MB per GPU

---

## 10. Comparison with Other Selectors

| Selector | Method | Cost | Memory | Cache |
|----------|--------|------|--------|-------|
| **LESS** | Train-eval similarity | High | High | Yes, per step |
| **NICE** | Reward model scoring | Very High | Very High | Yes, per step |
| **Loss** | Loss-based (fast) | Low | Low | Yes |
| **Delta Loss** | Loss change delta | Very Low | Very Low | No |
| **Random** | Random sampling | None | None | No |

---

## 11. Common Issues and Solutions

### Issue 1: CUDA Out of Memory During Similarity Computation

**Symptom:** Error when computing `train_grads @ eval_grads.T`

**Solution:**
- Reduce `proj_dim` in components.yaml (4096 → 2048)
- Reduce evaluation set size
- Use smaller batch sizes

### Issue 2: Selections Not Changing Between Rounds

**Symptom:** Same indices selected multiple times

**Possible Causes:**
- eval_dataset too small or too similar to train_dataset
- gradient_type mismatch (using "sgd" for eval when train uses "adam")
- proj_dim too small (information loss)

**Solution:**
- Use diverse evaluation set
- Ensure consistent gradient_type settings
- Increase proj_dim

### Issue 3: Slow Selection Due to Cache Misses

**Symptom:** Selection runs same computation repeatedly

**Solution:**
- Check cache_dir permissions
- Verify cache_dir is on fast storage
- Clear cache if corrupted: `rm -rf ../dataflex_saves/less_output/`

### Issue 4: Different Results on Different GPU Counts

**Symptom:** Selection differs with 1 GPU vs 4 GPUs

**Expected:** Some variance due to:
- Different batch ordering due to DDP sampling
- Floating point accumulation differences

**Mitigation:**
- Use deterministic algorithms: set `seed`, `torch.manual_seed()`
- Set `CUBLAS_WORKSPACE_CONFIG=:16:8` for cuBLAS determinism

---

## 12. Advanced Customization

### 12.1 Custom Gradient Type

To add a new gradient type (e.g., "normalize"):

1. Edit `src/dataflex/train/selector/less_selector.py`
2. Modify `_obtain_gradients()` (lines 100-141):
```python
elif gradient_type == "normalize":
    vectorized_grads = vectorized_grads / (vectorized_grads.norm() + 1e-12)
```

### 12.2 Custom Scoring Function

To change selection scoring:

1. Edit `select()` method (lines 315-388)
2. Modify scoring computation (lines 369):
```python
# Instead of: mean(similarities)
train_eval_similarities = (train_projected_grads @ eval_projected_grads.T).max(dim=1).values  # Max instead of mean
```

### 12.3 Dynamic Projection Dimension

To vary proj_dim based on model size:

1. Edit `src/dataflex/train/selector/less_selector.py`
2. Modify `__init__()` to calculate proj_dim:
```python
# Default: 4096, but scale with model
num_params = self._get_number_of_params(model)
self.proj_dim = min(8192, max(2048, num_params // 1000))
```

---

## 13. References

- **LESS Paper:** "Less is More: Pay Less Attention in Vision Transformers" (if applicable)
- **TRAK:** "Transformer Ranking of Axioms" (random projection technique)
- **Johnson-Lindenstrauss:** Distance-preserving dimensionality reduction
- **DataFlex Repository:** https://github.com/[path]/DataFlex
- **Key Source Files:**
  - `src/dataflex/train/selector/less_selector.py` - Core implementation
  - `src/dataflex/train/trainer/select_trainer.py` - Trainer integration
  - `src/dataflex/utils/selector_io.py` - Caching mechanism
  - `src/dataflex/configs/components.yaml` - Configuration templates

---

## Appendix: Quick Reference

### Config Template (less.yaml)
```yaml
train_type: dynamic_select
components_cfg_file: src/dataflex/configs/components.yaml
component_name: less
warmup_step: 10
update_step: 10
update_times: 2
eval_dataset: alpaca_zh_demo
```

### Components Template (components.yaml)
```yaml
less:
  name: less
  params:
    cache_dir: ../dataflex_saves/less_output
    gradient_type: adam
    proj_dim: 4096
    seed: 123
    save_interval: 16
```

### Selection Cache Format (step_N.json)
```json
{
  "indices": [0, 5, 12, 45, ...],
  "metric": {
    "train_eval_similarity": [0.87, 0.85, 0.84, 0.83, ...]
  }
}
```


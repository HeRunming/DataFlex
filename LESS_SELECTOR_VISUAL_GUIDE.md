# DataFlex LESS Selector - Visual & Architectural Guide

## System Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                    TRAINING LOOP (SelectTrainer)                     │
│                   select_trainer.py: _inner_training_loop()          │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                    ┌─────────────┴─────────────┐
                    │                           │
              ┌─────▼─────┐          ┌──────────▼────────┐
              │  WARMUP   │          │ SELECTION TRIGGER │
              │ (Random)  │          │ (Every N steps)   │
              └───────────┘          └──────────┬────────┘
                    │                          │
                    └──────────────┬───────────┘
                                  │
                    ┌─────────────▼──────────────┐
                    │ selector.select() called   │
                    │ with kwargs:               │
                    │ - optimizer_state         │
                    │ - scheduler_state         │
                    │ - current_update_times    │
                    │ - tokenizer               │
                    └─────────────┬──────────────┘
                                  │
                    ┌─────────────▼──────────────────────────┐
                    │    LESS SELECTOR PIPELINE BEGINS        │
                    └─────────────┬──────────────────────────┘
                                  │
           ┌──────────────────────┼──────────────────────┐
           │                      │                      │
       Phase 1             Phase 2                Phase 3
   (Gradient Calc)    (Merge & Norm)          (Scoring & Select)
       Per-Rank              Main Process Only      Main Process Only
           │                      │                      │
```

---

## Phase 1: Distributed Gradient Computation & Projection

```
┌──────────────────────────────────────────────────────────────┐
│  RANK-0  │  RANK-1  │  RANK-2  │  RANK-3   (4-GPU example)  │
│  GPU 0   │  GPU 1   │  GPU 2   │  GPU 3                    │
└──────────┼──────────┼──────────┼───────────────────────────┘
           │          │          │           (Parallel)
           ▼          ▼          ▼
    ┌─────────────────────────┐
    │ For each sample in batch:│
    │ (batch_size=1 per sample)│
    └────────────┬────────────┘
                 │
         ┌───────▼─────────┐
         │ Forward + Backward
         │ loss = model(**batch)
         │ model.backward(loss)
         └───────┬─────────┘
                 │
         ┌───────▼──────────────────────────┐
         │ Extract Gradient Vector           │
         │ for p in model.parameters():      │
         │   if p.requires_grad:             │
         │     grads.append(p.grad.view(-1)) │
         │                                  │
         │ vectorized_grads: [num_params]   │
         │ Shape: [4.2M] for LoRA-7B        │
         └───────┬──────────────────────────┘
                 │
         ┌───────▼──────────────────────────┐
         │ Adam Preconditioning (Optional)   │
         │ if gradient_type == "adam":       │
         │   v_bias_corrected = v/(1-β₂^t)  │
         │   g_precond = g/(√v + eps)       │
         │   g = (1-β₁)*g_raw + β₁*m        │
         │                                  │
         │ (Shapes preserved)               │
         └───────┬──────────────────────────┘
                 │
         ┌───────▼──────────────────────────┐
         │ TRAK Random Projection            │
         │ R ~ Rademacher(num_params×proj_dim) │
         │ sketch = g @ R                   │
         │                                  │
         │ INPUT:  [4.2M]                   │
         │ OUTPUT: [8192] (1/500th size!)   │
         │                                  │
         │ Projector: CudaProjector or      │
         │           BasicProjector         │
         └───────┬──────────────────────────┘
                 │
         ┌───────▼───────────────────────────┐
         │ Accumulate in Buffer (per-rank)   │
         │ grad_buffer[buf_pos] = sketch     │
         │ idx_buffer[buf_pos] = sample_idx  │
         │ buf_pos += 1                      │
         │                                  │
         │ When buf_pos == save_interval OR │
         │ at end of dataset:                │
         └───────┬───────────────────────────┘
                 │
         ┌───────▼───────────────────────────────────┐
         │ Save Local Chunk File (each rank)         │
         │ File: grads-{max_idx}-rank{rank_id}.pt   │
         │ {                                        │
         │   "grads": [buf_pos, proj_dim],          │
         │   "indices": [buf_pos]                   │
         │ }                                        │
         │                                          │
         │ All ranks do this in parallel            │
         └───────┬───────────────────────────────────┘
                 │
         ┌───────▼──────────────────────┐
         │ wait_for_everyone()          │
         │ (Synchronization barrier)     │
         └─────────────────────────────┘
```

**Memory Usage During Phase 1 (per-GPU)**:
```
                         On GPU:
grad_buffer:             [16 × 4.2M floats] = 268 MB
idx_buffer:              [16 integers] = negligible
m, v (Adam):             [4.2M × 2] = 34 MB
vectorized_grads:        [4.2M] = 17 MB
────────────────────────────────────
Peak:                    ~350 MB

                         Stored to Disk:
grads-{idx}-rank{i}.pt:  [batch_size × 8192] ≈ 32 KB per save
```

---

## Phase 2: Merge & Normalize (Main Process Only)

```
Only CPU (Main Process)
┌────────────────────────────────────────────────────┐
│ Load all rank files from disk:                     │
│                                                    │
│ rank 0: grads-1000-rank0.pt (samples 0-250)      │
│ rank 1: grads-1000-rank1.pt (samples 250-500)    │
│ rank 2: grads-1000-rank2.pt (samples 500-750)    │
│ rank 3: grads-1000-rank3.pt (samples 750-1000)   │
│                                                    │
│ Using glob.glob(save_dir, "grads-*-rank*.pt")    │
└────────────┬─────────────────────────────────────┘
             │
    ┌────────▼─────────────────────────────┐
    │ Initialize Empty Tensor               │
    │ final_grads = zeros([N, proj_dim])   │
    │ final_grads.shape = [1000, 8192]    │
    │                                      │
    │ dtype=float32                        │
    └────────┬─────────────────────────────┘
             │
    ┌────────▼──────────────────────────────────┐
    │ For each file:                             │
    │   chunk = torch.load(file)                 │
    │   grads_chunk = chunk['grads']  # [?, 8192]│
    │   indices = chunk['indices']    # [?]      │
    │                                           │
    │   final_grads[indices] = grads_chunk      │
    │   (Index-based assignment to restore      │
    │    original dataset ordering)             │
    └────────┬──────────────────────────────────┘
             │
    ┌────────▼──────────────────────────────────────┐
    │ L2 NORMALIZATION (Row-wise, Unit Hypersphere)│
    │                                              │
    │ norms = final_grads.norm(dim=1, keepdim=True)│
    │        # Shape: [1000, 1]                   │
    │        # Values: L2 norm of each row        │
    │                                              │
    │ norms = norms.clamp(min=1e-12)              │
    │ final_grads.div_(norms)                     │
    │                                              │
    │ After this step:                            │
    │ ∀i: ||final_grads[i]||₂ = 1                │
    │                                              │
    │ final_grads shape still [1000, 8192]        │
    └────────┬──────────────────────────────────────┘
             │
    ┌────────▼────────────────────────────────┐
    │ Save Merged Normalized Grads             │
    │                                          │
    │ torch.save(                              │
    │   final_grads,                          │
    │   "cache/train/step_1000/               │
    │    all_projected_grads.pt"              │
    │ )                                       │
    │                                          │
    │ Clean up intermediate files:            │
    │ for file in rank_files:                 │
    │   os.remove(file)                       │
    │                                          │
    │ ✓ Ready for next phase                  │
    └────────────────────────────────────────┘
```

**Key Insight**: Row-wise L2 normalization transforms gradients to unit vectors
- Each sample's gradient sketch becomes a point on unit sphere
- Inner products now represent cosine similarity

---

## Phase 3: Scoring & Selection

```
┌───────────────────────────────────────────────────────┐
│              Main Process Only - CPU/GPU              │
└───────────────────────────────────────────────────────┘

Step 1: Load Normalized Gradients
┌──────────────────────────────────────┐
│ train_grads = load("cache/train/     │
│               step_1000/             │
│               all_projected_grads.pt")│
│                                      │
│ train_grads.shape = [10000, 8192]   │
│ (all normalized to unit vectors)    │
│                                      │
│ eval_grads = load("cache/eval/      │
│              step_1000/              │
│              all_projected_grads.pt") │
│                                      │
│ eval_grads.shape = [1000, 8192]    │
│ (all normalized to unit vectors)    │
└──────────────────────────────────────┘
         │
Step 2: Compute Similarity Matrix
┌─────────────────────────────────────────────────────┐
│ sim_matrix = train_grads @ eval_grads.T            │
│                                                    │
│ Operations:                                        │
│ [10000, 8192] @ [8192, 1000] → [10000, 1000]     │
│                                                    │
│ Element [i,j] = dot(train[i], eval[j])           │
│ = cos(angle) between vectors (cosine similarity)  │
│                                                    │
│ Since normalized: values ∈ [-1, 1]               │
│ High value = similar gradient directions          │
└─────────────────────────────────────────────────────┘
         │
Step 3: Aggregate Scores (Mean Pooling)
┌──────────────────────────────────────────────────────────┐
│ scores = sim_matrix.mean(dim=1)                         │
│                                                         │
│ scores.shape = [10000]                                 │
│                                                         │
│ scores[i] = mean_j(sim_matrix[i,j])                   │
│           = average similarity of train[i] to all eval  │
│                                                         │
│ Interpretation:                                        │
│ High score = training sample aligns with eval set      │
│ Low score = training sample misaligned with eval       │
└──────────────────────────────────────────────────────────┘
         │
Step 4: Top-K Selection
┌──────────────────────────────────────────────────────────┐
│ num_samples = 2000  (e.g., select 2000 from 10000)     │
│                                                        │
│ topk_vals, topk_indices = torch.topk(                  │
│     scores,                                            │
│     k=num_samples,                                     │
│     largest=True  ← highest scores first               │
│ )                                                      │
│                                                        │
│ selected_indices = topk_indices.tolist()               │
│ selected_indices.length = 2000                         │
│                                                        │
│ Values might be: [453, 127, 8932, 12, ...]           │
└──────────────────────────────────────────────────────────┘
         │
Step 5: Save & Broadcast
┌─────────────────────────────────────────────────────┐
│ metric_payload = {                                 │
│   "train_eval_similarity": [s1, s2, ...]          │
│ }                                                  │
│                                                   │
│ save_selection(                                   │
│   "cache/step_1000.json",                        │
│   selected_indices,                              │
│   metric_payload,                                │
│   accelerator                                    │
│ )                                                │
│                                                   │
│ JSON saved only on rank-0                         │
└─────────────────────────────────────────────────────┘
         │
         ▼
    Broadcast to all ranks
┌────────────────────────────────────┐
│ obj_list = [selected_indices]      │
│ dist.broadcast_object_list(        │
│   obj_list, src=0                  │
│ )                                  │
│ # Now all ranks have the selection │
└────────────────────────────────────┘
```

**Scoring Example** (2 samples, 3 eval):
```
Train Sample #100:  [0.1, 0.2, -0.1, ...]  (normalized)
Eval Sample #1:     [0.12, 0.19, -0.08, ...]
Eval Sample #2:     [-0.1, 0.3, 0.2, ...]
Eval Sample #3:     [0.05, 0.1, 0.0, ...]

sim[100, 1] = 0.1×0.12 + 0.2×0.19 + (-0.1)×(-0.08) + ...
            = 0.012 + 0.038 + 0.008 + ... ≈ 0.85
sim[100, 2] = 0.1×(-0.1) + 0.2×0.3 + (-0.1)×0.2 + ... ≈ 0.42
sim[100, 3] = ... ≈ 0.78

score[100] = mean([0.85, 0.42, 0.78]) ≈ 0.68  ← High alignment!
```

---

## Data Flow Diagram: End-to-End

```
┌─────────────────────────────────────────────────────────┐
│           Training Data (10,000 samples)                │
└────────────┬────────────────────────────────────────────┘
             │
    ┌────────▼─────────────┐
    │ Warmup Phase (optional)
    │ Random selection from data
    └────────┬─────────────┘
             │
    ┌────────▼──────────────────────┐
    │ Training Steps 1-50            │
    │ (Model learns basic patterns)  │
    └────────┬──────────────────────┘
             │
             │ After warmup_step:
    ┌────────▼───────────────────────────────┐
    │ SELECTION ROUND 1                      │
    │ ├─ Gradient computation: 10K samples   │
    │ ├─ Projection: 4.2M → 8192 dims       │
    │ ├─ Merge & normalize (main process)   │
    │ ├─ Score: train vs eval gradients    │
    │ └─ Select top 2000 similar samples    │
    └────────┬────────────────────────────────┘
             │
    ┌────────▼──────────────────────────┐
    │ Training on Selected Data          │
    │ 2000 × 10 = 20k gradient updates  │
    │ (assuming 10 epochs on selection)  │
    └────────┬──────────────────────────┘
             │
             │ After update_step × 10:
    ┌────────▼───────────────────────────────┐
    │ SELECTION ROUND 2 (re-select)          │
    │ ├─ Gradient computation: 10K samples   │
    │ ├─ Model has now seen 20k steps       │
    │ ├─ New selection based on updated     │
    │ │  model gradients                   │
    │ └─ Select new 2000 samples           │
    └────────┬────────────────────────────────┘
             │
    ┌────────▼──────────────────────────┐
    │ ... (repeat until training ends)   │
    │ Each round: fresh sample selection │
    └────────────────────────────────────┘
```

---

## Distributed Execution Timeline (4-GPU Example)

```
Time →

RANK-0    ┌─ Forward (L)
          ├─ Backward (M)
GPU-0     ├─ Gradient Extract (S)
          ├─ Adam Precond (S)
          ├─ Project (M)
          ├─ Save Local (S)
          └─ wait_for_everyone() ───────────┐
                                             │
RANK-1    ┌─ Forward (L)                     │
          ├─ Backward (M)                    │
GPU-1     ├─ Gradient Extract (S)            │
          ├─ Adam Precond (S)                │
          ├─ Project (M)                     │
          ├─ Save Local (S)                  │
          └─ wait_for_everyone() ────────────┼──┐
                                             │  │
RANK-2    ┌─ Forward (L)                     │  │
          ├─ Backward (M)                    │  │
GPU-2     ├─ Gradient Extract (S)            │  │
          ├─ Adam Precond (S)                │  │
          ├─ Project (M)                     │  │
          ├─ Save Local (S)                  │  │
          └─ wait_for_everyone() ────────────┼──┼──┐
                                             │  │  │
RANK-3    ┌─ Forward (L)                     │  │  │
          ├─ Backward (M)                    │  │  │
GPU-3     ├─ Gradient Extract (S)            │  │  │
          ├─ Adam Precond (S)                │  │  │
          ├─ Project (M)                     │  │  │
          ├─ Save Local (S)                  │  │  │
          └─ wait_for_everyone() ────────────┼──┼──┼──┐
                                             │  │  │  │
RANK-0    ┌─ Load rank files (all 4)         │  │  │  │
          ├─ Reconstruct order               │  │  │  │
          ├─ Normalize vectors               │  │  │  │
CPU       ├─ Compute sim matrix              │  │  │  │
          ├─ Score & select top-k            │  │  │  │
          ├─ Save results (JSON)             │  │  │  │
          └─ broadcast_object_list ──────────┼──┼──┼──┘
                                             │  │  │
RANK-0    ┌─ Receive broadcast ──────────────┘  │  │
RANK-1    ├─ Receive broadcast ────────────────┘  │
RANK-2    ├─ Receive broadcast ─────────────────┘
RANK-3    ├─ Receive broadcast

Legend:
L = Large time (network communication)
M = Medium time (GPU computation)
S = Small time (trivial operations)
```

---

## Configuration Example

```yaml
# config in components_cfg.yaml
selectors:
  less:
    type: less
    dataset: ${runtime:dataset}
    eval_dataset: ${runtime:eval_dataset}
    accelerator: ${runtime:accelerator}
    data_collator: ${runtime:data_collator}
    cache_dir: ./cache/selections
    gradient_type: adam          # or "sgd"
    proj_dim: 8192              # projection dimension
    save_interval: 16           # samples before saving
    seed: 42                    # for reproducibility

# config in training args
train_args:
  warmup_step: 100              # samples for warmup
  update_step: 200              # samples between selections
  update_times: 5               # selection rounds per epoch
  train_step: 1000              # total training steps
```


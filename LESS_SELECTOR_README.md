# LESS Selector - Complete Technical Documentation

This directory contains comprehensive technical documentation of the **LESS (Low-rank Embedding Selectivity Strategy)** selector implementation in DataFlex.

## 📚 Documentation Files

### 1. **LESS_SELECTOR_ANALYSIS.md** (19 KB, 530 lines)
**Detailed technical deep-dive** with code references and full implementation details.

**Contents:**
- Gradient flow and dimension transformations (LoRA → full gradient → projected)
- TRAK projector mechanics (Rademacher random projection)
- Complete pipeline flow: gradient → sketch → normalize → score → select
- Optimizer state handling (Adam preconditioning)
- Distributed training patterns (multi-GPU synchronization)
- Core algorithm constants and hyperparameters
- Memory and performance characteristics
- Caching and resumption mechanisms
- Key technical insights

**Best for:** Understanding the "how" and "why" behind each component.

---

### 2. **LESS_SELECTOR_VISUAL_GUIDE.md** (26 KB, 456 lines)
**Visual flowcharts and architectural diagrams** with execution timelines.

**Contents:**
- System architecture overview
- Phase 1: Distributed gradient computation & projection (ASCII diagrams)
- Phase 2: Merge & normalize (main process)
- Phase 3: Scoring & selection (main process)
- End-to-end data flow diagram
- 4-GPU distributed execution timeline
- Configuration examples
- Memory breakdown examples

**Best for:** Understanding data flow and distributed execution patterns.

---

### 3. **LESS_SELECTOR_QUICK_REFERENCE.md** (13 KB, 447 lines)
**Quick lookup guide** with equations, code snippets, and troubleshooting.

**Contents:**
- Core mathematics (6 key equations with explanations)
- Implementation details (file locations, key functions)
- Parameter tuning guide
- Memory usage estimation
- Distributed training details
- Cache & resumption patterns
- Common issues & solutions (4 examples with fixes)
- Algorithm pseudocode
- Performance profiling tips
- Troubleshooting checklist

**Best for:** Quick lookups, debugging, and parameter tuning.

---

## 🎯 Quick Answers to Key Questions

### What is the gradient dimension after LoRA projection?
- **Initial**: `num_params_LoRA` (e.g., 4.2M for LoRA-7B with rank-8)
- **After random projection**: `proj_dim` (default 8192)
- **Compression**: ~500× reduction (e.g., 4.2M → 8192)

See: LESS_SELECTOR_ANALYSIS.md § 1, LESS_SELECTOR_QUICK_REFERENCE.md § 3

### How does the TRAK projector work?
Random Rademacher matrix (±1 entries) projects from `num_params` to `proj_dim` dimensions:
```
sketch = gradient @ Rademacher_matrix
```
- Deterministic based on `seed=42`
- Johnson-Lindenstrauss guarantee: preserves distances
- Two implementations: CudaProjector (GPU) and BasicProjector (fallback)

See: LESS_SELECTOR_ANALYSIS.md § 2, LESS_SELECTOR_VISUAL_GUIDE.md § Phase 1

### What is the exact flow: gradient → sketch → normalize → score → select?
1. **Gradient Computation** (per-rank, parallel on GPUs)
   - Forward/backward pass → extract full gradient vector
   - Optional Adam preconditioning
   - Random projection → sketch (8192 dims)

2. **Merge & Normalize** (main process only)
   - Load all rank-saved files
   - Reconstruct dataset order by index
   - L2 normalization (row-wise) → unit hypersphere

3. **Scoring & Selection** (main process only)
   - Compute similarity matrix: train_sketches @ eval_sketches.T
   - Average per-sample scores across eval set
   - Top-k selection by highest scores
   - Broadcast to all ranks

See: LESS_SELECTOR_ANALYSIS.md § 3, LESS_SELECTOR_VISUAL_GUIDE.md § Phase 2-3

### What optimizer state info is available in kwargs?
```python
extra_args = dict(
    optimizer_state=self.optimizer.state,          # Dict[param → {exp_avg, exp_avg_sq, step}]
    scheduler_state=self.lr_scheduler.state_dict(),  # LR scheduler state
    current_update_times=current_update_times,       # Selection round number
    update_times=effective_update_times,             # Total selection rounds
    tokenizer=self.tokenizer,                        # For processing
)
```

- **exp_avg**: first moment (momentum)
- **exp_avg_sq**: second moment (adaptive LR)
- **step**: training step count

See: LESS_SELECTOR_ANALYSIS.md § 4, select_trainer.py lines 821-827

---

## 🔧 Parameter Configuration Guide

| Parameter | Default | Tuning Strategy |
|-----------|---------|-----------------|
| `proj_dim` | 8192 | Reduce if slow; increase if accuracy needed |
| `save_interval` | 16 | Increase if GPU OOM; decrease if disk I/O is bottleneck |
| `gradient_type` | "adam" | "adam" for preconditioning; "sgd" for raw gradients |
| `seed` | 42 | Must match across runs for reproducibility |
| `warmup_step` | - | How many steps before first selection |
| `update_step` | - | Steps between selection rounds |
| `update_times` | 1 | Total number of selection rounds per epoch |

See: LESS_SELECTOR_QUICK_REFERENCE.md § Parameter Tuning Guide

---

## 📊 Performance Estimates

### Gradient Computation (per-GPU, LLaMA-7B + LoRA-8)
```
GPU Memory:  ~350 MB peak
Time/sample: 15-60 ms
```

### Full Selection Round (10K samples)
```
Time:        2.5-10 minutes
Disk:        ~650 MB (gradients + cache)
```

### Memory Breakdown
```
grad_buffer:     268 MB
Adam states:      34 MB
Rademacher:        0 MB (implicit, seeded)
─────────────────────
Peak GPU:       ~350 MB
```

See: LESS_SELECTOR_QUICK_REFERENCE.md § Memory Usage Estimation

---

## 🔍 Understanding the Code

### Core Files
```
src/dataflex/train/selector/
├── less_selector.py              # Main implementation
│   ├── _obtain_gradients()       # Extract & precondition gradients
│   ├── _prepare_optimizer_state() # Load Adam states
│   ├── _get_trak_projector()     # Select projector (CUDA or CPU)
│   ├── _collect_and_save_projected_gradients()  # Phase 1
│   ├── _merge_and_normalize_info()              # Phase 2
│   └── select()                  # Main entry point (Phases 2-3)
├── base_selector.py              # Abstract base class
│   └── broadcast_object_list()   # Synchronization
└── [other selectors...]

src/dataflex/train/trainer/
├── select_trainer.py             # Training loop integration
│   └── _inner_training_loop()    # Calls selector.select() at right times
└── [other trainers...]

src/dataflex/utils/
├── selector_io.py                # Save/load selections (JSON + PT)
└── [other utilities...]
```

---

## 🚀 Quick Start: Running a Selection

### Step 1: Configure in YAML
```yaml
# components.yaml
selectors:
  less:
    type: less
    cache_dir: ./cache/selections
    gradient_type: adam
    proj_dim: 8192
    save_interval: 16

# training args
warmup_step: 100
update_step: 200
update_times: 5
```

### Step 2: Training
```bash
python run_train.py \
  --config training_config.yaml \
  --finetuning_type lora \
  --lora_rank 8
```

### Step 3: Monitor
- Selection logs appear at `warmup_step`, then every `update_step`
- Cache stored in `./cache/selections/`
- Resuming from checkpoint automatically loads cached selection

---

## ⚠️ Common Pitfalls

1. **Different selection across runs**
   - Ensure `seed=42` and global torch/numpy seeds set
   - See: LESS_SELECTOR_QUICK_REFERENCE.md § Issue 4

2. **GPU out of memory**
   - Reduce `save_interval` or `proj_dim`
   - See: LESS_SELECTOR_QUICK_REFERENCE.md § Issue 1

3. **Selection takes too long**
   - Install `fast_jl` for CudaProjector
   - Reduce `proj_dim` to 4096
   - See: LESS_SELECTOR_QUICK_REFERENCE.md § Issue 2

4. **Constant cache misses**
   - Increase `save_interval` to reduce disk I/O
   - See: LESS_SELECTOR_QUICK_REFERENCE.md § Issue 3

---

## 📖 Reference Sections by Use Case

### I want to understand the math
→ LESS_SELECTOR_QUICK_REFERENCE.md § Core Mathematics

### I want to debug why selection is slow
→ LESS_SELECTOR_QUICK_REFERENCE.md § Issue 2 + Performance Profiling

### I want to understand how multi-GPU works
→ LESS_SELECTOR_VISUAL_GUIDE.md § Distributed Execution Timeline

→ LESS_SELECTOR_ANALYSIS.md § 5. Distributed Training Pattern

### I want to tune hyperparameters
→ LESS_SELECTOR_QUICK_REFERENCE.md § Parameter Tuning Guide

### I want to understand gradient preconditioning
→ LESS_SELECTOR_QUICK_REFERENCE.md § Adam Preconditioning

→ LESS_SELECTOR_ANALYSIS.md § 1. Gradient Flow

### I want to see code line-by-line
→ LESS_SELECTOR_ANALYSIS.md § 3. Exact Flow (with line numbers)

---

## 🔗 Cross-References

**Between documents:**
- ANALYSIS.md § 1 ↔ VISUAL_GUIDE.md § Phase 1 ↔ QUICK_REF.md § Gradient Extraction
- ANALYSIS.md § 3 ↔ VISUAL_GUIDE.md § Phase 2-3 ↔ QUICK_REF.md § Algorithm Pseudocode
- ANALYSIS.md § 5 ↔ VISUAL_GUIDE.md § Distributed Timeline ↔ QUICK_REF.md § Distributed Training

**To source code:**
- All sections include file paths and line numbers
- Example: `less_selector.py`, `lines 100-141` → LESS_SELECTOR_ANALYSIS.md § 1

---

## 📝 Document Statistics

| Document | Size | Lines | Coverage |
|----------|------|-------|----------|
| ANALYSIS.md | 19 KB | 530 | Complete technical deep-dive |
| VISUAL_GUIDE.md | 26 KB | 456 | Flowcharts & diagrams |
| QUICK_REFERENCE.md | 13 KB | 447 | Math + troubleshooting |
| **Total** | **58 KB** | **1,433** | **Comprehensive** |

---

## 🤝 Contributing

When modifying the LESS selector, update relevant sections in:
1. LESS_SELECTOR_ANALYSIS.md (technical details)
2. LESS_SELECTOR_VISUAL_GUIDE.md (flow diagrams if algorithm changes)
3. LESS_SELECTOR_QUICK_REFERENCE.md (equations, code references, parameters)

---

## 📞 Quick Help

**Need to find...**
- Line numbers: ANALYSIS.md
- Visual overview: VISUAL_GUIDE.md
- Specific equation: QUICK_REFERENCE.md
- Troubleshooting: QUICK_REFERENCE.md § Common Issues

---

Generated: May 13, 2026  
Repository: `/jizhicfs/karonhe/DataFlex`


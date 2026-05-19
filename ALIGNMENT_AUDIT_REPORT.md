# DataFlex Codebase Audit: MMD vs LESS Alignment
**Date**: May 19, 2026  
**Scope**: Comprehensive alignment audit of MMD experiment settings with LESS paper baseline

---

## EXECUTIVE SUMMARY

**Critical Finding**: The MMD experiment configurations have **significant misalignments** with the LESS paper baseline settings. While both use gradient-based selection, the implementations diverge in critical areas:

1. **Gradient Type Mismatch** (CRITICAL)
2. **Target Dataset Semantics** (MEDIUM)
3. **Projection Parameters Consistency** (LOW)
4. **Evaluation Protocol Issues** (MEDIUM)

---

## 1. GRADIENT TYPE ANALYSIS

### LESS Baseline Configuration
- **components.yaml (line 34-41)**: `gradient_type: adam`
- **less_selector.py (line 34)**: Default parameter `gradient_type: str = "adam"`
- **Computation (lines 125-133)**: 
  - Uses Adam preconditioner with β₁=0.9, β₂=0.999, ε=1e-8
  - Modifies gradient in-place: `vectorized_grads.mul_(1-β₁).add_(m, α=β₁)` then `vectorized_grads.div_(denom)`
  - **This is problematic**: In-place modification of gradient vector **during Adam preconditioning** can cause subtle bugs with optimizer state tracking

### MMD Gradient Type Settings

| Config | kernel_type | gradient_type | Issue |
|--------|-------------|--------------|-------|
| **mmd_grad_rbf.yaml** | grad_rbf | **sgd** | ❌ MISMATCH |
| **mmd_grad_cov.yaml** | grad_cov | **sgd** | ❌ MISMATCH |
| **components.yaml** (mmd_grad_rbf, line 64) | grad_rbf | **sgd** | ❌ MISMATCH |
| **components.yaml** (mmd_grad_cov, line 76) | grad_cov | **sgd** | ❌ MISMATCH |
| **less_baseline.yaml** | N/A (LESS) | adam | ✓ CORRECT |

### Specific Code Evidence

**LESS uses Adam (line 356 of less_selector.py)**:
```python
self._collect_and_save_projected_gradients(
    model, now_eval_save_dir, self.target_dataset, "sgd", None  # ← uses "sgd" for target!
)
```
Wait - **LESS also uses SGD for target_dataset!** But for training set uses Adam.

**mmd_selector.py (line 290-291)**:
```python
self._collect_and_save_projected_gradients(
    model, target_grads_dir, self.target_dataset, self.gradient_type, optimizer_state
)
```
- Uses `self.gradient_type` (which defaults to "sgd" from components.yaml)
- **For both candidate AND target sets**

### KEY FINDING: Gradient Type Asymmetry

**LESS does:**
- Training set: Adam (line 347, using optimizer_state)
- Target set: SGD (line 356, None for optimizer_state)
- **Rationale**: This matches the LESS paper - uses Adam-preconditioned gradients for training data, raw gradients for evaluation target

**MMD does:**
- Training set: SGD (components.yaml, gradient_type: sgd)
- Target set: SGD (both use self.gradient_type)
- **Problem**: Should use Adam for candidates to match LESS paper!

---

## 2. PROJECTION PARAMETERS ALIGNMENT

### components.yaml Comparison

| Parameter | LESS (line 39) | MMD Grad RBF (line 63) | MMD Grad Cov (line 75) | Status |
|-----------|---------------|----------------------|----------------------|--------|
| **proj_dim** | 4096 | 4096 | 4096 | ✓ Match |
| **save_interval** | 16 | 16 | 16 | ✓ Match |
| **seed** | 123 | 42 | 42 | ❌ MISMATCH |
| **proj_type** | (not in yaml) | (not in yaml) | (not in yaml) | Both use `ProjectionType.rademacher` (hardcoded) |

### Code Evidence

**less_selector.py (lines 186-195)**:
```python
projector = projector_class(
    grad_dim=num_params,
    proj_dim=self.proj_dim,           # 4096
    seed=self.seed,                   # 123 (from components.yaml)
    proj_type=ProjectionType.rademacher,
    max_batch_size=8,
    block_size=128,
    device=self.device,
    dtype=self.dtype,
)
```

**mmd_selector.py (lines 686-695)**:
```python
projector = projector_class(
    grad_dim=num_params,
    proj_dim=self.proj_dim,           # 4096
    seed=self.seed,                   # 42 ← DIFFERENT FROM LESS!
    proj_type=ProjectionType.rademacher,
    max_batch_size=8,
    block_size=128,
    device=self.device,
    dtype=self.dtype,
)
```

**Impact**: Different random seeds (123 vs 42) will produce different random projection matrices, affecting reproducibility and direct comparison.

---

## 3. TARGET DATASET HANDLING

### Problem: Semantic Confusion Between `eval_dataset` and `target_dataset`

**LESS paper semantics:**
- Target set = the held-out benchmark examples used to compute similarity for selection
- Evaluation set = separate held-out test set for measuring final performance

**DataFlex current implementation:**

**less_baseline.yaml (line 52)**:
```yaml
eval_dataset: alpaca_zh_demo
per_device_eval_batch_size: 1
metric_for_best_model: eval_loss
greater_is_better: false
load_best_model_at_end: false
eval_strategy: steps
eval_steps: 10
```
- Sets `eval_dataset` which LlamaFactory interprets as evaluation set
- **But in SelectTrainer (line 249)**, this is **passed as target_dataset for selection**!

**mmd_grad_rbf.yaml (line 53-55)**:
```yaml
eval_dataset: alpaca_zh_demo
per_device_eval_batch_size: 1
eval_strategy: "no"  # ← Disables evaluation!
```
- Sets `eval_dataset` 
- Sets `eval_strategy: "no"` to disable actual evaluation
- But still passes it to selector as target_dataset

### Critical Code Path (select_trainer.py, lines 247-254)

```python
# Determine target dataset for selection (separate from eval)
# If target_dataset is explicitly configured, load it; otherwise fall back to eval_dataset
target_dataset_for_selector = self.eval_dataset  # default fallback
if hasattr(finetuning_args, 'target_dataset') and finetuning_args.target_dataset:
    # target_dataset is already loaded by LlamaFactory as eval_dataset if
    # the user sets it in YAML. For now, we pass eval_dataset as target.
    # The key semantic change: selector sees it as "target_dataset", not "eval_dataset"
    pass

runtime = dict(
    dataset=self.train_dataset,
    target_dataset=target_dataset_for_selector,  # ← passes eval_dataset
    ...
)
```

**Issue**: The code always passes `self.eval_dataset` as `target_dataset` to selector, even when:
1. User intends `eval_dataset` for actual evaluation
2. User intends `target_dataset` for selection (per dynamic_params.py line 560-568)

### Alignment Status with LESS

**LESS doesn't have this confusion because:**
- It always passes target_dataset explicitly (line 356 in less_selector.py)
- The target dataset is used ONLY for selection, not evaluation

**MMD has both ambiguity AND evaluation pollution:**
```yaml
# mmd_grad_rbf.yaml
eval_strategy: "no"  # Tries to disable evaluation, but...
```
- If eval_strategy was NOT set to "no", the selector would try to compute selection against training target
- Evaluation targets are NOT held-out if they share data with selection targets

---

## 4. ADAM PRECONDITIONING BUG IN GRADIENT COMPUTATION

### LESS Implementation Issue

**less_selector.py (lines 125-133)**:
```python
if gradient_type == "adam":
    if m is None or v is None:
        raise ValueError("Adam optimizer states (m, v) must be provided...")
    beta1, beta2, eps = 0.9, 0.999, 1e-08
    denom = v.mul(beta2)              # ← in-place mutation of v!
    denom.addcmul_(vectorized_grads, vectorized_grads, value=(1 - beta2))
    denom.sqrt_().add_(eps)           # ← in-place mutations
    vectorized_grads.mul_(1 - beta1).add_(m, alpha=beta1)  # ← in-place mutation
    vectorized_grads.div_(denom)      # ← in-place mutation
    del denom
```

**Problem**: This modifies the input gradient in-place AND modifies v in-place without copying!
- Correct Adam formula should use: `adam_precond = (β₁m + (1-β₁)g) / (√(β₂v + (1-β₂)g²) + ε)`
- This creates: `adam_precond = (β₁m + (1-β₁)g) / √(β₂v + (1-β₂)g²)`
- But after line 130, it overwrites `v` in-place, corrupting the optimizer state!

### MMD Implementation Fix

**mmd_selector.py (lines 645-654)**:
```python
if gradient_type == "adam":
    if m is None or v is None:
        raise ValueError("Adam states (m, v) required for 'adam' gradient type.")
    # FIX: Do NOT modify m or v in-place. Compute Adam-preconditioned gradient
    # as a new tensor: adam_grad = (beta1*m + (1-beta1)*g) / (sqrt(beta2*v + (1-beta2)*g²) + eps)
    beta1, beta2, eps = 0.9, 0.999, 1e-08
    numerator = beta1 * m + (1.0 - beta1) * vectorized_grads
    denominator = torch.sqrt(beta2 * v + (1.0 - beta2) * vectorized_grads.pow(2)) + eps
    vectorized_grads = numerator / denominator
    del numerator, denominator
```

**Status**: ✓ MMD fixed this bug, LESS still has it

---

## 5. EXPERIMENT CONFIG ALIGNMENT

### Training Hyperparameters

All configs (less_baseline.yaml, mmd_grad_rbf.yaml, mmd_grad_cov.yaml, mmd_emb_rbf.yaml):

| Parameter | Value | Status |
|-----------|-------|--------|
| Model | Qwen/Qwen2.5-0.5B | ✓ Same |
| Learning rate | 1.0e-4 | ✓ Same |
| Warmup ratio | 0.1 | ✓ Same |
| LR scheduler | cosine | ✓ Same |
| Batch size | 1 | ✓ Same |
| Epochs | 1.0 | ✓ Same |
| LoRA rank | 16 | ✓ Same |
| LoRA alpha | 8 | ✓ Same |
| Precision | bf16 | ✓ Same |
| Seed | 42 | ✓ Same |

### Dynamic Training Parameters

| Parameter | LESS | MMD Grad RBF | Status |
|-----------|------|-------------|--------|
| warmup_step | 10 | 10 | ✓ Same |
| update_step | 10 | 10 | ✓ Same |
| update_times | 2 | 2 | ✓ Same |

### Selection Protocol Differences

**LESS (less_baseline.yaml, line 44-49)**:
```yaml
train_type: dynamic_select
warmup_step: 10
update_step: 10
update_times: 2
```
- **Dynamic selection**: Selection happens during training at multiple steps
- Selector is called at warmup_step and then every update_step

**MMD (mmd_grad_rbf.yaml, line 44-49)**:
```yaml
train_type: dynamic_select
warmup_step: 10
update_step: 10
update_times: 2
```
- **Also dynamic**: Identical to LESS

---

## 6. EVALUATION PROTOCOL ISSUES

### LESS Baseline (line 51-58)

```yaml
eval_dataset: alpaca_zh_demo
per_device_eval_batch_size: 1
metric_for_best_model: eval_loss
greater_is_better: false
load_best_model_at_end: false
eval_strategy: steps
eval_steps: 10
```

- **Evaluates during training**: eval_strategy=steps, eval_steps=10
- **Uses alpaca_zh_demo as target** (for selection) AND evaluation set (for metrics)
- **Problem**: Target and eval are the SAME dataset!
- This violates the LESS paper principle of held-out evaluation

### MMD Grad RBF (line 51-55)

```yaml
eval_dataset: alpaca_zh_demo
per_device_eval_batch_size: 1
eval_strategy: "no"
```

- **Disables evaluation**: eval_strategy="no"
- **Uses alpaca_zh_demo as target set for selection** 
- **Does not compute evaluation metrics**
- This is cleaner but breaks compatibility with LESS config for direct comparison

### LESS Paper Expected Setting

According to LESS paper (Xia et al., 2024):
1. Candidate pool: Large unlabeled or training data
2. Target set: Few-shot or small labeled examples (64 examples)
3. Evaluation set: Held-out test split (separate from both)

**Current DataFlex setup violates this**: alpaca_zh_demo is used for BOTH target and evaluation!

---

## 7. SELECTION SCORING METHOD

### LESS (less_selector.py, line 369)

```python
train_eval_similarities = (train_projected_grads @ eval_projected_grads.T).mean(dim=1)
topk = torch.topk(train_eval_similarities, k=num_samples, largest=True)
```

- **Scoring**: Cosine similarity between training and target gradients
- **Selection**: argmax(similarity)
- **Rationale**: Samples with high similarity to target are selected

### MMD Gradient Kernel (mmd_selector.py, line 320-326)

```python
local_selected = self._greedy_mmd_exact(
    candidate_features=train_grads,
    target_features=target_grads,
    num_samples=num_samples,
    sigma=sigma,
    kernel_type=kernel_type_for_select,
)
```

- **Scoring**: Exact marginal greedy MMD
- **Formula (lines 410-419)**: 
  ```
  Δ(x) = r_T(x) - (1/(m+1)) * [r_S(x) + k(x,x)/2]
  ```
  where r_T(x) = target relevance, r_S(x) = selected set redundancy
- **Selection**: argmax(Δ) iteratively

- **Difference**: MMD penalizes redundancy within selected set, LESS does not

---

## 8. LESS_ALIGNED EXPERIMENTS

### Experiment Structure (run_all.sh, experiments/less_aligned/)

```bash
METHODS: random, mmd_emb_rbf, embedding_nn
TARGETS: GSM8K, MMLU
RATIOS: 1%, 5%, 10%
SEEDS: 42, 123, 456
```

### Key Design:
1. **Static selection**: Not dynamic during training
2. **Held-out evaluation**: Test splits separate from target and candidate
3. **Real scale**: 100k+ candidates, 7B model, real benchmarks
4. **Ratio control**: Fixed percentage of candidates selected

### Alignment Issues:
- **No LESS or MMD gradient methods in this script!** 
  - Only: random, mmd_emb_rbf, embedding_nn
  - Missing: mmd_grad_rbf, mmd_grad_cov, LESS
- The less_aligned experiment tests ONLY embedding-based methods
- This is **insufficient for paper comparison** of gradient vs embedding methods

---

## 9. DATA FLOW ANALYSIS: How target_dataset Reaches Selector

### Path 1: LESS Baseline
```
YAML (less_baseline.yaml)
  ↓ eval_dataset: alpaca_zh_demo
  ↓ LlamaFactory loads as eval_dataset
  ↓ SelectTrainer.__init__ (line 249)
  ↓ target_dataset_for_selector = self.eval_dataset
  ↓ REGISTRY.build("selector", "less", runtime={target_dataset: ...})
  ↓ LessSelector.__init__ (line 30)
  ↓ self.target_dataset = target_dataset
  ↓ select() call
  ↓ self.target_dataset used in line 356
```

### Path 2: MMD Gradient
```
YAML (mmd_grad_rbf.yaml)
  ↓ eval_dataset: alpaca_zh_demo
  ↓ LlamaFactory loads as eval_dataset
  ↓ SelectTrainer.__init__ (line 249)
  ↓ target_dataset_for_selector = self.eval_dataset
  ↓ REGISTRY.build("selector", "mmd_grad_rbf", runtime={target_dataset: ...})
  ↓ MMDSelector.__init__ (line 97)
  ↓ self.target_dataset = target_dataset
  ↓ select() call
  ↓ _select_gradient_kernel() (line 254)
  ↓ self.target_dataset used in line 291
```

### Path 3: explicit target_dataset parameter (DynamicFinetuningArguments)

**dynamic_params.py (lines 560-568)**:
```python
target_dataset: Optional[str] = field(
    default=None,
    metadata={
        "help": (
            "Dataset used as the MMD selection target set. "
            "This is ONLY for guiding data selection (e.g., few-shot examples from the target task). "
            "It is NOT used for evaluation metrics. Set eval_dataset separately for actual evaluation."
        )
    },
)
```

**But in SelectTrainer (line 250-254)**:
```python
if hasattr(finetuning_args, 'target_dataset') and finetuning_args.target_dataset:
    # target_dataset is already loaded by LlamaFactory as eval_dataset if
    # the user sets it in YAML. For now, we pass eval_dataset as target.
    pass  # ← Does nothing!
```

**Critical Issue**: The `target_dataset` parameter is defined but **never actually used**!
- If user sets `target_dataset: dataset_name` in YAML, it's ignored
- Still defaults to `self.eval_dataset`

---

## 10. SUMMARY OF MISALIGNMENTS

### Critical Issues (Must Fix for Paper Validity)

| Issue | LESS | MMD | Impact | Severity |
|-------|------|-----|--------|----------|
| **Gradient type** | Adam for candidates, SGD for target | SGD for both | Different gradient representations | 🔴 CRITICAL |
| **Adam preconditioning bug** | Modifies v in-place (corrupts optimizer state) | Fixed: creates new tensor | Numerical instability in LESS | 🔴 CRITICAL |
| **Random projection seed** | 123 | 42 | Different random matrices, breaks reproducibility | 🟠 HIGH |
| **eval_dataset != target_dataset** | Uses eval as target (confusing semantics) | Same | Violates LESS paper protocol | 🟠 HIGH |
| **target_dataset parameter unused** | N/A | N/A | Cannot specify independent target set | 🟠 HIGH |
| **Evaluation leakage** | alpaca_zh_demo used for both selection AND eval | alpaca_zh_demo for selection, no eval | Biased evaluation metrics | 🟠 HIGH |

### Medium Issues (Design Differences)

| Issue | LESS | MMD | Impact |
|-------|------|-----|--------|
| **Selection scoring** | Cosine similarity | Exact marginal greedy MMD | Different selection criteria |
| **Redundancy penalty** | No | Yes (k(x,x) term) | MMD prefers diversity, LESS doesn't |
| **Dynamic selection protocol** | ✓ | ✓ | Both match |

### Low Issues (Configuration)

| Issue | LESS | MMD | Impact |
|-------|------|-----|--------|
| **Projection dimension** | 4096 | 4096 | ✓ Match |
| **Save interval** | 16 | 16 | ✓ Match |
| **Training hyperparams** | All match | | ✓ Match |

---

## RECOMMENDATIONS

### 1. Fix Gradient Type (Priority 1)
- **Change components.yaml**: mmd_grad_rbf and mmd_grad_cov should use `gradient_type: adam`
- **Verify LESS**: Check if LESS should use Adam for target_dataset too
- **Code change**: mmd_selector.py already supports Adam, just need config fix

### 2. Fix Random Seed (Priority 1)
- **Change components.yaml**: MMD variants should use `seed: 123` to match LESS
- **Enables fair comparison** of gradient projection matrices

### 3. Fix target_dataset Handling (Priority 1)
- **Implement target_dataset parameter**: Actually read `finetuning_args.target_dataset`
- **Separate from eval_dataset**: Only use target_dataset for selection, eval_dataset for metrics
- **Update SelectTrainer**: 
  ```python
  target_dataset_for_selector = self.eval_dataset
  if hasattr(finetuning_args, 'target_dataset') and finetuning_args.target_dataset:
      # Load target_dataset separately
      target_dataset_for_selector = load_target_dataset(finetuning_args.target_dataset)
  ```

### 4. Fix Adam Preconditioning Bug (Priority 1)
- **LESS**: Use MMD's fixed implementation (non-in-place mutation)
- **Prevents optimizer state corruption**

### 5. Fix Evaluation Protocol (Priority 2)
- **LESS baseline**: Remove alpaca_zh_demo from eval_dataset OR use separate held-out test set
- **MMD configs**: If enabling evaluation, use separate test set
- **Update run_all.sh**: Include gradient-based methods (LESS, mmd_grad_rbf, mmd_grad_cov)

### 6. Add Seed Tracking (Priority 2)
- **Document**: Why different seeds (42 vs 123) are used
- **Standardize**: Use 42 for all methods OR 123 for all methods
- **Report**: Seed values in experiment logs

### 7. Add Ablation Studies (Priority 3)
- **Gradient type**: Adam vs SGD impact on selection
- **Seed sensitivity**: How sensitive is MMD to random projection seed
- **Lambda/sigma**: Document hyperparameter choices

---

## CONCLUSION

The MMD implementation is **well-engineered** (fixes Adam bug, implements exact greedy MMD) but **configuration misalignment** with LESS baseline makes **direct comparison invalid**:

1. ✗ Different gradient types (Adam vs SGD)
2. ✗ Different random seeds (123 vs 42)
3. ✗ Evaluation data leakage
4. ✗ target_dataset parameter not implemented

**For paper publication**: Fix all Priority 1 items before running experiments.


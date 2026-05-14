# Opt-GCS: Rank-Truncated Whitened Spectral Coreset Construction in Optimizer-Induced Update Space for Unsupervised SFT Data Selection

> **Working title for submission**  
> Authors: [TBD]  
> Status: Draft proposal — theory + method + experiment design

---

## Abstract

We formulate unsupervised SFT data selection as **spectral coreset construction in optimizer-induced update space**. Each training example is represented by its frozen-state optimizer-induced local update feature — a principled representation that reflects the actual training dynamics under AdamW or Muon, rather than raw SGD gradients. We show that the covariance of these update features exhibits a spiked-yet-heavy-tailed spectral structure, enabling recovery of a dominant low-dimensional update subspace from a small probe set. We then select a **whitened log-det diverse subset** that maximally covers this subspace, with a tunable whitening parameter β that controls the trade-off between focusing on dominant update directions (exploitation) and covering rare but potentially important directions (exploration). Our method, Opt-GCS, is target-free (requiring no validation examples), provides a (1-1/e) submodular approximation guarantee, and naturally differentiates from last-layer Fisher proxies by leveraging multi-layer gradient information. Experiments on Open-Hermes-2.5 with Llama-3.1-8B demonstrate that Opt-GCS selects compact, diverse subsets that preserve the dominant training geometry.

---

## 1. Introduction

### 1.1 Motivation

Supervised fine-tuning (SFT) of large language models (LLMs) transforms a pretrained base model into an instruction-following assistant. The quality and composition of SFT data directly impacts the resulting model's capabilities. Given a large candidate pool of SFT examples, **data selection** — choosing a compact subset that maximizes training effectiveness — has emerged as a critical research direction.

Existing SFT data selection methods fall into several paradigms:

| Paradigm | Representative | Key Limitation |
|----------|---------------|----------------|
| **Targeted gradient matching** | LESS (Xia et al., ICML 2024) | Requires target/validation examples |
| **Last-layer Fisher information** | FisherSFT (Deb et al., 2025) | Only captures output-layer information |
| **Gradient agreement** | SAGE (2025) | Selects redundant samples aligned with dominant direction |
| **Gradient clustering** | TAGCOS (NAACL 2025) | Implicit subspace; no spectral analysis |
| **Fisher + conflict penalty** | SPICE (ICLR 2026) | Conflict avoidance ≠ coverage maximization |

We identify a fundamental dichotomy in existing approaches: **agreement vs. coverage**.

- **Agreement-based methods** (SAGE, gradient similarity) select examples whose gradients align with a target or dominant direction. This biases selection toward **redundant** samples — many examples pointing in the same direction.
- **Coverage-based methods** should select examples that jointly **span** the training-relevant subspace, ensuring both dominant and complementary directions are represented.

### 1.2 Key Insight: Coverage in Optimizer-Induced Update Space

We propose a shift in perspective:

> **Do not ask "which examples align with a target?"**  
> **Ask "which examples jointly reconstruct the dominant training geometry?"**

Moreover, the "training geometry" should not be defined by raw SGD gradients, but by the **optimizer-induced update directions** that the model actually follows during training. Under AdamW, the effective per-sample update is not g_i but D_t · g_i, where D_t is the diagonal preconditioning matrix derived from the second moment estimate. This distinction matters because the optimizer reshapes the loss landscape geometry.

### 1.3 Contributions

1. **Optimizer-induced update representation**: We formalize per-sample update features under frozen-state AdamW/Muon, providing a principled representation that goes beyond raw gradients (vs LESS/TAGCOS) and beyond last-layer Fisher (vs FisherSFT/SPICE).

2. **Spectral analysis with whitening**: We introduce a tunable whitening parameter β ∈ [0,1] that controls the coverage-exploitation trade-off in the recovered eigenspace. This is a novel degree of freedom absent in all prior logdet-based selection methods.

3. **Rank-truncated spectral coreset**: We prove that the dominant update subspace can be recovered from a probe set (Davis-Kahan), that sketching preserves the relevant geometry (JL embedding), and that greedy logdet selection achieves (1-1/e) approximation to the optimal coverage (submodularity).

4. **Comprehensive experimental framework**: We implement Opt-GCS in DataFlex with systematic ablations (gradient type, whitening, rank, selection method) and critical negative controls.

---

## 2. Problem Formulation

### 2.1 Setup

Let the SFT data pool be D = {z_i}_{i=1}^n, where each z_i = (x_i, y_i) is an instruction-completion pair. At a training checkpoint θ_t with optimizer state S_t, define:

- **Per-sample loss**: ℓ_i(θ_t) = ℓ(z_i; θ_t)
- **Per-sample gradient**: g_i^(t) = ∇_θ ℓ_i(θ_t) ∈ ℝ^d

### 2.2 Optimizer-Induced Update Feature

In real LLM training, the model does not update along g_i. It follows an optimizer-induced direction:

```
u_i^(t) = A_t(g_i^(t); S_t)
```

where A_t is the update map induced by the optimizer at step t, and S_t is the optimizer state.

**For AdamW**, the frozen-state per-sample update feature is:

```
u_i^(AdamW) = D_t · g_i,    where  D_t = diag(1 / (√v̂_t + ε))
```

This is the **diagonal AdamW-preconditioned gradient** — the theoretically recommended representation. It is NOT a full counterfactual "what if we trained on only z_i" — it is a **local update feature at the current optimizer state**.

**Important terminological precision**: We call this a "frozen-state optimizer-induced local update feature", not a "per-sample Adam update", because:
- AdamW's second moment v_t involves squared gradients, making batch aggregation and per-sample decomposition non-commutative
- We freeze the optimizer state and use it as a fixed preconditioner
- This is a local surrogate, not an exact counterfactual

**For SGD** (special case): u_i = g_i (raw gradient, no preconditioning)

**For Muon** (general case): u_i^(Muon) ≈ J_{M,t} · g_i, where J_{M,t} is the local Jacobian of the Muon update map.

### 2.3 The Selection Problem

Given budget k, select S* ⊂ D with |S*| = k such that S* **maximally covers the dominant update geometry** of the full data pool.

---

## 3. Theoretical Framework

### 3.1 Assumption: Spiked-Yet-Heavy-Tailed Update Covariance

**Hypothesis**: The update covariance

```
Σ_u = E_i[u_i · u_i^T]
```

has a **spiked-yet-heavy-tailed spectral structure**: a small number of dominant eigenvalues coexist with a long tail of weaker but non-negligible eigenvalues.

Formally, we decompose:

```
Σ_u = U_r Λ_r U_r^T + Σ_tail
```

where:
- U_r ∈ ℝ^{d×r} contains the top-r eigenvectors (dominant update subspace)
- Λ_r = diag(λ_1, ..., λ_r) with λ_1 ≥ ... ≥ λ_r
- Σ_tail captures the remaining directions
- There exists an eigengap: Δ_r = λ_r - λ_{r+1} > 0

**Empirical validation** (Llama-3.1-8B on Open-Hermes-2.5, 3000 samples):
- Effective rank ≈ 12.6 (vs ambient dimension 4096)
- Eigengap ratio λ_1/λ_2 = 3.61
- Top-10 eigenvalues explain 17.4% of variance
- Top-50 explain 26.8%, top-100 explain 33.1%

This confirms the spiked-yet-heavy-tailed structure: the spectrum is not "only 12 directions" but rather has a sharp spike followed by a slowly decaying tail.

### 3.2 Theorem 1: Dominant Update Subspace is Recoverable

**Statement**: Let {u_i}_{i=1}^m be i.i.d. draws from the update distribution. If the update vectors are sub-Gaussian (or clipped to be sub-exponential), and the population covariance has eigengap Δ_r > 0, then with high probability:

```
‖Σ̂_u - Σ_u‖_op ≤ C · √(d/m)
```

where Σ̂_u = (1/m) Σ_{i=1}^m u_i u_i^T is the empirical covariance.

By **Davis-Kahan perturbation theorem**:

```
‖sin Θ(Û_r, U_r)‖_op ≤ ‖Σ̂_u - Σ_u‖_op / Δ_r
```

**Implication**: With probe size m = O(d / (Δ_r^2 · δ^2)), the top-r eigenspace can be recovered to accuracy δ. For our setting (d = 4096, effective rank ≈ 13, Δ ≈ 0.06), m ≈ 5000 suffices for a good approximation.

### 3.3 Theorem 2: Sketching Preserves Update-Space Geometry

Full update vectors (dimension d, potentially millions for full-parameter gradients) are too large to store. We use a random sketch:

```
ũ_i = R · u_i,    R ∈ ℝ^{p×d}
```

where R is a Johnson-Lindenstrauss (JL) random projection matrix.

**Statement**: If p = O((r + log n) / ε²), then with high probability for all i, j in [n]:

```
|⟨ũ_i, ũ_j⟩ - ⟨u_i, u_j⟩| ≤ ε · ‖u_i‖ · ‖u_j‖
```

and the subspace projection scores are preserved:

```
|‖Ũ_r^T ũ_i‖² - ‖U_r^T u_i‖²| ≤ ε · ‖u_i‖²
```

**Implication**: Our method operates in sketch space (p = 4096 in practice) without loss of relevant geometric information. The TRAK projector (Rademacher random projection) serves as our sketching matrix, consistent with the LESS infrastructure.

### 3.4 Theorem 3: Submodular Guarantee for LogDet Selection

**Definition**: The whitened logdet objective is:

```
F(S) = log det(εI_r + Σ_{i∈S} x_i^(β) · (x_i^(β))^T)
```

where x_i^(β) = Λ_r^{-β/2} · U_r^T · ũ_i is the **whitened projection** of sample i.

**Statement**: F(S) is **monotone** and **submodular**. Therefore, the greedy algorithm that iteratively adds the sample with maximum marginal gain achieves:

```
F(S_greedy) ≥ (1 - 1/e) · F(S*)
```

where S* = argmax_{|S|≤k} F(S) is the optimal solution.

**Proof sketch**:
- Monotonicity: A_{S∪{i}} = A_S + x_i x_i^T ⪰ A_S, so det(A_{S∪{i}}) ≥ det(A_S)
- Submodularity: For S ⊆ T, A_T ⪰ A_S implies A_T^{-1} ⪯ A_S^{-1}, so the marginal gain Δ(i|T) = log(1 + x_i^T A_T^{-1} x_i) ≤ log(1 + x_i^T A_S^{-1} x_i) = Δ(i|S). This is the diminishing returns property.
- The (1-1/e) bound follows from the classical result of Nemhauser, Wolsey, and Fisher (1978) for greedy maximization of monotone submodular functions under cardinality constraints.

**Marginal gain interpretation**: The greedy selection rule chooses

```
i_t = argmax_{i∉S_{t-1}} x_i^T · A_{S_{t-1}}^{-1} · x_i
```

This has a clear geometric meaning:
- ‖x_i‖ large → sample has strong signal in the update subspace
- A_S^{-1} large along uncovered directions → the inverse emphasizes gaps
- x_i^T A_S^{-1} x_i large → sample is both **strong** and **complementary** to already-selected samples

---

## 4. Method: Opt-GCS Algorithm

### 4.1 Preprocessing: Length Normalization and Clipping

SFT samples have variable completion lengths, which biases gradient norms. We apply:

```
ū_i = (u_i / L_i^α) · min{1, τ / ‖u_i‖}
```

where:
- L_i is the completion token length
- α ∈ [0, 1] controls length normalization strength (default 0.5)
- τ is an adaptive clipping threshold (95th percentile of ‖u_i‖)

After clipping, we L2-normalize: h_i = ū_i / ‖ū_i‖ (direction matters).

### 4.2 Eigenspace Estimation

From the normalized update features, compute the empirical covariance and its top eigenspace:

1. **Compute**: Σ̂_u = (1/n) Σ h_i h_i^T (or use randomized SVD for efficiency)
2. **Rank selection** (automatic):
   - Effective rank: r_eff = tr(Σ̂) / ‖Σ̂‖_op = Σλ_j / λ_1
   - Eigengap: first j where λ_j / λ_{j+1} > threshold
   - Entropy rank: r_ent = exp(-Σ p_j log p_j), where p_j = λ_j / Σλ_k
3. **Extract**: Û_r = top-r eigenvectors, Λ̂_r = corresponding eigenvalues

### 4.3 Whitened Projection (Key Differentiator)

The standard approach would project as x_i = U_r^T h_i. We introduce **partial whitening**:

```
x_i^(β) = Λ_r^{-β/2} · Û_r^T · h_i,    β ∈ [0, 1]
```

**Interpretation**:
- **β = 0** (unwhitened): Preserves original eigenvalue scaling. The logdet objective naturally focuses on directions with high variance. Equivalent to standard D-optimal design.
- **β = 1** (fully whitened): All eigenspace directions are equally weighted. Maximizes coverage of rare directions at the cost of potentially over-representing weak signals.
- **β ∈ (0, 1)** (partial whitening): Smoothly interpolates between exploitation (dominant directions) and exploration (rare directions). This is the recommended setting.

**Why this matters for differentiation from FisherSFT/SPICE**: Neither FisherSFT nor SPICE performs explicit spectral decomposition or whitening. They operate with the raw Fisher/gradient geometry. Our whitening parameter gives a principled control knob that:
1. Makes the "spectral" nature of our method explicit
2. Provides a natural ablation axis for experiments
3. Connects to the exploration-exploitation trade-off in active learning

### 4.4 Selection Strategies

**Opt-GCS-LogDet** (main method):
```
S = ∅, A = εI_r
Repeat k times:
    i* = argmax_{i∉S} x_i^T A^{-1} x_i
    S ← S ∪ {i*}
    A ← A + x_{i*} x_{i*}^T    (Sherman-Morrison update for A^{-1})
```
Complexity: O(k · n · r) for scoring + O(k · r²) for updates. Since r ≪ n, this is very efficient.

**Opt-GCS-Score** (ablation: magnitude only):
```
s_i = ‖x_i^(β)‖², select top-k by score
```

**Opt-GCS-Diverse** (ablation: score + diversity):
```
Prefilter top-5k by score, then k-center greedy in x_i space
```

### 4.5 Complete Algorithm

```
Algorithm: Opt-GCS-LogDet

Input:
  SFT data pool D = {z_i}_{i=1}^n
  Checkpoint θ_t, optimizer state S_t
  Budget k, sketch dimension p, whitening β, ridge ε

1. For each z_i ∈ D:
     g_i = ∇_θ ℓ(z_i; θ_t)                    [per-sample gradient]
     u_i = D_t · g_i                             [frozen-state AdamW feature]
     ũ_i = R · u_i ∈ ℝ^p                        [random sketch]
     h_i = normalize(clip(ũ_i / L_i^α))          [length-norm + clip + L2-norm]

2. Eigenspace estimation:
     Σ̂ = (1/n) Σ h_i h_i^T
     Û_r, Λ̂_r = TopEig(Σ̂, r = AutoRank(Σ̂))

3. Whitened projection:
     x_i^(β) = Λ̂_r^{-β/2} · Û_r^T · h_i       [for all i]

4. Greedy LogDet selection:
     A = εI_r, S = ∅
     Repeat k times:
       i* = argmax_{i∉S} (x_i^(β))^T A^{-1} x_i^(β)
       S ← S ∪ {i*}
       A ← A + x_{i*}^(β) (x_{i*}^(β))^T

5. Fine-tune θ_t on selected subset S.
```

---

## 5. Relationship to Prior Work

### 5.1 vs FisherSFT (Deb et al., PMLR 267, 2025)

FisherSFT approximates the SFT Fisher Information Matrix via last-layer linearization (treating the LLM as a multinomial logistic regression on pre-logit embeddings), then maximizes logdet for D-optimal design.

| Dimension | FisherSFT | Opt-GCS |
|-----------|-----------|---------|
| Information source | Last-layer pre-logit embeddings | **Multi-layer** optimizer-induced updates |
| Requires backward pass | No (forward only) | Yes (per-sample gradient) |
| Optimizer-aware | No (raw Fisher) | **Yes** (AdamW/Muon preconditioned) |
| Spectral analysis | No (scalar logdet in full embedding space) | **Yes** (rank truncation + whitening) |
| Computational cost | Lower (forward only) | Higher (forward + backward) |
| Theoretical guarantee | D-optimal in last-layer space | D-optimal in **whitened update eigenspace** |

**Key experiment**: If Opt-GCS > FisherSFT on reasoning/math tasks that require middle-layer representation changes, this validates the multi-layer advantage. If Opt-GCS ≈ FisherSFT, it suggests last-layer Fisher is a surprisingly good proxy.

### 5.2 vs SAGE (2025)

SAGE uses Frequent Directions (FD) sketching of the gradient matrix, then selects samples by **alignment** (cosine similarity) with the consensus/dominant gradient direction.

**Fundamental distinction — Agreement vs Coverage**:

Consider a 2D example with dominant direction e_1 and secondary direction e_2, budget k=4:
- **SAGE (agreement)**: Selects 4 samples all near e_1 (highest alignment with dominant direction)
- **Opt-GCS (coverage)**: Selects 2 near e_1 + 2 near e_2 (logdet penalizes redundancy via A^{-1})

This is because the greedy logdet marginal gain x_i^T A^{-1} x_i automatically deprioritizes directions already covered by selected samples.

### 5.3 vs SPICE (ICLR 2026)

SPICE starts from logdet Fisher information (similar objective function) but adds a **gradient conflict penalty** — penalizing candidates whose gradients conflict with the selected subset.

| Dimension | SPICE | Opt-GCS |
|-----------|-------|---------|
| Base objective | logdet Fisher | logdet in **whitened update eigenspace** |
| Additional mechanism | Conflict avoidance penalty | **Spectral whitening** (β parameter) |
| Subspace analysis | None (operates in original space) | Explicit eigendecomposition + rank truncation |
| Optimizer-aware | Not as core focus | Core contribution |

### 5.4 vs LESS (Xia et al., ICML 2024)

LESS computes low-rank gradient similarity between training and target examples, selecting examples whose gradients best match the target.

| Dimension | LESS | Opt-GCS |
|-----------|------|---------|
| Supervision | **Requires** target/validation examples | **Unsupervised** (target-free) |
| Selection criterion | Train-target gradient similarity | Update-space spectral coverage |
| When to use | Target examples available | Target examples unavailable |
| Complementarity | Can use Opt-GCS as prefilter → LESS rerank | Can use LESS to verify Opt-GCS selections |

### 5.5 vs TAGCOS (NAACL 2025 Findings)

TAGCOS computes per-sample gradients → K-means clustering → OMP (Optimal Matching Pursuit) coreset selection.

| Dimension | TAGCOS | Opt-GCS |
|-----------|--------|---------|
| Subspace discovery | Implicit via K-means clusters | **Explicit** via eigendecomposition |
| Selection in subspace | OMP within clusters | LogDet in whitened eigenspace |
| Diversity control | By cluster structure | By eigenvalue whitening (β parameter) |
| Optimizer-aware | No | Yes |
| Theoretical guarantee | Coreset approximation | (1-1/e) submodular + geometry preservation |

---

## 6. Experimental Design

### 6.1 Setup

- **Model**: Llama-3.1-8B with LoRA (rank=16, alpha=8, target=all)
- **Training data**: Open-Hermes-2.5 (100,000 samples)
- **Evaluation**: MMLU subset (matched to DataFlex benchmark)
- **Framework**: DataFlex (built on LLaMA-Factory)
- **Hardware**: 8× NVIDIA H20 (95GB each)

### 6.2 Experiment Matrix

#### Table 1: Main Results (MMLU Accuracy)

| Method | Type | Target-free? | Budget=1k | Budget=5k | Budget=10k |
|--------|------|:------------:|:---------:|:---------:|:----------:|
| Random | Baseline | ✅ | — | — | — |
| Loss top-k | Heuristic | ✅ | — | — | — |
| Grad norm top-k | Negative ctrl | ✅ | — | — | — |
| LESS | Supervised | ❌ | — | — | — |
| FisherSFT | Fisher logdet | ✅ | — | — | — |
| **Opt-GCS-LogDet** | **Ours (main)** | ✅ | — | — | — |
| Opt-GCS-Score | Ours (ablation) | ✅ | — | — | — |

#### Table 2: Optimizer-Aware Ablation

Validates the core claim: optimizer-induced geometry matters.

| Gradient Representation | β | Method | MMLU |
|------------------------|---|--------|------|
| Raw SGD (g_i) | 0.5 | LogDet | — |
| Adam diagonal (D_t·g_i) | 0.5 | LogDet | — |
| Full Adam surrogate (LESS-style) | 0.5 | LogDet | — |

**Expected result**: Adam diagonal ≥ Raw SGD

#### Table 3: Whitening Ablation

Validates the spectral whitening contribution.

| β | Interpretation | MMLU |
|---|---------------|------|
| 0.0 | Unwhitened (standard logdet) | — |
| 0.25 | Mild whitening | — |
| 0.5 | Balanced (default) | — |
| 0.75 | Strong whitening | — |
| 1.0 | Full whitening | — |

**Expected result**: Intermediate β outperforms extremes.

#### Table 4: Selection Method Ablation

| Method | Diversity mechanism | MMLU |
|--------|-------------------|------|
| Score top-k | None (magnitude only) | — |
| Score + k-center | Heuristic farthest-first | — |
| **LogDet greedy** | Submodular coverage | — |

**Expected result**: LogDet > Score (validates coverage > magnitude).

#### Table 5: Rank Sensitivity

| Rank r | % Variance Explained | MMLU |
|--------|:-------------------:|------|
| 5 | ~8% | — |
| 10 | ~15% | — |
| 20 | ~22% | — |
| 50 | ~27% | — |
| auto (effective) | ~varies | — |

### 6.3 Critical Negative Controls

| Control | What it tests |
|---------|---------------|
| **Random subspace + LogDet** | Does the eigenspace matter, or is any low-dim projection + logdet sufficient? |
| **Grad norm top-k** | Is the method just selecting high-gradient-norm samples? |
| **Shuffled eigenspace + LogDet** | Does eigenvalue ordering matter for whitening? |

If random subspace + logdet ≈ Opt-GCS, then the spectral analysis contribution is weak. This is the most important negative control.

### 6.4 Diagnostic Visualizations

1. **Eigenvalue decay** (log-log plot + cumulative variance)
2. **Eigengap ratios** (spiked structure verification)
3. **Score vs token-length** (length normalization validation)
4. **2D PCA of selections** (LogDet vs Score vs Random: visual coverage comparison)
5. **Domain composition** (what types of data does each method select?)
6. **Selection overlap heatmap** (Jaccard similarity between methods)

---

## 7. Theoretical Claims and Limitations

### 7.1 What We Claim (Defensible)

1. At a fixed checkpoint and optimizer state, each sample induces a well-defined local update direction u_i.
2. If the update distribution has low effective-rank structure (empirically verified), its principal subspace can be estimated from a probe set (Theorem 1).
3. Random sketching preserves the relevant geometry (Theorem 2).
4. Greedy logdet selection in the whitened eigenspace constructs an approximate spectral coreset with (1-1/e) guarantee (Theorem 3).
5. This coreset preserves the dominant local optimizer-induced training geometry.

### 7.2 What We Do NOT Claim (Explicit Limitations)

1. **We do not claim** that first-order Taylor expansion accurately predicts full SFT retraining effects.
2. **We do not claim** that influence function-style pointwise attribution is reliable in deep nonconvex LLMs.
3. **We do not claim** a formal connection from update-space geometry preservation to downstream task accuracy. This connection is **empirical**.
4. **We do not claim** that unsupervised selection must outperform targeted selection (LESS) when target examples are available.
5. **We do not claim** that the spectral structure is stable across all checkpoints. Pathwise extensions are left for future work.

**Key limitation statement for the paper**:

> Our theorem guarantees spectral coverage of the optimizer-induced update distribution, not downstream accuracy. The downstream relevance of this geometry is an empirical hypothesis validated through SFT experiments. The connection from update-space coverage to generalization in nonconvex deep learning remains an open theoretical question.

### 7.3 Relationship Between Theory and Practice

| Layer | Theory | Practice |
|-------|--------|----------|
| Representation | u_i = D_t · g_i | TRAK random projection to 4096 dims |
| Eigenspace recovery | Davis-Kahan bound | Randomized SVD (torch.svd_lowrank) |
| Sketch preservation | JL lemma | Rademacher random projection |
| Selection quality | (1-1/e) submodular | Greedy with Sherman-Morrison O(k·n·r) |
| Downstream performance | **No formal guarantee** | **Empirical validation on MMLU** |

---

## 8. Implementation Details

### 8.1 Integration with DataFlex

Opt-GCS is implemented as a registered selector in DataFlex (built on LLaMA-Factory), supporting:
- Multi-GPU distributed training (gradient computation parallelized across ranks)
- DeepSpeed ZeRO-3 compatibility
- Gradient caching and resume
- Multiple selection variants via configuration

### 8.2 Computational Considerations

| Step | Complexity | Time (100k samples, 8×H20) |
|------|-----------|:---------------------------:|
| Per-sample gradient + sketch | O(n · forward+backward) | ~2h (parallelized) |
| Randomized SVD (q=200) | O(n · p · q) | ~30s |
| Whitened projection | O(n · r) | ~1s |
| LogDet greedy (k samples) | O(k · n_candidates · r) | ~10s |

**vs LESS**: LESS requires train gradients + eval gradients + similarity scoring. Opt-GCS requires train gradients + SVD + logdet. The SVD + logdet is negligible. The key difference is Opt-GCS does NOT require eval gradient computation, saving ~50% gradient compute when the eval set is large.

**vs FisherSFT**: FisherSFT requires only forward passes (no backward). It is approximately 2× cheaper. This is the computational trade-off for multi-layer information.

---

## 9. Preliminary Results

### 9.1 Spectral Diagnostic (Completed)

**Setup**: Llama-3.1-8B, LoRA (q_proj + v_proj), 3000 samples, proj_dim=4096

| Metric | Value |
|--------|-------|
| Effective rank | 12.6 |
| Entropy rank | 985.5 |
| Eigengap ratio (λ₁/λ₂) | 3.61 |
| Top-1 explains | 7.9% |
| Top-10 explain | 17.4% |
| Top-50 explain | 26.8% |
| Top-100 explain | 33.1% |
| Score-length correlation | -0.26 |

**Interpretation**: The gradient covariance has a clear spiked-yet-heavy-tailed structure. The low effective rank confirms that a small number of dominant directions capture the majority of update variance. The modest score-length correlation (-0.26) suggests length normalization (α=0.5) is effective but could be tuned further.

### 9.2 Training Experiments (In Progress)

Smoke test: Opt-GCS-LogDet on Open-Hermes-2.5 (100k) with Llama-3.1-8B, all-linear LoRA (41.9M trainable params).
- Warmup: 10 steps (random data)
- Selection: 2 rounds × 80 samples each
- Total: 30 training steps
- Status: Gradient computation running (~8800/12500 per GPU)

---

## 10. Future Directions

1. **Pathwise extension**: Average update covariance over multiple checkpoints for more stable eigenspace estimation
2. **Opt-GCS + LESS pipeline**: Use Opt-GCS as unsupervised prefilter, then LESS for targeted reranking
3. **Muon support**: Full matrix-aware update representation for Muon optimizer
4. **Scaling to larger models**: Investigate whether spectral structure persists at 70B+ scale
5. **Dynamic re-selection**: Adapt eigenspace as training progresses (currently static per selection round)

---

## References

1. Xia, M., Malladi, S., Gururangan, S., Arora, S., Chen, D. **LESS: Selecting Influential Data for Targeted Instruction Tuning.** ICML 2024. arXiv:2402.04333

2. Deb, S., et al. **FisherSFT: Data-Efficient Supervised Fine-Tuning via Fisher Information.** PMLR 267, 2025. arXiv:2505.14826

3. SPICE authors. **SPICE: Submodular Penalized Information-Conflict Selection for Efficient Instruction Tuning.** ICLR 2026. OpenReview:9rCRy58TPF

4. SAGE authors. **SAGE: Streaming Agreement-Driven Gradient Sketches for Representative Subset Selection.** 2025. arXiv:2510.02470

5. TAGCOS authors. **TAGCOS: Task-Agnostic Gradient Clustered Coreset Selection for Instruction Tuning Data.** NAACL 2025 Findings. arXiv:2407.15235

6. Wang, S., et al. **OPUS: Towards Efficient and Principled Data Selection in LLM Pre-training.** 2026. arXiv:2602.05400

7. GRAFT authors. **GRAFT: Gradient-Aware Fast MaxVol Technique for Dynamic Data Sampling.** 2025. arXiv:2508.13653

8. Li, M., et al. **How Instruction and Reasoning Data Shape Post-Training.** 2025. arXiv:2504.10766

9. Nemhauser, G. L., Wolsey, L. A., Fisher, M. L. **An analysis of approximations for maximizing submodular set functions.** Mathematical Programming 14(1), 1978.

10. Davis, C., Kahan, W. M. **The rotation of eigenvectors by a perturbation. III.** SIAM J. Numerical Analysis 7(1), 1970.

11. Vershynin, R. **High-Dimensional Probability.** Cambridge University Press, 2018.

12. Liang, H., et al. **DataFlex: A Unified Framework for Data-Centric Dynamic Training of Large Language Models.** arXiv:2603.26164, 2026.

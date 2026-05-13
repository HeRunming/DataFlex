# Opt-GCS 论文深度 Review 补充 & 修改方案 (v2)

> 日期: 2025-05-14  
> 基于: GPT 深度 review + Claude 初始 review + 12 篇近邻工作分析  
> 结论: **novelty pressure 比预期更强，需要方法升级 + 实验扩充**

---

## 核心判断

**当前最大风险不是理论错误，而是 novelty 不足。**

以下三个工作已经占据了关键 claim：

| 已有工作 | 已占据的 Claim | 出处 |
|---------|-------------|------|
| **FisherSFT** | logdet maximization + unsupervised SFT selection | ICML 2025 workshop / PMLR 267 |
| **SPICE** | log-det Fisher + submodular greedy + (1-1/e) guarantee + instruction tuning | ICLR 2026 poster |
| **TAGCOS** | unsupervised + sample gradients as representations + coreset + instruction tuning | NAACL 2025 Findings |

因此，"logdet + gradient + unsupervised" 这个组合**已经不新**。我们的方法必须钉死在以下差异化：

> 1. **Optimizer-induced update covariance spectrum**（不是 raw Fisher / raw gradient）
> 2. **Rank-truncated + whitened spectral coreset**（不是普通 logdet）
> 3. **Coverage-vs-Agreement 可诊断验证**
> 4. **与 Fisher/logdet last-layer proxy 的层级差异**

---

## 一、方法层面的必要修改

### 1.1 从普通 LogDet 升级为 Whitened Spectral LogDet

**问题**：如果我们只做 `x_i = U_r^T u_i` 然后 logdet，数学形式与 FisherSFT/SPICE 太像。

**修改**：引入 partial whitening parameter β ∈ [0,1]：

```
x_i^(β) = Λ_r^{-β/2} · U_r^T · u_i
```

然后选择目标变为：

```
S = argmax_{|S|=k} log det(εI + Σ_{i∈S} x_i^(β) (x_i^(β))^T)
```

**语义**：
- β = 0：保留原始方差强度（unwhitened），偏向主方向 → 更像 agreement
- β = 1：完全 whitening，每个主方向等权 → 最强 coverage
- β ∈ (0,1)：折中，coverage-agreement tradeoff

**这是核心差异化**：我们不只是做 logdet，而是在 optimizer-induced eigenspace 中做**可调谱白化的 coverage selection**。这比 FisherSFT（无谱分析）和 SPICE（无 whitening）更有"谱分析"的味道。

### 1.2 AdamW 术语修正

**不要说**："full Adam per-sample update"

**应该说**："frozen-state optimizer-induced local update feature"

严格表述：
```
u_i = D_t · g_i, where D_t = diag(1 / (√v̂_t + ε))
```

这是在**固定 optimizer state**下的 local preconditioned gradient feature，不是完整 AdamW 的反事实单样本 update。

AdamW 的 v_t 涉及平方项，batch aggregation 和 per-sample decomposition 不可交换。所以不能声称这是 "per-sample Adam update"。

### 1.3 推荐的方法命名

**主标题**：Opt-GCS: Rank-Truncated Spectral Coreset Selection in Optimizer-Induced Update Space

**方法定义**（一句话）：
> Opt-GCS is rank-truncated, optionally whitened spectral coreset construction in frozen-state optimizer-induced update space.

---

## 二、Effective Rank 现象的正确解释

### 当前数据

```
Llama-3.1-8B (LoRA q_proj+v_proj, 3000 samples, proj_dim=4096):
  Effective rank: 12.6
  Top-1 explains: 7.9%
  Top-10 explains: 17.4%
  Top-50 explains: 26.8%
  Top-100 explains: 33.1%
```

### 正确解释

这个看似矛盾：effective rank 低（12.6），但 cumulative variance 并不集中。

**正确的描述是**：

> The update spectrum exhibits a **spiked-yet-heavy-tailed** structure: a small number of dominant directions coexist with a long tail of weaker but potentially meaningful directions.

**不要写**："只有 12 个方向包含主信号"

**应该写**：
- Top eigenvalues 明显突出（eigengap = 3.61）
- 但 long tail 贡献不可忽略（top-100 仍只解释 33%）
- 这说明 whitening (β > 0) 可能是必要的——给 tail directions 更多权重

### 需要的实验验证

| 实验 | 目的 |
|------|------|
| Rank sensitivity: r = 5, 10, 20, 50, 100, 200 | 多少方向够？ |
| Whitening sensitivity: β = 0, 0.25, 0.5, 0.75, 1.0 | Whitening 是否帮助？ |
| Rare-domain retention analysis | Whitening 是否保留了 rare but useful 样本？ |
| Top-r cumulative variance vs downstream score | 谱覆盖与性能的关系 |

---

## 三、必须正面处理的近邻工作

### 3.1 FisherSFT (PMLR 267, 2025)

**核心**：通过最大化 SFT objective 的 Fisher information (last-layer linearization) 做 logdet greedy selection。

**我们的差异化**：
1. 全层 gradient 信息 vs last-layer embedding
2. Optimizer-aware (AdamW preconditioned) vs raw Fisher
3. 显式谱分解 + rank truncation + whitening vs scalar logdet
4. 可诊断的谱结构分析

**关键实验**：如果 last-layer Fisher 效果与全层 Opt-GCS 相当，那么：
> "Opt-GCS reveals that FisherSFT works because last-layer Fisher is a good low-cost proxy for optimizer-induced update geometry."

这个 fallback narrative 可以保底。

### 3.2 SPICE (ICLR 2026)

**核心**：log-det Fisher + gradient conflict penalty + submodular greedy。

**我们的差异**：
- SPICE = logdet + conflict avoidance penalty
- Opt-GCS = spectral truncation + whitened logdet coverage
- SPICE 不做谱分析，不做 rank discovery，不做 whitening

### 3.3 TAGCOS (NAACL 2025 Findings)

**核心**：unsupervised + sample gradients → K-means clustering → OMP coreset selection。

**我们的差异**：
- TAGCOS = clustering coreset（implicit subspace）
- Opt-GCS = spectral coreset（explicit eigenspace + logdet）
- TAGCOS 不 optimizer-aware，不做谱分析

### 3.4 SAGE (2025)

**核心**：Frequent Directions gradient sketch → alignment with consensus direction → top-k。

**我们的关键对比（paper 核心 narrative）**：
- SAGE = agreement（选与主方向对齐的样本 → 冗余）
- Opt-GCS = coverage（选覆盖子空间的样本 → 多样性）

---

## 四、更新后的实验 Baseline 表

| 类别 | 方法 | 必要性 | 实现难度 |
|------|------|:------:|:--------:|
| **Lower bound** | Random | 必须 | ✅ 已有 |
| **Lower bound** | Length-stratified random | 必须 | 简单 |
| **Heuristic** | Loss top-k | 必须 | ✅ 已有 |
| **Heuristic** | Gradient norm top-k | 必须 | 简单 |
| **Supervised** | LESS | 必须 | ✅ 已有 |
| **Fisher/logdet** | FisherSFT | 🔴 必须 | 中等（只需 forward + last-layer） |
| **Gradient coreset** | TAGCOS | 强烈建议 | 中等 |
| **Agreement** | SAGE-style (alignment scoring) | 建议 | 简单 |
| **Ours** | Raw-GCS-LogDet (β=0, no Adam) | 必须 | ✅ 已有 |
| **Ours** | AdamW-GCS-LogDet (β=0) | 必须 | ✅ 已有 |
| **Ours** | AdamW-GCS-Whitened-LogDet (β=0.5) | 🔴 必须 | 需改代码 |
| **Ours** | AdamW-GCS-Score | 必须 | ✅ 已有 |
| **Combination** | Opt-GCS prefilter + LESS rerank | 强烈建议 | 中等 |
| **Upper bound** | Full data training | 参考 | — |

---

## 五、Negative Control 实验（排除混淆因素）

| Control | 目的 |
|---------|------|
| Gradient norm top-k | 排除"只是选了高范数样本" |
| Loss top-k | 排除"只是选了高 loss 样本" |
| Length-stratified random | 排除 length bias |
| Domain-stratified random | 排除 domain composition bias |
| **Random orthogonal projection + logdet** | 🔴 关键！排除"任何 low-dim projection + logdet diversity 都有效" |
| **Shuffled eigenspace + logdet** | 🔴 关键！排除"logdet 本身有效而不是谱子空间有效" |

**最重要的两个 negative controls**：

1. **Random subspace + LogDet**：如果用随机正交投影（而非 top eigenvectors）做 logdet 选择，效果也很好，那说明谱子空间没提供特殊信息，只是 "diversity in any subspace" 有用。

2. **Shuffled eigenspace**：将 estimated eigenvectors 随机打乱列顺序后做 logdet。如果效果不变，说明 eigenvalue ordering 不重要。

---

## 六、实验规模问题

### 当前实验设置（DataFlex default）

```
warmup_step: 10, update_step: 10, update_times: 2
total_steps: 30
selection budget: 80 samples per round
```

### 问题

80 samples × 2 rounds = 160 samples，总共 30 steps 训练。这**太小了**，难以支撑 paper claim：
- Noise 太大，方法差异可能被 variance 淹没
- LogDet coverage 的优势在 budget 大时才明显
- Reviewer 会质疑 statistical significance

### 建议：分两档实验

#### Smoke Test（验证代码和方向）
- Selection budget: 80-160 samples
- Training: 30 steps  
- Eval: MMLU subset
- 用途：快速迭代

#### Paper-Scale Experiment（投稿用）
| Selection Ratio | 样本数 | 训练步数 | 评测 |
|:---------------:|:------:|:--------:|------|
| 1% | 1,000 | 200-500 | MMLU + BBH + GSM8K |
| 5% | 5,000 | 500-1000 | MMLU + BBH + GSM8K |
| 10% | 10,000 | 1000-2000 | MMLU + BBH + GSM8K |
| 20% | 20,000 | 2000-3000 | MMLU + BBH + GSM8K |

多个 selection ratio 能画出 **data efficiency curve**（x=selected data %, y=benchmark score），这比单点结果有说服力得多。

---

## 七、代码修改清单

### 必须修改

1. **`spec_gcs_selector.py`**:
   - 增加 `whitening_beta: float = 0.0` 参数
   - 在 `_project_to_eigenspace()` 中加入 `x_i = Λ^{-β/2} U_r^T u_i`
   - 增加 `gradient_type="raw_sgd"` 选项（纯 raw gradient，不 Adam precondition）
   - 增加 `gradient_type="adam_diag"` 选项（只做 `D_t g_i`）
   - 增加 adaptive clipping: `τ = percentile_95(||u_i||)`

2. **`components.yaml`**:
   - 增加 `spec_gcs_whitened` 变体配置 (β=0.5)
   - 增加 `spec_gcs_raw` 变体配置 (gradient_type=sgd)

### 强烈建议新增

3. **`fisher_sft_selector.py`**（FisherSFT baseline）:
   - 只需 forward pass → 提取 last-layer pre-logit embedding → logdet greedy
   - 不需要 backward，计算量小很多

4. **`scripts/analyze_selection.py`**:
   - 分析被选样本的 domain composition
   - 计算不同方法间的 selection overlap (Jaccard similarity)
   - 2D PCA visualization

5. **`scripts/negative_controls.py`**:
   - Random subspace + LogDet baseline
   - Shuffled eigenspace + LogDet baseline

---

## 八、理论层面补充

### 需要在 paper 中主动声明的 Limitation

> Our theorem guarantees spectral coverage of the optimizer-induced update distribution, not downstream accuracy. The downstream relevance of this geometry is an empirical hypothesis validated through SFT experiments. The connection from update-space coverage to generalization in nonconvex deep learning remains an open theoretical question.

**这句话必须写**。它显得诚实，减少 reviewer 追着要非凸泛化定理的风险。

### 需要加强的理论 contribution

1. **Partial whitening 的理论动机**：β controls the trade-off between focusing on dominant update directions (exploitation) and covering rare but potentially important directions (exploration).

2. **与 Fisher Information 的关系**：
   - 当 model 是 GLM (generalized linear model) 且 optimizer 是 SGD 时，update covariance = Fisher Information
   - Opt-GCS 的 optimizer-induced covariance 是 Fisher 的**推广**
   - FisherSFT 是特例（last-layer GLM approximation）

3. **Coverage vs Agreement 的形式化**：
   - Agreement score: `s_i = ||U_r^T u_i||^2`（SAGE 做的事）
   - Coverage score: `Δ(i|S) = x_i^T A_S^{-1} x_i`（LogDet marginal gain）
   - Coverage > Agreement 当存在 redundancy（多个样本指向同一方向时）

---

## 九、推荐的 Paper Narrative

### Introduction 主线

```
Hook: SFT 数据选择方法如何在不依赖 target examples 的情况下发现有价值的训练信号？

Existing approaches:
- Targeted methods (LESS): 需要 target examples，不 general
- Fisher/logdet methods (FisherSFT, SPICE): last-layer proxy，信息不完整
- Agreement methods (SAGE): 选与主方向对齐的样本 → 冗余
- Gradient coreset methods (TAGCOS): clustering-based, 没有利用谱结构

Gap: 没有方法同时做到：
1. Unsupervised (不需要 target)
2. Multi-layer (不只是 last-layer)  
3. Optimizer-aware (适配 AdamW/Muon)
4. Coverage-oriented (而非 agreement/magnitude)
5. Spectrum-aware (利用 eigenvalue 结构做 whitened selection)

Our contribution: Opt-GCS — rank-truncated, whitened spectral coreset 
in optimizer-induced update space.
```

### 核心实验 Story

```
1. Diagnostic: SFT gradients have spiked-yet-heavy-tailed spectral structure
2. Optimizer matters: AdamW-GCS > Raw-GCS (validates optimizer-induced theory)
3. Coverage matters: LogDet > Score (validates coverage-vs-agreement)
4. Whitening matters: β=0.5 > β=0 (validates spectral control)
5. vs Fisher proxy: Opt-GCS ≥ FisherSFT (multi-layer helps)
6. Complementarity: Opt-GCS + LESS > either alone
```

---

## 十、风险优先级排序（更新版）

| 优先级 | 风险 | 概率 | 影响 | Mitigation |
|:------:|------|:----:|:----:|-----------|
| 1 | FisherSFT/SPICE 已占据 logdet+SFT | 确定 | 高 | Whitening + multi-layer + optimizer-aware 差异化 |
| 2 | TAGCOS 已占据 unsupervised gradient coreset | 确定 | 中高 | Spectral (explicit eigenspace) vs clustering (implicit) |
| 3 | LogDet 不加 whitening 与 Fisher design 太像 | 高 | 高 | **必须加 whitening β 作为核心 contribution** |
| 4 | 实验预算太小 (160 samples) 无法证明 claim | 高 | 高 | 扩大到 paper-scale (1k-10k samples) |
| 5 | AdamW-GCS ≈ Raw-GCS 实验上无差异 | 中 | 中 | 准备 fallback narrative |
| 6 | FisherSFT (forward-only) 效果相当但成本低 | 中 | 中高 | 强调 multi-layer 在 reasoning 任务上的优势 |
| 7 | Random subspace + LogDet 也很强 | 低 | 极高 | Negative control 实验必须做 |

---

## 十一、下一步 Action Items

### 立即执行（代码修改）
- [ ] 在 `spec_gcs_selector.py` 中加入 `whitening_beta` 参数和 whitened projection
- [ ] 加入 `gradient_type="raw_sgd"` 选项
- [ ] 加入 adaptive clipping

### 短期（1-2天）
- [ ] 实现 FisherSFT baseline selector（forward-only, last-layer logdet）
- [ ] 实现 random subspace + logdet negative control
- [ ] 扩大 DataFlex 实验配置（selection ratio 1%-20%）

### 中期（1周）
- [ ] 跑完整 paper-scale 实验矩阵
- [ ] Whitening sensitivity ablation (β=0, 0.25, 0.5, 0.75, 1.0)
- [ ] Rank sensitivity ablation (r=5, 10, 20, 50, 100)
- [ ] FisherSFT vs Opt-GCS overlap analysis

### 产出
- [ ] Main results table (all baselines × multiple selection ratios)
- [ ] Data efficiency curve (x=selection ratio, y=benchmark score)
- [ ] Diagnostic visualizations (eigenvalue plot, coverage illustration)
- [ ] Ablation tables (optimizer-aware, whitening, rank, selection method)
- [ ] Negative control results

---

*End of v2 Review*

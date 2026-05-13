# Opt-GCS 论文 Review & 实验计划

> 作者: Claude (Review Agent)  
> 日期: 2025-05-14  
> 目的: 综合评审 Opt-GCS 理论框架，识别潜在质疑点，制定实验修改计划

---

## 一、理论分析总评

### 1.1 理论更新的核心改进

`optimizer_induced_gcs_theory_update.md` 对原始 proposal 做了三个关键改进：

| 改进点 | 原始版本的问题 | 更新后的解决方案 |
|--------|--------------|----------------|
| Optimizer mismatch | 基于 vanilla SGD 的 θ⁺ = θ - ηg | 引入通用 optimizer-induced update map A_t(g_i; S_t) |
| 过强的理论 claim | 声称 first-order Taylor 预测 retraining 效果 | 降级为 "local geometry-preservation guarantee" |
| 缺乏近似比 | 没有选择质量的数学保证 | LogDet submodularity → (1-1/e) 近似保证 |

**评价**：方向完全正确。把 claim 从 "influence-style counterfactual" 降为 "geometry preservation in optimizer update space" 是最安全的理论立场。

### 1.2 理论框架的完整性

当前理论覆盖了以下层次：

```
Layer 1: 建模 — u_i = A_t(g_i; S_t)
    ↓
Layer 2: 结构假设 — Σ_u 具有 spiked/low-rank 结构（eigengap Δ_r > 0）
    ↓
Layer 3: 可恢复性 — Davis-Kahan 保证 eigenspace estimation 误差 bounded
    ↓
Layer 4: 可 sketch — JL/subspace embedding 保证 sketch 后 geometry 保持
    ↓
Layer 5: 可选择 — LogDet greedy (1-1/e) approximation guarantee
    ↓
Layer 6: 几何保持 — 选出的 S 使 U_r^T(Σ_S - Σ)U_r 有界
```

**缺失的层**：Layer 6 → downstream performance 之间没有形式化的连接。这是 by design 的（避免过强 claim），但需要在 paper 中明确说明这个 gap 由实验来弥补。

### 1.3 AdamW-GCS 的数学推导验证

Section 2.2 的推导是正确的：

```
u_i^{AdamW} ≈ D_t(β₁m_t + (1-β₁)g_i)
            = D_t β₁ m_t + (1-β₁)D_t g_i
              ^^^^^^^^    ^^^^^^^^^^^^^^
              共同项       sample-distinguishing 项
```

对于 ranking，只需看 `(1-β₁)D_t g_i`，因此相关协方差是 `Σ_AdamW = E[D_t g_i g_i^T D_t]`。

**注意**：这个推导假设了 "frozen optimizer state"（固定 m_t, v_t），这在单次 selection 时是合理的。但如果做多轮 selection（update_times > 1），optimizer state 会变，需要重新计算。当前框架中每轮 selection 都重新 forward+backward，所以这不是问题。

---

## 二、可能被 Reviewer 质疑的关键点

### 🔴 Critical Issue 1: 与 FisherSFT 的高度重叠

**威胁级别: 最高**

FisherSFT (Deb et al., arXiv:2505.14826, 2025) 几乎在同一时期做了类似的事情：

- 也用 **log-det maximization** 做 SFT data selection
- 也是 submodular greedy
- 也不需要 target set
- 也声称是 D-optimal design

**必须回答的问题**："你和 FisherSFT 的本质区别是什么？"

**我们的差异化论证**：

| 维度 | FisherSFT | Opt-GCS |
|------|-----------|---------|
| 信息源 | Last-layer Fisher (pre-logit embeddings) | **全层** optimizer-induced update 协方差 |
| 需要 backward | ❌ 否（只需前向传播） | ✅ 是（per-sample gradient） |
| Optimizer-aware | ❌ 否 | ✅ 是（适配 AdamW/Muon） |
| 谱分析 | ❌ 无（直接 scalar logdet） | ✅ 显式 eigendecomposition → rank discovery |
| 子空间投影 | ❌ 无（在原始 embedding 空间操作） | ✅ 投影到 top-r update subspace |
| 计算代价 | 低（only forward） | 高（forward + backward + projection） |
| 信息丰富度 | 只有 last-layer logit 信息 | 全层梯度 → 更完整的训练信号 |

**论文中的 positioning 策略**：
- FisherSFT 是一个高效的 last-layer proxy
- Opt-GCS 是一个 principled full-gradient 方法，捕获了 multi-layer 训练动态
- 如果实验中 Opt-GCS > FisherSFT，说明 "全层信息确实比 last-layer proxy 更好"
- 如果 FisherSFT ≈ Opt-GCS，说明 "last-layer 可能已经足够"，但我们的方法更 principled

### 🔴 Critical Issue 2: 与 SAGE 的哲学冲突

SAGE (arXiv:2510.02470, 2025) 用 Frequent Directions 做梯度矩阵的 low-rank 近似，然后选择**与主方向对齐**的样本。

**核心冲突**："coverage vs agreement"
- SAGE: 选择 alignment 最高的样本（agreement）
- Opt-GCS: 选择覆盖子空间的样本（coverage/diversity）

**这是我们的核心 narrative**：

> Agreement-based selection (如 SAGE) 倾向于选择冗余样本——它们都指向同一个主方向。Coverage-based selection (Opt-GCS LogDet) 保证选出的样本在梯度子空间中张成最大体积，避免冗余。

**2D 直观例子**：假设 top eigenspace 有两个方向 (1,0) 和 (0,1)，budget=4：
- SAGE 会选 4 个都接近 (1,0) 的样本（因为 λ₁ 最大）
- Opt-GCS LogDet 会选 2 个 (1,0) + 2 个 (0,1)（因为选第 3 个 (1,0) 的 marginal gain 远小于选第 1 个 (0,1)）

### 🟡 Moderate Issue 3: Unsupervised 能否超越 Supervised

"如果有 target examples，LESS 一定比你好。那为什么不直接用 LESS？"

**回应策略**：
1. **场景论证**：Target examples 不总是可用的（新任务/隐私/general-purpose SFT）
2. **互补性论证**：Opt-GCS prefilter (10w→2w) + LESS rerank (2w→800) 可能是最佳 pipeline
3. **Generality 论证**：在多个 diverse benchmark 上，unsupervised 方法可能比 targeted 方法更稳定（不依赖 target set 的选择偏差）

### 🟡 Moderate Issue 4: Effective rank 很低（12.6），是否信息不足？

Diagnostic 显示 effective rank ≈ 12.6。意味着只有约 12 个方向包含主信号。

**潜在质疑**：
- "12 个方向真的够描述 SFT 数据的复杂性吗？"
- "剩下的 4084 个方向是 noise 还是 rare but important signals？"

**回应**：
1. Effective rank 是 trace/max ratio，不代表只有 12 个方向有用
2. Top-50 解释 26.8% 方差，top-100 解释 33.1% → 长尾结构
3. LogDet 选择在 r 维子空间中操作，r 由算法自适应确定
4. **实验验证**：做 rank sensitivity ablation (r=5,10,20,50,100)

### 🟡 Moderate Issue 5: 计算开销

"你需要对 10w 样本做 per-sample backward pass，这太贵了"

**数据**：
- 10w 样本 × 8B 模型 × single GPU ≈ 15h
- 8 GPU 并行 ≈ 2h
- LESS 也需要同样的计算（train gradients + eval gradients）
- Opt-GCS 实际上**更省**（不需要 eval gradient 计算）

### 🟢 Minor Issue 6: Clipping threshold τ 的选择

Section 5.1 的 clipping 没有理论指导。

**建议**：使用自适应规则 `τ = percentile_95(||u_i||)` 或 `τ = median(||u_i||) × 3`。

### 🟢 Minor Issue 7: Pathwise Extension 可行性

Section 10 的多 checkpoint 方案计算量太大（T 倍），在 paper 中最好弱化为 future work。

---

## 三、实验修改与完整计划

### 3.1 实验 Baselines 更新

| 方法 | 类型 | 需要 Target? | 核心机制 | 备注 |
|------|------|:-----------:|---------|------|
| **Random** | Baseline | ❌ | 随机采样 | Lower bound |
| **Loss** | Heuristic | ❌ | Loss-based scoring | DataFlex 已有 |
| **LESS** | Supervised gradient | ✅ | Train-eval gradient similarity | DataFlex 已有, 最强 targeted baseline |
| **FisherSFT** | Unsupervised Fisher | ❌ | Last-layer logdet(FIM) | **需新增实现** |
| **Opt-GCS-Score** | 本文变体 | ❌ | Top-k by \|\|x_i\|\|² | Ablation: magnitude only |
| **Opt-GCS-Diverse** | 本文变体 | ❌ | Score + k-center | Ablation: heuristic diversity |
| **Opt-GCS-LogDet** | 本文主方法 | ❌ | LogDet greedy coverage | 主方法 |
| **Raw-GCS-LogDet** | Ablation | ❌ | LogDet on raw g_i (no Adam) | 验证 optimizer-aware 的增益 |
| Full Data | Upper bound | — | 全量训练 | Reference |

### 3.2 核心实验矩阵

#### Experiment 1: Main Results（最重要）

**设置**：Llama-3.1-8B, LoRA rank=16, Open-Hermes-2.5 (10w), 评测 MMLU

| 方法 | Selection Budget | 训练步数 | 评测 |
|------|:----------------:|:--------:|------|
| Random | 80/round × 2 rounds | 30 steps | MMLU subset |
| Loss | 80/round × 2 rounds | 30 steps | MMLU subset |
| LESS | 80/round × 2 rounds | 30 steps | MMLU subset |
| Opt-GCS-LogDet | 80/round × 2 rounds | 30 steps | MMLU subset |
| Opt-GCS-Score | 80/round × 2 rounds | 30 steps | MMLU subset |

#### Experiment 2: Optimizer-Aware Ablation（验证核心 claim）

验证 "optimizer-induced geometry matters"：

| Gradient Representation | Description |
|------------------------|-------------|
| Raw SGD gradient `g_i` | 不做任何预处理 |
| L2-normalized `g_i / \|\|g_i\|\|` | 只保留方向 |
| Adam-preconditioned `D_t g_i` | 理论推荐的 optimizer-aware 版本 |
| Full Adam update `(β₁m + (1-β₁)g_i) / (√v + ε)` | 当前实现的 LESS-style |

**预期结果**：Adam-preconditioned ≥ Raw gradient，验证 Section 2.2 的理论。

#### Experiment 3: Selection Method Ablation（验证 coverage > magnitude）

固定 gradient representation = Adam-preconditioned，对比：

| Selection | Expected Ranking |
|-----------|:----------------:|
| Random subset from all | Weakest |
| Top-k by score \|\|x_i\|\|² | Moderate (redundant) |
| Score + k-center | Good |
| LogDet greedy | **Best** |

**预期结果**：LogDet > Score，验证 "coverage 比 magnitude 更重要"。

#### Experiment 4: Rank Sensitivity

| Rank r | 方法 | 评测 |
|--------|------|------|
| 5 | LogDet | MMLU |
| 10 | LogDet | MMLU |
| 20 | LogDet | MMLU |
| 50 | LogDet | MMLU |
| 100 | LogDet | MMLU |
| auto (effective) | LogDet | MMLU |

#### Experiment 5: Diagnostic Visualizations（用于 Paper）

1. **Eigenvalue decay plot** — log-log + cumulative variance
2. **Eigengap ratio plot** — 验证 spiked structure
3. **Score vs token-length scatter** — 验证 length normalization 有效
4. **Selected data composition** — 被选中的样本的 domain 分布 (math/code/general/safety)
5. **Eigenspace semantic analysis** — top eigenvectors 对应的样本类型
6. **Coverage illustration** — 2D PCA 上展示 LogDet vs Score 的选择差异

### 3.3 代码实现修改计划

```
需要修改的代码:
├── spec_gcs_selector.py
│   ├── 增加 gradient_type="raw_sgd" 选项（纯 raw gradient，不做 Adam preconditioning）
│   ├── 增加 gradient_type="adam_diag" 选项（只做 D_t g_i 对角预处理）
│   ├── 增加 clipping_method="adaptive_quantile" 选项
│   └── 增加 rank sensitivity 的 fixed_rank 配置支持
│
├── 新增: fisher_sft_selector.py（如果决定加入 FisherSFT baseline）
│   └── 只需前向传播 → last-layer embedding → logdet greedy
│
└── 新增: scripts/analyze_selection.py
    ├── 分析被选中样本的 domain composition
    ├── 2D PCA visualization of selected vs unselected
    └── 与 LESS 选择结果的 overlap 分析
```

### 3.4 时间线估算

| 阶段 | 内容 | 时间估算 (8×H20) |
|------|------|:-----------------:|
| Phase 0 | 代码修改（增加 ablation 变体） | 2-3h 编码 |
| Phase 1 | 全量梯度计算 (10w × Llama-8B) | 3-4h |
| Phase 2 | 谱分析 + 各方法 selection | < 30min |
| Phase 3 | 训练 (6种方法 × 30 steps) | 6h |
| Phase 4 | MMLU 评测 | 2h |
| Phase 5 | Diagnostic plots + analysis | 1h |
| **总计** | | **~15h** |

---

## 四、Paper 结构建议

### 推荐标题
> **Opt-GCS: Spectral Coreset Construction in Optimizer-Induced Update Space for Unsupervised SFT Data Selection**

### 推荐 Abstract 核心句

> We formulate unsupervised SFT data selection as spectral coreset construction in optimizer-induced update space. Unlike targeted methods (LESS) that require evaluation examples, and unlike last-layer proxies (FisherSFT) that only capture output-level information, Opt-GCS discovers the intrinsic low-rank structure of full-layer optimizer-induced updates and selects a logdet-diverse subset that maximally covers the dominant training geometry.

### Section 结构

1. **Introduction**: "Agreement vs Coverage" narrative + motivation
2. **Related Work**: LESS, OPUS, FisherSFT, SAGE, TAGCOS — 详细对比
3. **Preliminaries**: Optimizer-induced update, spiked covariance, Davis-Kahan
4. **Method**: Opt-GCS algorithm (probe → SVD → project → LogDet)
5. **Theoretical Analysis**: Eigenspace recovery + sketch preservation + submodularity
6. **Experiments**:
   - 6.1 Setup (model, data, baselines)
   - 6.2 Main Results (Table 1: MMLU comparison)
   - 6.3 Optimizer-Aware Ablation (Table 2: Raw vs AdamW vs Muon)
   - 6.4 Selection Strategy Ablation (Table 3: Score vs Diverse vs LogDet)
   - 6.5 Diagnostic & Analysis (Figures: eigenvalue, composition, coverage)
7. **Discussion**: Limitations, when to use vs LESS, computational considerations
8. **Conclusion**

---

## 五、关键风险与 Mitigation

| 风险 | 概率 | 影响 | Mitigation |
|------|:----:|:----:|-----------|
| FisherSFT 效果相当或更好 | 中 | 高 | 强调 multi-layer + optimizer-aware 的 principled 优势；加 FisherSFT+Opt-GCS 组合实验 |
| Raw-GCS ≈ AdamW-GCS | 中 | 中 | 改为 "optimizer-aware 是 theoretically principled，empirically L2-norm 是一个 strong heuristic" |
| LESS 远超 Opt-GCS | 低 | 高 | 强调 unsupervised setting；展示 Opt-GCS prefilter + LESS rerank 的互补性 |
| 谱结构在 all-linear LoRA 上不存在 | 低 | 极高 | 如果 diagnostic 失败，考虑换 gradient representation（last-2-layers, lm_head only） |
| LogDet ≈ Score | 低 | 中 | 分析 selection overlap；可能需要更大的 budget 才能看出差异 |

---

## 六、当前实验进展

### 已验证

- ✅ Qwen3-4B 上 500 samples：effective rank = 5.6, eigengap = 4.34
- ✅ Llama-3.1-8B 上 3000 samples：**effective rank = 12.6, eigengap = 3.61**
- ✅ Score-length correlation = -0.26（length norm 有效但可能需要调小 alpha）
- ✅ 代码在 DataFlex 框架中注册并通过单元测试
- ✅ 8 GPU 分布式训练启动成功

### 当前运行

- 🔄 完整 Spec-GCS LogDet 训练 (Llama-3.1-8B × Open-Hermes 10w)
  - Log: `/jizhicfs/karonhe/dataflex_saves/spec_gcs_logdet_train.log`
  - 预计完成时间: 3-5 hours

### Diagnostic 关键发现

```
Llama-3.1-8B (LoRA q_proj+v_proj, 3000 samples, proj_dim=4096):
  - Effective rank: 12.6
  - Top-1 eigenvalue explains: 7.9% of variance
  - Top-10 explain: 17.4%
  - Top-50 explain: 26.8%
  - Eigengap ratio (λ₁/λ₂): 3.61
  - Score-length correlation: -0.26

Verdict: MODERATE spectral structure detected.
正式实验使用 all-linear LoRA (更多参数), effective rank 可能更高。
```

---

## 七、近期相关文献（必须 cite）

| Paper | Year/Venue | 关系 | 差异化 |
|-------|-----------|------|--------|
| **FisherSFT** (Deb et al.) | 2025, arXiv:2505.14826 | 🔴 最高重叠：logdet + SFT + unsupervised | 我们用全层梯度 + optimizer-aware |
| **SAGE** (Xiang et al.) | 2025, arXiv:2510.02470 | 🔴 高重叠：gradient sketching + spectral | 我们做 coverage，他们做 agreement |
| **SPICE** | 2026, ICLR | 🔴 高重叠：Fisher + gradient conflict | 我们做 volume max，他们做 conflict avoidance |
| **GRAFT** | 2025, arXiv:2508.13653 | 🟡 中度：MaxVol + gradient features | Online/vision tasks，我们是 offline/LLM |
| **LESS** (Xia et al.) | 2024, ICML | 🟡 主要 baseline | Targeted vs unsupervised |
| **OPUS** (Wang et al.) | 2026, arXiv:2602.05400 | 🟡 理论邻居 | Dynamic pretraining vs offline SFT |
| **TAGCOS** | 2025, NAACL | 🟡 gradient coreset | Clustering+OMP vs spectral+logdet |
| **GradientSpace** | 2024, arXiv:2512.06678 | 🟡 gradient SVD | Expert routing vs data selection |
| **Li et al.** | 2025, arXiv:2504.10766 | 🟡 gradient SVD analysis | 诊断性论文，不做选择 |

---

*End of Review*

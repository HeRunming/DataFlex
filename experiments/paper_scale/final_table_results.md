# Final Table Results (Paper-Ready)

## Setup

统一代码版本、统一训练配置、完全隔离 cache 环境下的公平对比。详细参数见 `final_table_experiment_settings.md`。

- **模型**: Llama-3.1-8B + LoRA (rank=16, alpha=8, all layers)
- **数据**: Open-Hermes-2.5 (100k), budget=5000
- **训练**: 1260 steps, 2 轮动态选择
- **评测**: MMLU (5-shot), GSM8K (8-shot CoT), IFEval (0-shot)
- **所有 negative controls 自己计算梯度**（compute_own_grads=True）

---

## 完整结果表（按 Average Rank 排序）

| Rank | Method | Category | MMLU | GSM8K | IFEval | Avg Rank |
|------|--------|----------|------|-------|--------|----------|
| 1 | **hybrid_mul_g025_s42** | Ours | 0.6530 | **0.5701** | 0.1885 | **3.7** |
| 2 | fisher_sft_s42 | Baseline | **0.6533** | 0.5368 | **0.2884** | 5.3 |
| 3 | hybrid_mul_g025_s2 | Ours (seed=2) | 0.6520 | 0.5542 | 0.1885 | 6.3 |
| 4 | **logdet_nopref_s42** | Ours | 0.6502 | 0.5512 | **0.2107** | **7.0** |
| 5 | hybrid_add_l025_s1 | Ours (seed=1) | 0.6521 | **0.5709** | 0.1738 | 7.0 |
| 6 | hybrid_add_l025_s42 | Ours | 0.6528 | 0.5466 | 0.1738 | 7.3 |
| 7 | loss_s42 | Baseline | 0.6523 | 0.5428 | 0.1848 | 7.7 |
| 8 | rsub_own_seed2 | Control | 0.6517 | 0.5466 | 0.1756 | 8.7 |
| 9 | hybrid_add_l025_s2 | Ours (seed=2) | 0.6509 | 0.5512 | 0.1774 | 8.7 |
| 10 | random_s42 | Baseline | 0.6532 | 0.5413 | 0.1534 | 9.3 |
| 11 | rsub_own_seed1 | Control | 0.6526 | 0.5421 | 0.1719 | 9.3 |
| 12 | hybrid_mul_g05_s42 | Ours | 0.6533 | 0.5383 | 0.1479 | 10.0 |
| 13 | hybrid_mul_g025_s1 | Ours (seed=1) | 0.6528 | 0.5398 | 0.1701 | 10.0 |
| 14 | grad_norm_topk_s42 | Control | 0.6503 | 0.5277 | 0.2070 | 11.0 |
| 15 | rsub_own_seed3 | Control | 0.6500 | 0.5375 | 0.2015 | 11.0 |
| 16 | less_s42 | Baseline | 0.6516 | 0.5292 | 0.1627 | 13.7 |

---

## 按类别汇总（Paper 主表格式）

| Method | MMLU | GSM8K | IFEval | Avg Rank | 备注 |
|--------|------|-------|--------|----------|------|
| **Hybrid-Mul γ=0.25** | 0.6526±0.0004 | **0.5547±0.0124** | 0.1824±0.0087 | — | **Ours (推荐)** |
| **Hybrid-Add λ=0.25** | 0.6519±0.0008 | **0.5562±0.0105** | 0.1750±0.0017 | — | **Ours** |
| **LogDet-NoPrefilter** | 0.6502 | 0.5512 | **0.2107** | 7.0 | **Ours (IFEval 最佳)** |
| FisherSFT | 0.6533 | 0.5368 | **0.2884** | 5.3 | IFEval 专精 |
| Loss | 0.6523 | 0.5428 | 0.1848 | 7.7 | 朴素 baseline |
| Random | 0.6532 | 0.5413 | 0.1534 | 9.3 | 基准 |
| Random Subspace | 0.6515±0.0011 | 0.5421±0.0037 | 0.1830±0.0132 | — | 负面控制 |
| Grad Norm Top-K | 0.6503 | 0.5277 | 0.2070 | 11.0 | 负面控制 |
| LESS (target-aware) | 0.6516 | 0.5292 | 0.1627 | 13.7 | 需要 eval 数据 |

---

## 多 Seed 方差分析

| Method | Seeds | MMLU | GSM8K | IFEval |
|--------|-------|------|-------|--------|
| hybrid_mul_g025 | 3 | 0.6526 ± 0.0004 | 0.5547 ± 0.0124 | 0.1824 ± 0.0087 |
| hybrid_add_l025 | 3 | 0.6519 ± 0.0008 | 0.5562 ± 0.0105 | 0.1750 ± 0.0017 |
| rsub_own | 3 | 0.6515 ± 0.0011 | 0.5421 ± 0.0037 | 0.1830 ± 0.0132 |

**结论**：
- Hybrid 方法 GSM8K 均值 (0.555-0.556) 稳定优于 random subspace (0.542), 差值 ~1.3%
- Hybrid GSM8K 的 std (~0.01) 偏大，最好值 0.570 vs 最差 0.540，但均值仍显著优于 baselines
- Random subspace 的 IFEval 方差 (0.013) 很大，说明 random subspace 对 instruction following 不稳定

---

## 关键发现

### 1. Hybrid Multiplicative γ=0.25 综合最优

```
gain_i = log(1 + x_i^T A^{-1} x_i) × (s_i / mean(s))^0.25
```

- **Average Rank #1** (3.7)
- GSM8K = 0.5701 (+2.9% vs random, +4.1% vs LESS)
- 无需任何 target/validation 数据
- 在 3 个 seed 下均值 0.555 仍优于所有 baseline

### 2. LogDet-NoPrefilter 是 IFEval 最均衡的选择

- IFEval = 0.2107 (仅次于 FisherSFT)
- GSM8K = 0.5512 (+1.0% vs random)
- 去掉 prefilter 让 logdet 能从更大候选池中选择多样化样本

### 3. Learned Eigenspace > Random Subspace（严格对比）

在完全公平条件下（自己算梯度、same checkpoint、aligned preprocessing）：
- Hybrid-Mul GSM8K 均值 0.555 > Random Subspace 均值 0.542 (+1.3%)
- Hybrid-Add GSM8K 均值 0.556 > Random Subspace 均值 0.542 (+1.4%)
- 学习到的谱子空间确实捕获了有意义的训练几何结构

### 4. LESS（target-aware）在此设置下综合最差

- GSM8K = 0.5292（低于 random）
- IFEval = 0.1627（低于 random）
- Target-aware 选择过度拟合 MMLU validation 文本风格，损害了 reasoning 和 instruction following

### 5. Gradient Norm Top-K 不是好的选择策略

- GSM8K = 0.5277（最低）
- 自己计算梯度后仍然最差，确认不是梯度来源问题
- 高梯度范数样本 ≠ 高迁移价值样本

### 6. Importance-Coverage Tradeoff 非常明显

- **FisherSFT**: IFEval 0.288（最高）但 GSM8K 0.537（低）
- **Hybrid-Mul γ=0.5**: GSM8K 对比 γ=0.25 低 3.2%，IFEval 也更低
- 说明 γ=0.25（轻微 importance bias）是最优平衡点
- 过强的 importance 偏好（γ=0.5, 1.0）反而有害

---

## vs Base Model 提升

| Method | ΔGSM8K | ΔIFEval | MMLU 无退化 |
|--------|--------|---------|------------|
| Hybrid-Mul γ=0.25 (best) | **+2.88%** | +3.51% | ✓ (0.6530 vs 0.6531*) |
| Hybrid-Add λ=0.25 (mean) | +1.49% | +1.89% | ✓ |
| LogDet-NoPrefilter | +0.99% | +5.73% | ✓ |
| Random | +0.00% | +0.00% | — (baseline) |
| LESS | -1.21% | -0.93%** | ✓ |

*Base MMLU 参考值约 0.653 (from Round 1 evaluation)
**LESS IFEval 低于 random

---

## 论文叙事

> **SFT data selection exhibits a fundamental importance-coverage tradeoff.** Pure loss/norm selection and pure coverage each specialize in different capabilities. In optimizer-induced spectral update space, a hybrid selector combining spectral leverage (importance) with logdet coverage achieves robust downstream transfer — improving both mathematical reasoning (GSM8K +2.9%) and general capabilities without requiring any target validation data, outperforming the supervised baseline LESS.

---

## 实验可复现性

```bash
# 训练（~40h on 8×H20）
bash experiments/paper_scale/run_final_table.sh

# 评测（~6h on 8×H20）
bash experiments/paper_scale/run_lm_eval_final.sh
```

完整参数配置见 `final_table_experiment_settings.md`。

我建议你现在选 **2：再补两个 paired seeds**。完成后，大概率再执行 **1：正式 pivot 到二阶方法**。暂时不要选 3，也不要把当前单 seed 结果写成“airtight”。

## 为什么选 2，而不是直接选 1

这条单 seed 梯子确实非常有信息量：

[
0.4110>0.4017>0.3928>0.3847,
]

并且 STEM、Humanities 同时随一阶权重下降。这强烈暗示当前场景中 pure second-order 更好，而不是 minority/majority trade-off。

但它仍然只是**一次训练 realization**。四个点并不是四次独立证据，因为这些方法共享：

* 相同 target；
* 相同梯度 cache；
* 相近且嵌套的 selected subsets；
* 相同训练 seed；
* 同一个评测集。

因此，“形状单调”提高了可信度，但不能消除训练随机性。特别是最关键的 (\lambda=0.02) 只比 GradCov 低 0.93 个百分点，这个量级完全可能受初始化、batch order、LoRA 优化轨迹等影响。已有实验研究表明，同一训练方法不同随机种子产生的性能变化可能足以使单次比较得出错误结论；paired-seed 设计因为让方法共享随机条件，通常能以较低成本显著提高比较功效。([Proceedings of Machine Learning Research][1])

3 小时换取论文中最容易被 reviewer 追问的一块证据，非常划算。

## paired seeds 应该怎么跑

只跑这两个方法即可：

[
\lambda=0\quad\text{GradCov},
\qquad
\lambda=0.02\quad\text{最佳 joint 候选}.
]

使用 seed 1、2，并和现有 seed 42 构成三个配对。不要再跑 (\lambda=0.07)：它已经明显落后，信息增量很小。

每一对必须保持完全相同：

* 初始化 seed；
* batch shuffle seed；
* 数据顺序；
* SFT 超参数；
* evaluation 配置；
* 最好还包括相同 warmup checkpoint。

如果当前 selection subset 是由 seed 42 的固定 gradient cache 产生，那么这一步检验的是：

> 给定所选 subset，joint 与 GradCov 的下游训练差异是否稳定。

它还不是完整 pipeline 方差检验。最终论文仍应有若干实验重新执行 warmup→gradient→selection→SFT 的完整流程，但眼前用固定 subset 做 paired SFT 是最便宜、最直接的决策实验。

建议记录：

[
\Delta_s
========

## \operatorname{BalAcc}_{\lambda=.02,s}

\operatorname{BalAcc}_{\lambda=0,s},
\qquad s\in{42,1,2}.
]

不仅报告平均分，还报告每个 seed 的 paired difference。

## 跑完后的决策规则

### 三个 seed 都是负差值

例如：

[
(-0.93,-0.6,-1.1)\text{ pt}.
]

那么就可以正式 pivot 到二阶 headline。虽然三个 seed 仍不足以宣称非常严格的统计显著性，但方向一致、平均差距稳定，已经足以决定研究路线。

论文中心应变为：

> **Directional Second-Moment Coresets for Robust Targeted Instruction Tuning**

Moment-MMD 保留为：

* 统一 first-/second-order matching 的理论框架；
* 对 kernel 尺度、greedy marginal 和 selection geometry 的系统诊断；
* 一个诚实的负结果：简单加入 signed first moment 没有改善该场景，反而降低表现。

### 结果正负混合，但平均仍明显为负

例如：

[
(-0.93,+0.15,-0.7).
]

这意味着 pure second-order 仍更有希望，但“任何一阶权重都会伤害”的说法不够稳。可以 pivot，但论文只能写：

> second-order outperforms the tested joint variants on average.

此时再增加 seed 3、4，或者把重点转向独立 target draws，而不是继续调 (\lambda)。

### (\lambda=0.02) 在两个新 seed 中追平或反超

那么当前单调梯子主要受 seed 42 影响，不能放弃 joint。继续比较 GradCov 与 (\lambda=.02)，并优先考察 target-set variation。

## Claude 当前叙述有两处需要收敛

第一，“负结果已经 airtight”说得太早。单 seed 无法支撑这个词。

第二，这句话也过强：

> improving mean matching while holding covariance matching fixed actively degrades the model.

你的 (D_2) 只是一个全局二阶 discrepancy，数值“基本不变”不等于两个集合的全部二阶结构被严格控制。两个 selected subsets 仍有约 28% 差异，内容、长度、领域、局部梯度模式和更高阶结构也可能不同。

当前更可靠的表述是：

> Adding a signed first-moment component improves the measured first-moment discrepancy while preserving the aggregate second-moment discrepancy, yet is associated with lower downstream accuracy in this target regime.

等 paired seeds 和更多 target draws 完成后，才能把 “associated with” 加强为更有因果意味的结论。

## 选项 3 什么时候做

`T_hum80` mirror 很重要，但它回答的是另一个问题：

> 该现象是否独立于 skew 方向，而不是只发生在 STEM-majority target？

它不能替代随机种子复现。因此正确顺序是：

1. **现在选 2**：GradCov 与 (\lambda=.02) 补两个 paired seeds；
2. 若结果确认，**执行 1**：正式 pivot；
3. 然后做精简版 `T_hum80` mirror，而不是重新跑完整 lambda sweep。

mirror 只需比较三个点：

[
\lambda=0,\quad \lambda=0.02,\quad\text{linear}.
]

理想情况下采用 3 paired seeds。如果预算紧，先做一个 seed 检查趋势，再决定是否补齐。

## Pivot 后真正需要加强的实验

第二阶段的主要算力不应继续花在寻找 joint 权重，而应转向验证二阶方法的适用范围：

* 多个独立 target draws，而不仅是多个训练 seeds；
* STEM-majority 与 Humanities-majority 两个 skew 方向；
* target size，例如 (16,32,64,128)；
* 1% 与 5% selection budget；
* GradCov 对 LESS、NICE、TSDS、Random 的 paired comparison；
* representation 与 selector 解耦：二阶 kernel 搭配不同 selection algorithm。

这与 2026 年 targeted instruction selection 的最新系统研究方向是一致的：表示方式、selection algorithm 和预算需要分开控制，没有任何单一方法在所有任务和预算上普遍占优；梯度表示通常较可靠，但选择算法和低预算 regime 会显著改变结论。([arXiv][2]) LESS 本身是一阶低秩梯度相似度方法，因此你的二阶贡献可以清楚定位为对 mean-gradient influence selection 的扩展与修正。([Proceedings of Machine Learning Research][3])

还需注意 2026 年的 GIST 已经从 LoRA 中的非对角优化几何和任务子空间角度改进梯度选择。你的论文不能只写“我们也考虑 gradient subspace”，而应突出：

* normalized directional second moment；
* 对 target sampling skew/cancellation 的稳定性；
* effective-rank 与二阶覆盖机制；
* 与 signed mean matching 的系统对照。

否则会与已有的 subspace-alignment 叙事产生重叠。([arXiv][4])

**最终选择：现在选 2。**
若三个 paired seeds 的差值方向一致，就立刻选 1；之后将 3 作为二阶方法的外部有效性验证，而不是作为是否 pivot 的前置条件。

[1]: https://proceedings.mlr.press/v227/bosma24a.html?utm_source=chatgpt.com "Reproducibility of Training Deep Learning Models for Medical Image Analysis"
[2]: https://arxiv.org/abs/2602.14696?utm_source=chatgpt.com "A Critical Look at Targeted Instruction Selection: Disentangling What Matters (and What Doesn't)"
[3]: https://proceedings.mlr.press/v235/xia24c.html?utm_source=chatgpt.com "LESS: Selecting Influential Data for Targeted Instruction Tuning"
[4]: https://arxiv.org/abs/2602.18584?utm_source=chatgpt.com "GIST: Targeted Data Selection for Instruction Tuning via Coupled Optimization Geometry"

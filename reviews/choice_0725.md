这个结果说明了一件比“Phase-1 gate failed”更重要的事：

> **当前问题主要不是 Moment-MMD 的两个分量无法联合，而是 (\beta) 的参数化完全没有可解释性；但继续换一种固定尺度归一化，并不会创造新的方法，只会重新标记同一条 kernel-mixture 路径。**

所以四个选项中，我不建议直接选 2，也不建议选 3。最合理的是一个更精确的“选项 1.5”：

> **做一次很便宜的低权重 selection-only sweep，然后只训练 1–2 个真正不同、位于 Pareto 前沿的 joint 集合。**

## 结果具体意味着什么

Random-MMD normalization 确实把两个完整目标的典型数值拉近了：

[
s_1=0.2221,\qquad s_2=0.1679,
]

只有 (1.32\times) 差距。

但 greedy 实际看的是候选之间的**边际分数差异**。归一化后：

[
\sigma_{\text{lin}}=0.0131,\qquad
\sigma_{\text{quad}}=0.00119,
]

仍然相差约 (11\times)。

因此 (\beta=0.5) 并不代表两个分量各贡献一半。一阶项仍然控制了大多数候选排序，导致：

* (\beta=0.25) 已经和 linear endpoint 有 90.2% Jaccard；
* (\beta=0.5) 达到 95.5%；
* 这些点基本是在重复 linear-MMD，而不是测试真正的 joint。

这和 MMD 的一般性质一致：MMD 的行为高度依赖 kernel 及其尺度，kernel 选择本身是核心建模决策，而不是无关紧要的常数。([机器学习研究杂志][1]) 多 kernel MMD 工作也会专门归一化、选择和组合不同 kernel；但这些方法主要针对两样本检验的统计功效与校准，不能保证归一化后的 kernel 在 greedy coreset 的逐步 argmax 中拥有相同区分度。([UCL Discovery][2])

## 一个关键数学事实：marginal-std normalization 只是重新参数化

当前归一化 kernel 是

[
k_\beta
=======

\frac{\beta}{s_1}k_{\rm lin}
+
\frac{1-\beta}{s_2}k_{\rm quad}.
]

假设改成 marginal-std normalization：

[
k'_\gamma
=========

\frac{\gamma}{q_1}k_{\rm lin}
+
\frac{1-\gamma}{q_2}k_{\rm quad}.
]

只要 (s_1,s_2,q_1,q_2>0)，对于任意 (\gamma)，总能找到一个 (\beta)，使两者的系数比例完全相同。整体乘以正数不会改变 greedy argmax。

也就是说：

> **固定的 marginal-std normalization 不会扩展可选 kernel 的集合，也不会产生原先不可能产生的 selected subset；它只是把“真正均衡的位置”从一个 (\beta) 映射到另一个 (\gamma)。**

因此，重新实现 marginal normalization 再跑完整 7-point sweep，科学上新增的信息很少。

根据你测得的 (11\times) 比例，当前参数化下两个 marginal 区分度大致相等的位置不是 (\beta=0.5)，而是

[
\frac{\beta}{1-\beta}\times10.98\approx1,
]

所以

[
\beta_\star\approx\frac{1}{1+10.98}\approx0.0835.
]

你已经跑过的 (\beta=0.1) 与它非常接近。因此，**(\beta=0.1) 基本就是 marginal-balanced joint 候选**。把它重新命名为 marginal-normalized (\gamma=0.5)，selected set 只会有很小变化。

## (\beta=0.1) 值得训练吗？

值得，而且是目前唯一明显值得训练的 interior 点。

和 pure GradCov 比较：

[
D_1: 0.1719\rightarrow0.1596,
]

下降约 (7.2%)；

[
D_2: 0.1456\rightarrow0.1459,
]

只恶化约 (0.2%)；

effective rank：

[
2397\rightarrow2440.
]

也就是说，(\beta=0.1) 在 selection diagnostics 上是一个相当漂亮的 Pareto 改进：

* 明显改善一阶误差；
* 几乎不损失二阶误差；
* effective rank 反而提高；
* selected set 与两个 endpoint 都不完全相同。

它虽然与 linear endpoint 的 Jaccard 达到 0.806，但这不等于它的下游行为一定等同于 linear。Jaccard 0.806 仍意味着约 19.4% 的样本不同；在 13,533 条训练集上，约有 2,600 条不同样本，足以改变 SFT 结果。

Greedy MMD 本来就是序列化、路径依赖的选择过程，kernel 的细小变化可能在早期 argmax 中产生变化，并在后续累积。相关理论工作也将 greedy MMD minimization 作为独立的迭代优化过程分析，而不是仅由最终 MMD 数值决定。([arXiv][3])

# 下一步怎么做

## 第一步：不要训练 (\beta=0.25,0.5,0.75)

它们分别与 linear endpoint 有约 90%、96% 和接近 100% 的重合。训练这三个点大概率只会重复之前 0.385–0.388 左右的 linear 结果，计算价值很低。

选项 3 可以直接排除。

## 第二步：用直接的系数比 (\lambda) 代替 (\beta)

建议把 kernel 写成：

[
k_\lambda
=========

k_{\rm quad}
+
\lambda k_{\rm lin}.
]

整体尺度不影响选择，所以 (\lambda) 直接代表一阶相对于二阶的 kernel 权重，不再受任意的 (s_1,s_2) 参数化影响。

做一个便宜的 selection-only sweep：

[
\lambda\in
{0,;0.005,;0.01,;0.02,;0.04,;0.07,;0.10,;0.20}.
]

其中根据当前 marginal std，

[
\lambda_{\rm balanced}
\approx
\frac{0.00040}{0.00581}
\approx0.069.
]

所以 (\lambda=0.07) 是理论上最值得看的点。

这与当前 random-MMD-normalized 参数化的换算关系是：

[
\lambda
=======

# \frac{\beta/s_1}{(1-\beta)/s_2}

\frac{\beta}{1-\beta}\frac{s_2}{s_1}.
]

你的 (\beta=0.1) 对应：

[
\lambda\approx
\frac{0.1}{0.9}\frac{0.1679}{0.2221}
\approx0.084.
]

因此新 sweep 主要是在 (\lambda=0.01\sim0.08) 间寻找一个比当前 (\beta=0.1) 更保守的 joint。

## 第三步：从 sweep 中只选两个集合训练

不要根据 (\lambda) 是否“漂亮”选择，而是根据 selection geometry 选择。

候选 A：**marginal-balanced joint**

[
\lambda\approx0.07,
]

也就是当前 (\beta\approx0.083)，和现有 (\beta=0.1) 很接近。

候选 B：**GradCov-preserving joint**

选择一个满足以下条件的较小 (\lambda)：

* 与 GradCov Jaccard 大约 0.65–0.80；
* (D_1) 明显低于 GradCov；
* (D_2) 增幅不超过约 0.5%；
* effective rank 不下降。

它很可能落在：

[
\lambda\approx0.01\sim0.04.
]

如果多个点满足，只选 (D_1) 最低的那个。这样最多增加两次 SFT。

## 第四步：用单 seed 结果决定论文路线

GradCov 当前 balanced score 是 0.411，linear 大约是 0.385–0.388。

可以采用下面的决策规则。

### joint 达到约 0.408 以上

说明 joint 至少接近 GradCov，而且可能存在互补性。

下一步扩大到：

* 3 个独立 `T_stem80` target draws；
* 每个 draw 3 个 training seeds；
* 比较 GradCov、最佳 joint、linear。

若 joint 稳定领先，Moment-MMD 可以成为主方法。

### joint 在 0.400–0.408

属于不确定区间。再为最佳 joint 和 GradCov 各补两个训练 seed，用 paired seeds 比较，不要立即扩大所有实验。

### joint 低于约 0.400，尤其接近 linear

这时基本可以停止继续校准 joint。

论文主线应转为：

> **Directional Second-Moment Coresets for Robust Targeted Instruction Tuning**

Moment-MMD 保留为：

* 统一一阶与二阶匹配的框架；
* 一个充分诊断过的 ablation；
* 说明简单的一阶—二阶 kernel 混合无法稳定提升；
* 支撑“二阶方向矩是 skew/small-target regime 中更稳健统计量”的结论。

这不是失败。MMD 文献本身就强调 kernel 的选择应与具体目标匹配，不存在任意 kernel 混合必然优于单 kernel 的结论。([NeurIPS 会议论文集][4])

# 应该选择哪个选项

我的明确建议是：

> **不要直接选截图中的 1、2 或 3；选择“低权重 selection sweep + 只训练两个 Pareto joint 候选”。**

具体执行：

1. 不必先重构成 marginal-std normalization；
2. 改用 (\lambda) 参数化；
3. selection-only 扫 (\lambda=0.005\sim0.1)；
4. 训练现有 (\beta=0.1) 或等价的 (\lambda\approx0.07)；
5. 再训练一个更靠近 GradCov 的 Pareto 点；
6. 两者都失败，再正式 pivot 到 pure second-order。

当前结果还不足以支持立刻选选项 2，因为 (\beta=0.1) 已经显示出“改善 (D_1)、几乎不伤害 (D_2)、提高 effective rank”的强烈诊断信号；不做一次下游训练就放弃，证据链会缺一环。

但也绝对不应该选选项 3，训练 (\beta=0.25,0.5,0.75)：它们在 selected-set 层面已经基本退化成 linear endpoint，属于高概率浪费算力。

[1]: https://jmlr.org/papers/v13/gretton12a.html?utm_source=chatgpt.com "A Kernel Two-Sample Test"
[2]: https://discovery.ucl.ac.uk/id/eprint/10181227/?utm_source=chatgpt.com "MMD-Fuse: Learning and Combining Kernels for Two-Sample Testing Without Data Splitting - UCL Discovery"
[3]: https://arxiv.org/abs/2101.07564?utm_source=chatgpt.com "Performance analysis of greedy algorithms for minimising a Maximum Mean Discrepancy"
[4]: https://papers.nips.cc/paper_files/paper/2012/hash/dbe272bab69f8e13f14b405e038deb64-Abstract.html?utm_source=chatgpt.com "Optimal kernel choice for large-scale two-sample tests"

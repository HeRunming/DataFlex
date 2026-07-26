这个结果不是“工作失败”，而是把原本混在一起的两个假设清楚地拆开了：

1. **二阶梯度方向矩确实有效。**
2. **一阶与二阶联合匹配在当前真实场景中并没有额外收益，反而明显有害。**

我也重新检查了最新 `fa` 分支：(\alpha) 范围、NaN/Inf、空集与零 target、有效候选预算、subsample 局部到全局映射等保护都已加入。  按代码推断，这些修改不会改变你已经用完整 270k、正常 gradient cache 跑出的结果，因为核心 kernel 和 greedy 更新没有变化。

## 这个结果具体说明了什么

从 (\alpha=1) 到 (\alpha=0.25)，结果几乎形成平台：

* Balanced：0.385 → 0.388
* STEM：0.362 → 0.369
* Humanities：0.407 → 0.408

但到了 (\alpha=0)，突然变成：

* STEM：0.392，约提高 2.3–3.0 个点；
* Humanities：0.430，约提高 2.1–2.3 个点；
* Balanced：0.411，约提高 2.3–2.6 个点。

所以它不是“二阶牺牲 target-majority，换来 minority 平衡”。**二阶方法在 STEM 和 Humanities 上都更好。**这说明 balanced metric 不是结果的主要原因。

更准确的结论是：

> 在固定的 `T_stem80`、当前梯度表征、5% selection budget 和单个训练 seed 下，任何非零的一阶 signed-gradient matching 都会降低二阶方向矩 coreset 的效果。

目前不能进一步说“一阶信息普遍有害”，更不能说“所有联合匹配都失败”，因为还有三个未区分因素。

### 第一，联合目标的两个分量并没有被校准

对单位梯度 (u)，你的 kernel 对应

[
\operatorname{MMD}*{k*\alpha}^{2}
=================================

\frac{\alpha}{2}
\left|\mu_S-\mu_T\right|_2^2
+
(1-\alpha)
\left|M_S-M_T\right|_F^2,
]

其中

[
\mu_P=\mathbb E_P[u],\qquad
M_P=\mathbb E_P[uu^\top].
]

因此，(\alpha=0.25) 并不意味着优化过程中“一阶贡献 25%、二阶贡献 75%”。一阶项的显式系数其实是 (0.125)，而且两个 discrepancy 的自然数值尺度、边际分数方差和排序敏感性可能完全不同。MMD 是标准 RKHS 分布距离，但具体 kernel 及其混合权重会决定它对何种分布差异敏感；多核 MMD 工作也通常会对 kernel 权重进行选择，而不是默认原始 kernel 已经在同一尺度。([机器学习研究杂志][1])

即便一阶项权重看似很小，也可能改变 30%–40% 的离散选择集合，造成明显性能变化。

### 第二，`T_stem80` 可能根本不是 joint 所需的场景

可能出现以下情况：

* target mean gradient 是小样本噪声；
* target 内部存在一定 cancellation，使 (\mu_T) 不稳定；
* 一阶 signed direction 与最终评测收益相关性较弱；
* 二阶矩已经充分捕获了需要覆盖的梯度子空间；
* 候选池中没有足够多“二阶相同但符号错误”的 decoy，因此 sign invariance 没有暴露缺陷。

LESS 本身利用低维梯度相似度进行 targeted selection，属于一阶影响近似。([Proceedings of Machine Learning Research][2]) 但 2026 年一项系统研究也发现：梯度表征总体最可靠，却没有单一 selection method 在所有任务和预算下占优；Random 在较大预算下经常追平部分选择方法，而 greedy round-robin 在低预算下往往更强。([arXiv][3]) 所以你的负结果其实符合当前领域逐渐形成的认识：**好的梯度表征不等于某一种组合方式或 selector 必然普适。**

### 第三，目前还是单 seed、单 target draw

现在的差距很大，值得认真对待，但还不能写成论文结论。训练 seed 只是其中一种随机性；更重要的是重新采样 `T_stem80` 后，target mean 和 second moment 是否稳定。

---

# 下一步应当怎么做

我不建议立即把主要算力投入 toy model。Claude 提出的 toy experiment 有价值，但在它之前，应先确定现在看到的是：

* 真正的一阶信息伤害；
* 一阶/二阶尺度失配；
* 离散 greedy 对很小 kernel 扰动的跳变；
* 还是单次训练偶然性。

## 第一优先级：低成本诊断，不训练模型

先做更细的近零 sweep：

[
\alpha\in
{0,;0.01,;0.03,;0.05,;0.10,;0.15,;0.25,;1}.
]

先只选择数据，不做 SFT。对每个 (\alpha) 输出：

1. 与 (\alpha=0) 的 Jaccard overlap 和 top-(K) overlap；
2. 一阶误差
   [
   D_1=|\mu_S-\mu_T|^2;
   ]
3. 二阶误差
   [
   D_2=|M_S-M_T|_F^2;
   ]
4. 每一步一阶和二阶 marginal score 的均值、标准差及相对大小；
5. selected gradient effective rank；
6. selected token 数、长度分布和数据领域分布。

这会区分三种情况：

* **性能在 (\alpha=0.01) 就崩掉**：更像 signed information 与任务冲突，或选择排序高度不连续；
* **存在 (\alpha=0.01\sim0.1) 的甜点区**：之前的 sweep 太粗；
* **归一化后 joint 恢复**：主要是 kernel scale 问题。

建议增加一个 scale-normalized 版本，而不是马上发明复杂 adaptive selector：

[
\widetilde D_\beta
==================

\beta\frac{D_1}{\mathbb E_{\text{random}}[D_1]}
+
(1-\beta)\frac{D_2}{\mathbb E_{\text{random}}[D_2]}.
]

分母可以由同预算 random subsets 在 gradient cache 上估计，不需要使用评测标签。这能让 (\beta) 更接近“相对重视程度”，同时保持一阶、二阶矩解释。

## 第二优先级：确认统计稳定性

不必重跑所有五个 (\alpha)。先保留：

[
\alpha\in{0,;0.25,;1},
]

或者在细 sweep 后把 (0.25) 替换为最有希望的小 (\alpha)。

最低配置：

* 3 个独立 `T_stem80` target draws；
* 每个 target draw 跑 3 个训练 seeds；
* 所有方法共享相同初始化与训练 seed，做 paired comparison。

即每个方法 9 次 SFT。最终论文最好增加到 5 个 target draws，训练 seed 可以根据方差决定是否保持 3 个。

这里 target draw 比单纯增加训练 seed 更重要，因为你的论文中心命题正是“小而偏斜的 target set 下如何稳定选择”。

同时再做一次 `T_hum80`。如果 (\alpha=0) 在两个 skew 方向都稳定获胜，二阶 skew-robust 的故事就会非常扎实。

## 第三优先级：分离“representation”和“selection algorithm”

当前 alpha sweep 同时涉及 kernel representation 和 MMD-herding selector。建议至少补下面四个组合：

[
\begin{array}{c|cc}
& \text{Round-Robin} & \text{MMD Greedy}\
\hline
k_1(u,v)=u^\top v & \checkmark & \checkmark\
k_2(u,v)=(u^\top v)^2 & \checkmark & \checkmark
\end{array}
]

也就是：

* signed-gradient + RR；
* signed-gradient + MMD；
* squared-gradient + RR；
* squared-gradient + MMD。

不需要显式构造 (d^2) 二阶特征，RR 中直接使用 ((u^\top v)^2) 即可。

这一步很重要，因为最新系统研究明确显示，selection algorithm 本身会显著影响结果：同一梯度 representation 下，RR 常在低预算占优，而部分 top-k/doubly-greedy 方法会落后。([arXiv][3])

做完以后，你才能回答：

> GradCov 的优势究竟来自二阶表示，还是来自 MMD 的 diversity/repulsion，还是两者缺一不可？

另外建议增加 1% budget。你当前 5% 大约是 13.5k 条，已经属于较大预算；已有研究观察到 targeted selection 的优势通常在低预算更明显，预算增大后 Random 会变得更有竞争力。([arXiv][3])

---

# Toy model 应该怎么做

Claude 的方向对，但“完全对称的 (+v/-v) cancellation toy”本身不足以证明 joint 优于二阶。

若 target 恰好是等权的 (+v) 与 (-v)，那么：

[
\mu_T=0,\qquad M_T=vv^\top.
]

这时一阶矩本来就不提供有效信息，二阶矩已经足够识别目标方向。joint 最多与二阶持平，不应期待它严格更好。

一个能够严格展示 joint 必要性的 toy，应同时放置两类 decoy：

* (Q_{\text{mean}})：与 target 的一阶均值相同，但二阶结构错误；
* (Q_{\text{moment}})：与 target 的二阶矩相同，但符号或混合权重错误；
* (Q_\star)：一阶与二阶都正确。

例如 target 是两个不共线方向、非等权混合：

[
P_T=p,\delta_{v_1}+(1-p)\delta_{v_2},
\qquad p\neq \frac12.
]

构造：

[
\mu_{Q_{\text{mean}}}=\mu_T,\quad M_{Q_{\text{mean}}}\neq M_T,
]

以及

[
M_{Q_{\text{moment}}}=M_T,\quad
\mu_{Q_{\text{moment}}}\neq\mu_T.
]

这样：

* pure first-order 无法排除 (Q_{\text{mean}})；
* pure second-order 无法排除 (Q_{\text{moment}})；
* joint 才能唯一识别 (Q_\star)。

更有说服力的版本不是人工向量，而是从真实 gradient cache 中寻找：

* 与 target 正向接近的候选；
* 与 target 反向接近但平方余弦相同的候选；
* 均值匹配但 effective rank 很低的 collapsed candidates。

这会成为“controlled real-gradient stress test”，比普通二维 toy 更接近 ICLR 机制实验。

---

# 论文现在应该如何定位

接下来存在两条都合理的路线。

### 路线 A：如果复现后始终是 (\alpha=0) 最优

那就不要勉强把 Moment-MMD 写成主方法。论文主线改成：

> **Second-Order Gradient Direction Matching for Robust Targeted Instruction Selection**

或者：

> **Beyond Mean Gradient Matching: Directional-Moment Coresets for Robust Targeted SFT**

Moment-MMD 仍然有用，但角色变为：

* 统一一阶与二阶方法的理论框架；
* 解释为什么 second-order endpoint 更稳定；
* 展示 joint mixture 是自然但失败的 ablation；
* 证明 target cancellation、skew 与 effective rank 之间的联系。

这并不比“joint 新方法”弱。相反，**一个清晰、可复现的负结果加上强机制解释，往往比硬凑出的组合增益更可信。**

### 路线 B：如果归一化或特定 regime 下 joint 获胜

那就把 claim 写成条件性的：

> 一阶与二阶矩具有互补性，但最优权重取决于 target gradient geometry；二阶在 cancellation/skew 下占优，signed mean 在 sign-sensitive target 下有用。

随后考察一个非常简单的 target statistic：

[
c_T=|\mu_T|_2^2,
]

因为对单位梯度有 (\operatorname{tr}(M_T)=1)，所以 (c_T\in[0,1]) 可以视为 gradient directional concentration：

* (c_T) 小：均值 cancellation 强，更应依赖二阶；
* (c_T) 大：方向集中，一阶信号可能更可靠。

先画“不同 target draw 的最佳 (\alpha)”与 (c_T) 的关系。只有确实出现稳定相关性后，再考虑 adaptive (\alpha)，不要现在直接添加更多机制。

## 我对当前结果的最终判断

Claude 的大方向是对的：**现在没有证据支持“joint 比二阶更好”**。但下一步不应直接跳到 toy 并试图证明预设结论。

优先顺序应当是：

1. 近零 alpha 与 component-scale 诊断；
2. 3 target draws × 3 training seeds 的复现；
3. squared-gradient representation 与 MMD selector 的解耦实验；
4. 1% 低预算实验；
5. 再做能够同时让两个 endpoint 失败的 joint-identifiability toy。

只要多 target draw 后 (\alpha=0) 仍然稳定领先，你现在实际上已经得到了一条比 Moment-MMD 更明确的论文主线：**不是“更多 moment 总是更好”，而是小样本、异质和偏斜 target 下，signed mean gradient 是脆弱统计量，二阶方向矩才是稳定的 selection signal。**

[1]: https://www.jmlr.org/papers/v13/gretton12a.html "A Kernel Two-Sample Test"
[2]: https://proceedings.mlr.press/v235/xia24c.html?utm_source=chatgpt.com "LESS: Selecting Influential Data for Targeted Instruction Tuning"
[3]: https://arxiv.org/abs/2602.14696 "A Critical Look at Targeted Instruction Selection: Disentangling What Matters (and What Doesn't)"

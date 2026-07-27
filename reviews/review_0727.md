Claude 现在的判断基本正确。**这组结果足以支持“停止继续调 (\lambda)，把 pure second-order 设为默认方法”的研究决策，但还不足以宣称二阶方法已被统计显著地证明优于 joint。**

最合理的选择是：

> **允许它把结果推送到 GitHub，并启动精简版 `T_hum80` mirror；不再做任何额外的 (\lambda) 调参。Mirror 完成后，主算力转向独立 target draws 和强 baseline，而不是继续研究 Moment-MMD 的混合权重。**

## 这三组 paired seeds 到底说明了什么

三个差值是：

[
-0.93,;-0.77,;-0.07\ \text{个百分点},
]

平均：

[
\overline{\Delta}=-0.59\ \text{个百分点}.
]

它支持的可靠结论是：

> 在固定的 `T_stem80`、固定 selected subsets 和三个配对训练随机种子下，加入经过校准的 signed first-moment 分量没有带来收益；pure directional second moment 在三个运行中都不低于 joint，平均高约 0.59 个百分点。

Claude 修改后的表述很好：

> Pure directional second-moment is at least as strong as the best calibrated joint on average, and the joint never outperforms it across three paired seeds in this target regime.

但不要写成：

* “一阶信息必然有害”；
* “任何非零一阶权重都会严格降低性能”；
* “二阶显著优于 joint”；
* “负结果已经 airtight”。

原因是 seed 2 基本平局，而且只有三个训练 seed。简单估算这三个 paired differences 的标准差约为 0.46 个百分点，标准误约为 0.26 个百分点；样本量太小，传统置信区间仍会覆盖零。深度学习训练的随机性确实可能改变方法排序，因此多次运行很重要。([ACL Anthology][1])

所以应当把它理解成：

> **足以停止一个低回报的研究分支，但不能单独承担论文的主要统计证据。**

## 现在立即做什么

### 1. 推送结果，并冻结 Moment-MMD 分支

让 Claude 把以下内容提交：

* 三个 paired-seed 原始结果；
* 每个 seed 的 STEM、Humanities、balanced 分数；
* paired difference；
* (\lambda=0,0.02,0.07,\text{linear}) 的 selection diagnostics；
* kernel-scale mismatch 和 (\lambda) 重参数化过程；
* 精确实验配置、checkpoint、subset hash 和评测命令。

然后明确冻结：

[
\lambda_{\mathrm{default}}=0.
]

后面不再扫 (0.005,0.01,0.03) 等更多权重，也不再尝试新的静态归一化。现有结果已经充分说明，继续调权重的预期收益很低，并且容易形成 reviewer 眼中的 post-hoc tuning。

### 2. 启动精简版 `T_hum80` mirror

做，但只做：

[
\lambda=0,\qquad \lambda=0.02,\qquad \text{linear endpoint}.
]

使用三个 paired seeds。若 GradCov 和 linear 的对应 runs 已经在完全相同的配置下存在，直接复用，不要重复训练。

这个实验回答的是一个重要问题：

> pure second-order 的优势是否只出现在 STEM-majority target，还是对相反 skew 方向也成立？

判定方式：

* 如果 `T_hum80` 下也是
  [
  \text{GradCov}\geq \lambda=.02>\text{linear},
  ]
  则可以较有把握地关闭 joint 分支。
* 如果 `T_hum80` 下 joint 明显反超，说明一阶信息的价值依赖 target geometry，论文可能转为条件性方法，而不是单纯二阶方法。
* 如果 GradCov 与 joint 基本相同，则写成“joint 未稳定改善 endpoint”，而不是“一阶有害”。

Mirror 是值得花 7–8 小时的，因为它测试的是外部有效性，而不是继续调参。

## Mirror 之后最重要的实验不是更多训练 seed

真正的下一个核心变量应该是 **target draw**。

你现在所有三次 paired runs 共用同一个 `T_stem80`。它们只测量了训练随机性，没有测量小 target set 的抽样随机性。但你的核心叙事恰恰是：

> signed mean gradient 在小样本、异质、偏斜 target 下不稳定；directional second moment 更稳健。

因此论文的统计单位最终应当是“独立 target draw”，而不是仅仅“训练 seed”。

推荐的最小设计是：

* 两个 skew 方向：STEM 80/20、Humanities 80/20；
* 每个方向 5 个独立 target draws；
* 每个 draw 先使用一个共享 training seed；
* 方法至少包括：

  * Directional Second-Moment Coreset；
  * LESS；
  * NICE；
  * Random；
  * 当前最强 gradient round-robin baseline；
  * GIST，若代码和环境允许。

随后在每个方向选择一个代表性 draw，再增加 2 个训练 seeds，用于估计训练方差。

这比“对一个 target 再跑十个训练 seed”更有价值，因为它能直接检验方法对 target sampling noise 的鲁棒性。

## 必须补 representation–selector 解耦

2026 年一项针对 targeted instruction selection 的系统研究明确区分了两个组成部分：

1. 数据 representation；
2. selection algorithm。

该工作发现，梯度 representation 通常最可靠，但没有任何 selector 在所有任务和预算下始终最好；低预算时 gradient representation 配合 greedy round-robin 往往很强。([arXiv][2])

因此你的论文不能只比较：

[
\text{LESS}\quad\text{vs}\quad\text{GradCov-MMD}.
]

否则 reviewer 无法判断增益究竟来自：

* 二阶 representation；
* MMD/herding 的 diversity；
* 或两者共同作用。

至少需要下面这个 (2\times2)：

| Representation              | Relevance / round-robin |       MMD coreset |
| --------------------------- | ----------------------: | ----------------: |
| signed first-order (u)      |    First-RR / LESS-like |        Linear-MMD |
| second-order ((u^\top v)^2) |               Second-RR | Second-Moment MMD |

最好再加：

* target-subspace scoring；
* second-order top-k relevance；
* second-order MMD without diversity/repulsion 的消融。

如果只有 MMD+second-order 有效，你的贡献是 representation 与 coreset objective 的结合；如果所有 second-order selectors 都改善，那么核心贡献主要是二阶 representation。

## GIST 是必须正面处理的相关工作

GIST 在 2026 年提出了面向 LoRA 优化几何的 target subspace 方法：它不采用逐坐标的对角近似，而是利用 SVD 恢复 target-specific subspace，再根据训练梯度与目标方向的对齐选择数据。([arXiv][3])

它和你的工作叙事有明显重叠：

* 都使用梯度；
* 都强调低维结构；
* 都涉及 target subspace；
* 都针对 PEFT/LoRA。

你的区别必须非常明确：

> GIST 估计一个 target subspace，并按对齐程度评分；你匹配单位梯度方向分布的二阶矩，并通过 coreset objective 同时控制 target coverage 与候选冗余。

建议加入两个机制性对比：

1. **Subspace recovery**
   [
   |\widehat P_S-\widehat P_T|_F
   ]
   或 principal-angle error。

2. **Skew stability**
   在不同 target draws 下比较：

   * selected-set Jaccard；
   * estimated subspace variance；
   * effective rank；
   * downstream variance。

这样才能避免论文被评价为“GIST 的另一种 kernel 版本”。

## 主实验矩阵怎样收缩才不会爆算力

不建议一次性完整跑：

[
4\text{ target sizes}\times
2\text{ budgets}\times
2\text{ directions}\times
5\text{ draws}\times
6\text{ methods}\times
3\text{ seeds}.
]

可以分阶段。

第一阶段，验证主结论：

* target size：80；
* budget：5%；
* skew：STEM80、HUM80；
* target draws：5；
* methods：GradCov、LESS、NICE、Random、RR、GIST；
* training seed：1 个共享 seed。

第二阶段，验证适用范围：

* target size：16、64、128；
* budget：1%、5%；
* 只比较 GradCov、最强 baseline、Random；
* 每个设置 3 个 target draws；
* 先一个 training seed。

第三阶段，对关键 setting 增加 paired training seeds。

这能把算力集中在最有判别力的实验上。

## 论文主线应该怎样改

建议不再把方法正式称为 `GradCov`。由于梯度经过逐样本 L2 normalization，它匹配的不是通常意义上的原始 covariance，而是

[
M_P=\mathbb E_{u\sim P}[uu^\top],
\qquad
u=\frac{\Pi g}{|\Pi g|}.
]

更准确的名称可以是：

* **Directional Second-Moment Coreset，DSMC**；
* **Gradient Directional-Moment Coreset，GDMC**；
* **Directional-Moment MMD，DM-MMD**。

论文核心命题建议写成：

> Mean-gradient matching uses a signed first moment that can be unstable under target heterogeneity, cancellation, and finite-sample skew. We instead match the second moment of normalized gradient directions, preserving target-relevant update subspaces while remaining invariant to sign cancellation.

LESS 是 optimizer-aware 的低维梯度相似度选择方法，因此它是非常自然的一阶对照。([Proceedings of Machine Learning Research][4]) 但当前大规模 data-selection 研究也提醒，复杂方法可能在预算和候选池扩张后输给 Random，所以你必须保留 Random、token-matched Random，并至少验证一个较低 selection budget。([arXiv][5])

## Moment-MMD 在论文里保留到什么程度

保留，但不要让负结果占据主线。

主文中可以有一个统一公式：

[
k_\lambda(u,v)
==============

(u^\top v)^2+\lambda\frac{1+u^\top v}{2}.
]

然后报告：

* endpoint (\lambda=0) 是 directional second moment；
* (\lambda\to\infty) 接近 signed first-moment matching；
* 原始 convex parameterization 存在 kernel-scale mismatch；
* 重参数化后测试了真正 interior 的 (\lambda)；
* interior joint 未稳定改善 pure second-order endpoint。

一句话结论即可：

> The unified family reveals that the second-order endpoint is consistently at least as strong as the best calibrated joint on the studied skewed target.

详细 scale diagnosis、Jaccard、(D_1/D_2) 和完整 sweep 放附录。

## 现在给 Claude 的具体回复

可以直接回复它：

> Push the paired-seed results and freeze (\lambda=0) as the default method. Start the slim T_hum80 mirror with ({\lambda=0,\lambda=0.02,\text{linear}}) and paired seeds 42/1/2, reusing any endpoint runs that exactly match the configuration. Do not run additional lambda sweeps. In parallel, prepare the independent-target-draw protocol and the representation-vs-selector ablations. After the mirror, shift compute to second-order external validity rather than further Moment-MMD tuning.

最终决定就是：

**确认 pivot；现在启动精简 `T_hum80` mirror；然后将主要资源投入独立 target draws、强 baseline、representation–selector 解耦和 GIST 对比。**

[1]: https://aclanthology.org/2023.findings-acl.172/?utm_source=chatgpt.com "Exploring Variation of Results from Different Experimental Conditions - ACL Anthology"
[2]: https://arxiv.org/abs/2602.14696?utm_source=chatgpt.com "A Critical Look at Targeted Instruction Selection: Disentangling What Matters (and What Doesn't)"
[3]: https://arxiv.org/abs/2602.18584?utm_source=chatgpt.com "GIST: Targeted Data Selection for Instruction Tuning via Coupled Optimization Geometry"
[4]: https://proceedings.mlr.press/v235/xia24c.html?utm_source=chatgpt.com "LESS: Selecting Influential Data for Targeted Instruction Tuning"
[5]: https://arxiv.org/abs/2503.01807?utm_source=chatgpt.com "Large-Scale Data Selection for Instruction Tuning"

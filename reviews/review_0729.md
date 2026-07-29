Claude 的结论总体正确，仓库中的 CSV 也与它汇报的数字一致。HUM80 三个 paired seeds 下，GradCov 的 balanced mean 为 0.4069，joint 为 0.4016；三个差值全部为负。和 STEM80 合在一起，现在共有 6 个 paired comparisons，方向全部一致，平均差约为 (-0.56) 个百分点。

但对结果的科学含义需要再收紧一点。

## 这组结果真正说明了什么

第一，它确实排除了一个重要替代解释：

> STEM80 上 pure second-order 较好，不只是因为 target 偏向 STEM。

在反方向的 HUM80 target 上，aggregate ordering 仍然是：

[
\text{second-order}

>

\text{calibrated joint}

>

\text{linear}.
]

而且 joint–GradCov 的平均差在两个方向几乎相同：

[
-0.59\text{ pp}
\quad\text{vs}\quad
-0.54\text{ pp}.
]

这是一种很干净的方向一致性结果。

第二，现在足以作出**研究资源决策**：

> 冻结 (\lambda=0)，不再继续调静态的一阶—二阶 kernel mixture。

这并不证明所有可能的一阶信息都无用，也不排除自适应、非线性或任务依赖的组合；它证明的是：

> 在目前考察的固定 kernel family、LESS-aligned gradient protocol、小型 skewed target 和 5% budget 下，加入经过认真校准的 signed first-moment 分量没有稳定改善 second-order endpoint。

因此“joint branch closed”可以作为项目决策，但论文里不要写成普适定理。

第三，这个结果还不是“DSMC 已经击败现有方法”的主证据。当前只完成了：

* second-order 对 joint/linear 的内部消融；
* 两个固定 target sets；
* 同一个 MMLU STEM/Humanities 问题结构；
* 同一个模型、候选池、warmup cache、预算和 Adam-candidate/SGD-target protocol。

尚未证明：

* 对独立 target draws 稳定；
* 对 LESS、NICE、GIST、gradient-RR、Random 稳定领先；
* 优势来自 second-order representation，还是 MMD diversity；
* 在其他 target size、budget、模型和任务上成立。

## Claude 有一句话需要更精确

它说两个方向上 ordering 完全相同，这对 **balanced aggregate** 是正确的，但不是每个子分数都逐项占优。

例如 HUM80 seed 42：

[
\text{STEM}_{\rm joint}=0.3853

>

\text{STEM}_{\rm GC}=0.3809,
]

joint 在 STEM 子分数上反而高一些；只是 Humanities 上下降更多，所以 balanced score 更低。

因此论文应写：

> DSMC achieves the best balanced aggregate under both skew directions.

不要写：

> DSMC dominates the joint on every subdomain and every seed.

不过我按 HUM80 的真实 20/80 比例重新计算了 target-weighted score：

[
0.2,\text{STEM}+0.8,\text{Humanities}.
]

三个 seed 的平均值是：

[
\text{GradCov}=0.4198,\qquad
\text{joint}=0.4124,
]

差约 (-0.74) 个百分点。因此 HUM80 的结论也不是由 balanced metric 人为制造的。建议把这个 target-weighted 指标补进 summary，和 balanced score 同时报告。

## 现在最大的概念问题：先定义“skew”是什么

在继续跑实验之前，必须明确论文中的 skew 设定究竟表达什么。

一种解释是：

> 80/20 就是真实目标分布。

这时主要指标应是 80/20 target-weighted performance。

另一种解释是：

> 潜在目标能力分布是平衡的，但观察到的小 target set 因抽样偏差变成 80/20。

这时 balanced performance 是合理的主要指标，论文研究的是：

> robustness to biased finite target samples。

我认为第二种解释更适合你现在的结果和论文故事。否则，一个方法忠实匹配 80/20 target 并不一定是缺点。

下一阶段 protocol 应明确写成：

[
P^\star=\text{balanced latent evaluation distribution},
]

但 query/target set 来自受控偏斜分布

[
Q_\rho,\qquad
\rho\in{0.5,0.8,0.9}.
]

这样你的 claim 才是：

> DSMC is robust when the observed few-shot target set is a skewed finite sample of a broader target capability distribution.

## 下一步不应直接启动完整 target-draw 大矩阵

Claude 提议设计 target-draw protocol 并先带回来 review，这个选择是对的。

但在跑

[
2\text{ directions}
\times5\text{ draws}
\times6\text{ methods}
=60
]

次 SFT 之前，还应该先做一个很便宜、但对论文贡献归因至关重要的实验。

### 第一步：先做 representation × selector 的 (2\times2)

当前方法同时改变了两件事：

1. representation 从 signed first moment 变成 second directional moment；
2. selector 从 relevance/top-k 变成 MMD coreset。

至少比较：

[
\begin{array}{c|cc}
&\text{relevance/RR}&\text{MMD coreset}\
\hline
\text{first-order }u
&\text{First-RR/LESS-like}
&\text{Linear-MMD}\
\text{second-order }(u^\top v)^2
&\text{Second-RR}
&\text{DSMC}
\end{array}
]

先在现有 STEM80 和 HUM80 上各跑 seed 42。DSMC 和 Linear-MMD 已经有结果，所以真正新增的主要是 First-RR 和 Second-RR，可能只需约 4 次 SFT。

这个实验决定论文到底应该声称什么：

* 如果所有 second-order selectors 都好，贡献主要是 **second-order representation**；
* 如果只有 DSMC 好，贡献是 **second-order representation + coreset diversity**；
* 如果 Second-RR 比 DSMC 更好，就必须调整方法，不能先跑几十次 DSMC external-validity 实验。

这一步很重要，因为 2026 年最新系统研究明确指出，targeted instruction selection 必须拆开 representation 和 selector；其结果显示 gradient representation 较可靠，而 greedy round-robin 在低预算下经常很强，没有任何单一 selector 普遍占优。([arXiv][1])

## 第二步：设计 independent-target-draw protocol

完成 (2\times2) gate 后，再执行 target draws。

建议第一版固定：

* target size：80；
* skew：STEM 80/20、Humanities 80/20；
* 每个方向：5 个预先生成的独立 draws；
* 每个 draw 内所有方法共享同一个 training seed；
* 方法间使用 paired comparison；
* 不允许根据这 10 个 draws 的结果再调 DSMC。

每个 draw 应保存：

* target example IDs；
* subject/category composition；
* target data hash；
* target gradient hash；
* target-set overlap matrix；
* shared training seed；
* candidate cache hash；
* selection indices hash。

如果 reservoir 足够大，优先生成互不重叠的 draws；做不到时，允许重叠，但必须报告 pairwise overlap。

训练 seed 不必全部固定为 42，可以预先轮换：

[
42,1,2,42,1,
]

但一个 draw 内所有方法必须使用相同 seed。这样能在不增加运行数的情况下，避免所有 draw 都依赖同一种训练轨迹。然后选择一个代表性 draw，再补 3 个 paired training seeds，单独估计训练方差。

统计分析的单位应是 target draw：

[
\Delta_d
========

## \text{score}_{\rm DSMC,d}

\text{score}_{\rm baseline,d}.
]

报告：

* mean paired difference；
* median difference；
* 多少个 draw 获胜；
* 以 target draw 为 cluster 的 bootstrap CI；
* 两个 skew 方向的 interaction。

不要把 evaluation items 当成独立重复来夸大显著性。

## 第三步：方法矩阵分两阶段跑

不建议一上来就 60 次。

先做 pilot：

[
2\text{ directions}
\times2\text{ draws}
\times6\text{ methods}
=24\text{ SFT runs}.
]

方法建议为：

1. DSMC；
2. Second-RR；
3. LESS；
4. GIST；
5. NICE；
6. token-matched Random。

如果你希望保留普通 Random，则可以让 Random subset 同时满足样本数与 token 数两个版本；主文优先 token-matched Random。

LESS 是最直接的一阶 optimizer-aware gradient-similarity baseline。([Proceedings of Machine Learning Research][2]) GIST 则利用 target validation gradients 的 SVD 构造低秩 task subspace，再按 candidate 对该 subspace 的 alignment 打分，它与“second-order/subspace”叙事高度接近，属于必须正面对比的 baseline。([arXiv][3])

Random 也不能省略。较大规模的 instruction selection 研究发现，多种复杂方法在扩大 pool 或 budget 后可能不如 Random，因此任何 data-selection claim 都应包含严格匹配预算的随机基线。([arXiv][4])

Pilot 没有工程问题且 DSMC 仍有竞争力后，再扩展到每个方向 5 draws。

## 第四步：最后再扫 target size 和 budget

不要现在同时把所有轴展开。

先在 (n=80,K=5%) 建立主结论。之后只保留：

* DSMC；
* 最强 gradient baseline；
* GIST；
* token-matched Random。

再测试：

[
n_T\in{16,64,128},
\qquad
K\in{1%,5%}.
]

1% budget 很重要，因为最新 controlled study发现 gradient representation 与 greedy-RR 的优势通常在低预算更明显，预算增大后方法差异会缩小。([arXiv][1])

## 当前论文主线可以冻结了

方法名称建议正式从 `GradCov` 改掉，因为它匹配的是单位梯度方向的 uncentered second moment：

[
M_P=\mathbb E_{u\sim P}[uu^\top],
\qquad
u=\frac{\Pi g}{|\Pi g|}.
]

我更推荐：

> **Directional Second-Moment Coreset，DSMC**

当前最稳妥的 headline 是：

> **Directional Second-Moment Coresets for Robust Targeted Instruction Tuning**

核心论点暂时写成：

> Signed mean-gradient matching can be unstable under heterogeneous and skewed few-shot target sets. DSMC instead matches the directional second moment of normalized projected gradients, capturing target-relevant update subspaces while encouraging nonredundant coverage.

Moment-MMD 的整个 calibration arc 很有价值，但放在 appendix 或 mechanism section：

* 原始 (\alpha) 尺度失衡；
* random-MMD normalization 不控制 greedy marginal；
* (\lambda) 重参数化；
* genuine interior joint；
* 两个 skew 方向都未稳定改善 endpoint。

## 对 Claude 问题的明确回复

现在应该选择：

> **让 Claude 先设计 independent-target-draw protocol，并带回来 review；不要立即运行完整矩阵。与此同时，先实现并跑完 representation × selector 的 (2\times2) attribution gate。**

可以直接告诉 Claude：

> Freeze DSMC and stop all Moment-MMD tuning. Before launching the full target-draw matrix, prepare a preregistered target-draw protocol that defines skew as sampling bias relative to a balanced latent evaluation distribution. Also run the first-order/second-order × relevance-RR/MMD 2×2 attribution pilot on the existing STEM80 and HUM80 targets. After that gate, run a 2-draw-per-direction pilot with DSMC, Second-RR, LESS, GIST, NICE, and token-matched Random; expand to five draws only after validating the pipeline and attribution story.

另外有一个很小的仓库问题：manifest 中记录的 `git_commit` 是 `809593d`，结果提交是 `286f44d`。 如果前者是实际运行代码版本，这是合理的，但建议字段改名为 `run_git_commit`，并额外记录 `results_commit`，避免以后误解。

[1]: https://arxiv.org/abs/2602.14696?utm_source=chatgpt.com "A Critical Look at Targeted Instruction Selection: Disentangling What Matters (and What Doesn't)"
[2]: https://proceedings.mlr.press/v235/xia24c.html?utm_source=chatgpt.com "LESS: Selecting Influential Data for Targeted Instruction Tuning"
[3]: https://arxiv.org/abs/2602.18584?utm_source=chatgpt.com "GIST: Targeted Data Selection for Instruction Tuning via Coupled Optimization Geometry"
[4]: https://arxiv.org/abs/2503.01807?utm_source=chatgpt.com "Large-Scale Data Selection for Instruction Tuning"

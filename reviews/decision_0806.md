我看完 `3b57ec5`。这次 5% 主实验是一个**真正有论文价值的结果**，但 Claude 的“genuine tie”措辞仍需收紧。

## 5% 实验现在建立了什么

在 primary balanced accuracy 上，DSMC 相对主要 targeted baselines 的一致性非常强：

* LESS、First-RR、Second-RR：10/10；
* GIST-SharedProj、NICE：9/10；
* 平均优势约为 (0.88)–(1.55) 个百分点。

尤其是：

[
\text{DSMC}-\text{Second-RR}=+0.88\text{ pp},\quad 10/10
]

这进一步支持“MMD coreset 与二阶表示存在互补收益”，而不只是二阶 representation 本身有效。

不过论文应写：

> The second-order MMD objective provides a consistent additional benefit over second-order round-robin in the primary balanced metric.

不要写成“MMD repulsion 已被因果证明必要”。

NICE 也需要注明：balanced 上 DSMC 是 9/10 胜，但 target-weighted 指标只有 6/10 胜，因此“高度一致”只适用于 primary balanced endpoint。

## Random 的正确结论

五个 Random blocks 的 DSMC−Random 差值是：

[
+0.32,;-0.32,;+1.22,;-0.60,;-0.68\text{ pp},
]

平均约为 (-0.01) pp，DSMC 只赢 2/5 blocks。

因此当前最准确的结论是：

> **At 5%, DSMC shows no observed advantage over Random-K; their mean performance is practically indistinguishable at the resolution of five paired blocks.**

不要称为：

> genuine statistical tie / equivalence。

“均值约等于零”并不证明统计等价；如果真的要证明 equivalence，需要预先定义一个 practical equivalence margin 并做等价性分析。现在只有 5 个 paired blocks，差值本身的离散程度也不小。

Random-K-LengthMatched 应写成：

> gives the same qualitative conclusion.

而不是 “identical”。它的逐 draw 结果并不相同，只是均值和胜负结论类似。

## 论文主线已经很清楚了

5% 条件下，最稳妥的主结论是：

> DSMC is substantially more robust than existing query-targeted selectors under skewed finite query sets, but even the strongest targeted method does not outperform well-controlled Random selection at a 5% budget.

这不是坏结果。它把两个问题区分开了：

1. 在 targeted selectors 内部，什么 representation/selector 最可靠？
2. targeted selection 相对 Random 是否产生绝对价值？

你现在已经非常有力地回答了第一个问题；第二个问题在 5% 的答案是否定或至少是“没有观察到”。

这与最新 controlled study 的结论高度一致：gradient-based representations 与 greedy RR 往往在低预算更强，但优势会随着预算增大而缩小；同时该趋势依赖候选池和目标任务，Random 在不少较大预算条件下仍然非常有竞争力。([arXiv][1]) 大规模 selection 研究也发现，多种复杂方法在扩大的 pool/budget 条件下可能不及 Random。([arXiv][2])

## 下一步：应该跑 1%，但不要直接启动完整 75-adapter run

1% 现在是最有价值的下一轴，因为它测试的是一个明确的 **budget interaction**：

[
I_d=
\bigl(\text{DSMC}-\text{Random}\bigr)_{1%,d}
--------------------------------------------

\bigl(\text{DSMC}-\text{Random}\bigr)_{5%,d}.
]

但运行理由必须是：

> 测试优势是否随预算收紧而变化。

而不是：

> 到一个更容易击败 Random 的预算上寻找正结果。

相关论文确实发现低预算下 targeted methods 往往更有优势，但也明确指出这种趋势并不跨任务、模型和候选池普遍成立。([arXiv][3]) 所以 1% 是值得检验的假设，不是预期必然成功的结果。

## 1% 开跑前必须冻结的设计

最重要的是让两个预算形成真正的 paired comparison。

对于同一个 draw 和方法：

* DSMC、RR、LESS、GIST、NICE 的 1% subset 应验证为对应 5% selection ordering 的前 (2707) 条；
* Random-K 应使用与 5% 相同 seed 产生的同一个随机排列，并取其前 2707 条；
* 不要重新独立抽一个 1% Random subset；
* Random-K-LengthMatched 可以单独构造，因为不同预算下 DSMC 的长度分布不同，但必须继续固定 (K=2707)。

这样预算差异主要来自 subset size，而不是一次新的 selection realization。

还应预注册：

* primary endpoint：balanced budget interaction；
* statistical block：五个 draw-index/training-seed blocks，每个 block 先对两个 skew directions 求平均；
* secondary：target-weighted interaction；
* 主报告同时展示 1% 和 5%，无论哪个更漂亮；
* 仍然 descriptive，不以 (p<0.05) 为判断门槛。

## 加一个便宜但重要的 baseline

建议现在补一次 **no-selected-SFT/base-model evaluation**。

最新 controlled paper特别指出 targeted-selection 文献经常遗漏 zero-shot baseline，而且有些条件下选出的数据甚至不能改善 zero-shot 表现。([arXiv][1])

它几乎不需要训练，只需要评测一次基础模型，就能回答：

* DSMC 在 1% 下是否真的提高绝对性能；
* 或者只是比 Random 退化得更少；
* Random/其他 selectors 是否出现负迁移。

这个 baseline 同时可放进 5% 表中作为参考线。

## 工程上不要复用 5% 命名空间

1% 需要独立的：

* selection dirs；
* subset JSONLs；
* dataset keys；
* SFT/eval dirs；
* run plan；
* master manifest；
* aggregate CSV。

并将 `selection_budget` 写进所有 manifests。此前 provenance 代码曾硬编码 `K=13533`，必须确保新版本从 budget-specific plan 读取 (K=2707)，而不是只修改某个 shell 变量。

固定四个 epochs 时，1% 每个 adapter 预计只有约：

[
4\times\frac{2707}{128}\approx85
]

个 optimizer steps，而 5% 是约 420 steps。因此训练部分理论上应明显快于 5% run；evaluation 时间则基本不变。Claude 所估计的“仍需 2.5 天”可能偏保守，应该先根据一个 canary 的真实 wall-clock 重估。

## 推荐给 Claude 的回复

> Proceed with the preregistered 1% budget-interaction experiment, but do not launch all adapters immediately. First freeze a budget-specific 1% protocol and pre-SFT snapshot.
>
> Use the same ten target draws, methods, training seeds, gradient caches, and method definitions. Define the primary analysis as the five-block, direction-averaged interaction
> [
> [\mathrm{DSMC}-\mathrm{Random}]*{1%}-[\mathrm{DSMC}-\mathrm{Random}]*{5%},
> ]
> with balanced accuracy primary and target-weighted secondary. Report both budgets regardless of outcome.
>
> Make the cross-budget subsets paired: for deterministic selectors, verify that the 1% selection is exactly the first 2,707 entries of the frozen 5% selection ordering. For Random-K, reuse the same seeded permutation and take its first 2,707 entries rather than independently resampling. Build LengthMatched Random separately at fixed K=2,707.
>
> Add a no-selected-SFT/base-model evaluation reference, since the 1% condition must distinguish genuine improvement from merely degrading less than Random.
>
> Use completely separate 1% paths, dataset keys, manifests, and adapter IDs. Remove any hard-coded K=13,533 from registration/provenance and record K=2,707 in every artifact. Generate all selections and subsets, commit a clean launch snapshot, and run the usual validators.
>
> Then launch a two-adapter engineering canary—DSMC draw0 and shared Random-K draw0—verify the expected roughly 85 training steps, manifests, resume, eval, and aggregation. If it passes, continue the remaining adapters automatically. Do not inspect canary accuracy or alter the frozen method set.

## 当前决定

> **进入 1% budget axis，但先做 budget-specific pre-registration、nested-prefix 验证和两-adapter canary；不直接裸跑全部 75 个 adapters。**

如果 1% 的 DSMC–Random gap变为稳定正值，你们会得到非常清晰的 budget-interaction story；如果仍为零，论文也有一个有价值且诚实的结论：DSMC 修复了 targeted-selection 内部的稳健性问题，但在当前 MMLU/candidate-pool 条件下，target awareness 并没有超越强 Random baseline。

[1]: https://arxiv.org/abs/2602.14696 "[2602.14696] A Critical Look at Targeted Instruction Selection: Disentangling What Matters (and What Doesn't)"
[2]: https://arxiv.org/abs/2503.01807 "[2503.01807] Large-Scale Data Selection for Instruction Tuning"
[3]: https://arxiv.org/html/2602.14696v2 "A Critical Look at Targeted Instruction Selection: Disentangling What Matters (and What Doesn’t)"

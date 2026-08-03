结论很明确：

> **选择 2：先把 5% 主实验扩展到每方向 5 个 draws。暂时不要先跑 1%，也不要选择“哪个预算信号更清楚就扩哪个”。**

当前 pilot 是一个很好的结果，但 Claude 对其中两处表述略有过度。

## 这组结果真正建立了什么

在 4 个 draw–seed replicates 上，DSMC 相对以下方法表现非常一致：

* LESS：4/4 胜，平均 (+0.68) pp；
* First-RR：4/4 胜，平均 (+1.36) pp；
* Second-RR：4/4 胜，平均 (+0.88) pp；
* GIST-SharedProj：4/4 胜，平均 (+1.58) pp。

这已经是很有希望的 external-validity pilot：二阶表示带来的优势没有停留在固定 STEM80/HUM80 target 上，而是在四个新的、无重叠 target draws 中继续出现。

而且它补强了此前的机制结论：

[
\text{DSMC}

>

\text{Second-RR}
]

说明结果不只是“用了二阶表示”；MMD coreset selector 至少在当前四个 replicates 中也 consistently 优于二阶 RR。

但现在仍然只能称为 **pilot-scale consistency evidence**，因为每个方向只有两个 draws。

## Claude 的第一处过度表述：不能笼统说“击败所有 targeted selectors in both directions”

NICE 是例外。

NICE 的 balanced 差值为：

* STEM：(+1.71,+2.79) pp，DSMC 胜；
* HUM：(-0.95,+0.45) pp，一胜一负。

所以 HUM 方向平均是：

[
\frac{-0.95+0.45}{2}
====================

-0.25\text{ pp},
]

即 NICE 在 HUM 方向的两-draw平均值略高于 DSMC。target-weighted 指标下，这个 HUM 方向的 NICE 优势还稍大。

更准确的 headline 应是：

> DSMC outperforms LESS, First-RR, Second-RR, and GIST in every pilot replicate; NICE is mixed, while DSMC is better on average across all four replicates.

不要写：

> DSMC beats every targeted selector in both skew directions.

## 第二处过度表述：Random tie 目前不是四个完全独立的 Random comparisons

Random-K 在相同 draw index 的两个方向之间复用了同一个 adapter，因此这里只有两个 unique Random adapters：

* draw index 0 / seed 42；
* draw index 1 / seed 1。

其结果有很明显的 block pattern：

[
\begin{aligned}
\text{idx0:}&\quad +0.43,,+0.21\text{ pp},\
\text{idx1:}&\quad -0.43,,-0.20\text{ pp}.
\end{aligned}
]

换言之：

* seed 42 / draw0 block：DSMC 两个方向都赢；
* seed 1 / draw1 block：DSMC 两个方向都输。

这意味着当前“DSMC 与 Random 打平”的证据，实际上主要来自 **两个 draw-index/training-seed blocks 正负抵消**，而不是四个独立 Random realizations。

这正是为什么下一步应扩展 5% draws：draw2、3、4 会增加三个新的 Random subsets、三个新的 training seeds，以及六个新的 DSMC target-draw runs。

所以 Claude 所说：

> 扩到 5 draws 不解决 Random tie

并不正确。它虽然不测试 budget interaction，但会直接回答一个更基础的问题：

> 当前 Random tie 是稳定现象，还是仅由两个 draw-index blocks 偶然抵消产生？

## LengthMatched Random 的措辞也要收紧

Random-K-LengthMatched 与 Random-K 的**总体结论相同**，但不能称作“identical”。

它们的逐 draw 差异不同，例如：

* stem1：LengthMatched Random 比 DSMC 高 (1.00) pp；
* hum0：DSMC 比 LengthMatched Random 高 (0.86) pp。

而且当前 control 匹配的是长度 bucket counts，不是精确总 token 数。因此应该写：

> The Random tie persists after matching the coarse post-tokenization length histogram.

不要写：

> This proves the tie is not a token-count artifact.

目前只能排除明显的粗粒度长度分布解释，不能完全排除剩余 token exposure 差异。

## 为什么现在不应该先跑 1%

1% 轴值得做，而且它确实是合理的预注册 secondary experiment。最新 controlled study在 (B\in{500,1000,2500,5000,10000}) 上发现，targeted gradient methods 和 RR 的优势通常在低预算更明显，而 Random 在预算增加后更容易变得有竞争力。([arXiv][1])

但这只是一个跨任务经验趋势，不保证 DSMC 在你们的 MMLU skew setting 下到了 1% 就必然击败 Random。该研究自己也强调，没有单一方法在所有任务和预算上占优，Random 经常能匹配或击败复杂方法。([arXiv][1])

如果现在因为 5% 没赢 Random，就马上跳到“预计能赢”的 1%，论文审稿人可能会将其理解成：

> 在主预算没有获得优势后，转而寻找一个更有利的预算。

即使 1% 原本在计划里，也最好先完成冻结的 5% 主条件，再把 1% 明确作为 budget-interaction follow-up。

## 为什么不要选 option 3 的当前写法

Option 3 说：

> 先跑 1%，然后扩展“信号最清楚”的预算。

这是 outcome-dependent selection。它会让最终报告偏向表现最好看的预算。

可以最终两个预算都做，但必须：

* 预先定义两个预算都会报告；
* 不根据哪一个结果更好决定主表保留谁；
* 不只扩展“最清楚”的那个。

所以 option 3 目前不应选择。

## 下一步的正确实验

现在选择：

> **2. Expand to 5 draws @ 5%.**

完成剩余：

[
2\text{ directions}
\times3\text{ draws}
\times7\text{ draw-specific methods}
+
3\text{ shared Random-K}
========================

45
]

个 unique adapters。

继续保留全部八种方法。不要根据 pilot 删除表现差的 baseline，因为这 5% 条件是主 external-validity experiment，完整方法矩阵有助于避免 outcome-dependent baseline screening。

这轮运行后，主分析应同时报告：

1. 每方向 5 个 draw 的 mean、median 和 win count；
2. 每个 draw 的完整 paired differences；
3. 将两个方向按 draw index 组成 block 后的 direction-averaged difference；
4. Random-K 明确按 **5 个 unique Random adapters** 分析，而不是把十个方向 cells 当成十个独立 Random runs；
5. descriptive interval，不作强显著性宣称。

由于每个 draw index 同时绑定了一个预先指定 training seed，最准确的统计单位名称是：

> target-draw/training-seed replicate

而不只是纯粹的 target draw。

## 完成 5% 后再做什么

若完整 5% 结果仍显示：

* DSMC 对 LESS/RR/GIST 稳定领先；
* DSMC 与 Random 仍接近；

那么 1% 预算轴会成为非常自然且有价值的机制问题：

[
\text{Does DSMC’s advantage emerge as selection becomes more constrained?}
]

届时再跑 1%，并将其解释为 **budget interaction**，而不是用于挽救“DSMC 必须击败 Random”的 headline。

即使 1% 最后仍与 Random 打平，这也不是项目失败。论文仍然可以有一个诚实且有价值的结论：

> Directional second-moment matching substantially improves over existing targeted selectors under skewed query sets, but targeted selection as a whole does not consistently outperform well-controlled Random selection at a 5% budget.

这实际上与近期对 instruction-selection 方法脆弱性的研究高度一致。([arXiv][2])

## 可以直接回复 Claude

> Choose option 2: expand the frozen 5% primary condition to five draws per direction before starting the 1% axis.
>
> The current Random comparison contains only two unique shared Random adapters, and its sign is perfectly blocked by draw index/training seed: DSMC wins both direction cells at index 0 and loses both at index 1. Adding draw indices 2–4 directly resolves whether the apparent tie is stable or just cancellation between two blocks.
>
> Keep all eight methods and all configurations frozen for the remaining six draws. Do not stop or modify anything based on interim accuracy. In the final analysis, report per-direction five-draw results and also direction-averaged differences by draw index; treat Random-K as five unique Random/training-seed blocks, not ten independent direction cells.
>
> Before launch, patch the writeup in three places:
>
> * DSMC beats LESS, First-RR, Second-RR, and GIST in 4/4, but NICE is mixed and slightly ahead on average in the HUM direction;
> * say Random-K-LengthMatched supports the same overall conclusion, not that it is identical;
> * say the tie is not explained by coarse length-bucket composition, rather than claiming all token-count confounding is eliminated.
>
> After the complete 5% result is reviewed, run the preregistered 1% condition as a separate budget-interaction experiment. Do not expand only whichever budget produces the clearest DSMC signal; both reported budgets must remain visible.

**当前决策：选择 option 2，完成 5% 的五-draw主实验。1% 是下一阶段，不是现在用来追求更漂亮结果的替代条件。**

[1]: https://arxiv.org/abs/2602.14696 "A Critical Look at Targeted Instruction Selection: Disentangling What Matters (and What Doesn't)"
[2]: https://arxiv.org/abs/2503.01807?utm_source=chatgpt.com "Large-Scale Data Selection for Instruction Tuning"

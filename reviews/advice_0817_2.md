是。**如果目标明确是 ICLR main，而且现在还能并行写论文、算力也允许，我现在会建议补一组 Llama-3.2 × MMLU。** 但我只建议补 **5%**，不建议再把 1% 也复制一遍。

这不是因为现有故事还不成立。现在核心负面结论已经有 Llama-2 和 Llama-3.2 两个 stack 的 BBH confirmation，已经足以支持“更好的 target-gradient alignment 不是 downstream improvement 的充分条件”。真正还存在的不对称是：**你们唯一比较正面的 method-side 结果——directional second moment / DSMC 在 MMLU 上相对 first-order targeted selection 的优势——目前仍然只有 Llama-2 一个 stack。**

这会是我现在作为 ICLR reviewer 最自然的一个问题：

> 你已经证明“失败现象”能跨模型，但你论文里的方法贡献——second-order representation 在 MMLU 上更好——是不是只存在于 Llama-2？

ICLR 2027 reviewer guide明确允许要求“范围有限、用于更充分验证已有结果”的追加实验，而不是要求作者无限扩张实验矩阵。([ICLR][1]) 你现在考虑的这个实验正好符合这种类型。邻近的 *Critical Look* 也专门做 additional-model experiments，并且发现 selector/representation 的表现具有明显 task/model dependence；例如它报告 Llama-3.2/Qwen 在 BBH 上的趋势和 Olmo 在 MMLU-Pro 上的趋势。([arXiv][2])

所以我会把之前的 stop rule做一次**公开、事前、最后一次 amendment**：

> “After paper-level audit, we identified one asymmetric evidence gap: the central negative result is cross-stack, but the positive MMLU second-order attribution is single-stack. We therefore add one frozen Llama-3.2 MMLU-5% confirmation before seeing any Llama-3.2 MMLU result. No 1% follow-up or further experiment is permitted regardless of outcome.”

这样比偷偷违反 stop rule干净得多。

为什么我选 5%，不是 1%？因为 **5% 正好检验你们最需要外推的正面结果。**

Llama-2 的 MMLU 5% 上，DSMC 相对 Second-RR 是很清楚的额外收益，而且相对 First-RR/其他 targeted baselines 也非常稳定。相反，在 1% 上 DSMC 对 Second-RR 已经只剩很小差距、结果混合；1% 最突出的发现其实是 Random 更强。后者现在已经被 BBH 两个模型充分支撑了。因此：

* Llama-3.2 MMLU **5%** 回答：“second-order representation / MMD coreset 的正面方法结果能不能跨 stack？”
* Llama-3.2 MMLU **1%** 主要回答一个已经回答很多次的问题：“低预算 Random 会不会仍然强？”

第二个的边际价值明显低。

而且不要为了“budget completeness”把两个都跑了。邻近工作本身就显示 selection trends 随 budget 改变，而且低预算与高预算行为可能不同；它们做预算 sweep是论文中心之一，而你们现在不是。([arXiv][2]) 对你们来说，追加实验越有限、科学问题越单一，越符合 ICLR reviewer 对补实验的期待。([ICLR][1])

我会把实验冻结成下面这样。

**Llama-3.2-3B × MMLU × 5%，只跑四个方法：**

* DSMC
* Second-RR
* First-RR
* Random-K
* 外加一个共享 no-SFT reference

不要 LESS/GIST/NICE，不要 LengthMatched，不要新的 control。

复用原来完全相同的 MMLU 十个 target draws：

[
5\ \text{stem-majority}+5\ \text{hum-majority}.
]

而且我建议**完全复刻原 MMLU 的 seed mapping**，不要这时突然改成 BBH 的 2-seed crossed design：

[
d=0,1,2,3,4
\longrightarrow
seed=42,1,2,3,4.
]

这样 Llama-2 ↔ Llama-3.2 的比较最干净：实验设计不变，stack变。

于是分析上有：

[
10\text{ direction cells}\times4\text{ methods}=40\text{ cells},
]

但真正训练量不是40个adapter。

三个 target-aware methods需要：

[
10\times3=30
]

个adapter。

Random-K继续沿用 MMLU 原设计，同一个 draw index 的 STEM/HUM 方向共享一个target-independent subset，因此只要：

[
5
]

个Random adapters。

总共：

[
\boxed{35\text{ unique adapters}}
]

外加一个no-SFT reference。

这已经是相当合理的规模。

另一个好消息是：**不用重新生成270k candidate datastore。**

你们刚才为了 Llama-3.2 BBH 已经生成并hash-pin了该stack自己的：

[
(270679,8192)
]

candidate gradient datastore。Candidate features和target task无关，只依赖模型stack、warm-up、candidate pool，因此可以直接复用。

需要新做的只有：

* 十套 MMLU target/query gradients；
* DSMC / First-RR / Second-RR selections；
* 35个SFT adapters；
* MMLU eval。

这让这个追加实验比第二模型BBH便宜不少。

但目标梯度必须重新算，而且必须严格复制原 MMLU protocol。不要“顺便修”旧设计：

* candidate = Adam-aware；
* target = SGD；
* projection 8192 / seed123；
* 同一批64-example target draws；
* target-gradient仍使用原来 MMLU 的 single-example supervised、`num_fewshot=0` 格式；
* downstream eval仍是原来的5-shot MMLU；
* Llama-3.2使用它自己的 `llama3` serialization；
* target-aware subsets必须用 Llama-3.2 MMLU gradients重新选择；
* Random-K则复用 Llama-2 5% 原来的exact indices。

尤其不要因为我们后来发现 BBH prompt alignment问题，就把 Llama-3 MMLU改成“更合理”的5-shot target gradient。那会变成同时改 stack + target protocol，失去直接cross-stack replication意义。

我甚至建议像此前一样，不手写config。直接：

> 从已验证的 Llama-2 MMLU 5% configs派生，只允许覆盖 model-stack相关字段、Llama-3 candidate-cache路径、Llama-3 target-cache/output路径和template；其余键集合必须不变。

这能避免 fresh Claude/上下文压缩再制造一次隐藏漂移。

统计上也不要因为这次有10个方向cells就突然声称 (n=10)。

继续用你们现在已经很成熟的：

[
5\text{ draw-index blocks}
]

作为主要描述单位——对每个 index 先平均 stem-majority 和 hum-majority 两个方向，再报告五个 paired block differences。

主问题我会预注册成两个：

[
\Delta_{\text{rep}}
===================

\text{DSMC}-\text{FirstRR},
]

和

[
\Delta_{\text{MMD}}
===================

\text{DSMC}-\text{SecondRR}.
]

Random是target-awareness anchor：

[
\Delta_{\text{rand}}
====================

\text{DSMC}-\text{Random}.
]

Primary metric仍是 balanced MMLU：

[
(\mathrm{STEM}+\mathrm{HUM})/2.
]

no-SFT只是一条共享reference，不进入block counts。

而且在任何计算之前就把结果解释写死：

* **DSMC > Second-RR + First-RR**：说明 MMLU 中 second-order / MMD 的正面方法结果至少能跨两个 model stacks；
* **Second-RR ≈ DSMC，但两者 > First-RR**：说明跨stack稳定的是 second-order representation，而不是MMD coreset额外收益；
* **First/Second-RR ≥ DSMC**：说明 MMLU 中DSM C的方法优势也是 stack-dependent，论文应进一步降低method claim；
* **Random ≥ targeted methods**：进一步强化“target awareness本身不可靠”，但不允许因此调方法。

无论结果是什么：

> **绝不追加 Llama-3.2 MMLU 1%。**

这是现在就必须写进amendment的。

我甚至认为，如果5%结果是“DSMC在Llama3上不再胜SecondRR”，它也很有价值，因为那会让论文更统一：

> second moments were useful in one stack, but even the representation advantage is model-stack dependent; what does replicate robustly is the failure of geometric target alignment to guarantee utility.

这不会毁论文。

如果5%结果又复现DSM C > SecondRR，则方法贡献明显变强：

> directional second-moment matching provides a reproducible targeted-selection improvement on MMLU across two model stacks, yet that improvement still does not imply an advantage over strong Random or transfer to BBH.

这会是非常好的“positive method + negative general principle”组合。

所以，我现在与上一轮“实验停止”的建议相比会做一个有限调整：

**如果目标只是把当前核心negative story投出去，我仍然认为不补也成立。**

但既然你明确说目标是 **ICLR main**，而且时间和算力允许，我认为：

[
\boxed{\text{Llama-3.2 MMLU 5% 是值得补的最后一块。}}
]

不是1%+5%，只做5%。

它把目前证据结构从：

> “negative finding 跨模型；positive method finding 单模型”

升级成一个更完整的问题：

> “在第二个stack上，MMLU里到底是second-order representation本身能复现，还是只有geometry→utility failure能复现？”

这正是 reviewer 真正会关心的，而不是“你为什么没有再多跑一个budget”。

同时论文**现在就继续写，不等结果**。ICLR 2027 full-paper deadline是9月16日，主文限制9页；reviewer也明确不要求论文必须SOTA，而更看重claim、rigor和new knowledge是否匹配。([ICLR][1]) 所以这组实验是加强证据矩阵，不应该重新改变论文中心。

我会直接给Claude这样的指令：

> Amend the stop rule once, transparently and before any Llama-3.2 MMLU computation, to add exactly one final scoped validation: **Llama-3.2-3B × MMLU at 5% only**. The reason is to close the asymmetric evidence gap identified during paper drafting: the geometry→utility failure is already cross-stack, while the positive MMLU second-order attribution is currently Llama-2-only. This is validation of an existing claim, not a search for a favorable setting.
>
> Freeze: the same ten MMLU target draws, original draw→training-seed mapping {42,1,2,3,4}, K=13,533, four epochs, and methods DSMC / First-RR / Second-RR / Random-K plus one shared no-SFT reference. This gives 40 analysis cells but 35 unique adapters because Random remains shared between STEM/HUM directions for each draw index.
>
> Reuse the already pinned Llama-3.2 warm-up and full 270,679×8192 candidate datastore. Recompute all ten MMLU target-gradient caches and all three target-aware selections using the exact historical MMLU feature/query protocol; derive configs structurally from the frozen Llama-2 MMLU configs and change only model-stack/cache/output/template fields. Random-K must reuse the exact frozen Llama-2 5% indices.
>
> Pre-register the five direction-averaged draw-index blocks as the primary descriptive unit and the three comparisons DSMC−First-RR, DSMC−Second-RR, and DSMC−Random. Report the shared Llama-3.2 no-SFT MMLU reference.
>
> Explicitly prohibit any Llama-3.2 MMLU 1% follow-up, regardless of outcome, and prohibit tuning/new controls. Continue drafting the paper in parallel; this experiment may refine the scope of the positive DSMC claim but may not change the paper’s central “matching the target is not enough” framing.

**所以我的答案是：补，但只补 Llama-3.2 的 MMLU 5%。** 这是现在最有性价比、最能提升 ICLR main 完整度的一组；1% 不值得再开。

[1]: https://iclr.cc/Conferences/2027/ReviewerGuidelines?utm_source=chatgpt.com "ICLR 2027 Reviewer Guidelines"
[2]: https://arxiv.org/abs/2602.14696 "A Critical Look at Targeted Instruction Selection: Disentangling What Matters (and What Doesn't)"

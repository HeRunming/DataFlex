这两个问题建议分别这样选：

> **GIST timing：选“Type something”，采用并行方案。**
> **Random baseline：选 2，“Both random variants”，但修正 token-matched 的定义。**

## 先看 2×2 结果意味着什么

这组结果确实很好，但 Claude 的“cleanest possible outcome”稍微说早了一点。

最稳固的结论是：**二阶 representation 是主要贡献来源。**因为无论 selector 是 TopK 还是 MMD，二阶都在 STEM80 和 HUM80 上优于一阶：

[
\text{Second-TopK}-\text{First-TopK}
====================================

+0.91,,+0.61\text{ pp},
]

[
\text{DSMC}-\text{Linear-MMD}
=============================

+2.63,,+2.30\text{ pp}.
]

而 MMD 与 representation 的交互也非常整齐。difference-in-differences 分别是：

[
(0.42)-(-1.30)=1.72\text{ pp},
]

[
(0.62)-(-1.07)=1.69\text{ pp}.
]

两个 skew 方向几乎完全一致。

但 DSMC 相对 Second-TopK 的直接增益只有 (0.42/0.62) pp，且目前单 seed，和此前测得的训练波动同量级。因此论文现在可以把：

* **二阶表示是主驱动因素**写成较强结论；
* **MMD diversity 与二阶表示互补**写成很有希望的机制假设；
* 暂时不要写成“MMD repulsion 已被统计证明必不可少”。

接下来的 independent target draws 会自然提供更多 paired observations，不必现在专门再为 2×2 所有格子补多个 seed。

# 决策一：GIST timing

## 不建议选 2：Pilot now, GIST later

GIST 不是普通的附加 baseline，而是目前与你的方法**概念上最接近的竞争方法**：

* 从 target validation gradients 中做 SVD；
* 恢复低维 task-specific subspace；
* 将 candidate gradients 投影到该子空间；
* 按 target direction alignment 打分。([arXiv][1])

DSMC 的核心解释也是从单位梯度方向的二阶矩中捕获 target-relevant subspace。因此 reviewer 很可能直接问：

> DSMC 相比 GIST 的 task-subspace alignment 到底增加了什么？

而且 GIST 自己报告了很强的存储和计算效率，因此不仅要比较 accuracy，还应比较 selection storage、selection FLOPs 和 wall-clock。([arXiv][2])

如果先跑完不含 GIST 的 pilot，后来 GIST 明显领先，那么方法排序、论文定位和后续矩阵都要重新调整，之前的 pilot 价值会下降。

## 也不必单纯选 3：全部停住

target-draw protocol 的确必须先审定，但 GIST 实现可以在协议审查期间并行进行，没有必要让工程工作等待。

## 最佳选择：选 4，输入并行方案

可以直接回复 Claude：

> Implement GIST now in parallel with finalizing and reviewing the target-draw protocol. Do not launch the target-draw SFT pilot until both conditions are met: (1) the protocol is approved and frozen, and (2) the GIST implementation passes a focused code and numerical review on the existing STEM80/HUM80 targets. Once both gates pass, run the pilot with GIST included from the start.

也就是：

1. target-draw protocol 起草并 review；
2. 同时实现 GIST；
3. 在现有 STEM80/HUM80 cache 上做 correctness validation；
4. 两边都通过后，一次性启动完整 pilot。

界面必须在 1–3 中选择时，选 **1 Implement GIST first** 最接近，但要额外声明 pilot 仍需等 protocol 审定。

## GIST 实现必须通过哪些检查

目前我没有在论文的 arXiv 页面和搜索结果中找到明确的官方代码仓库链接，因此自己实现时应避免只做“GIST-like”。论文当前版本是 2026 年 5 月修订的 v2，应以 v2 公式和 TeX source 为准。([arXiv][1])

至少检查：

* SVD 使用的是哪类梯度、是否中心化、是否归一化；
* rank 的选择规则与 spectral filtering 阈值；
* candidate score 的准确公式；
* candidate 与 target 是否使用相同 projection；
* projector seed、checkpoint 和参数子集；
* 子空间基是否正交；
* score 对正交基旋转是否不变；
* rank 取满时是否退化到预期的全空间评分；
* 与 LESS-aligned Adam/SGD protocol 如何公平对齐。

在正式 pilot 前，先返回 code diff、公式对应关系和 STEM80 的 score sanity checks。

# 决策二：Random baseline

这里选：

> **2. Both random variants**

但我不同意截图中把“可变 example count 的 token-matched Random”直接设为唯一主 Random。

## 为什么两种都需要

targeted selection 的标准问题定义是固定选择预算：

[
|\mathcal S|=B.
]

最新的 controlled study 也是让 Random 从 candidate pool 中均匀、不放回地抽取恰好 (B) 个样本，并使用多个随机抽样运行；它还发现 Random 在不少任务和较大预算上非常有竞争力。([arXiv][3]) ([arXiv][3])

大规模 instruction-selection 研究也发现，许多复杂方法可能输给 Random，因此 Random 不能只是一个随意配置的弱基线。([Hugging Face][4])

两种 Random 分别控制不同 confound：

1. **Example-count Random**
   固定 (K=13{,}533)，与所有 selection methods 使用相同样本预算和相同优化步数。

2. **Length/token-matched Random**
   控制 DSMC 可能偏向更长或更短样本，从而改变实际训练 token exposure。

只跑第一种，reviewer 会问提升是否来自更多 token；只跑第二种且改变样本数量，reviewer又会问提升是否来自更多或更少的独立训练样本。

## token-matched 应怎样定义

不要让 token-matched Random 改变 (K)。

推荐定义成：

> 从 candidate pool 中随机选择恰好 (K) 个样本，同时匹配 DSMC subset 的 post-tokenization length histogram 或有效 token 总量。

也就是同时满足：

[
|\mathcal S_{\rm rand}|=K,
]

以及

[
\left|
\operatorname{Tokens}(\mathcal S_{\rm rand})
--------------------------------------------

\operatorname{Tokens}(\mathcal S_{\rm DSMC})
\right|
\leq \epsilon.
]

更稳妥的是匹配长度分桶，例如：

```text
[0,256), [256,512), [512,1024), [1024,1536), [1536,2048]
```

每个桶的样本数尽量与 DSMC 一致，而不是只匹配总 token 数。这样可以避免“少量超长样本加大量短样本”与 DSMC 总 token 相同、但训练动态完全不同。

应以经过当前 tokenizer、template 和 `cutoff_len=2048` 之后的有效长度为准，而不是原始字符串长度。

主表可以报告：

* `Random-K`：标准均匀随机；
* `Random-K-LengthMatched`：样本数和长度分布都匹配。

我建议 **Random-K 作为主 Random row**，因为它严格对应固定 selection budget；LengthMatched 作为 compute/length-control baseline。

# Pilot 方法矩阵怎么调整

2×2 gate 已经完成，因此 external-validity pilot 不必继续完整重复四格。它现在应该优先比较方法本身与强 baseline。

建议 pilot 为：

1. DSMC；
2. true Second-RR；
3. LESS + true RR；
4. GIST；
5. NICE；
6. Random-K；
7. Random-K-LengthMatched。

注意，最新研究中的 RR 是逐 query 循环：每个 query 选择当前最相似且尚未选择的 candidate，再循环直到预算耗尽，不是 mean-relevance TopK。该工作发现 gradient representation 配合 RR 在低预算下往往表现较好。([arXiv][3]) ([arXiv][3])

现有 Second-TopK 可以：

* 保留在 2×2 mechanism ablation；
* 不必作为每个 target draw 都跑的主要 baseline；
* 用 true Second-RR 替换它进入 external-validity pilot。

这样 pilot 回答的是更有价值的问题：

> DSMC 能否击败最强的二阶 relevance selector、最强的一阶 gradient selector，以及最接近的 subspace baseline GIST？

## Random 的运行数可以优化

`Random-K` 不依赖 target draw，只依赖：

* random subset seed；
* training seed；
* candidate pool。

因此，相同 training seed 下可以跨不同 target draws 复用 Random-K adapter，不必每个 draw 都重新训练。

`Random-K-LengthMatched` 因为需要匹配每个 DSMC subset 的长度分布，通常是 draw-specific，需每个 draw单独生成。

这能降低选择“两种 Random”带来的额外成本。

# 最终选择

对两个界面：

### GIST timing

选择 **4. Type something**，填入：

> Implement GIST in parallel with target-draw protocol review. Hold the SFT pilot until both the protocol and the GIST implementation are reviewed and frozen; then launch the pilot with GIST included from the start.

### Random baseline

选择 **2. Both random variants**，但补充：

> Use uniform fixed-K Random as the primary baseline, and a second fixed-K length/token-distribution-matched Random as the compute/length control. Do not change the example count for token matching.

2×2 gate 已经成功完成了它的任务：现在没有理由改变 DSMC headline，但它也使 **GIST 和 true RR 更重要，而不是更不重要**。接下来应该停止内部调参，转向一次设计严谨、baseline 完整的 external-validity pilot。

[1]: https://arxiv.org/abs/2602.18584 "[2602.18584] GIST: Targeted Data Selection for Instruction Tuning via Coupled Optimization Geometry"
[2]: https://arxiv.org/abs/2602.18584?utm_source=chatgpt.com "GIST: Targeted Data Selection for Instruction Tuning via Coupled Optimization Geometry"
[3]: https://arxiv.org/abs/2602.14696 "A Critical Look at Targeted Instruction Selection: Disentangling What Matters (and What Doesn't)"
[4]: https://huggingface.co/papers/2503.01807?utm_source=chatgpt.com "Paper page - Large-Scale Data Selection for Instruction Tuning"

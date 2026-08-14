这次结果对“DSMC 是一个普遍更好的 selector”这条故事确实不利，但对整个项目未必是坏消息。相反，它把论文应该讲什么变得更清楚了。

我核对了 `85b8e83` 和完整 `bbh_external_results.md`。工程和预注册执行都很干净：36/36 train、36/36 eval，全部 84 steps、27 subtasks、5,209 held-out examples，18 个 subset hash 没变，accuracy 直到全部完成后才一次性解封。 因此这里没有一个明显的工程理由可以拿来“解释掉”结果。

先给结论：

> **不要继续救 DSMC、不要调 BBH 的 LR/epoch/budget，也不要发明 DSMC-v2。**
>
> 先做一轮几乎不需要重新训练的机制诊断，然后重构论文。
> 如果还愿意投入一组大实验，下一组最高价值的是“第二个 base model”，而不是第三个任务、第三个 matched Random 或继续调参。

### 这次究竟说明了什么

第一，**“target awareness 能胜过 Random”现在已经是一个跨 family 的负面结果。**

MMLU：

* 5%：DSMC − Random ≈ −0.01 pp；
* 1%：DSMC − Random ≈ −0.77 pp。

BBH：

* DSMC − Random = **−2.94 pp**；
* 六个 seed-level cells 全负；
* DSMC 只在 4/27 subtasks 上超过 Random。

而 BBH 这次不是之前的 80/20 skew setting。query reservoir 与 evaluation 是同 family、held-out、正常抽样的，因此之前还能提出的：

> “target selector 被 skewed query 误导了”

已经不能解释 BBH。

这与近期文献的大方向并不异常。2026 年的 targeted-selection controlled study就明确发现没有单一方法稳定占优，gradient representation + greedy RR主要是在低预算平均更强，而优势高度依赖任务和模型。([Hugging Face][1]) 更大规模的 instruction-selection 研究也发现复杂 selector 经常无法超过 Random，甚至在更大 candidate pool 上反而下降。([ACL Anthology][2])

所以：

> **Random 很强不是你们实验出了问题，而是这个领域现在越来越真实的 empirical picture。**

---

第二，**DSMC 自身的正面 claim 必须明显降级。**

BBH 上：

* First-RR = 0.3715
* Second-RR = 0.3687
* DSMC = 0.3631
* LESS = 0.3601

也就是说 DSMC 在这里不是“best targeted selector”。

尤其 DSMC 对 Second-RR：

[
-0.56\text{ pp}
]

六个 cells 全输。换成更合理的“先在同一 draw 内平均两个 SFT seeds”的三个 selection blocks，差值大约是：

[
[-0.44,-0.54,-0.71]\text{ pp},
]

仍然是 **0/3**。

这个结果很重要，因为它告诉我们：

> MMLU 5% 上观察到的“second-order MMD 比 second-order RR 进一步好”，不是一般规律。

它是 task/budget dependent 的。

所以原来：

> directional second moments substantially improve robustness among target-aware selectors

现在也不能无条件写。

更准确的是：

> **Directional second moments improved target-aware selection on the MMLU family, particularly at 5%, but that advantage did not transfer to BBH.**

这是一个正面结果，但已经不是 universal method contribution。

---

### 有几个 Claude 当前总结里的措辞需要马上收紧

最重要的是这句：

> “format explains 40.7% of the gap”

**这个不能作为因果结论。**

SeqLabelMatched 的确非常有价值：

[
\text{DSMC}=0.3631,
]

[
\text{SeqLabelMatched}=0.3805,
]

[
\text{Random}=0.3925.
]

所以把 Random 限制到 DSMC 类似的 coarse sequence × label-length histogram 后，performance下降 1.20 pp；这个下降在数值上相当于 DSMC–Random 2.94 pp gap 的约41%。

但这个 control同时改变了：

* source composition；
* provenance entropy；
* task/format composition；
* lexical/content distribution。

我们之前就记录过 SeqLabelMatched 的 source entropy 变得更像 DSMC，而不是 plain Random。

因此正确表述是：

> The shift induced by the coarse sequence/label-matched Random control is numerically about 41% of the raw DSMC–Random gap, showing that instruction-format composition is a plausible contributor; it is not a causal decomposition because matching length also changes correlated source/content composition.

不要写：

> format explains 41%.

同理：

> “over-matching short-response format causes specialization”

也太强。

应该写：

> **The degradation of SeqLabelMatched relative to Random is consistent with harmful specialization toward the BBH-like long-context/short-response regime.**

这是相关证据，不是 causal identification。

近期 ROSE 工作其实和你们这里非常契合：它专门指出 task-specific instruction selection通常用 next-token cross-entropy 或类似 surrogate，但 instruction-tuning loss与实际 task metric并不保持单调关系，这种 surrogate–outcome misalignment正是现有 selection 方法的核心困难之一。([ACL Anthology][3])

---

### “所有方法都把 base 搞坏了”也需要稍微收紧

所有**方法均值**确实都低于 no-SFT：

[
0.396429.
]

Random 均值是 0.3925，只低 0.39 pp；但其中一个 Random cell是 0.3974，实际上略高于 base。

所以论文不要写：

> every Random run degrades the base.

而是：

> **All method means fall below the shared no-SFT reference; Random-K remains close to base, while every target-aware method shows a substantially larger mean degradation.**

DSMC 的 −3.33 pp就和 Random 的 −0.39 pp不是一个量级了。

---

## 一个统计展示问题也应该现在修

我建议不要把“6 cells”继续当 headline paired unit。

设计实际上是：

[
3\text{ query/subset realizations}\times2\text{ training seeds}.
]

两个 SFT seeds共享同一个 query draw和同一个 selected subset。所以六个 cells不是六个独立 selection replicates。

BBH 主文最好先对 seed 求平均，然后展示三个 draw-level blocks。

这样 DSMC−method 是：

| comparator      |    draw0 |    draw1 |    draw2 | DSMC wins |
| --------------- | -------: | -------: | -------: | --------: |
| Random          | −2.94 pp | −2.32 pp | −3.57 pp |   **0/3** |
| SeqLabelMatched |    −1.15 |    −1.82 |    −2.28 |   **0/3** |
| Second-RR       |    −0.44 |    −0.54 |    −0.71 |   **0/3** |
| First-RR        |    +0.13 |    −1.14 |    −1.51 |   **1/3** |
| LESS            |    +0.83 |    +0.11 |    −0.05 |   **2/3** |

六个 seed-level cells可以留作 secondary stability evidence。

这样反而比“0/6”更严谨，而且结论完全没变。

另外：

> “27 subtasks (primary unit)”

建议也改掉。27 subtasks是 **diagnostic evaluation breakdown**，不是统计实验单位。Primary metric仍是 5,209-example micro aggregate。

---

# 现在最值得做什么？

我认为下一步不是新训练，而是做三个 **post-hoc、明确标成 exploratory 的零/低成本诊断**。

这三项很可能决定论文最后的核心机制故事。

### 1. BBH 上把 D2–accuracy dissociation 真正补完整

MMLU 已经有非常漂亮的结论：

> DSMC 的 D2 明显优于 Random，而且甚至更接近 balanced (P^\star)，但 downstream 不更好。

现在 BBH summary反而没有报告：

[
D_2(S,Q_d)
]

跨方法到底是什么样。

这应该成为第一优先级。

对三个 draws分别算：

[
D_2(S_m,Q_d)
============

|M_{S_m}-M_{Q_d}|_F^2
]

for:

* DSMC
* Second-RR
* First-RR
* LESS
* Random
* SeqLabelMatched

然后做：

* 每 draw method ranking；
* seed-average downstream ranking；
* within-draw Spearman；
* pooled只作为 secondary descriptive。

最关键的问题是：

> **BBH 上 DSMC 是否真的仍然最小化了我们声称的 target geometry？**

如果答案是：

[
D_2(\mathrm{DSMC}) \ll D_2(\mathrm{Random})
]

但

[
Acc(\mathrm{DSMC}) \ll Acc(\mathrm{Random}),
]

那你们的论文故事会突然非常强：

> **The surrogate failure externally replicates: DSMC successfully matches the target second moment, yet the better match produces markedly worse downstream performance.**

这比“DSMC输了”本身有学术价值很多。

如果 BBH 上DSM C连D2都没最好，那我们就要换解释：MMLU 的 geometric property本身也没有 transfer。

所以这个分析必须先做。

---

### 2. 测“query loss 是否真的改善，而 BBH accuracy 反而下降”

这是我现在最推荐的新 diagnostic。

不需要重新训练。

对：

* base；
* 36 个现有 adapters；

在各自64条 query draw上，用**和 target-gradient extraction完全相同的 final-answer supervision**计算：

[
L_Q(\theta).
]

最好还记录：

[
\Delta L_Q
==========

## L_Q(\theta_{\mathrm{adapter}})

L_Q(\theta_{\mathrm{base}}).
]

然后和 held-out exact-match比较。

如果出现：

> DSMC / RR 对 query final-answer CE改善最多，但 held-out CoT exact-match最差；

这会直接把故事从：

> “gradient matching mysteriously fails”

推进成：

> **the selection surrogate itself is misaligned with the downstream generation objective.**

这和 ROSE 2025指出的 cross-entropy/task-reward non-monotonicity形成非常直接的对话。([ACL Anthology][3])

而且你们 BBH有一个特别有意思的结构：

target gradient监督的是短 final answer：

> `(C)`, `14`, `Yes`

而真实 BBH evaluation要求模型生成 CoT reasoning再输出最终答案。

所以现在很可能存在：

[
\text{final-answer CE alignment}
\neq
\text{reasoning-generation utility}.
]

这可能比“source diversity”更接近问题核心。

但是先测，不要先下结论。

---

### 3. 检查“specialization”到底是不是 task-level specialization

现在 Claude根据：

* SeqLabelMatched下降；
* 高-base tasks受损；

推测 specialization。

这是合理 hypothesis，但还可以用现有结果做一个更直接的检验。

对每个 draw (d)、每个 BBH subtask (t)：

记录 query draw里该 task出现多少次：

[
n_{d,t}.
]

然后把两个 training seeds平均，计算：

[
\Delta_{d,t}
============

Acc^{DSMC}*{d,t}-Acc^{Random}*{d,t}
]

以及：

[
Acc^{DSMC}_{d,t}-Acc^{base}_t.
]

看：

[
corr(n_{d,t},\Delta_{d,t}).
]

如果 query里出现次数越多的 task越“少受损/更容易改善”，而低 exposure task明显下降，那会非常支持：

> finite-query targeting causes narrow specialization.

如果完全没有关系，那说明“specialization”更可能发生在 format/source level，而不是具体 BBH task level。

另外再算：

[
corr(Acc^{base}_t,;
Acc^{DSMC}_t-Acc^{base}_t).
]

但这个只能 exploratory，因为会有 ceiling/regression-to-mean等问题，不能因果解释。

---

# 做完这三个诊断以后，论文应该怎么转向？

我认为现在已经不能把它作为：

> **“DSMC: a new targeted selector”**

来写。

这会被 BBH 自己击穿。

更好的定位是一个**方法 + critical empirical study**：

> **Target matching is not enough: directional gradient moments reveal when targeted instruction selection fails.**

核心逻辑可以变成：

1. 我们从 controlled attribution发现，MMLU 中 directional second moments明显优于一阶 representation；
2. DSMC因此在 MMLU 两个 skew directions上成为最稳的 targeted selector；
3. 但它从未可靠超过 strong Random；
4. tightening budget反而让 Random优势扩大；
5. equal-step不能 rescue；
6. geometric forensic发现更精确的 D2 matching不保证 downstream utility；
7. query-aligned BBH external validation进一步显示：

   * Random仍最好；
   * DSMC不再是 targeted methods中最好；
   * format-matched Random介于 Random和DSMC之间；
   * targeted SFT尤其伤害一些 base已经会做的任务；
8. 因此 representation、selector和task utility之间不存在一个简单的 monotonic chain。

这其实和最近文献形成很好的位置：

* 2026 Critical Look：**no single method dominates**，gradient+RR低预算通常最好；([Hugging Face][1])
* EMNLP 2025 Random-at-Scale：复杂 data selection经常赢不了 Random，diversity/coverage可能更重要；([ACL Anthology][2])
* ROSE：instruction loss surrogate与真实 task performance可能不单调。([ACL Anthology][3])

而你们独特的贡献会变成：

> **我们不仅观察到 Random 很强，还通过 second-moment objective、base reference、budget interaction、geometry audit和 Seq×Label-matched Random，展示了 target-matching surrogate可以被成功优化，却仍然导致负迁移。**

这个故事我认为比“DSMC比LESS高1个百分点”更有价值。

---

# 还要不要继续做大实验？

我的建议是：

**现在先不要。**

先做上面三个 diagnostics + 重写 consolidated story。

然后再决定是否值得花一组大 compute补最后一个轴。

如果准备冲比较强的 venue，当前最大的剩余漏洞已经从“只有MMLU”变成：

> **只有 Llama-2-7B。**

你们现在已经有：

* MMLU；
* BBH；
* 两个 budget；
* skew 与 aligned query；
* Random；
* base；
* format control。

任务轴已经比之前强得多。

如果只允许再做一组真正大的实验，我会选：

> **第二个 base model，同一个 BBH split、同一个 Tulu pool、同一个 K。**

这样只改变 model axis。

不要再跑全部六方法。

最小有价值 set：

* DSMC；
* First-RR；
* Second-RR；
* Random-K；
* no-SFT。

如果资源允许：

[
3\text{ draws}\times2\text{ seeds}\times4=24
]

adapters。

如果很贵，可以把它明确降级成 model-sensitivity：

[
3\text{ draws}\times1\text{ seed}\times4=12
]

但那就只做 descriptive confirmation。

为什么 model axis现在最有价值？因为近期 targeted-selection controlled work本身强调跨模型、跨任务、跨预算比较，而结果高度依赖这些因素。([Hugging Face][1]) 大规模 selection工作也显示方法排名会随模型和 candidate setting改变。([ACL Anthology][2])

但这是**诊断之后**再决定。

如果三个 cheap diagnostics已经形成非常完整的 surrogate-misalignment故事，我甚至觉得可以先写论文，再让第二-model实验作为“是否需要补强”的最后决定，而不是自动继续烧算力。

---

所以我现在给 Claude 的任务会非常明确：

> Do not launch any new SFT or modify DSMC. First clean the BBH statistical/reporting language and run three post-hoc diagnostics on existing artifacts only.
>
> 1. Make the draw-level unit primary: average the two SFT seeds within each of the three draws, report the three paired DSMC−method blocks in the main BBH table, and keep six seed-level cells as secondary stability evidence. Replace “27 subtasks are the primary unit” with “27-subtask diagnostic breakdown.” State that all **method means** are below the single shared no-SFT reference; do not imply every Random cell is below base.
>
> 2. Tighten the SeqLabelMatched interpretation. Do not say “format explains 40.7%” or “format matching causes specialization.” Say the control-induced shift is numerically ~41% of the raw DSMC−Random gap and is consistent with a format/provenance contribution, but is not a causal decomposition because sequence/label matching also changes correlated source/content composition.
>
> 3. Compute BBH (D_2(S,Q_d)) for every method and draw from the frozen gradient caches, and report within-draw geometry ranking versus seed-averaged held-out accuracy ranking. This is the key test of whether the MMLU surrogate–outcome dissociation externally replicates.
>
> 4. Evaluate the base and all existing BBH adapters on their own 64-example query draws using the exact final-answer supervised loss used for target-gradient extraction. Report query-loss changes versus held-out BBH exact-match. This is diagnostic only; no model or protocol decisions may depend on it.
>
> 5. Perform a task-level specialization diagnostic: for each draw/subtask, correlate query-set task frequency with seed-averaged DSMC−Random and DSMC−base subtask deltas; separately report the descriptive relation between base subtask accuracy and post-SFT degradation. Label all of this exploratory.
>
> Then rewrite the consolidated paper framing around the cross-family result. Do not propose or launch a second-model experiment until these diagnostics are reviewed.

我的总体判断是：

**结果“对 DSMC 当英雄方法”不好，但对论文作为一篇有洞察力的 targeted-selection 研究，可能反而更好。**

现在最不应该做的是继续找条件让 DSMC赢。最应该做的是回答：

> **为什么一个成功匹配 query gradient geometry 的 selector，在 query-aligned BBH 上仍然比 Random差近3个百分点？**

如果现有 artifacts就能把这个问题回答到一定程度，这会成为整篇论文真正有价值的中心。

[1]: https://huggingface.co/papers/2602.14696 "Paper page - A Critical Look at Targeted Instruction Selection: Disentangling What Matters (and What Doesn't)"
[2]: https://aclanthology.org/2025.findings-emnlp.146/ "Rethinking Data Selection at Scale: Random Selection is Almost All You Need - ACL Anthology"
[3]: https://aclanthology.org/2025.findings-emnlp.710/ "ROSE: A Reward-Oriented Data Selection Framework for LLM Task-Specific Instruction Tuning - ACL Anthology"

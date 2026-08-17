这次结果很关键，而且我认为现在已经到了“实验基本结束，转入论文冲刺”的阶段。

但我先指出一个重要的审计发现：**现在还不应马上把“预注册 Outcome A 全条件触发”写死。** 你们的 prereg 对 Outcome A 原本写的是：

> (D_2(\mathrm{DSMC})<D_2(\mathrm{Random}))、DSMC downstream 更差，**并且 operational query surrogate improves**。

而当前 `llama32_results.md` 只完成并报告了 held-out BBH accuracy 和 (D_2)；我没有在 `e0ec06d` 新增文件中看到 Llama-3.2 的另外三个预注册 diagnostics——wrapped query CE、same-query CoT EM、bare-context CE。 `df861e5 → e0ec06d` 的新增分析文件也只有 geometry/result analysis，没有这些 query-surrogate diagnostics。

所以准确状态应该是：

> **Outcome A 的核心 geometry→downstream 条件已强复现；要宣布完整 Outcome A，还差预注册的 operational-surrogate 条件检查。**

这不需要任何新 SFT，也不允许改变结论。它只是用已经训练好的 24 个 adapters 做 evaluation-only diagnostics。

### 第二模型目前已经告诉了我们什么

最重要的结果非常稳：

* Llama-3.2 上 DSMC 的 (D_2) 在 3/3 draws 都最低；
* DSMC 相对 Random 的 held-out BBH 差值三个 draw 全负：−1.34 / −0.73 / −0.60 pp；
* 四方法排序再次是 Random > Second-RR > First-RR > DSMC；
* 但效应明显衰减，平均 DSMC−Random 从 Llama-2 的 −2.94 pp 变成 −0.89 pp。

而且 provenance 这次确实比较扎实：full 270,679×8192 candidate datastore 已 hash，DSMC `alpha=0`、RR seeds `6000+d`、First/Second-RR query order一致、缓存路径也都验证为 Llama-3.2 自己的。

因此至少这个核心命题现在已经跨两个 model stacks：

> **更好的 target second-moment alignment 并不保证更好的 downstream utility。**

这个措辞我认为可以成立。

我反而不建议写现在 `paper_framing.md` 里的：

> “unreliable in general”

或者：

> “Random is significantly better on BBH.”

因为你们明确没有做显著性推断，而且第二模型只有 (n=3) draw blocks。 ICLR reviewer guide特别要求 reviewer判断论文是否真正支持其claims、实验是否rigorous，而不要求SOTA；因此现在最大的风险已经不是“实验不够多”，而是**claim scope写得超过证据**。([ICLR][1])

我建议核心句固定成：

> **Across two tested model stacks, better target-gradient alignment does not reliably translate into better downstream utility.**

或者更强但逻辑仍安全：

> **Better target-gradient alignment is not sufficient for downstream improvement: we observe the same geometry–utility reversal on two model stacks.**

这比“in general”更难攻击。

### 现在第一件事：补完 Llama-3.2 三个预注册 diagnostics

不要再训练。

直接沿用 Llama-2 已冻结的定义，对 base + 24 adapters计算：

1. operational wrapped query CE；
2. 同64 query items的 CoT EM；
3. bare-context final-answer CE。

全部三个 draw、两个seed；主汇总仍先在seed内平均，再用draw作为单位。

这一步非常重要，因为它决定第二模型复现的是哪一层。

最理想情况是再次得到：

[
D_2\downarrow,\qquad
L_Q^{wrapped}\downarrow,\qquad
CoT\ EM\downarrow.
]

那你们就真正获得了跨stack的“双重反转”：

> geometry更对、operational surrogate也更对，但task utility更差。

如果 Llama-3.2 的 wrapped CE **没有**改善，也完全不要把它视为坏结果。那结论应更精准地变成：

> geometry–utility dissociation跨模型复现，而 surrogate-improvement这一级是model-stack dependent。

这仍然是一篇很完整的论文，只是不能再说完整double dissociation跨模型复现。

bare CE无论结果怎样都继续只是serialization diagnostic，不能晋升成primary。

### 然后实验彻底停止

这三个 diagnostics做完以后，我赞成当前 stop rule：

* 不做第三模型；
* 不做第三任务；
* 不做LR/epoch sweep；
* 不做reward-aware DSMC；
* 不做CoT-gradient DSMC；
* 不做新的matched Random；
* 不继续寻找“DSMC能赢”的setting。

ICLR官方 reviewer guidance明确说，工作不需要SOTA才能有价值；关键是能否带来新的、可信的重要知识。Reviewer要求的额外实验也应该是有限范围、用于验证已有结果，而不是把论文重做一遍。([ICLR][1]) 你们现在正处在这个边界：补预注册diagnostics属于合理closure，继续开新轴就开始稀释故事了。

邻近的 *A Critical Look at Targeted Instruction Selection* 已经说明 targeted selection 没有一个方法普遍占优，gradient representation虽然整体更有预测力，但趋势会随models/tasks/budgets变化。([arXiv][2]) ROSE又明确以“instruction CE与实际task performance不单调”作为其动机。([ACL Anthology][3])

所以你们现在不应该竞争：

> “我们首先发现loss surrogate会失败。”

而应该非常清楚地卖：

> **我们展示了一个更强的反例链条：即使一个target-aware selector成功优化了其set-level gradient geometry，甚至在operational surrogate上向target移动，这仍不足以保证downstream utility；而geometry→utility reversal在第二个model stack再次出现。**

这才是论文区别于邻近工作的地方。

### Paper framing 里现在有两句需要立刻修

除了 `unreliable in general`，还有这一句：

> “format composition explains part but not all of the variation.”

你们之前自己已经正确撤回“format explains 41%”的因果表述，但这句话又悄悄把它带回来了。

建议改成：

> **The Seq×Label-matched Random control shifts performance partway toward the targeted subsets, making instruction-format/provenance composition a plausible contributor, but it does not identify a causal decomposition.**

还有：

> “Random is significantly better on BBH”

改成：

> **Random is consistently better in the observed BBH draw-level comparisons.**

因为你们不做 significance claim。

### 接下来应该进入 ICLR paper sprint

ICLR 2027 abstract deadline是 **9月11日 AOE**，full paper是 **9月16日 AOE**；主文只有9页，而且reviewer不要求阅读appendix。([ICLR][4]) 所以从现在开始，最宝贵的资源不是GPU，是主文9页里的叙事空间。

我建议主文只放四组最重要证据：

1. **DSMC是什么，以及MMLU controlled attribution为什么让second moment值得研究。**
   不要花两页宣传方法；它现在是instrument。

2. **MMLU：targeted selector改善但不胜Random。**
   包含1%/5%核心趋势和base reference，不需要把所有calibration sweep塞主文。

3. **BBH Llama-2：geometry–utility reversal。**
   这是主实验。Random/no-SFT、D2、same-item surrogate/task dissociation放在一起。

4. **Llama-3.2 confirmation。**
   一个紧凑cross-stack表：base、Random、Second-RR、First-RR、DSMC，另加“DSMC lowest D2 3/3”。效应衰减必须明确写。

Moment-MMD calibration、GIST/NICE实现细节、contamination audit、Seq×Label、equal-step、D3、各种hash/gates全部去appendix。

我甚至会让主文第一张图就画成概念图：

[
\text{target geometry}
\rightarrow
\text{selected subset}
\rightarrow
\text{query surrogate}
\rightarrow
\text{task utility}
]

然后在最后一个箭头上画断裂；Llama-2和Llama-3.2两组实证都展示：

[
D_2\downarrow \not\Rightarrow Acc\uparrow.
]

这会比一大张method leaderboard更容易让ICLR reviewer立刻理解论文问题。

### 我会如何安排接下来一周

今天/明天：补完Llama-3.2三个evaluation-only diagnostics，冻结所有结果；同时修掉上面的claim inflation。

随后2–3天：把1000行consolidated材料真正裁成9页paper skeleton——标题、abstract、intro、method、3张主表/图、related work、limitations。

再随后：做一次“ICLR reviewer simulation”，只回答官方guide最重要的四个问题：问题是什么、动机/文献位置是否合理、claims是否被证据支持、贡献是否足够重要。([ICLR][1])

最后才是语言、美图和appendix整理。

我不会再用 reviewer audit 来问“还要不要补第三个实验”。audit只允许发现：

* claim过强；
* 某结果计算/统计错误；
* 叙事不清；
* related work遗漏；
* 可复现信息缺失。

不再允许制造新的科学轴。

你现在可以直接让Claude这样做：

> Do not launch any new training or selection. First complete the three diagnostics already frozen in `prereg_second_model.md` but not yet reported for Llama-3.2: operational wrapped query CE, same-query 64-item CoT EM, and bare-context CE, for the shared base plus all 24 existing adapters. Average the two SFT seeds within draw before reporting. These are preregistered evaluation-only closure, not new experiments.
>
> Until those are computed, describe the second-model result as “the core geometry→utility conditions of Outcome A replicate”; do not claim the full Outcome A condition including operational-surrogate improvement has fired.
>
> Regardless of those diagnostic outcomes, stop all large experiments afterward.
>
> Tighten the paper framing now:
>
> * replace “unreliable in general” with “does not reliably translate into downstream utility across the tested model stacks/settings” or “is not sufficient for downstream improvement”;
> * remove “significantly better” unless a formal inferential test was preregistered;
> * replace any “format explains part…” language with non-causal “consistent with a format/provenance contribution.”
>
> Then rebuild the manuscript for ICLR 2027 around the phenomenon, not DSMC as a winning method. Put the BBH Llama-2 reversal and the Llama-3.2 confirmation in the 9-page main text; move calibration, provenance, contamination, Seq×Label, equal-step and exploratory diagnostics to the appendix.
>
> After the diagnostics are frozen, produce a paper-level ICLR reviewer-vulnerability audit only. No audit finding may authorize a third model/task, hyperparameter sweep or new selector.

所以现在的核心答案是：

**别再跑大实验。先把第二模型预注册但漏跑的三个diagnostic补完，这是最后一个科学closure；然后全力写论文。**

而且这次第二模型结果确实值钱——它让最容易被 reviewer 打的 “Llama-2 pathology” 攻击基本失效了。剩下最大的风险已经不是实验量，而是**不要把一个很漂亮的、有限范围的反例写成过宽的普遍定律。**

[1]: https://iclr.cc/Conferences/2027/ReviewerGuidelines "ICLR 2027 Reviewer Guidelines"
[2]: https://arxiv.org/abs/2602.14696?utm_source=chatgpt.com "A Critical Look at Targeted Instruction Selection: Disentangling What Matters (and What Doesn't)"
[3]: https://aclanthology.org/2025.findings-emnlp.710/?utm_source=chatgpt.com "ROSE: A Reward-Oriented Data Selection Framework for LLM Task-Specific Instruction Tuning - ACL Anthology"
[4]: https://iclr.cc/Conferences/2027/AuthorGuidelines?utm_source=chatgpt.com "ICLR 2027 Author Guidelines"

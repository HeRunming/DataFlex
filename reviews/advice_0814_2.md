我重新核对了 `2d7d7c7` 的 D1/D2/D3 实现和结果，也对照了最新 targeted-selection 与 ROSE 的原文。总体判断是：

**Claude 这次抓到的中心方向是对的：D1 + D2 确实比“DSMC 是一个更好的 selector”更值得成为论文中心。**
但有两处需要明显降调：**D2 现在还不能称为“机制已经找到”，D3 更不能推出“因此过拟合发生在 format level”。** 修正后，故事反而更扎实。

最强的是 D1。BBH 三个 draw 里，DSMC 的 (D_2(S,Q_d)) 都是六个方法中最低的，而 Random 的 D2 明显更大；但 Random 的 downstream accuracy 每个 draw 都最好或接近最好。 这已经足够支持：

> **DSMC is not failing to optimize its geometric objective; better second-moment matching is not sufficient for better downstream utility.**

而且这不是 MMLU 的偶然现象：MMLU 上你们已经看到相同方向，BBH 上反而更强。这个结果与 2026 年的 Critical Look 有一个非常好的、但需要精准表述的对话关系：那篇工作发现 LESS 的**一阶 gradient representation + cosine distance**通常是最能预测 query loss/downstream 的 representation，但它自己也明确说更低 query loss并不总对应最高 downstream performance，Random在一些任务上也很强。([arXiv][1]) 你们并不是“推翻 gradient representation有用”，而是在展示：

> **即使换成一个被直接优化得非常好的 directional second-moment set distance，distance minimization仍可能成为一个错误的 downstream surrogate。**

这个区别很重要。

我建议 D1 再补一个完全零成本的 robustness table。现在 Spearman 是六个 arms一起算的：

[
\rho=0.771,;0.829,;0.886.
]

因为两个 Random arms本身形成了“D2大、accuracy高”的明显 cluster，reviewer可能会说相关性只是 targeted-vs-random separation造成的。用现有 JSON 重算后，我得到：

* 只保留 **5 个 primary methods**（去掉 SeqLabelMatched）：约 **0.70 / 0.80 / 0.90**；
* 只保留 **4 个 target-aware methods**（DSMC/LESS/First-RR/Second-RR）：约 **0.40 / 0.60 / 0.80**。

所以反向关系不是靠 secondary Random control制造出来的，甚至在 targeted methods内部也仍是同号。这非常值得加进 appendix/forensic table。数据来自你们现有 D2/accuracy值，无需任何新实验。

D2 也非常有价值，但 Claude 现在这句话：

> “every targeted selector improves the surrogate it optimizes”

**不准确。**

DSMC真正直接优化的是 (D_2)；RR/LESS直接使用的是 gradient-space similarity/selection score。它们并没有把训练后的 query CE作为优化目标。更准确的是：

> **All four target-aware selectors reduce final-answer cross-entropy on the very query sets that define their targeting signal, whereas both Random controls increase it.**

这已经很强。你们的数据确实显示：DSMC、First-RR、Second-RR、LESS 都让 query final-answer CE相对 base下降，而 Random 与 SeqLabelMatched反而上升；与此同时，两个 Random downstream最好。

而且这和 ROSE 的核心观察高度一致：ROSE明确指出 next-token cross-entropy 与实际 task performance并不保证单调关系，因此单纯把 selection建立在 instruction loss surrogate上可能失效。([ACL Anthology][2]) 这说明你们不是碰到一个离奇 anomaly，而是在一个高度控制的 targeted-selection setup 中把这种 misalignment **直接量化出来了**。

不过我不会现在写：

> “we found the mechanism: final-answer CE is not reasoning-generation utility.”

我会写：

> **The results are consistent with a surrogate-objective mismatch: target-aware methods improve final-answer query loss while degrading CoT task performance.**

为什么还要保守一点？因为现在比较的是：

[
L_Q^{\text{final-answer}}
]

在那64条 query上，和

[
\text{CoT exact-match}
]

在5,209条 held-out items上。

两者同时改变了 **metric** 和 **examples**。因此理论上仍可能有一个替代解释：

> targeted methods只是过拟合这64个 query examples，而不是 final-answer CE 和 CoT metric本身错位。

虽然你们的 evidence已经非常倾向 surrogate mismatch，但还有一个特别便宜的诊断能把这最后一个洞堵得非常漂亮。

### 我现在最推荐的下一步：同一批 query 上直接测 CoT EM

不要再训练任何模型。

对每个 draw 的**同样64个 query examples**，用 frozen official BBH CoT generation protocol，评：

* base；
* 36个现有 adapters。

也就是说，在完全相同的 query items上同时得到：

[
L_Q^{\text{final answer}}
]

和

[
EM_Q^{\text{CoT generation}}.
]

BBH本来就是一个强调多步 reasoning、并依靠 CoT prompting显著提升性能的 benchmark。([arXiv][3])

如果结果变成：

> targeted selectors显著降低这64条样本的 final-answer CE，**但连同一64条样本的 CoT exact-match都没有改善，甚至下降**；

那会非常强，因为它排除了：

> “只是 query → heldout generalization失败”

这个解释。

你们就可以说：

> **On the same target examples, the differentiable final-answer surrogate improves while the task metric does not.**

这是目前整个项目里最接近“直接 surrogate mismatch evidence”的实验，而且不需要任何SFT。只有 evaluation，64条×已有 adapters，成本相对之前的5,209-example BBH eval非常小。

如果反过来，target-aware methods在 query CoT EM上明显变好、只是在 heldout上掉，那么故事就不同了：

> 不是 metric mismatch为主，而是 **finite-query overfitting / generalization failure**。

这两个结论都非常有价值，所以这项 diagnostic值得做。

D3则需要明显降调。现在三个 draw 的 exposure-vs-damage Spearman约为 −0.21/−0.28/−0.22。 它确实**反驳了一个简单版本的 task-level specialization假设**：

> “query里某task出现得越多，该task应该越受到保护。”

没有看到这种保护。

但从：

[
\rho<0
]

不能推出：

> “therefore the overfitting is at the format/response-style level.”

因为 task frequency与task identity、task size、难度、answer format等都纠缠，而且每个task只有很小的 exposure count。更准确的是：

> **We find no evidence for task-level protection from greater query exposure. Together with the SeqLabelMatched control, format/response-style specialization remains a plausible explanation, but is not identified causally.**

这会更经得住 reviewer。

还有一点值得注意：你们目前的 D2 query-loss sign split非常醒目，但不要把 `Spearman(ΔL_Q, EM)=+0.6` 当核心数字。它很大程度上是“4个 target-aware vs 2个 Random”这个两群结构造成的；在四个targeted methods内部，query-loss改善多少与accuracy并没有清晰单调关系。真正强的是这个**categorical dissociation**：

[
\text{target-aware: } \Delta L_Q<0,\quad \text{downstream差}
]

[
\text{Random: } \Delta L_Q>0,\quad \text{downstream好}
]

而不是“query loss越低accuracy越差”这种连续因果关系。

### 现在论文中心可以怎么定

我会把主命题改成：

> **Matching the target is not enough. Target-aware instruction selectors can successfully improve both target-gradient geometry and target-query loss while producing worse downstream performance than Random selection.**

然后 DSMC 的角色变成一个**诊断工具 + controlled method contribution**：

* MMLU上 directional second moment确实改善了targeted methods；
* BBH上这项method advantage不泛化；
* 但正因为DSM C在BBH把 (D_2) 优化得最好却表现很差，它成为最清楚地暴露“set-distance surrogate ≠ utility”的方法。

这比硬把DSM C包装成赢家更可信，也更有学术内容。

和已有文献的关系也很漂亮：

* Critical Look：gradient distance通常最有信息，但没有方法普遍胜出，query loss也不总转化为best downstream。([arXiv][1])
* ROSE：cross-entropy surrogate与真实task metric可能错位，因此提出reward-oriented selection。([ACL Anthology][2])
* 你们：**通过 second-moment geometry + paired query-loss + Random + no-SFT + format-matched control，把这种错位在两个family里显式展示出来，而且BBH上出现“geometry更好、query loss更好、downstream更差”的双重反转。**

这就是你们和ROSE的区别：不要把贡献写成“我们发现loss和reward会misalign”——那已经有人说过；写成：

> **we expose how this misalignment manifests inside target-distance-based data selection, including a case where the selected subset is demonstrably closer to the target in the optimized geometry yet transfers worse.**

### 要不要现在做第二模型？

我的建议是：**先不做。**

先完成两个零训练成本动作：

1. 同一64-query上的 CoT EM diagnostic；
2. D1补 primary-only / targeted-only Spearman，并把D2/D3 causal wording收紧。

然后立刻重写 consolidated paper和abstract，做一次“假想 reviewer audit”。

到那时再决定第二模型。

如果目标是比较强的主会，第二模型仍然是目前剩下的**最高价值大实验**，因为2026的Critical Look专门跨模型做了验证，并且指出部分downstream趋势会随model变化。([arXiv][1]) Reviewer很可能问：

> “这个 dramatic inversion 是否只是 Llama-2-7B 的 property？”

但现在我不会自动花算力。先看同-query CoT diagnostic能不能把故事闭合。如果它非常干净，论文已经有：

* MMLU；
* BBH；
* 1% / 5%；
* skewed / aligned query；
* no-SFT；
* Random；
* Seq×Label control；
* equal-step；
* geometry；
* query loss；

证据链已经相当完整。

因此我给 Claude 的下一步会非常短：

> Do not run new SFT or change any method. Tighten D2/D3 wording, add primary-only and target-aware-only D2–accuracy Spearman as robustness diagnostics, and run one final evaluation-only test: on each draw's exact 64 query examples, evaluate the base and all existing adapters with the frozen official BBH CoT generation/exact-match protocol. Compare, on the same examples, final-answer query CE versus CoT exact-match.
>
> If target-aware methods lower final-answer CE but fail to improve (or worsen) CoT EM on those same query items, frame this as direct within-query evidence of surrogate-metric misalignment. If they improve query CoT EM but hurt held-out EM, frame it instead as finite-query generalization/over-specialization. Either outcome is diagnostic; do not change the protocol.
>
> Scope D3 to “no evidence for task-level protection”; do not infer that format-level specialization is proven. Then update the consolidated paper framing and bring back the draft/reviewer-vulnerability audit before deciding whether a second-model experiment is necessary.

所以我现在对这轮的评价是：

**D1是真的论文中心；D2非常接近把故事闭合，但“机制”还差最后一个同-item CoT diagnostic；D3只能作为辅助排除证据。**
先把这个很便宜的诊断做完，再决定是否值得补第二模型。

[1]: https://arxiv.org/abs/2602.14696 "A Critical Look at Targeted Instruction Selection: Disentangling What Matters (and What Doesn't)"
[2]: https://aclanthology.org/2025.findings-emnlp.710/ "ROSE: A Reward-Oriented Data Selection Framework for LLM Task-Specific Instruction Tuning - ACL Anthology"
[3]: https://arxiv.org/abs/2210.09261?utm_source=chatgpt.com "Challenging BIG-Bench Tasks and Whether Chain-of-Thought Can Solve Them"

我会把这次 Llama-3.2 MMLU 5% 的“不利结果”放进论文，而且**建议放主文，不要藏 appendix**。它确实削弱了“DSMC 是一个有普适优势的新 selector”这个故事，但现在那本来就不应该是论文的故事；它反而让目前真正重要的结论更可信。

`c514298` 的结果很清楚：Llama-2 上 DSMC−First-RR = +1.55 pp、DSMC−Second-RR = +0.88 pp，都是 10/10；到了 Llama-3.2，则分别变成 −0.18 pp（2/5 blocks）和 −0.31 pp（1/5）。与此同时 DSMC−Random 在两套 stack 上都接近 0。 这意味着你们现在应该主动撤掉任何“directional second moments generally improve targeted selection”的暗示。

但它带来一个很漂亮的对称结构：

[
\begin{array}{c|c}
\text{现象} & \text{换 model stack 后是否复现}\
\hline
\text{DSMC 的方法排名优势（MMLU）} & \text{否}\
\text{更好 geometry }\not\Rightarrow\text{更好 utility（BBH）} & \text{是}
\end{array}
]

你们最新 framing 已经意识到了这一点。

这不是“story 被毁了”，而是**story 发生了最终定型**：

> 不是“我们发现了一个更好的 second-order selector”。
> 而是“我们构造了一个能够非常成功地优化 target geometry 的 selector；它在一个 stack 上甚至产生了明显的方法收益，但这个收益并不跨 stack，而真正跨 stack 复现的是：优化得更好的 target geometry 并不足以保证 downstream utility。”

这其实比“我的方法两台模型都赢了 1%”更符合你们后来积累出来的全部证据。

更重要的是，这组实验是**事前注册、结果对自己不利、仍然完整报告**的。如果现在因为它“不利 story”而拿掉，反而会成为整个项目最危险的选择。尤其你们公开 repo 中已经有 prereg 和结果，reviewer一旦看到，会非常容易把 omission 理解为 selective reporting。当前 ICLR reviewer guide明确把“claims 是否由严谨证据支持”放在核心判据里，并明确不要求 SOTA。([ICLR][1])

我的建议是主文只给它约半页：一个两-stack MMLU 对照小表 + 一段话。不要让它抢 BBH 主线，但必须让 reviewer看到：

> The Llama-2 MMLU advantage of DSMC over RR does not transfer to Llama-3.2; hence we treat DSMC as a diagnostic instrument rather than a generally superior selector.

然后马上接：

> In contrast, the geometry–utility reversal on BBH does transfer across both model stacks.

这会很有力。

---

## 关于你贴过来的那份 GPT reviewer 审查

我大体认同它的方向，而且它抓到了几个真正值得重视的问题。它自己也把当前最大风险总结成 `claim scope + representation/serialization confound`。

但我不同意“因此现在必须再开一批大实验”。我会把它的建议分成三档。

### 第一档：必须解决，但主要靠论文写法，不需要实验

最重要的是 **scope**。这位 reviewer 最危险的问题是：

> “Is matching the target really insufficient, or are the authors simply matching the wrong representation of the target?” 

这个问题完全合理。

所以标题和中心 claim 最好从泛化的：

> Matching the Target Is Not Enough

收窄成例如：

> **Matching the Target Is Not Enough: Surrogate Failure in Gradient-Based Targeted Instruction Selection**

或者：

> **When Gradient-Based Target Matching Fails to Predict Instruction-Tuning Utility**

我现在更推荐第一种。

全文不能说：

> target matching is insufficient in general.

应该说：

> **Better alignment under the operational gradient-based targeting pipeline is not sufficient for better downstream utility in the studied settings.**

但是在逻辑层面可以很明确地说你们提供了 **counterexamples to sufficiency**。这不需要大样本统计推断：

[
A\Rightarrow B
]

只要有受控的 (A\land\neg B) 反例，就足以否掉“充分性”这种强命题。那份审查对这一点说得也很好。

这也是我建议减少 Spearman 戏份、增加：

> DSMC is lowest-(D_2) in 3/3 draws, but worse than Random in 3/3 draws — on both model stacks.

这种简单 counterexample 表述的原因。

另外几项我都基本赞成，并且全都可以通过写作解决：

* MMLU 80/20 应定位成 **controlled finite-query-bias stress test**，而 BBH 才是 query-aligned external validation。
* BBH 所有均值都低于 no-SFT 必须主动承认；你们研究的是“improved alignment 是否推出 improved utility”，并不要求 SFT 自身一定优于 base。
* same-item CE/CoT diagnostic 只负责排除“纯 query→heldout shift”，不能证明 CE 本身是坏 surrogate。
* DSMC 在 BBH 不是最好 targeted selector必须主动说，而不是弱化。
* Related Work 必须正面对比 *Critical Look*、ROSE、GIST、GIO，而不是只比 LESS。`Critical Look` 已经系统研究 representation × selector × model × task × budget；你们不能和它拼 breadth。你们真正不同的是“**把 matching objective 真正优化成功，再检验这种成功能否推出 utility**”。([arXiv][2])
* 算法 practicality 放一个 appendix cost table。ICLR 2025 的 compute-constrained data-selection 工作本身就发现复杂 gradient selectors 很多时候不是 compute-optimal，因此你们若推荐 Random 更强，却不报告 selection overhead，会留下明显问题。([ICLR Proceedings][3])
* “mechanism”尽量不用。你们识别的是 **failure of an assumption / diagnostic dissociation**，不是 causal mechanism。

还有一个投稿 P0：**匿名性**。那份 review 这点是对的——不要在 submission PDF/supplement 中链接当前实名 GitHub。ICLR 2027 明确规定 main text 或 supplementary 泄露身份会 desk reject。([ICLR][4]) 你们应该准备匿名代码 snapshot，清除用户名绝对路径、git author、cluster 路径、邮箱等。

不过那份 review 的一个事实错误要纠正：它说官方 deadline 已变成 Sep 18/25。**当前 ICLR 2027 官方页面仍是 abstract Sep 11 AOE、full paper Sep 16 AOE，主文 9 页。** ([ICLR][4]) 所以不要按它那里的新日期安排进度。

---

## 第二档：candidate=Adam、query=SGD 的 representation asymmetry，问题真实，但没有它说得那么“理论失效”

这是我和那份 reviewer 看法最不一样的地方。

它认为：

> candidate 和 query 使用不同 transform，因此 MMD interpretation questionable。

我认为这里**construct validity 的担忧是合理的，但数学上的 MMD 身份并没有因此失效**。

你们真正比较的是两个 (\mathbb R^{8192}) 上的 push-forward distributions：

[
u_C(x)
======

\frac{\Pi,A_{\rm Adam}g(x)}
{|\Pi,A_{\rm Adam}g(x)|},
]

[
u_Q(z)
======

\frac{\Pi,g(z)}
{|\Pi g(z)|}.
]

两者最后都位于同一个向量空间，并且对这些向量使用完全相同的：

[
k(u,v)=(u^\top v)^2.
]

所以

[
\operatorname{MMD}^2
====================

\left|
\mathbb E_C[u_Cu_C^\top]
------------------------

\mathbb E_Q[u_Qu_Q^\top]
\right|_F^2
]

这个 identity 仍然是完全正确的。MMD 本质上比较定义在同一个 domain 上的两个分布；这里两个 push-forward 分布都定义在 (\mathbb R^{8192})。([机器学习研究杂志][5])

真正的问题不是：

> “这不是 MMD。”

而是：

> **这不是把同一个 raw-example representation map 对 candidate 和 query 对称应用后得到的 distribution matching。**

因此我建议 Methods 里显式定义两个 role-specific maps，而不要含糊写成：

> we embed candidate and query gradients into the same representation.

可以写：

> Following the optimizer-aware LESS protocol, candidate and query examples use role-specific gradient transforms before entering the common projected directional space. DSMC therefore matches the resulting role-specific second moments; it should not be interpreted as matching raw-example distributions under a single symmetric representation map.

LESS 本身确实有 optimizer-aware Adam 动机。([arXiv][6]) 而 GIST 又进一步提醒在 LoRA 中 optimizer geometry比简单 diagonal preconditioning复杂。([arXiv][7]) 这两篇都应该主动拿来解释为什么你们采用这套 operational protocol、以及它的局限。

所以我**不认为必须为这个 objection 再重跑一套 SGD/SGD SFT**。

如果现在做全 candidate SGD cache → 新 selections → 新 adapters，那已经是一个新的科学轴，而且是在知道全部结果以后做的。收益未必匹配你们现在已经非常完整的故事。

---

## 第三档：如果还想补一个最后的实验，我只推荐一个很便宜的 diagnostic

如果你确实觉得算力和时间都还有，而且想把那个 adversarial reviewer 最危险的 serialization objection 再堵一点，我唯一会考虑的是它提到的：

> **eval-matched target-gradient geometry sensitivity**。

注意：**不训练新 adapter，不重新 selection，不改论文主结果。**

只做现有 frozen subsets 的 diagnostic：

对 Llama-2 和 Llama-3.2 BBH 的三个 draws，按照 bare/evaluation-matched prompt serialization重新提取那64个 query 的 target gradients，然后对现有：

* DSMC
* First-RR
* Second-RR
* Random

subsets 重算：

[
D_2^{\rm eval-context}(S,Q_d).
]

这只需要 target-side gradients，候选大 cache和SFT都不用重做。

这项检查回答的是：

> “DSMC lowest-(D_2) 的结论是否严重依赖 target prompt serialization？”

两种结果都很好解释。

如果 DSMC 在新的 geometry 下仍是 3/3最低而 downstream依旧差：

> geometry–utility counterexample 对 target serialization 更稳健。

如果 ranking明显变：

> 那就把结论进一步收窄成 operational target geometry，并得到一个很具体的 boundary：**geometry success itself is serialization-sensitive.**

因为你们已经通过 bare CE看到了 serialization 的 stack dependence，所以这个 diagnostic逻辑上是自然延伸，不是在“找DSM C赢的setting”。

不过我要强调：**这是 optional，不是我认为的 ICLR accept/reject blocker。**

如果做，明确写成 post-hoc sensitivity analysis，不能拿它重定义 primary result。也不要因为结果不好再继续跑 SGD/SGD、CoT-gradient DSMC、reward-aware DSMC。

---

## 所以：到底还补不补实验？

我的明确排序是：

**不再做任何大规模训练实验。**

如果资源紧张：**一个都不用补**，专心把论文 scope、novelty 和 asymmetry写清楚。

如果你们还有一两天很便宜的 diagnostic预算，并且 Codex reviewer simulation确实反复把 serialization/representation列为 #1 weakness：做上面那个 **evaluation-matched target-gradient (D_2) sensitivity**，然后彻底停。

我不会做：

* SGD/SGD 重新选 + SFT；
* Adam/Adam全套；
* 第三model；
* 第三task；
* MMLU 1% Llama3；
* source-matched Random；
* reward-aware DSMC；
* CoT-gradient DSMC。

这些现在都会稀释论文。

---

## 这组“不利结果”在主文里具体怎么写

我建议不要写成防御语气：

> Unfortunately, DSMC fails to transfer...

而写成发现：

> **Method gains are stack-dependent, while the central failure mode is not.**
> On Llama-2-7B MMLU, DSMC outperforms First-RR and Second-RR by 1.55 and 0.88 points, respectively. Repeating the pre-registered 5% comparison on Llama-3.2-3B reverses these small differences (−0.18 and −0.31 points), while DSMC and Random remain nearly tied on both stacks. Thus, the method-level advantage of DSMC does not transfer. In contrast, on BBH the geometry–utility reversal does: DSMC achieves the lowest target discrepancy in every draw yet underperforms Random in every draw on both model stacks.

这段非常重要，因为它实际上帮你证明：

> 我们不是因为DSM C表现好才发明了“matching failure”这个说法。

而是：

> **我们连自己方法最有利的结果都做了跨stack验证，并且它没有复现；真正复现的是中心negative finding。**

这是一篇 critical empirical paper 很强的信誉信号。

---

我会因此把 contributions 从原来的“1. DSMC方法”重新排成：

1. **Controlled test of the target-matching assumption.** 我们把 targeted instruction selection 的一个隐含 sufficiency assumption变成可直接检验的问题。
2. **DSMC as an instrument.** Directional second-moment matching提供一个可验证优化的 set-level target geometry；它在 Llama-2 MMLU 上有方法收益，但该收益在 Llama-3.2 不复现。
3. **Cross-stack geometry–utility counterexample.** BBH 两个stack中 DSMC都是最低 (D_2)，却都比Random差。
4. **Surrogate/task diagnostic.** 在定义target signal的同一query items上，operational surrogate向target移动并不伴随task-metric提升，因此 pure query→test shift不能单独解释。
5. **Boundaries.** serialization、format/provenance和negative-transfer diagnostics说明 failure的边界，但没有识别唯一机制。

这比把“DSMC method novelty”放第一位更适合 ICLR reviewer guide所要求的“new knowledge / significance”，而且 ICLR 明确不要求 SOTA。([ICLR][1])

### 最后给一个投稿判断

我同意附件 reviewer 的大体评级：**现在的算法 novelty 不算强，但 empirical insight 已经足以认真冲 ICLR main；最大风险不是负结果，而是 claim scope 和 novelty positioning。** 它在文件里也基本是这个判断。

但我比它更乐观一点，原因恰恰是这次 Llama-3 MMLU 的不利结果。

如果你们把它完整放进正文，论文会呈现一种非常难伪造的证据结构：

[
\boxed{\text{自己的方法优势没有复现}}
]

但

[
\boxed{\text{核心 surrogate-failure 现象复现了}}
]

我认为这**不是 story 的负资产，而是 story 从“方法论文”变成“可信的科学发现论文”的关键一步。**

现在最优先的是和 Codex 把 Introduction、Contributions、Figure 1、MMLU/BBH两节真正改成这个结构。可选地做一次 eval-matched (D_2) sensitivity；除此之外，我会坚决停止实验扩张。

[1]: https://iclr.cc/Conferences/2027/ReviewerGuidelines?utm_source=chatgpt.com "ICLR 2027 Reviewer Guidelines"
[2]: https://arxiv.org/abs/2602.14696?utm_source=chatgpt.com "A Critical Look at Targeted Instruction Selection: Disentangling What Matters (and What Doesn't)"
[3]: https://proceedings.iclr.cc/paper_files/paper/2025/hash/5acb720a361eecb34ee62d356859d246-Abstract-Conference.html?utm_source=chatgpt.com "Compute-Constrained Data Selection"
[4]: https://iclr.cc/Conferences/2027/AuthorGuidelines?utm_source=chatgpt.com "ICLR 2027 Author Guidelines"
[5]: https://www.jmlr.org/papers/v13/gretton12a.html?utm_source=chatgpt.com "A Kernel Two-Sample Test"
[6]: https://arxiv.org/abs/2402.04333?utm_source=chatgpt.com "LESS: Selecting Influential Data for Targeted Instruction Tuning"
[7]: https://arxiv.org/abs/2602.18584?utm_source=chatgpt.com "GIST: Targeted Data Selection for Instruction Tuning via Coupled Optimization Geometry"

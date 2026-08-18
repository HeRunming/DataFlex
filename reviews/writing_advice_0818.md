可以，而且我觉得现在正是最适合把全文“串起来”的时候。你们目前最有力的故事已经不是“我们提出 DSMC，然后它赢了”，而是一个更完整、也更像 ICLR 的问题驱动故事：

> **Targeted instruction selection implicitly assumes that matching a small target/query set more faithfully should produce more useful training data. We construct a stronger way to match that target, verify that the matching objective is actually optimized, and then show that better target alignment can still fail to produce downstream utility.**

这个问题和现有工作接得很自然。LESS把 targeted instruction tuning明确建模为用 few-shot target examples寻找相关训练数据，并通过 optimizer-aware gradient similarity实现它。([arXiv][1]) 最近的 *Critical Look* 又把 targeted selection拆成 representation 和 selection algorithm 两部分，发现 gradient representations整体最有预测力，但没有单一方法普遍占优，而且优势会随模型、任务和预算变化。([arXiv][2]) ROSE从另一侧指出 instruction-tuning cross-entropy 与最终 task metric可能不单调。([ACL Anthology][3]) 另有大规模研究发现复杂 selector 往往很难稳定超过 Random。([ACL Anthology][4])

你们最好的定位不是重复其中任何一句，而是把这些现象串成一个**可直接检验的因果链假设**：

[
\boxed{
\text{Target Geometry}
\rightarrow
\text{Selected Data}
\rightarrow
\text{Operational Surrogate}
\rightarrow
\text{Task Utility}
}
]

然后论文的核心发现是：**前面的箭头可以明显成立，最后一箭头仍然可以断。**

你们仓库现在的 `paper_framing.md` 已经基本走到这个方向了：它把 DSMC 定位成“instrument”，而不是赢家方法，也明确把主逻辑写成 `geometric alignment + surrogate improvement ⇏ task improvement`。这个判断我赞成。

### 我建议全文按这样的 7 幕来讲

1. **先提出问题，而不是先提出 DSMC。**
   Targeted instruction selection 的吸引力很直观：有少量目标数据 (Q)，从大 candidate pool 中挑出最“像目标、对目标最有帮助”的训练数据。LESS、GIST 等方法虽然实现不同，本质都依赖某种 target alignment。现有 controlled study 也表明 gradient representation 往往是其中较有信息量的一类。([arXiv][1])
   但这里有一个很少被直接检验的前提：**如果我们真的把 selected set 与 target 匹配得更好，downstream performance 就应该更好。** 你们论文就是检验这个前提。

2. **为了把这个前提检验得足够锋利，我们构造一个更强的 target-matching instrument：directional second moments。**
   这里引出 DSMC，但不要一上来卖成 SOTA。先解释你们为何怀疑 first-order gradient mean 不够：direction matters，(+u) 和 (-u) 在线性平均中会抵消，但在 (uu^\top) 中保留同一方向子空间信息。然后给出最漂亮的理论身份：
   [
   M_P=\mathbb E[uu^\top],
   \qquad
   \mathrm{MMD}_{\langle u,v\rangle^2}^2
   =====================================

   |M_S-M_T|_F^2.
   ]
   DSMC 因此不是“又一个 heuristic selector”，而是**显式最小化 target directional second-moment discrepancy 的 coreset construction**。这一步的功能是让后面“objective 真的被优化了”变得可测量。

3. **先在 MMLU 上做 controlled attribution，证明 second-order representation 不是凭空编出来的。**
   这里放你们最干净的 2×2：first/second-order representation × TopK/MMD。Llama-2 上两个 skew directions都显示 second-order representation 是主要增益来源；5% 下 DSMC 又进一步胜 Second-RR。这里的正面结论要故意写窄：

   > directional second moments can materially improve target-aware selection in the MMLU setting.

   不要提前写“DSMC is a generally better selector”。正在跑的 Llama-3.2 MMLU 5% 正好只负责决定这句话能否扩成“across two stacks”。它**不会改变整篇论文中心**。

4. **然后故事第一次反转：targeted selector 变好了，但 Random 没被打败。**
   Llama-2 MMLU 5% 中 DSMC 对 targeted baselines明显更强，却与 Random 几乎持平；到1%，Random反而领先，而且 tightening budget 并没有让 target awareness 的优势出现。base/no-SFT reference又揭示：很多 targeted methods 实际是在 negative transfer，DSMC更多是“伤得少”，而不是“大幅创造新能力”。Equal-step arm又说明简单增加 optimizer steps不会 rescue，反而两者都过训练。

   这一幕的作用不是宣布“Random wins”。而是提出一个新的、更重要的问题：

   > **DSMC 是没有真正匹配好 target，还是 target matching 本身不是可靠的 downstream surrogate？**

5. **用 geometry forensic 回答这个问题：不是优化失败。**
   这是全文真正进入核心的地方。MMLU 上 DSMC 比 Random 更接近自己的 skewed query，而且在 leave-one-draw-out balanced reference 上也更近；所以“DSMC只是在追逐偏斜 query、牺牲真正目标分布”这个解释被你们自己的实验否掉了。然后 BBH 做 external validation，把 query/evaluation 改成同一 family 的 held-out design，结果更强：DSMC 在 3/3 draws 都取得最低 (D_2)，downstream却明显不如 Random。第二个 Llama-3.2 stack 又复现相同方向。当前 cross-stack表和这个 scope 已经在 framing 文档里冻结得比较好了。

   这里的主结论应是：

   > **Better target-gradient alignment is not sufficient for downstream improvement.**

   我非常建议用 `not sufficient`，而不是 `targeted selection fails`、`unreliable in general`。两个 stack 都出现了 counterexample，已经足够否定“更好 matching 必然意味着更好 utility”，但没有必要声称 targeted selection 永远无效。

6. **再回答 reviewer 一定会问的第二个问题：也许 geometry 不够，但至少它让模型在 target query 上学得更好了吧？**
   这就是 same-item diagnostic 的意义。最漂亮的地方不是 held-out test：是在定义 targeting signal 的**同一64条 query items**上，target-aware methods让 operational wrapped final-answer CE 显著下降，但 CoT task metric并没有同步改善；Llama-2上甚至明显下降，Llama-3.2上平均也呈相同的 targeted-vs-Random分离。于是你们不是只看到：
   [
   \text{query surrogate good}
   \quad\text{but}\quad
   \text{held-out test bad},
   ]
   而是看到同题：
   [
   \text{operational surrogate}\downarrow
   \quad\text{while}\quad
   \text{task metric}\not\uparrow.
   ]
   这使得单纯的 query→test distribution shift 无法独自解释结果。

   但一定保留 D2c 带来的自我限制：bare-context CE 在两个 model stacks 上行为不同，所以不能说“cross-entropy 本身与 CoT inherently misaligned”。你们能说的是**pipeline 实际使用的 operational surrogate 与 task utility发生了 dissociation，而且 serialization 的影响本身具有 stack dependence**。ROSE 已经指出 instruction CE 与 task metric可能不单调；你们真正的新东西是，把这一 mismatch 与“set-level target geometry 确实被成功优化”连在了一起。([ACL Anthology][3])

7. **最后不是再解释一个机制，而是给出一个更成熟的结论。**
   Seq×Label-matched Random 说明 instruction format/provenance composition 是 plausible contributor，但不能做 41% causal decomposition；task exposure diagnostic又没有支持简单的“某个 task 被 query exposure保护”的 specialization story。所以结尾最好不是：

   > We found why targeted selection fails.

   而是：

   > **We identify a failure of the surrogate assumption, not a unique failure mechanism.**

   Target-aware selectors能够成功做到它们“应该做到”的事情——匹配 target geometry、在 operational surrogate上向 target移动——但这些中间目标并不是 utility 的充分代理。这为为什么 Random有时如此难打败，提供了比简单 leaderboard 更细的解释。

---

我觉得这条故事有一个特别好的“正—反—更深”的结构：

> **正面：** second-order target representation 确实比 first-order 更有用。
> **反面：** 即使把 targeted selection 做得更好，也不一定超过 Random。
> **更深：** 不是因为 matching objective 没优化好；恰恰是 matching 已经做得很好，但 intermediate surrogate 与 task utility 之间的假设会断。

这比从头就讲“targeted selection 不好”强很多。因为 reviewer 会看到你们不是先抱着 negative thesis 去找反例，而是**先做出了一个确实改善 targeted selection 的方法，然后这个更强的 instrument 反过来暴露了领域更基础的问题。**

这也恰好能与 *Critical Look* 区分。它系统拆分 representation/selector，并发现 gradient representation较有预测性、RR低预算平均较强，但没有单一方法普遍占优。([arXiv][2]) 你们的推进是：**不是只问某种 distance 是否 predictive，而是把一种 set-level distance明确最小化，然后观察“优化成功但 utility失败”。** 与 Random-at-scale 的区别则是：他们给出“Random难打败”的 broad empirical observation；你们给出一个被 instrumented 的 target-matching counterexample chain。([ACL Anthology][4])

### 我会怎么安排 9 页主文

ICLR 2027 主文严格最多9页，而且 reviewer 不要求看 appendix，所以最关键的证据必须留在正文。([ICLR][5]) 我会大概用这样的空间逻辑，而不是按实验发生时间罗列：

Introduction 用约1页，把上面的链条和反例直接讲清。

Problem + DSMC约1.25页。只给最必要的 gradient representation、second-moment identity和greedy rule。Moment-MMD calibration全过程不要进主文。

MMLU controlled attribution约1页。2×2 + 5%/1%最核心结果，把DSM C作为instrument建立起来。

BBH external reversal约1.5页。Random/base、targeted methods、SeqLabel control一句话带过。

Geometry→utility dissociation约1页。这里应该有主图。

Same-item surrogate→metric dissociation约1页。

Llama-3.2 cross-stack confirmation约0.75页。

Related work + limitations + conclusion塞进剩下约1.5页。

GIST/NICE实现、Moment-MMD λ/α calibration、equal-step、contamination、all hashes、D3、prompt audits、SeqLabel细节全部appendix。ICLR reviewer guide关注的是 claim、evidence和新知识是否匹配，而不是要求你把所有工程过程塞进正文。([ICLR][6])

### 主图我建议不要画 leaderboard

我最想看到 Figure 1 是一个两层图。

上面是概念链：

[
\boxed{\text{query set}}
\rightarrow
\boxed{\text{target-gradient geometry}}
\rightarrow
\boxed{\text{selected subset}}
\rightarrow
\boxed{\text{operational query surrogate}}
\not\Rightarrow
\boxed{\text{task utility}}
]

下面放两个 model stacks 的实数：

Llama-2：
[
D_2(\mathrm{DSMC})<D_2(\mathrm{Random}),
\quad
\Delta CE_{\rm wrapped}<0,
\quad
\Delta EM_Q<0,
\quad
\Delta EM_{\rm heldout}<0.
]

Llama-3.2：
同样显示 (D_2) 最低、wrapped CE明显改善、task utility仍不更好，同时明确注明 effect attenuated。

一张图就把论文90%的 novelty表达出来。

### 正在跑的 Llama-3.2 MMLU 5% 怎么嵌进去

正文现在先留一个很小的 `Cross-stack MMLU validation` 槽位，不要让它决定文章结构。

如果最终 **DSMC > First-RR 且 > Second-RR**，你们的正面 claim升级成：

> Directional second-moment matching improves targeted instruction selection on MMLU across two model stacks, yet this improvement neither guarantees an advantage over Random nor transfers uniformly to BBH.

这会是最好看的“positive method contribution + negative general principle”。

如果 **DSMC≈Second-RR，而两者 > First-RR**，故事也非常好：

> The transferable component is the second-order representation; the additional MMD coreset gain is stack-dependent.

这其实会让你们的2×2 attribution更漂亮，因为核心变成 representation。

如果 **First/Second-RR ≥ DSMC**，不要慌：

> Even the positive MMLU method ranking is model-stack dependent; the result that transfers most robustly is the failure of better target matching to guarantee utility.

中心故事完全不动。

如果 Random再度最好，也只是继续加固 target-awareness anchor，不能让文章变成“Random paper”。

### 有三句我会让 Codex 永久禁用

你们现在 `paper_framing.md` 还有两个残留的内部冲突，写正文前最好直接修掉。第一，它在 contribution 5 仍写“CE improvement disappears under bare serialization”，但同一个文件后面已经正确记录 Llama-3.2 targeted arms 的 bare CE 会改善；应该统一改成“serialization sensitivity is model-stack dependent”。第二，related work 里还写了 `mechanistic-level counterexample chain`，而你们自己明确没有找到 mechanism；改成 **instrumented empirical counterexample chain** 或 **diagnostically resolved counterexample**。

第三句虽然文档已经基本避免，但 Codex 很容易自动生成回来：

> “DSMC improves the surrogate the most.”

Llama-3.2 上 First-RR 的 wrapped CE下降实际上更大。DSM C真正特别的是 **最低 (D_2)**，不是最大CE gain。

### 我会用的标题

目前最稳的一版是：

**Matching the Target Is Not Enough: A Controlled Study of Targeted Instruction Selection**

如果想强调你们独特的 gradient contribution：

**Matching the Target Is Not Enough: When Gradient-Aligned Instruction Selection Fails to Deliver Utility**

我暂时更喜欢第一版。第二版的 `fails` 容易被读成一个普遍否定，而你们最严谨的 claim 是 `not sufficient`。

Abstract 甚至可以围绕5句话写：

> Targeted instruction selection assumes that training subsets better aligned with a small target set should yield greater downstream utility.
> We introduce directional second-moment matching, a coreset objective that explicitly matches the target distribution in projected gradient geometry, and show that it improves targeted selection on controlled MMLU experiments.
> Yet stronger target matching does not reliably translate into stronger task performance: on held-out BBH, DSMC achieves the closest target geometry while underperforming Random selection, a reversal that replicates across Llama-2-7B and Llama-3.2-3B model stacks.
> On the same query examples used to define the targeting signal, target-aware methods also improve the operational differentiable surrogate without corresponding gains in the task metric.
> These results show that successfully optimizing target alignment is not sufficient for downstream utility, motivating targeted selection objectives that are better aligned with the task quantities we ultimately care about.

最后一句要注意：可以说“motivating better-aligned objectives”，但不要暗示你们已经解决了它。ROSE已经在 reward-oriented方向上做了明确方法，因此 related work里要主动承认，并把你们区别定位成**diagnosis/instrumentation rather than proposing the next objective**。([ACL Anthology][3])

我现在会建议你和 Codex **直接按这个叙事写 Introduction + Figure 1 + Results section skeleton**，不要先从 Methods 开写。等正在跑的 Llama-3.2 MMLU 5%出来，只需要填“正面方法结论到底能扩到哪一层”，不应该重排整篇论文。

换句话说，全文最应该让 reviewer 记住的不是 **DSMC** 这四个字母，而是这一句：

> **We made target matching stronger, verified that it worked, and found that utility still did not follow.**

这就是目前最适合把你们所有正结果、负结果、falsification checks 和跨模型验证真正串成一篇论文的中心。

[1]: https://arxiv.org/abs/2402.04333?utm_source=chatgpt.com "LESS: Selecting Influential Data for Targeted Instruction Tuning"
[2]: https://arxiv.org/abs/2602.14696?utm_source=chatgpt.com "A Critical Look at Targeted Instruction Selection: Disentangling What Matters (and What Doesn't)"
[3]: https://aclanthology.org/2025.findings-emnlp.710/?utm_source=chatgpt.com "ROSE: A Reward-Oriented Data Selection Framework for LLM Task-Specific Instruction Tuning - ACL Anthology"
[4]: https://aclanthology.org/2025.findings-emnlp.146/?utm_source=chatgpt.com "Rethinking Data Selection at Scale: Random Selection is Almost All You Need - ACL Anthology"
[5]: https://iclr.cc/Conferences/2027/AuthorGuidelines?utm_source=chatgpt.com "ICLR 2027 Author Guidelines"
[6]: https://iclr.cc/Conferences/2027/ReviewerGuidelines?utm_source=chatgpt.com "ICLR 2027 Reviewer Guidelines"

我看完这次 equal-step 结果后，判断其实比之前更清楚了：**MMLU/Tulu 这条优化线应该正式结束，不再调 LR、epochs、LoRA 或训练步数。下一步应该转向 external validity，而不是继续“救”1% 结果。**

Equal-step 这次非常关键，因为它回答了最后一个合理的优化侧疑问。84 steps 时：

[
\text{DSMC}=0.4017,\qquad \text{Random}=0.4095,
]

420 steps 时：

[
\text{DSMC}=0.3826,\qquad \text{Random}=0.3827.
]

也就是说 DSMC 没有因为更多优化获得收益，反而下降约 1.91 pp；Random 降得更厉害，约 2.67 pp。两者最后都比 no-SFT base 的 0.4003 低约 1.8 pp。

所以那个正的

[
J=+0.0076
]

不能解释成“更多训练帮助 DSMC”。它只是因为 Random 过拟合得更严重。这正好落在你们预注册的 rule #4，而不是事后找理由。

这意味着：**“1% DSMC 输 Random 是因为 84 steps 没训够”这个简单解释已经没有支持。** 但也不能说已经证明“优化完全无关”，因为 420 steps 又引入了明显 over-training；更精确的说法是：

> DSMC–Random 的相对排序对 optimization horizon 有一定敏感性，但没有观察到一个训练步数条件使 DSMC 在改善绝对性能的同时超过 Random。

这个措辞很重要。

与此同时，forensic 的核心结论也站住了。leave-one-draw-out 后，DSMC 仍然在 10/10 draws 上比 Random 更接近 balanced (P^\star)，所以“DSMC 只是过度追逐 skewed query、牺牲 balanced geometry”基本可以关闭。

而且这个现象和近期 targeted-selection 文献形成了一个很好的对照：最新 controlled study确实发现 gradient-based distance通常比 embedding distance更能预测 query loss/downstream，但也明确指出更低 query loss 或更近 query 并不保证最高 downstream accuracy；Random 也经常在部分任务和预算下匹配或超过复杂方法。([arXiv][1])

不过 Claude 的 Spearman 结果还需要最后一个零成本修正。它现在报告的是把 70 个 method×draw points 混在一起做相关：

[
\rho(D_2,\text{acc})=+0.389\quad(1%).
]

这很有意思，但不能直接称为“D2 mildly anti-predictive”，因为这些点不是独立的，而且 pooled correlation 可能混入 method identity 和 draw difficulty。建议论文前再算两个 descriptive statistics：

* 每个 draw 内，对 7 个方法做 (D_2) 与 accuracy 的 Spearman，然后报告 10 个 draw 的 median/mean；
* 对 pooled data 做 draw fixed-effect residualization 后再看 correlation。

如果 within-draw 也普遍是正相关，那“更优的 D2 ranking 并不转化为更好的 downstream ranking”会非常有力。如果 pooled positive、within-draw 接近零，那应该写成 method-level dissociation，而不是 anti-predictive。

同样，source entropy 不应继续作为主要解释。GIST 的 source entropy 约 1.228，几乎和 Random 的 1.227 一样，却明显表现更差，这已经说明 coarse source entropy 不足以解释结果。Instruction-tuning diversity 的系统研究也发现，单一的简单 diversity metric 往往不足，需要同时考虑样本间差异与样本空间的信息分布。([ACL Anthology][2])

所以现在最合理的论文机制结论不是：

> Random 赢因为 source diversity。

而是：

> **Gradient-geometric coverage and instruction-data coverage are distinct notions, and neither the optimized target distance nor coarse source diversity alone is sufficient to explain downstream utility.**

这比一个未经因果验证的 source story稳很多。

接下来我建议分两步。

第一步，**现在先把全部结果合成一个 paper-ready consolidated document**。不是因为要停止实验，而是因为到这里你们已经有一条完整、不会再因一个新 MMLU 超参结果而改变的主线：

* Moment-MMD calibration → pure second moment；
* STEM/HUM mirror；
* representation × selector 2×2；
* 10 independent/non-overlap replicates；
* 5%：DSMC 明显优于 targeted baselines，但不优于 Random；
* 1%：DSMC 仍优于多数 targeted baselines，但 Random 优势扩大；
* base reference：大量 targeted selectors 发生 negative transfer；
* equal-step：更多训练没有 rescue DSMC，而是让 DSMC/Random 都过拟合；
* forensic：DSMC 确实优化了目标几何，甚至更接近 balanced (P^\star)，但 geometry 本身不足以保证 downstream utility。

这个故事已经很完整。

第二步，**如果还要投入下一轮大实验，优先做一个真正的 external-validity setting，不要再做 MMLU/Tulu 参数实验。**

我会优先选择“第二个 target/eval family + query 与 evaluation 同分布”，而不是第二个 candidate pool。

理由是你们当前最重要的外推缺口是：所有结论都来自 MMLU STEM/HUM skew setting。标准 targeted selection 问题通常假设 query set 是从实际 target-task distribution 抽来的；近期 controlled study也在 BBH、Codex、GSM8K、TyDiQA、MMLU-Pro 等多个 target tasks 上进行这种比较。([arXiv][1])

一个特别干净的下一轮可以是：

* candidate pool 保持 Tulu，避免同时改变两个轴；
* 换一个 target family，比如 BBH 或 MMLU-Pro；
* query 从该 target distribution 正常抽样，不人为 80/20 skew；
* 只比较四个方法：

  * DSMC；
  * Second-RR；
  * LESS/RR；
  * Random；
* 先一个预算，例如 1% 或文献常用的约 2,500 examples；
* 3–5 个 query draws；
* no-SFT base reference保留。

这样回答的是一个非常关键的问题：

> DSMC 的优势是“针对 skewed query 的鲁棒性”，还是它在标准 query-aligned targeted selection 中也能成为更好的 gradient representation？

这是比第二 candidate pool 更直接的问题。

如果 DSMC 在 query-aligned BBH/MMLU-Pro 上明显超过 Random，那论文故事会变成：

> DSMC 在标准 target-aligned setting 中能兑现 target awareness，而在 skewed finite-query setting 中比其他 targeted selectors更稳，但 Random 的 broad coverage 仍可占优。

如果 DSMC 仍然输 Random，但继续超过 LESS/RR，那么结论反而更统一：

> **Second moments improve targeted selection relative to existing targeted methods, but target-aware selection itself remains brittle versus Random across tasks.**

这与近期工作对 Random 强基线的发现非常吻合。大规模研究也发现许多复杂 selection methods 在更现实的规模上难以超过 Random。([arXiv][3])

我不建议下一步马上做“source-balanced DSMC”。那确实是一个有趣的探索，例如约束 DSMC 保留 candidate-pool source proportions，但它是在看到 Random source composition 后产生的 post-hoc method modification。可以作为 exploratory appendix 或下一篇工作的方向，但现在拿来当 main method upgrade，会让整个已经很干净的冻结实验链重新变成 outcome-driven method tuning。

所以我会给 Claude 这样的任务：

> Stop all MMLU/Tulu optimization tuning permanently. First consolidate the full experiment chain into one paper-ready results document, including the fixed-epoch 5%, fixed-epoch 1%, equal-step sensitivity, base-model reference, 2×2 attribution, mirror study, and forensic geometry analysis. Tighten the D2 correlation analysis by reporting within-draw Spearman correlations in addition to the pooled descriptive correlation.
>
> In parallel, scope one external-validity experiment on a second target/evaluation family with query and evaluation drawn from the same task distribution. Keep the Tulu candidate pool fixed so only the target/eval axis changes. Prefer a task used in the recent targeted-selection literature, such as BBH or MMLU-Pro. Start with DSMC, Second-RR, LESS/RR, and Random plus a no-SFT reference, 3–5 independent query draws, and one frozen low budget. Bring back the exact protocol and cost before launching.
>
> Do not modify DSMC based on the source-provenance forensic finding yet; treat source balancing as an exploratory follow-up, not part of the headline method.

如果只问“现在最重要的判断是什么”，我会概括成一句：

> **DSMC 已经证明自己是一个比现有 targeted selectors 更稳健的 representation/coreset 方法，但目前没有证据证明“更精确的 target-gradient matching”本身会带来比 Random 更高的 downstream utility；下一步应该验证这个结论能否跨 target family 成立，而不是继续在同一个 MMLU/Tulu setup 中寻找能赢 Random 的训练参数。**

[1]: https://arxiv.org/abs/2602.14696 "A Critical Look at Targeted Instruction Selection: Disentangling What Matters (and What Doesn't)"
[2]: https://aclanthology.org/2025.acl-long.908/?utm_source=chatgpt.com "Measuring Data Diversity for Instruction Tuning: A Systematic Analysis and A Reliable Metric - ACL Anthology"
[3]: https://arxiv.org/abs/2503.01807?utm_source=chatgpt.com "Large-Scale Data Selection for Instruction Tuning"


new advice after chat with codex:

我把最新 `d0bbe02` 的 consolidated 文档、你们当前 pipeline 的已知实现，以及近期 targeted-selection 论文重新对了一遍。整体判断是：

**Codex 的总结大体有道理，而且它没有推翻我们之前的方向，反而把“下一步为什么要做 external validity”说得更清楚了。** 真正需要调整的是：下一轮实验不能只是把同一套设计机械搬到第二个任务上，而应该顺便修掉当前最容易被 reviewer 抓住的统计和 provenance 缺口。

有一点先纠正：Codex 给出的“borderline / weak reject”只能算一种 reviewer-simulation，不是可以可靠预测的结果。ICLR 2027 的官方 reviewer guide 要求主要判断技术正确性、实验严谨性、可复现性、claim 是否被证据支持以及工作是否贡献新知识，并明确说并不要求 SOTA。([ICLR][1]) 所以“没有赢 Random”本身不是主要危险；**只有一个完整 target family、baseline fidelity 和 artifact inconsistency** 才更可能影响可信度。

### 我同意 Codex 的主论文定位，但建议再收紧一句

目前最好的故事确实不是“DSMC 是普适更好的 selector”，而是：

> directional second moments improve robustness within target-aware selection, while exposing that better target-gradient matching is not sufficient for downstream utility.

这个方向与最新 targeted-selection 系统研究非常契合。那篇工作本身就发现：gradient representation 的 query distance 确实比其他 representation 更有信息，但 lower query loss / closer query 不保证最佳 downstream；Random 在 TyDiQA、MMLU-Pro 等设置下也经常非常强。([arXiv][2])

不过 consolidated 中如果写“DSMC consistently outperforms the examined target-aware selectors”仍然略强。你们自己的 1% 结果里，DSMC 对 Second-RR 只有很小的平均优势，而且 cell-level 是 mixed；真正非常稳的是 5%。因此论文里最好拆开：

> At 5%, DSMC consistently outperforms all examined target-aware variants. At 1%, its advantage over first-order/adapted baselines remains clear, while the additional gain over Second-RR largely disappears.

这样也与 equal-step 和 2×2 的预算依赖结果完全一致。

---

### Codex 提出的七个漏洞，重要性其实不一样

**“只有一个 model / pool / MMLU family”是目前最大的科学漏洞。** 这点我完全同意。近期最有代表性的 controlled study用 Llama-2-7B + Tulu V2，但跨 BBH、Codex、GSM8K、TyDiQA、MMLU-Pro 五个 target tasks，并且还做 additional models；这说明在这个研究方向里，跨 target family 的证据已经是一个很自然的 reviewer expectation。([arXiv][2]) 你们 consolidated 也已经正确把这一点列为最高价值补实验。

**“十个 draws 不是 iid”是统计措辞问题，不会推翻结果。** 这些 draw 是从同一个有限 validation reservoir 联合、无放回地构造，因此存在负相关，确实不应称作 ten independent draws。 但这不会使 paired method differences 无效；只是不能使用假设十个 iid observations 的 CI 或显著性语言。现在统一成五个 direction-averaged blocks 做 descriptive inference 是合理的。

**“query draw 和 SFT seed 完全耦合”比上面这个更值得下一轮修。** 当前每个 draw index 固定一个 seed 42/1/2/3/4。 因为同一个 block 内所有方法共享 seed，所以 DSMC-vs-baseline 的 paired comparison仍然公平；但是你不能知道 block-to-block variation 到底来自 query realization 还是 training seed。因此现有论文最好称：

> five paired query-draw/training-seed blocks

而不是声称估计了纯 target-draw variance。

这会**直接改变下一轮实验设计**：下一 target family 最好采用 crossed design，而不是继续一 draw 一 seed。

**Random 跨方向共享 adapter → 有效单位是五个 blocks，这个判断完全正确，而且已经处理了。** 现在无需重跑现有实验，只需所有表格、CI、win count都按 block-level 解释。Codex 在这一点上只是把已有事实正式化。

**LESS/GIST/NICE 是 adaptation，这个提醒也完全正确。** 官方 LESS 不只是一个单 checkpoint mean-gradient TopK：它使用 optimizer-aware candidate gradients、SGD validation gradients，并可跨多个 warmup checkpoints按 learning-rate权重聚合。官方 LESS repo明确展示了这一 pipeline。([GitHub][3]) GIST 原方法又有自己的 target-gradient subspace / spectral filtering geometry。([arXiv][4]) NICE 则是针对非可微 downstream metric 的 policy-gradient方法。([Proceedings of Machine Learning Research][5])

所以这不会否定你们的 controlled comparison，但会限制 claim：

> 不要说 “DSMC beats LESS/GIST/NICE”。

应该说：

> “DSMC outperforms the examined LESS-style / GIST-SharedProj / NICE-MMLU-EM variants under a controlled shared-feature protocol.”

这两句话科学含义差很多。

**candidate-pool contamination audit 是现在必须补的，而且应该在新大实验前完成。** 你们的 candidate pool正是 LESS 使用的 Flan v2 + CoT + Dolly + OASST1，而 LESS 同样拿 MMLU、TyDiQA、BBH做 evaluation。([GitHub][3]) 但“原论文也这样做”并不能证明无 contamination。MMLU 本身的 contamination 风险已有大量研究，甚至出现了专门的 contamination-free MMLU-CF。([arXiv][6])

对 data selection 论文这个问题尤其敏感，因为不同 selector 可能以不同概率把 benchmark-like examples 拉进 subset。如果 DSMC、LESS、Random的 test-near-duplicate exposure 不一样，downstream ranking可能被影响。

这项审计成本很低，我会把它提升到**新实验之前的硬 gate**。

**alignment/provenance 4/6 是工程 artifact 问题，不是当前结果自动失效。** consolidated 的判断比较准确：verifier 里 Adam 检查可能已经 stale，但 target-dataset handling与旧文档确实不同。 更明确的两个事实是：

* YAML 写 alpha 256、batch 16、accum 8、3 epochs，但实际 driver override 为 alpha 512、batch 4、accum 4、4 epochs；
* equal-step 的 manifest 没记录 shell 注入的 `max_steps=420`，虽然 trainer/log明确跑到了 420。

这意味着“结果错了”吗？目前看不是。它意味着**仓库不能仅靠 YAML + manifest 完整重建实际 run**。

这个必须在 release 和下一大实验前修掉。

---

### contamination audit 我建议具体这么做

不是简单 grep 一遍 MMLU question string 就结束。

至少做四层：

1. normalized exact match：question、question+choices分别 canonicalize 后 hash；
2. long n-gram overlap，例如问题文本的高比例 8/13-gram containment；
3. fuzzy lexical retrieval / MinHash-Jaccard；
4. embedding/semantic nearest-neighbor retrieval，对最接近的若干百条人工或规则复核。

关键是最终不只报告：

[
\text{candidate pool contamination rate}
]

还要报告：

[
\text{contamination rate of DSMC / Random / baselines}.
]

也就是检查 selector 是否**富集污染样本**。

如果整个 pool 几乎没有 overlap，那么当前 MMLU 结果明显变得更可信。

如果有 overlap，但 DSMC/Random/其他 selector 的 exposure 基本一致，可以披露为 limitation。

如果有明显 method-differential contamination，那当前 MMLU downstream comparison 就要降级，external clean target family 需要成为主 evidence。

---

### Codex 的 “early BBH/MMLU/TyDiQA five-seed” 分层也合理

旧结果中不同 target 的 winner 会轮换，TyDiQA 上二阶方法曾经很有竞争力；但这些 run 的 target construction、representation variant、baseline set 和 variance decomposition 都和最终 DSMC protocol 不一致。

因此这些数据可以写成：

> preliminary cross-task evidence / motivation.

不能拿来宣称：

> DSMC 已经在三个 target families 上 external validated。

这一点 Codex 判断是对的。

而且因为我们**已经看过 TyDiQA 旧结果且二阶方法表现不错**，如果现在单纯选择 TyDiQA 做最终 external validation，reviewer甚至可以怀疑你挑了一个已知 favorable task。

所以我不建议现在直接说“下一站就是 TyDiQA”。

---

## 这会怎样改变下一轮实验？

方向没变：**仍然应该做第二个 query-aligned target/eval family。**

但我会修改 Codex 给出的 protocol。

首先，在任何新 SFT 前做三件零/低成本工作：

* 修 `verify_alignment` / 旧 completion reports；
* 生成真正 resolved 的 SFT/warmup/equal-step provenance；
* 做 MMLU candidate↔test contamination audit。

完成后再冻结 external experiment。

对于 target family，我建议不要根据现有 accuracy选择。先对 BBH / TyDiQA / MMLU-Pro做一个**不训练模型的 feasibility audit**，按预先规定的标准选择：

> clean query/test split、query reservoir足够大、candidate-test contamination低、现有 eval pipeline可靠。

最近的 controlled study也正好使用 BBH、TyDiQA 和 MMLU-Pro，因此无论最终是哪一个，都有直接的文献对照。([arXiv][2])

如果这三个条件差不多，我反而会倾向**不使用你们已经看过 favorable result 的 task作为唯一 confirmation target**，避免 cherry-picking 的观感。

---

### 下一轮最重要的设计变化：把 query draw 和 SFT seed 解耦

我不建议再做：

[
\text{draw}_0\to seed42,\quad
\text{draw}_1\to seed1,\quad\ldots
]

而是直接 crossed：

[
3\text{ query draws}\times2\text{ SFT seeds}.
]

例如每个 query draw 都在 seed 42 和 seed 1 下训练。

于是对每种方法有 6 个 paired observations，你可以分别看到：

* 同一个 query 改 seed 会怎样；
* 同一个 seed 改 query 会怎样。

如果资源充足，可以做 (4\times2)；但 (3\times2) 已经比当前的 5 个完全 confounded blocks科学上干净得多。

query draws 本身也不要再强制“十组全局完全无重叠”然后称 independent。若新 target reservoir 足够大，可以让每个 draw**独立随机采样**，draw 内无放回，draw 间允许自然产生小 overlap，并报告 overlap matrix。这样统计解释反而更标准。

---

### baseline set 我会稍微改 Codex 的方案

Codex 给的是：

DSMC / Second-RR / LESS-style TopK / Random / no-SFT。

这是合理的最小集，但考虑到 2026 的最新系统研究发现 **gradient representation + greedy RR 在低预算通常非常强**，([arXiv][2]) 我觉得如果算力允许，最好再保留 First-RR。

于是训练方法是：

**DSMC、Second-RR、First-RR、LESS-style TopK、Random-K**，外加一个共享 no-SFT reference。

这样 external experiment可以回答三个层次：

[
\text{first-order rep}
\leftrightarrow
\text{second-order rep}
]

[
\text{RR}
\leftrightarrow
\text{MMD coreset}
]

以及

[
\text{target-aware}
\leftrightarrow
\text{Random}.
]

如果做 (3) draws × (2) seeds：

[
5\times3\times2=30
]

个 SFT adapter，仍是可控规模。

如果必须压到四个方法，我会保留 **First-RR 而不是 LESS-style TopK**，因为这是当前低预算文献中更强、更现代的 controlled comparator；现有 MMLU 主实验已经有 LESS-style TopK。

---

### 预算我建议只跑一个

不要再做 1%/5% 两个预算轴。

为了与现有结果和近期文献的低预算 regime 对齐，我会冻结：

[
K\approx2707;(1%)
]

或直接 2500。

近期 controlled work用 (500,1000,2500,5000,10000) 并发现 RR/gradient methods 的优势主要在低预算。([arXiv][2]) 选择约 2500 不需要根据当前 DSMC 结果解释，是一个自然的 literature-aligned budget。

固定四 epochs，base reference保留。

同时记录：

* query loss；
* (D_2)；
* downstream metric；
* no-SFT performance。

这样你能直接测试“surrogate failure”是否跨 family 复现。

---

## MMLU 50/50 control 和 Adam/SGD ablation 要不要做？

Codex 把它们放“其次”，我赞同。

**50/50 MMLU query control 很有价值，但不是第二-family experiment的替代品。** 它直接回答：

> 当 observed query distribution真正与 balanced evaluation distribution一致时，Random 是否仍然赢？

这个问题非常贴合你们的故事。

但做完它，你仍然只有 MMLU 一个 family。因此如果只能选一组大实验，我还是优先 external family。

Adam-candidate / SGD-query asymmetry也是一个真实概念漏洞。官方 LESS 本来就是 candidate Adam-aware / validation SGD 的 influence formulation。([GitHub][3]) 所以作为“LESS-aligned feature protocol”它并不是 bug。

但对 DSMC 来说，论文又在讲“两个 distribution 的 directional second moment matching”，role-dependent preconditioning会让这个几何解释不那么纯。

这项我建议先做**selection-only cheap gate**：

> 把 query 也施加同样 Adam preconditioner，得到 Adam/Adam DSMC，先看和当前 Adam/SGD DSMC 的 Jaccard、D2 和 source composition。

如果 selection 几乎不变，就不值得烧 SFT。

如果 selection 巨变，再考虑少量 downstream ablation。

---

## 我认为 Codex 这版最重要的影响

不是让我们改论文主结论，也不是让我们重新做 MMLU。

而是把下一步从：

> “再找一个任务看看 DSMC 能不能赢”

升级成：

> **做一次真正解决当前 inferential weaknesses 的 external validation。**

具体执行顺序我会定成：

**先不启动 SFT。**

先让 Claude/Codex做 artifact cleanup + MMLU decontamination + external-task feasibility audit。完成后我们 review 一次。

然后冻结第二-family protocol：

[
3\text{ independent query draws}
\times
2\text{ SFT seeds}
\times
5\text{ methods}
]

一个 budget、一个 clean target family、一个 no-SFT reference。

这样下一轮实验同时解决：

* single target family；
* draw/seed coupling；
* Random block解释；
* stronger first-order comparator；
* contamination；
* provenance。

这比单纯“再跑 3–5 draws”价值大很多。

---

最后，Codex 关于论文 readiness 的方向性判断我同意，但我会换一种表达：

> **现在不是“结果不够漂亮”，而是“claim 的外推范围比 evidence 的覆盖范围大”。**

ICLR 的官方标准并不要求 SOTA，而要求论文的 claim 被严谨、可复现的证据支撑，并具有足够的新知识价值。([ICLR][1])

你们已经有了很不错的新知识：

> second-order target geometry比一阶 target geometry稳，但 target-geometry alignment本身并不足以决定 downstream utility。

现在缺的是证明：

> **这个现象不是 MMLU STEM/HUM + Tulu + Llama-2-7B 的偶然 case study。**

所以我建议我们下一步**先不跑实验**，先把 Codex 暴露出的 artifact/decontamination 问题全部清掉，并让 Claude拿出 2–3 个 external target family 的“split / contamination / query-reservoir / runtime feasibility表”；然后我们只选一次，不看任何新的 accuracy，再冻结外部验证实验。

[1]: https://iclr.cc/Conferences/2027/ReviewerGuidelines "ICLR 2027 Reviewer Guidelines"
[2]: https://arxiv.org/abs/2602.14696 "A Critical Look at Targeted Instruction Selection: Disentangling What Matters (and What Doesn't)"
[3]: https://github.com/princeton-nlp/LESS?utm_source=chatgpt.com "GitHub - princeton-nlp/LESS: [ICML 2024] LESS: Selecting Influential Data for Targeted Instruction Tuning · GitHub"
[4]: https://arxiv.org/abs/2602.18584 "GIST: Targeted Data Selection for Instruction Tuning via Coupled Optimization Geometry"
[5]: https://proceedings.mlr.press/v267/wang25bm.html "NICE Data Selection for Instruction Tuning in LLMs with Non-differentiable Evaluation Metric"
[6]: https://arxiv.org/abs/2412.15194?utm_source=chatgpt.com "MMLU-CF: A Contamination-free Multi-task Language Understanding Benchmark"

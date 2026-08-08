我看完 `ac7ecd0` 后，建议是：

> **15-adapter equal-step sensitivity 值得跑，而且现在可以跑。**
> 但在启动前/并行中，建议把 forensic 的两句因果性措辞收紧，并补两个零训练成本的 robustness check。不要再扩成 LR/LoRA/epoch 网格。

Claude 这次最重要的发现是真的：**“DSMC 在 1% 输给 Random，是因为它在梯度几何上追 skew 追得太狠”这个解释基本站不住。** DSMC 不仅比 Random 更接近自己的 query，在当前构造的 balanced (P^\star) 下也更接近，而且 1% 并没有出现预想的几何恶化。

但它随后又往前走了半步，走得有点太远。

### 第一，不能写成“我们的 objective 不预测 downstream accuracy”

目前能证明的是：

> **更小的 directional second-moment distance 并不足以保证更高的 downstream MMLU accuracy。**

或者更具体：

> DSMC dominates Random under the optimized (D_2) criterion while underperforming it downstream, demonstrating that (D_2) is not a sufficient surrogate for downstream performance in this setting.

这很强，而且已经足够有论文价值。

但“does not predict downstream accuracy”听起来像是说 (D_2) 完全没有 predictive value。你们还没有做跨 method、跨 draw 的 correlation / rank analysis来证明这一点。实际上最新的 controlled targeted-selection study发现 gradient-based distance 往往是各种 representation 中最能预测 query loss/downstream 的，但也明确观察到“更低 query loss / 更近 query”并不总转化成最高 downstream performance。([arXiv][1])

所以你们现在的结果更像是：

[
D_2 \text{ captures something real, but is incomplete.}
]

而不是：

[
D_2 \text{ is useless.}
]

这反而是更有意思的 framing。

### 第二，“source/format coverage is what pays off”目前是相关性，不是机制结论

Claude 找到的 source composition 差异很值得追：

* DSMC source entropy ≈ 0.965；
* Random ≈ 1.227；
* DSMC 1% 中约 63% 是 `flan_v2`，`oasst1` 只剩 68 条；
* Random 的 source mixture 明显更均衡。

但有一个非常关键的反例就在它自己的表里：

* GIST source entropy = **1.228**，几乎和 Random 的 1.227 一模一样；
* LESS = **1.205**，也很高；
* 可是 GIST 和 LESS downstream 都明显差于 Random。

因此：

> **高 source entropy 本身显然不是 Random 成功的充分解释。**

这点一定要指出。

可能真正重要的是更细的东西：

* **具体 source proportions**，而不是 entropy；
* source × task compatibility；
* format mix，例如 CoT / dialogue / short QA；
* answer style；
* candidate pool 原始 mixture 的 preservation；
* quality；
* source 与 pretrained model distribution 的 alignment。

Instruction-tuning diversity 的研究也强调，“diversity”不是一个简单 entropy 或 pairwise distance 就能概括的量；可靠的 diversity measure需要同时考虑样本间差异和样本空间里的信息分布。([arXiv][2])

所以当前 forensic 最好改成：

> Source provenance is a strong candidate explanatory axis distinguishing DSMC from Random, but the present analysis is correlational and source entropy alone is insufficient.

不要写：

> source coverage is what pays off.

---

还有一个小的 forensic robustness 问题：当前 (P^\star) 是十个 draw 的 union，所以对每个 (Q_d)，它自身就是 (P^\star) 的一部分。每个 draw 只占 10%，问题不算严重，但会让“DSMC 更接近自己的 (Q_d)”机械地贡献一点“更接近 (P^\star)”。

建议补一个完全免费的检查：

对每个 draw (d)，构造 leave-one-draw-out reference，然后重新按 STEM/HUM 50/50 reweight：

[
P^\star_{-d}
============

\frac12 P_{\mathrm{STEM},-d}
+
\frac12 P_{\mathrm{HUM},-d}.
]

重新算：

[
D_2(S_d,P^\star_{-d}).
]

如果仍然：

[
D_2(\mathrm{DSMC},P^\star_{-d})
<
D_2(\mathrm{Random},P^\star_{-d})
]

在绝大多数/全部 draws 成立，那么几何解释就真的可以关闭了。

第二个免费检查是把所有现有 method × draw 点拿出来，计算：

[
\mathrm{corr}(D_2\to P^\star,\text{downstream accuracy})
]

以及：

[
\mathrm{corr}(\text{source entropy},\text{downstream accuracy}).
]

用 Spearman 即可，最好分 1%/5%，并避免把 80 cells 当独立显著性样本，只作为 descriptive correlation。

我预期你们很可能会看到：

* (D_2) 有一定局部关联，但不能正确排序 DSMC vs Random；
* source entropy 也不能单独正确排序，因为 GIST/LESS 是反例。

那论文里最有价值的机制结论就变成：

> **Neither target-gradient geometry nor coarse source diversity alone fully explains downstream utility.**

这很有分量，也和现有 targeted-selection 文献的复杂结果相符：gradient distance通常有信息，但 lower distance 并不保证最佳 downstream，而 Random 仍能频繁匹配或超过复杂 selectors。([arXiv][1])

---

## Equal-step 这 15 个 adapter 该不该跑？

**该跑。**

这个 prereg 写得很好，而且目的非常明确。它只改：

[
84\text{ steps}\rightarrow420\text{ steps}
]

其他都冻结：

* 同一 1% subset；
* 同一 draw；
* 同一 seed；
* 同一 batch；
* 同一 LoRA；
* 同一 LR；
* 同一 scheduler；
* 同一 evaluation。

并预先固定了：

[
J_i=
[\mathrm{DSMC}-\mathrm{Random}]_{420,i}
---------------------------------------

[\mathrm{DSMC}-\mathrm{Random}]_{84,i}.
]

这是现在唯一还能干净回答的重大 confound。

而且我特别赞成 Claude 写进去的四种 interpretation rules：如果 Random 只是因为 20 epochs 过拟合掉下来，那**不能算 DSMC 胜利**。

这避免了“看到结果以后再解释”。

### 为什么 equal-step 依然有信息？

因为现在的主实验确实同时改变了：

[
K:13533\rightarrow2707
]

和

[
\text{steps}:420\rightarrow84.
]

所以观察到：

[
(\mathrm{DSMC}-\mathrm{Random})*{1%}
<
(\mathrm{DSMC}-\mathrm{Random})*{5%}
]

可能包含：

1. 数据预算效应；
2. 优化 horizon 效应；
3. 两者 interaction。

Equal-step 正好把第二项隔离出来。

不过也要明确，420-step 1% 相当于把同一 2707 条数据重复约 20 epochs，所以它不是“更公平的主实验”，只是 sensitivity。主结果仍然是固定 epochs 的原始 1%/5%。

---

## 我建议现在的执行顺序

先让 Claude做两个免费修补，然后直接启动 15 adapters：

1. 将 forensic wording 从
   “objective does not predict downstream accuracy”
   改成
   “objective distance is not sufficient to predict/rank downstream accuracy”。

2. 将
   “source/format coverage is what pays off”
   改成
   “source provenance is a candidate explanatory axis; current evidence is correlational”。

3. 补：

   * leave-one-draw-out reweighted (P^\star_{-d})；
   * (D_2)、source entropy 与 downstream 的 descriptive Spearman correlations。

4. 然后**直接启动预注册的 15-adapter equal-step run**。

这两个 forensic 补充不应该改变 equal-step 是否启动，也不要根据它们修改 experiment。

可以直接回复 Claude：

> The equal-step preregistration passes review; launch the 15-adapter run after two zero-cost forensic robustness checks, without changing the preregistered training design.
>
> First tighten two claims. Replace “our objective does not predict downstream accuracy” with “lower D2 is not sufficient to predict or rank downstream accuracy in this setting.” Replace “source/format coverage is what pays off” with “source provenance is a candidate explanatory axis; the evidence is correlational.” In particular, GIST has source entropy essentially equal to Random while performing much worse, so source entropy alone cannot explain Random’s advantage.
>
> Before launch, recompute D2 to a leave-one-draw-out, 50/50-reweighted P* for each draw, so the evaluated query is not itself part of its reference moment. Also report descriptive Spearman correlations across the existing method×draw points between downstream performance and (a) D2-to-P*, and (b) source entropy, separately at 1% and 5%. These are diagnostics only and must not alter the preregistered equal-step design.
>
> Then launch the frozen 15-adapter 1%-equal-step sensitivity exactly as preregistered: 10 DSMC + 5 shared Random-K, max_steps=420, no other hyperparameter changes. Do not add LR/LoRA/epoch sweeps. Report the original fixed-epoch results alongside it regardless of outcome.

---

## Equal-step 结果出来以后怎么办

这次应该是 MMLU/Tulu 优化线的**最后一道门**。

如果 DSMC 在 420 steps 下真正上升，而 Random 基本稳定：

> 可以说原始 1% negative budget interaction **部分由 under-training 导致**。

然后论文同时报告 fixed-epoch 和 equal-step，不替换主结果。

如果 Random 掉、DSMC不涨：

> 这是 Random 被重复训练过拟合，不是 DSMC 成功。

如果 Random 仍然领先：

> **关闭 optimization explanation。不要再扫 LR、epochs、LoRA。**

此时下一步不是“找一个 DSMC 能赢的参数”，而是 external validity：

* 第二个 candidate pool；
* 第二个 eval family；
* 或者 query distribution 与 evaluation distribution 真正一致的任务。

如果两者都掉：

> equal-step sensitivity 本身说明 20 epochs 过训，不解决 confound；原 fixed-epoch 结果仍然是主结果。

---

我现在对这个项目的看法反而比 1% 出来之前更积极一点，因为故事已经不再是简单的：

> “我们发明一个 metric，它比别人的 metric 高。”

而变成了一个更成熟的结论：

> **Directional second moments make targeted selection substantially more robust than existing targeted selectors, but matching the target gradient geometry—even very well—is not sufficient for downstream utility. Random data can outperform all target-aware methods, exposing a gap between geometric alignment and useful instruction-tuning coverage.**

这和最近 targeted-selection 工作指出的“distance 有信息、但不完全决定 downstream；Random 经常很强”形成了非常自然的对话。([arXiv][1])

所以现在的选择是：**修 forensic 措辞 + 两个免费 robustness checks，然后跑这最后 15 个 equal-step adapters；之后不再在 MMLU/Tulu 上调参。**

[1]: https://arxiv.org/abs/2602.14696 "A Critical Look at Targeted Instruction Selection: Disentangling What Matters (and What Doesn't)"
[2]: https://arxiv.org/abs/2502.17184 "[2502.17184] Measuring Data Diversity for Instruction Tuning: A Systematic Analysis and A Reliable Metric"

有必要认真查“训练步数/优化配置是不是把 1% 条件压坏了”，但我不建议现在开始泛化地扫 learning rate、LoRA rank、alpha、epochs 等一堆超参。最有价值的是做一个**非常窄、预先定义好的 equal-step sensitivity experiment**，然后就停止这条调参线。

先说最重要的判断：**目前 1% 输给 Random 不能简单解释成“84 steps 太少”。** 1% 下 DSMC balanced 是 0.4017，只比 no-SFT base 的 0.4003 高约 +0.14 pp；Random-K 却达到 0.4095，提升约 +0.92 pp。也就是说，同样只有 84 optimizer steps，Random 已经学出了明显收益，而 DSMC 没有。LESS/GIST/NICE 更是掉到了 base 以下。 这与近期 targeted-selection 的系统研究很一致：Random 经常很强，gradient similarity 即使能很好地拟合 query，也不保证 downstream 一定优于 Random，甚至某些 regime 下连未微调模型都无法超过。([arXiv][1])

但是，“训练 compute 不同”确实是一个真实 confound。你们当前 recipe 是 `lr=2e-5 + linear decay + warmup_ratio=0.03`。 于是 1% 的 4 epochs 只有 84 steps，大约 3 个 warmup steps；5% 是 420 steps，大约 13 个 warmup steps。线性 schedule 的整个优化 horizon 缩短了 5 倍，粗略看累计 learning-rate exposure 也少了约 5 倍。因此现在的 1% vs 5% 确实研究的是“**data + compute budget interaction**”，而不是纯粹只改变 unique-data 数量。LoRA/SFT 的最终结果对 learning rate 和训练配置可能相当敏感，近期也有系统研究专门展示这一点。([arXiv][2])

所以，步数可能是**部分原因**，但它更可能是一个“method × optimization”交互：Random 的广覆盖数据在很少的更新中就有收益，而 DSMC 这种高度定向的 subset 可能需要更多重复 exposure 才能把方向性信息转化为 downstream gain。另一方面，也完全可能相反——多训 DSMC 只是进一步过拟合 skewed query。最近甚至有工作发现，在某些 SFT 任务中反复训练少量高质量数据可以显著提高泛化，但这是强烈依赖任务的现象，并不能预设在你们这里也成立。([arXiv][3])

我反而觉得目前有一个比“步数不足”更值得怀疑的机制：**DSMC 在忠实地匹配一个有偏的 query distribution，而 Random 在保持候选池的广覆盖。** 你们故意给 selector 一个 80/20 skewed target，却用 balanced STEM/HUM 作为 primary evaluation。这正是 robustness setting，但在 (K=2707) 极小预算时，target-aware selection 可能把有限容量过多花在匹配观测到的 skew；Random 不看 query，反而天然保留 broader Tulu coverage。这也解释了为什么 DSMC 是所有 targeted methods 中“最不坏”的一个，却仍然赶不上 Random。Large-Scale Data Selection 的系统实验同样发现，很多复杂 selection method 在扩大场景后会落后于 Random，说明这种 coverage/diversity 效应并不罕见。([arXiv][4])

还有几个信号支持这个解释。第一，Random-K-LengthMatched 也比 DSMC 高约 0.63 pp，因此粗粒度 sequence-length/token composition 不是主要原因。第二，DSMC 与 Second-RR 在 5% 的差距是 +0.88 pp、10/10，但到了 1% 只剩 +0.17 pp、5/10；也就是说，MMD diversity 的额外价值本身随着预算缩小几乎消失。第三，其他 target-aware selector 普遍低于 base model，更像 selection-induced distribution shift / negative transfer，而不像一个简单的“所有 1% 模型都没训够”。

我建议接下来按下面四步做，不要直接开始大规模调参：

1. **先做零训练成本的机制分析。** 利用现有十个 target draws 的并集——它恰好是 320 STEM + 320 Humanities——构造一个 validation-only 的 balanced (P^\star) reference。然后对 DSMC 和 Random 在 1%/5% 分别计算“到当前 skewed query (Q_d) 的距离”和“到 balanced (P^\star) 的距离”。尤其看二阶矩
   [
   D_2(S,Q_d)=|M_S-M_{Q_d}|*F^2,\qquad
   D_2(S,P^\star)=|M_S-M*{P^\star}|_F^2.
   ]
   如果出现“DSMC 显著更贴近 (Q_d)，但 1% 下反而比 Random 离 (P^\star) 更远；到了 5% 这个差距缩小”，那几乎就是你们想要的机制解释。同时看 Tulu source entropy、unique sources、near-duplicate rate、gradient-space effective rank、pairwise similarity，以及 MMLU subject-level 的 base→DSMC/Random delta。这些都不需要重新训练。

2. **然后做一个唯一的 training sensitivity：1% equal-step。** 不扫超参，只比较 DSMC 与 Random-K。沿用完全相同的 1% subsets、5 个 draw indices、training seeds、LoRA、batch、`lr=2e-5`、linear scheduler、warmup 3%；唯一变化是把 1% 从 84 steps 拉到**恰好 420 optimizer steps**，也就是约 20 epochs。最好直接 `max_steps=420`，如果框架不方便再用 20 epochs 并 assert 最终正好 420 steps。由于 Random adapter 按方向共享，只需 10 个 DSMC + 5 个 Random = **15 个新 adapters**，而不是重新跑全部八种方法。

3. **预先固定这个 sensitivity 的判断量。** 对五个 draw-index blocks 比较
   [
   J_i=
   [\mathrm{DSMC}-\mathrm{Random}]_{1%,420\text{ steps},i}
   -------------------------------------------------------

   [\mathrm{DSMC}-\mathrm{Random}]_{1%,84\text{ steps},i}.
   ]
   同时始终报告相对 base-model 0.4003 的绝对变化。如果 420 steps 后 DSMC 上升、Random 变化不大，说明 under-optimization 确实是重要因素；如果 Random 因 20 epochs 过拟合而掉下来、DSMC没涨，也只能说“equal-compute 改变了排序”，不能说 DSMC 学得更好；如果 Random 仍稳定领先，那就基本可以关闭“步数不足”解释。

4. **equal-step 之后不要继续追 LR/LoRA 超参。** 如果 Random 仍赢，就停止在 MMLU/Tulu 上调 recipe，转向真正有论文价值的 external-validity：另一个 target/eval family、另一个 candidate pool，或者一个 query 与 evaluation distribution 更一致的 setting。当前文献也强调 targeted selection 的收益高度依赖模型、任务、candidate pool 和 budget，并没有单一方法普遍占优。([arXiv][5])

我不建议现在做 `lr ∈ {1e-5,2e-5,5e-5} × epochs ∈ {...} × LoRA rank × alpha` 这种网格。因为 1% 结果已经看过，再去寻找某个“DSMC 赢 Random”的组合，很容易变成 outcome-driven tuning；即使找到一个漂亮点，论文上也会很难解释。equal-step 不一样：它回答的是当前设计中一个**明确、事先可陈述的 confound**，所以是合理的 post-hoc sensitivity analysis，只要明确它不替代原先 preregistered fixed-epoch 结果。

还有一点很重要：不要觉得这次 1% 是“实验失败”。它其实让论文故事更有内容。当前结果已经说明：

> 在 skewed few-shot query 下，directional second moment 显著提高了 targeted selection 的稳健性；但 target-awareness 本身不保证比 Random 更有价值。随着预算从 5% 降到 1%，这种局限反而更明显。

这和 2026 年那篇 Critical Look 的核心警告高度呼应，只是你们进一步找到了一个**在 targeted selectors 内部明显更稳的二阶表示**。([arXiv][1])

如果让我决定现在给 Claude 的具体任务，我会让它**先做 balanced-(P^\star) geometry/source-diversity forensic analysis，同时设计并预注册 15-adapter 的 1% equal-step DSMC-vs-Random sensitivity；不要跑任何其他超参。** 这两项结果出来后，我们基本就能判断该继续研究 optimization interaction，还是彻底关闭这条线并转向第二个外部任务/候选池。

[1]: https://arxiv.org/abs/2602.14696 "A Critical Look at Targeted Instruction Selection: Disentangling What Matters (and What Doesn't)"
[2]: https://arxiv.org/abs/2602.04998?utm_source=chatgpt.com "Learning Rate Matters: Vanilla LoRA May Suffice for LLM Fine-tuning"
[3]: https://arxiv.org/abs/2602.11149?utm_source=chatgpt.com "Data Repetition Beats Data Scaling in Long-CoT Supervised Fine-Tuning"
[4]: https://arxiv.org/abs/2503.01807?utm_source=chatgpt.com "Large-Scale Data Selection for Instruction Tuning"
[5]: https://arxiv.org/abs/2602.14696?utm_source=chatgpt.com "A Critical Look at Targeted Instruction Selection: Disentangling What Matters (and What Doesn't)"

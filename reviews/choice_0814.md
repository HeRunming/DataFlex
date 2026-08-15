我会把这次 `b65efe8` 视为一个很好的终点：D2c 没有“帮故事”，反而主动砍掉了过强的解释。现在能支持的核心命题非常干净——**在 Llama-2-7B 上，target-aware selection 确实让所优化的二阶几何更贴近 query，并改善了 targeting pipeline 自己的 operational surrogate，但这些改善不足以转化成 task utility；这个 CE 改善又依赖于 targeting pipeline 的序列化方式，因此不能泛化成“cross-entropy 本身与 CoT 不一致”。** 这个 scope 我认为是对的。

如果目标明确是 **ICLR 2027 main conference，我现在会建议补第二个 base model，而且把它视为最高优先级的大实验。** 它不是 ICLR 的硬性要求：官方 reviewer guide明确要求的是 claim 是否被严谨证据支持、工作是否贡献新的重要知识，并明确说不要求 SOTA；同时 reviewer 可以要求范围有限、用于验证现有结论的补实验。([ICLR][1]) 但你们现在已经把 task、budget、query distribution、Random、base、format control、optimization confound、geometry 和 same-item diagnostics都补得很多了，**“完整结果只来自 Llama-2-7B”已经几乎成为唯一非常干净的 major-reviewer attack。**

这点在你们的直接邻近工作面前尤其明显。2026 那篇 *A Critical Look at Targeted Instruction Selection* 的主实验也是 Llama-2-7B + Tulu V2，但它专门在 appendix 加了 additional-model experiments，并明确写到 Llama-3.2-3B、Qwen3-4B-Base 和 OLMo-3-7B-Base；论文还强调 newer models 上 downstream distance trends 会更不一致。它甚至报告 BBH 上 LESS+RR 在 Llama-3.2-3B 和 Qwen3-4B-Base 上表现最好。([arXiv][2]) 所以如果我们只留一个模型，reviewer 很容易问：“你们这个 dramatic double reversal 是 targeted selection 的现象，还是 Llama-2-7B 的特殊 pathology？”

而且时间上还能做。ICLR 2027 full-paper deadline 是 **2026-09-16**；现在是 8 月 15 日，大约还有一个月。官方主文限制 9 页，appendix不限页但 reviewer不必读，因此如果加第二模型，关键结果应该进主文，不要只丢 appendix。([ICLR][3])

我不建议再复制一遍36-cell大实验。第二模型应该是一个**预注册的 model-sensitivity confirmation**，只回答中心问题，不再扩方法矩阵。我的首选设计是保持 Tulu candidate pool、已经冻结的 BBH 5209 held-out split、三个64-query draws、(K=2707)、4 epochs、effective batch和SFT seeds完全不变；唯一科学轴就是 base model。

方法我会保留四个：

* DSMC；
* First-RR；
* Second-RR；
* Random-K；
* 再加一个共享 no-SFT base reference。

这样有三个好处：DSMC测试中心现象；First-RR是近期文献里最自然的强 gradient-targeted comparator；Second-RR保留你们“first vs second representation / MMD vs RR”的结构；Random负责真正的 target-awareness baseline。没有必要再跑 LESS、GIST、NICE 或 SeqLabelMatched。

如果做完整 crossed：

[
3\ \text{draws}\times 2\ \text{SFT seeds}\times4\ \text{methods}
=24\ \text{adapters}.
]

这是我针对 ICLR 更推荐的版本。如果计算预算非常紧，最小可接受版可以删掉 Second-RR，变成18 adapters；我不太建议只用一个SFT seed，因为你们现在终于有了一个很干净的 crossed design，再退回单seed会主动留下一个容易问的问题。

还有一个非常重要的点：**不能把 Llama-2 选出来的 DSMC/RR subsets拿去训练第二模型，然后称作 second-model replication。** 你们中心 claim涉及 model-specific gradient geometry，因此新模型必须重新建立自己的 warm-up / candidate gradient datastore、三套 target gradients和 target-aware selections。Random-K则反而应该复用相同的 frozen candidate indices，这样 model axis最干净。

关于具体模型，我会这样选。

**科学上首选 Qwen3-4B-Base。** 它和 Llama-2 是明显不同的模型家族，4B规模也不大，而且邻近的 Critical Look 正好在 BBH 上用了 Qwen3-4B-Base，因此文献对照非常自然。官方 model card说明它是4B pretrained base model、32k context、Apache-2.0；LLaMA-Factory v0.9.3 的 release也列出了 Qwen3支持。([Hugging Face][4])

但它有一个工程风险：Qwen3官方明确要求 Transformers ≥4.51，否则会出现 `KeyError: 'qwen3'`。([Hugging Face][4]) 如果你们当前被冻结的 DataFlex 环境低于这个版本，我**不建议为了第二模型直接升级现有环境**。应该做独立的新环境；如果迁移 candidate-gradient/LoRA pipeline需要大量修改，就放弃Qwen，转向 **Llama-3.2-3B**。Meta官方提供 base 3B模型，而 Critical Look 同样在BBH additional-model experiment用了它，所以它虽然同属Llama family，但仍然能有效堵住“只有一个checkpoint”的攻击，而且工程风险通常更低。([Hugging Face][5])

因此我建议不要今天直接拍脑袋选模型，而是让 Claude 做一个**不看任何 accuracy 的24小时内 feasibility gate**：Qwen3-4B-Base优先、Llama-3.2-3B fallback。选择标准只能是环境兼容、模型/LoRA支持、warm-up能否正常跑、candidate gradient可否提取并投影8192维、target gradient canary、token/truncation完整性以及实测throughput；**不能根据BBH base accuracy或任何SFT downstream结果选模型。**

第二模型的分析端也应现在一次性冻结。primary仍然是 held-out BBH micro exact-match；同时预注册四个 diagnostic：

[
D_2(S,Q_d),
]

operational wrapped query CE，

same-query 64-item CoT EM，

以及 bare-context CE作为secondary serialization diagnostic。

这样无论第二模型结果是什么，都有明确解释。若再次出现“DSMC最低D2，但Random downstream最好”，你们的中心结论就从一个模型的case study跃升成相当有力的 cross-model evidence。若第二模型上target matching开始真正帮助 downstream，那也不是坏结果——论文就应明确说 **surrogate failure is model-dependent**，这实际上比假装普遍失败更符合邻近文献发现的模型依赖性。([arXiv][2])

如果第二模型上是混合结果，例如：

[
D_2(\mathrm{DSMC}) < D_2(\mathrm{Random}),
]

但 downstream DSMC≈Random，那么也仍然支持比较保守的核心：

> **Better target matching is not sufficient to guarantee downstream improvement.**

注意这个 claim本来就不是要求每个模型上 target matching都必须失败，只需要说明“更近 ⇒ 更好”不是一个可靠的一般原则。

ROSE也让第二模型更值得补，而不是更不值得。ROSE已经把“instruction CE 和 task metric可能不单调”作为出发点，并宣称跨多个benchmark和不同model architectures有generalization。([ACL Anthology][6]) 所以你们若想把贡献写成一个更广泛的 critical empirical finding，仅用一个 model会明显弱于相关工作的证据范围。你们真正新的地方是 set-level gradient geometry 被成功优化但 utility仍可能失败；**加第二模型能证明这个新观察不是只绑在Llama-2-7B上。**

我会把接下来一个月的执行顺序定成这样：

1. **现在就开始 paper rebuild，不等第二模型。** Abstract/Introduction直接围绕 “Matching the target is not enough”，把 D2c 的负面check也写进去，展示你们主动收窄claim。
2. **并行做 Qwen3-4B-Base / Llama-3.2-3B 的 engineering-only feasibility audit**，不看任何 accuracy。
3. 按事先标准选一个模型，冻结 second-model prereg。
4. 重新生成该模型自己的 candidate/target gradients和 DSMC/RR selections。
5. 跑 24-adapter confirmation；如果时间/预算确实不允许，再降到18 adapters，而不是继续删draw或只做一个target draw。
6. 第二模型完成后**停止所有大实验**。不做第三模型、不做第三任务、不再调方法。
7. 最后做一次 ICLR reviewer audit，把“single model”这一条删除或改成“两模型仍有限但覆盖不同generation”的 limitation。

所以，如果你问我一个明确的 yes/no：

> **Yes。既然目标是 ICLR main，我认为补第二模型现在是值得的，而且是目前唯一明显值得继续投入的大实验。**

现有论文即使不补，也已经有一个可投稿的 critical empirical story；但第二模型很可能是把它从“很有意思但可能被一句 single-model case study 打成 borderline”推向“主要claim经过 task × budget × query regime × model 多轴验证”的最有效一笔投入。

我会优先让 Claude做 **Qwen3-4B-Base vs Llama-3.2-3B 的纯工程 feasibility + 精确成本表**，然后我们只选一个，马上冻结实验，不再讨论新的科学轴。

[1]: https://iclr.cc/Conferences/2027/ReviewerGuidelines?utm_source=chatgpt.com "ICLR 2027 Reviewer Guidelines"
[2]: https://arxiv.org/abs/2602.14696 "A Critical Look at Targeted Instruction Selection: Disentangling What Matters (and What Doesn't)"
[3]: https://iclr.cc/Conferences/2027/AuthorGuidelines?utm_source=chatgpt.com "ICLR 2027 Author Guidelines"
[4]: https://huggingface.co/Qwen/Qwen3-4B-Base?utm_source=chatgpt.com "Qwen/Qwen3-4B-Base · Hugging Face"
[5]: https://huggingface.co/meta-llama/Llama-3.2-3B?utm_source=chatgpt.com "meta-llama/Llama-3.2-3B · Hugging Face"
[6]: https://aclanthology.org/2025.findings-emnlp.710/?utm_source=chatgpt.com "ROSE: A Reward-Oriented Data Selection Framework for LLM Task-Specific Instruction Tuning - ACL Anthology"

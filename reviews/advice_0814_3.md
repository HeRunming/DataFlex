我重新核对了 `c50e353`。这次结果确实把证据链补得很完整，但 Claude 有一句话还是说得过强：

> “competing explanation is ruled out”

我不会这么写。更准确的是：**“query→held-out 的泛化失败不能单独解释结果，因为 task metric 在定义 targeting signal 的同一批 query items 上也没有改善。”** 这已经很强，但还不是把所有“overfitting”概念都排除了。

我现在认为论文的中心已经基本确定了，而且不应该再把 DSMC 当作“赢家方法”来组织。

你们现在最强的证据是三层同时成立：

[
\text{better target geometry}
]

[
\Downarrow
]

[
\text{better final-answer query CE}
]

但却

[
\Downarrow!!!!!!!!\diagup
]

[
\text{better task utility}.
]

BBH 上 DSMC 在 3/3 draws 都有最低的 (D_2(S,Q_d))，而 Random 的 (D_2) 明显更差；即使去掉 secondary matched-Random，甚至只在四个 target-aware methods 内部比较，(D_2) 与 downstream accuracy 的 rank association 仍然保持“更好 geometry 对应更差 accuracy”的方向。

与此同时，四个 target-aware methods 都显著降低了定义 targeting signal 的那 64 条 query 上的 final-answer CE，而两个 Random controls 反而提高 CE；但 downstream 排序几乎反过来。

现在 D2b 又进一步把“examples 改变”这个因素拿掉了：在**同样的 64 个 query items**上，DSMC 的 final-answer CE 平均下降约 1.16 nats，但 CoT exact-match 反而比 base 下降约 5.2 pp；其他 targeted methods也是相同方向。

所以现在可以相当有底气地写：

> **Target-aware selection can improve the differentiable target surrogate on the very examples used to define the target, while simultaneously degrading the task metric on those same examples.**

这是我认为你们整篇论文现在最重要的一句话。

这和已有工作的位置也非常清楚。2026 年的 *Critical Look* 发现 gradient-based representation 的 distance 是各种 representation 中最可靠地预测 query loss/downstream 的，但它也明确观察到更低 query loss并不必然带来最高 downstream performance，而且 Random 在一些任务上很强；他们还指出这种趋势在不同模型上并非完全一致。([arXiv][1]) ROSE 则直接把 cross-entropy 与实际 task metric 的非单调关系作为现有 targeted-selection 的核心问题，并因此转向 reward-oriented selection。([ACL Anthology][2])

你们的增量不是“发现 CE 和 reward 可能不一致”——这个 claim 已经有人做了。真正独特的是：

> **你们把 mismatch 放进了一个可测量的 target-set geometry 链条里：目标集合在被优化的二阶梯度几何中真的更近了，训练后的 target CE 也真的更低了，但 utility 仍然更差。**

也就是：

[
\boxed{
\text{geometric alignment}
\not\Rightarrow
\text{surrogate improvement}
\not\Rightarrow
\text{task improvement}
}
]

严格说第一条在你们这里其实是：

[
\text{geometric alignment}
\quad+\quad
\text{surrogate improvement}
\quad\not\Rightarrow\quad
\text{task improvement}.
]

这是比单纯“Random beats selectors”更有研究价值的结果。

不过在正式封稿前，我还建议做**一个最后的、非常便宜的 evaluation-only robustness check**。之后就停止 forensic 扩张。

原因是 D2b 虽然已经固定了“same items”，但还没有完全固定“same executable input”。

你们的 final-answer CE仍然沿用 target-gradient extraction pipeline：

* LlamaFactory `llama2` template；
* 因此模型真正看到 `<s>[INST] ... [/INST] ...`。

而 CoT EM 走的是 pinned lm-eval BBH generation：

* pre-wrapper prompt内容一致；
* 但并没有那个 Llama2 chat wrapper。

这个 caveat之前已经确认过。所以 D2b现在证明的是：

> **the operational targeting surrogate improves while official BBH task performance worsens on the same examples.**

这本身完全成立。

但如果论文再往前说：

> “final-answer CE itself is misaligned with CoT generation”

reviewer可能会指出还有 chat-wrapper/input-serialization 的差别。

最好现在用现有模型、现有64条query再算一个额外量：

[
L_Q^{\text{bare}}
]

也就是：

* 使用和 lm-eval CoT generation **完全相同的裸 prompt prefix**；
* 不加 `[INST]` wrapper；
* 不生成 CoT；
* 只 teacher-force final bare answer `(C)` / `14` / `Yes`；
* 算 answer-token CE。

这样你会有：

[
L_Q^{\text{target-pipeline}}
]

和

[
L_Q^{\text{eval-context-aligned}}
]

两套 CE。

如果两套都出现：

[
\Delta CE<0,
\qquad
\Delta CoT\ EM<0,
]

那 surrogate-metric mismatch 的解释就非常难被 input-template confound攻击。

如果 bare-prompt CE 不再改善，也没关系。那论文应明确说：

> **the exact surrogate actually used by the targeting pipeline is misaligned with downstream utility; we do not attribute the entire mismatch specifically to cross-entropy independent of prompt serialization.**

这项只需 forward pass，不训练，不改任何结果。我认为它值得作为 forensic 的真正最后一个 check。

除此之外，我不会再做更多机制实验。特别不建议：

* 再做 response-format matched control；
* 再做 source-matched Random；
* 改 CoT target gradients；
* 用 rationale gradients重新选择；
* 尝试 reward-aware DSMC；
* 调 DSMC kernel；
* 改 budget / LR / epoch。

这些都已经进入“看到结果后发明新方法”的区域，而且会把现在非常干净的 critical empirical story重新弄脏。

D3 也应该继续保持现在这种降调。三个 draw 中 task exposure 与 DSMC−Random 的 correlation都是轻微负值，说明**没有观察到“query里出现越多的 task越受到保护”这一简单 task-specialization模式**。 但不能因此推出“所以一定是在 format level overfit”。SeqLabelMatched只说明 format/provenance是一条 plausible axis，不是 causal identification。

所以我会把 D3 放 appendix，主文最多一句：

> We find no evidence that greater per-task query exposure protects that task, arguing against a simple task-frequency specialization account.

### 第二模型现在还要不要做？

我的判断和上一轮相比稍微更倾向于：**不是现在必须做，但如果目标是强主会，它仍然是唯一还值得考虑的大实验。**

原因很简单。

现在 task axis 已经足够好了：

* MMLU；
* BBH；
* skewed queries；
* query-aligned queries；
* 1% / 5%；
* base reference；
* Random；
* matched Random；
* geometry；
* same-query CE；
* same-query task metric。

继续第三个 task的边际价值非常小。

但目前所有完整证据仍然只有一个 base model：

[
\text{Llama-2-7B}.
]

而最新 *Critical Look* 特意跨模型检查，并明确报告 newer/over-trained models 上 downstream distance trends会更不一致；他们还在 Llama 3.2 3B、Qwen3 4B、Olmo 3等模型上观察不同 selector patterns。([arXiv][1]) 大规模 data-selection 工作同样强调 selector表现随规模和 setting变化，且很多复杂方法会被 Random超过。([arXiv][3])

所以 reviewer现在最自然的问题已经变成：

> “这个 double reversal 是 targeted selection 的一般问题，还是 Llama-2-7B 特有的现象？”

如果你们准备冲一个要求比较高的 venue，这个问题会真实存在。

但我**不建议现在立即跑第二模型**。先做 paper-first decision：

1. 做上面那个 bare-prompt CE robustness，成本很低；
2. 把整个 consolidated document真正改成 paper structure；
3. 重写 abstract/introduction/contributions；
4. 做一次非常严格的 reviewer simulation；
5. 再判断“single model”是不是 reviewer audit 中唯一剩余的 major weakness。

如果 audit 的结论仍然是“除了 single-model，没有明显 major hole”，那第二模型就值得投入。

而且如果真做第二模型，不需要重复这整套36-cell工程。目标不是证明DSM C排名，而是确认中心现象：

[
\text{target match improves}
\quad\text{but}\quad
\text{utility does not}.
]

我会建议届时设计一个**最小 confirmation study**，例如：

* 一个现代 base model；
* BBH；
* 同一个 frozen split和3 query draws；
* DSMC；
* First-RR 或 Second-RR；
* Random；
* no-SFT；
* 一个 training seed就够做 model-sensitivity，两个更好。

重点同时报告：

[
D_2,\quad
\Delta L_Q,\quad
\Delta CoT\ EM.
]

不要再跑 LESS/GIST/NICE/SeqLabelMatched整套。

但这项实验要先评估：如果换模型意味着必须重新生成整个 270k candidate-gradient datastore，那么成本可能显著大于“12 adapters”本身。不要为了多一个model column随便复用 Llama2 selections然后称为 model replication；那只能叫 **cross-model transfer of Llama2-selected subsets**，科学问题不同。

### 论文 framing 我现在会彻底改成这个方向

标题方向可以从 DSMC 名字主导，转成现象主导，比如：

> **Matching the Target Is Not Enough: When Gradient-Aligned Instruction Selection Hurts Downstream Performance**

DSMC变成你们用来建立严格 test 的方法，而不是整篇论文唯一卖点。

贡献逻辑建议是：

第一，提出 directional second-moment matching，并在 MMLU controlled attribution里发现它相对 first-order targeted selection有明显优势。

第二，系统发现这个 advantage不等于 target-aware selection advantage：Random在 MMLU不弱，在BBH显著更好。

第三，也是核心：

> 在 MMLU 和 query-aligned BBH 中，更好的 target-gradient geometry不保证更好的 downstream utility；BBH上甚至出现明确的反向 ranking。

第四：

> 在同一BBH query items上，target-aware methods降低 final-answer CE却降低 CoT task metric，说明 failure不是单纯“query→test shift”就能解释。

第五：

> base/no-SFT与Seq×Label control显示 targeted SFT可能发生负迁移，format composition解释一部分变化但不能解释全部。

这就不是一篇“我们方法没赢”的论文了，而是一篇：

> **用一个更强的 matching objective揭示 targeted instruction selection 的基础 surrogate assumption何时失效。**

这和 *Critical Look*、ROSE构成的是互补关系，而不是被它们覆盖：前者显示gradient distance通常更有预测性但并非万能，后者从reward角度指出CE surrogate不可靠；你们展示的是**成功优化set-level gradient geometry也不足以保证utility，并提供跨family的实证反例链条**。([arXiv][1])

所以现在我的明确建议是：

**不要启动新SFT。做一个最后的 bare-prompt final-answer CE robustness check，然后停止实验设计，进入论文重构 + reviewer audit。第二模型作为 audit 后唯一可能追加的大实验，而不是默认下一步。**

你可以直接让 Claude 做：

> Run one final evaluation-only robustness diagnostic, then stop forensic expansion. On the exact same 64 BBH query items, compute final-answer teacher-forced CE using the **bare pinned lm-eval prompt context without the LlamaFactory `[INST]` wrapper**, while keeping the gold final answer and token-level loss definition fixed. Compare this with the existing operational target-pipeline CE and same-item CoT exact-match. No training or selection changes.
>
> If targeted methods still reduce bare-context final-answer CE while query CoT EM falls, state that the surrogate/task-metric dissociation survives removal of the chat-wrapper confound. If not, scope the claim to the exact operational targeting surrogate rather than cross-entropy generically.
>
> After that, freeze all experimental/forensic work. Rebuild the paper around “Matching the target is not enough”: controlled MMLU evidence, external BBH reversal, D2 geometry/outcome dissociation, same-query surrogate/task-metric dissociation, Random/no-SFT baselines, and the Seq×Label secondary control. Keep D3 exploratory and do not infer a causal format mechanism.
>
> Then perform a paper-level reviewer vulnerability audit whose main decision is whether a second base model is necessary. Do not launch it automatically. If single-model generality is the only remaining major weakness, scope a minimal second-model BBH confirmation around DSMC / one RR / Random / no-SFT and the three quantities D2, query loss and task metric before any compute.

我现在认为你们已经到了**“停止寻找结果，开始写出结论”**的阶段。第二模型可以让论文更强，但它不再是为了救故事；故事本身已经成立。

[1]: https://arxiv.org/abs/2602.14696 "A Critical Look at Targeted Instruction Selection: Disentangling What Matters (and What Doesn't)"
[2]: https://aclanthology.org/2025.findings-emnlp.710/ "ROSE: A Reward-Oriented Data Selection Framework for LLM Task-Specific Instruction Tuning - ACL Anthology"
[3]: https://arxiv.org/abs/2503.01807?utm_source=chatgpt.com "Large-Scale Data Selection for Instruction Tuning"

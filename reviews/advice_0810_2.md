我仔细看了 `c356569 → 5cd4514 → fe154ea`，也重新对照了当前 instruction-selection 文献和 LlamaFactory 的训练语义。我的判断是：

**Phase A 可以正式关闭，Phase B 工程上 PASS；但我赞成现在加 `Random-K-LengthMatched`，而且这是最后一个“干净地加”的时点。** 不过在加之前还有一个我认为很重要的小坑：**不要直接复用现有 `select_randk_lenmatch.py`，需要先把 length 定义修成真正执行时的 LlamaFactory 长度。**

先说 Phase A。这个结果已经足够好了：四个 target-aware selector 在两份 target-gradient cache 之间最多只换 5/2707 条，预先冻结的门槛是 27 条；RR 确实稍微敏感，但完全没有 cascade。 因此冻结 run1 `10c92b25…` 作为 canonical cache 的理由也很干净——“approved canary 的第一次成功 extraction，在任何 downstream BBH accuracy 出现前冻结”，不是因为它表现更好。

Phase B 的五个 subset 也是真正 distinct 且工程正确：K、unique/range、RR query order、Random seed 都通过；DSMC/Random 的 Jaccard 只有约 0.0046，接近随机交集水平。 这部分不用再改。

但 length finding 确实值得控制。当前 artifact 记录的是 DSMC 1,459,218 个 `post_template_tokens`，Random 950,783；即使把 2048 cutoff 后被删掉的 token 扣掉，实际保留的 sequence-token exposure 仍大约是：

[
\frac{1,459,218-28,734}{950,783-2,326}
\approx 1.51\times.
]

所以这并不是少数超长样本尾巴造成的假象。

而且这个问题在 targeted instruction selection 里是有实质意义的。LESS 本身就明确强调它要处理 variable-length instruction data；近期 controlled targeted-selection work 又特别强调比较 selector 时应避免把 representation、selection procedure 和 budget 等因素纠缠在一起。([arXiv][1]) ([arXiv][2]) 既然 BBH 这轮本来就是为了提供更干净的 external validation，那么明知 DSMC 与 Random 的 sequence exposure 相差 50% 却不做 sensitivity control，会留下一个很容易被 reviewer 抓住的洞。

不过 Claude 的一句话需要马上纠正：

> “DSMC adapters would see ~1.5× more supervised tokens”

**现在还不能这么说。**

commit 里真正统计的字段是 `post_template_tokens`，不是 `#labels != IGNORE_INDEX`。 LlamaFactory 把 user prompt 和 model response视为不同角色，SFT pipeline 还专门支持“effective tokens”的统计；因此完整 sequence token 数、真正参与 loss 的 response/label token 数、以及 GPU compute exposure不是一个概念。([GitHub][3]) ([GitHub][4])

所以当前最准确的说法应该是：

> DSMC has ~1.5× greater **post-template sequence-token exposure** than Random at fixed K; the difference in loss-bearing supervised tokens has not yet been measured.

而且我发现一个比措辞更具体的问题。你们现有的 `scripts/select_randk_lenmatch.py` docstring声称：

> “Llama-2 tokenizer length of the full sharegpt example (user+assistant, template applied)”

但实现实际上只是：

```python
"\n".join(m["content"] for m in messages)
```

然后直接 tokenizer；它**没有真正经过 LlamaFactory 的 `llama2` chat template**。

这意味着：

**如果现在直接拿这个旧脚本生成 BBH LengthMatched Random，控制变量和 Phase-B diagnostic 可能不是同一个量。**

这不会让我推翻过去 MMLU 的 qualitative length-matched result——那个 arm 本来就是 coarse sensitivity，而且结果与 Random qualitatively一致——但论文里以后最好不要再声称旧 MMLU arm是“exact post-LlamaFactory-template matched”，除非重新 audit。它更接近“coarse tokenizer/content-length matched”。

所以我建议现在先做一个完全不需要 SFT 的修补，然后正式把第六个 arm加进去。

具体我会这样冻结：

1. 用**实际 LlamaFactory v0.9.3 preprocessing + llama2 template + cutoff 2048**，对 270,679 candidates生成一次 immutable length cache。对每条记录保存至少三个量：

   * `sequence_tokens_before_cutoff`
   * `sequence_tokens_after_cutoff = len(input_ids)`
   * `supervised_label_tokens = count(labels != IGNORE_INDEX)`

   然后把 draw0 五个 subset 的三个指标都重新报告。不要再用“supervised tokens”代指 sequence length。

2. 把 BBH 的 `Random-K-LengthMatched`定义成一个**secondary control，而不是第六个 primary selector**。主分析仍是原来的 DSMC / Second-RR / First-RR / LESS-style / Random-K；LengthMatched只回答：

   > DSMC-vs-Random 的差异是否可能由 selector诱导的 sequence-length distribution解释？

3. 每个 draw基于该 draw的 DSMC subset做 fixed-K length matching：
   [
   K=2707
   ]
   两个 SFT seeds重用同一个 subset。seed现在就冻结，例如：
   [
   7000+d.
   ]
   为了与旧 MMLU control保持概念一致，我会继续用固定 length buckets，但这次 bucket assignment必须来自**真正 LlamaFactory post-template、post-cutoff sequence length**，不能来自简单拼接文本。

4. 现在就把 interpretation rules写进 prereg，不看 BBH accuracy：

   * DSMC > Random 且 DSMC > LengthMatched-Random：length distribution不足以解释 DSMC优势；
   * DSMC > Random，但 ≈ LengthMatched-Random：不能把优势归因于 target awareness，length exposure是一个充分的候选解释；
   * DSMC ≤ Random 和 LengthMatched-Random：target-awareness negative result更强；
   * LengthMatched-Random < Random：说明“更长”本身不是普遍好处，因此不能简单用 length解释 targeted selector表现。

   无论结果怎样都报告 plain Random 和 LengthMatched Random。

我尤其不建议做另一种看似更“公平”的操作，例如给 Random增加 epochs直到总 token数相同，或者给 DSMC减少 steps。这会同时改变 optimizer horizon，又重新引入我们在 1% equal-step arm里已经见过的问题。固定 K、固定4 epochs、只改变 Random subset composition，是更干净的 control。

还有一个理由让我倾向于现在加：你们在 MMLU 主线里已经有 `Random-K-LengthMatched`。如果 BBH external validation恰好是唯一没有这个 control 的地方，而 BBH 又出现了 **1.53×** 的明显长度差异，那么 reviewer 很自然会问：

> 为什么在已有 length control 的 MMLU 上做了，而在长度差异更大的 external-validation setting 中反而不做？

现在还没有任何 BBH adapter accuracy，所以加它没有 outcome-driven selection 的问题；等两-adapter canary结果出来后再决定，反而会变脏。

额外成本约：

[
6\times111.4\text{ min eval}+6\times15\text{ min train}
\approx12.64\text{ GPU-h},
]

把剩余总量从约 63.2 提到约 75.8 GPU-h。这个代价相对于它堵掉的 reviewer vulnerability，我觉得值得。评测又能在 8 张 H20 上并行，所以主要是 GPU-hour 增长，而不是不可接受的 wall-clock增长。

另外 source entropy这件事继续只做 diagnostic。Phase B 显示 Random 1.201 → DSMC 0.833，同时 DSMC 70% `flan_v2`。 LengthMatched Random的 source mix很可能也会因为“长样本”条件发生变化，这恰好说明 length和provenance是相关的；所以它只能控制 length axis，不能声称因此“因果隔离 source”。不要再加 source-matched Random——那会开始变成无穷控制实验。

我的下一步建议因此是：

**先不要跑 2-adapter SFT canary。用半小时左右把 exact LlamaFactory sequence/label-token audit + 正确的 BBH LengthMatched-Random protocol冻结，并把 run plan从 30 adapters扩成 36。之后 selection-only验证三个 LengthMatched subsets；然后再跑原来的 DSMC+Random 两-adapter engineering canary。工程通过后直接全量36，不再等 accuracy作任何实验决策。**

你可以直接给 Claude：

> Add the BBH Random-K-LengthMatched arm now, but as a **pre-registered secondary sensitivity control**, not as a sixth primary selector. No BBH adapter accuracy has been observed, and the measured DSMC-vs-Random sequence-length difference is large enough (~1.5×) that leaving it uncontrolled would create an avoidable reviewer vulnerability.
>
> Before constructing it, fix the length definition. The current Phase-B report measures `post_template_tokens`, but do not call these “supervised tokens” until loss-bearing labels are counted. Using the exact pinned LlamaFactory v0.9.3 preprocessing, `llama2` template, tokenizer, and SFT `cutoff_len=2048`, build and hash a candidate-level cache containing: full pre-cutoff sequence length, post-cutoff `input_ids` length, and `#labels != IGNORE_INDEX`. Recompute these three quantities for the five draw0 subsets and report DSMC/Random ratios.
>
> Do **not** reuse `scripts/select_randk_lenmatch.py` unchanged: its docstring says the Llama-2 template is applied, but the implementation currently joins raw message contents and tokenizes them directly. For BBH, length bucket assignment must use the exact executed LlamaFactory post-template, post-cutoff sequence length.
>
> Freeze one LengthMatched-Random subset per draw at K=2707, shared across the two SFT seeds; use a new pre-registered selection seed such as `7000+draw_id`. Keep the original plain Random-K unchanged. Record subset hashes, exact bucket counts, total post-cutoff sequence tokens, supervised-label tokens, source composition, and overlap with DSMC/Random.
>
> Pre-register the interpretation before training: DSMC beating both Random controls argues against length alone; DSMC beating plain Random but not LengthMatched-Random means the target-awareness advantage cannot be separated from length exposure; losing to both strengthens the negative target-awareness result. LengthMatched-Random is secondary only and must not alter the original primary comparisons.
>
> Update the frozen run plan from 30 to 36 adapters and the remaining budget accordingly (~75.8 GPU-h total from here). Then selection-validate all three new LengthMatched subsets. After that, run the already planned two-adapter DSMC-vs-Random engineering canary; do not use its interim accuracy for any decision. If engineering is green, launch all remaining adapters including LengthMatched-Random without further protocol changes.

所以我的明确 call 是：

**加。** 但先把“length”测对、把现有 length-match script 的假 post-template 假设修掉，再加；不要把 1.53×叫成“supervised tokens”。完成这一步后，BBH protocol就真的应该冻结，不再加任何新 control。

[1]: https://arxiv.org/abs/2402.04333 "https://arxiv.org/abs/2402.04333"
[2]: https://arxiv.org/abs/2602.14696 "https://arxiv.org/abs/2602.14696"
[3]: https://github.com/hiyouga/LlamaFactory/blob/main/data/README.md "https://github.com/hiyouga/LlamaFactory/blob/main/data/README.md"
[4]: https://github.com/hiyouga/llamafactory/releases "https://github.com/hiyouga/llamafactory/releases"

这轮我建议先 **HOLD 2-adapter engineering canary**。不是因为 DSMC/BBH 设计又出了问题，而是我检查 `8ac5d0a / 1bc4bce` 后发现：Claude 新建的所谓 “exact LlamaFactory length cache” **实际上还不是 exact**。这会直接影响刚加入的 LengthMatched control，所以应该趁现在还没有任何 BBH adapter accuracy 时一次修干净。

核心问题在 `scripts/build_candidate_length_cache.py`。它现在把一条样本中的所有 `user` 内容先拼成一个字符串、所有 `assistant` 内容再拼成一个字符串，然后手工构造一个单轮：

```python
u = " ".join(... user ...)
a = " ".join(... assistant ...)
src = tok(f"[INST] {u} [/INST]", ...)
tgt = tok(a, ...) + [eos]
```

再调用 `infer_seqlen`。

但 pinned 的 LlamaFactory v0.9.3 并不是这样处理一般 ShareGPT conversation 的。它的 template 会**按消息顺序逐条编码不同 role**；`encode_multiturn` 会保留每一轮 prompt/response pair，而 `Llama2Template` 还分别处理 user、assistant、observation、function，并融合 system 信息。([GitHub][1]) 官方数据格式也明确支持多轮 ShareGPT、system、observation 和 function，并规定 GPT/function response 是训练目标。([GitHub][2])

所以当前 cache 只有在一个尚未证明的条件下才是 exact：

> 270,679 条 candidate 全部恰好只有一个 user + 一个 assistant，没有 system/tool/observation，也不存在多轮消息，并且手工 tokenizer 拼接与真实 template 输出完全一致。

我们现在不知道这个条件是否成立。尤其不应该因为脚本跑出了很有意思的 47,438 vs 268,850 就默认它成立。

因此我会把目前这几个数字——

[
\text{DSMC label positions}=47{,}438
]

[
\text{Random}=268{,}850
]

以及 draw0 当前 `randk_lenmatch` hash——都先标成 **provisional diagnostic**，不能再向下训练。

不过，这个发现本身很可能是真的，而且如果复核后仍成立，会非常有价值。

还有两个措辞也需要一起修。

第一，`sequence_tokens_after_cutoff` 不能写成 `== GPU cost`。它是很好的 **sequence-token exposure** 指标，但实际 batch 会 padding；不同样本放进同一 batch 后，tensor 形状由 batch 中最长序列决定，所以简单求未 padding 的 token 总数不等于真实 FLOPs/GPU cost。Hugging Face 的 collator 文档明确说明动态 batching 会 pad 到 batch 内最长序列。([Hugging Face][3])

因此写：

> post-cutoff sequence-token exposure

不要写：

> GPU cost.

第二，`#labels != -100` 确实是非常有意义的 **loss-bearing label positions**，但也不要叫“supervised signal amount”。Transformers v4.50 的 causal-LM loss会忽略 `-100`，然后对剩余 token 做 cross-entropy，并按 batch token count进行 mean/normalized reduction；因此 5.67× label positions 并不等于“5.67× gradient signal”或“5.67× effective learning”。([GitHub][4])

所以如果复核后数字成立，正确写法是：

> DSMC exposes the model to 1.51× as many post-cutoff sequence positions but only 0.18× as many loss-bearing label positions per epoch as Random-K.

这已经足够强，不需要再往“signal”上推。

更重要的是，这个 response-length / supervision-density 差异不能当成无关 bookkeeping。已有 SFT data-selection 研究发现，仅仅偏向长 response 就可能显著改变 fine-tuning 结果；另一些 reasoning-data selection 工作也直接把 reasoning-trace length 当作有效 selection signal。([arXiv][5]) 大规模 instruction-selection 工作甚至发现 token-length filtering 本身就是一个很强的 selector。([arXiv][6])

因此 Claude 说“这可能是一个真实机制轴”方向是对的，但：

> “这是 faithful targeting，因为 BBH answer 很短”

目前仍然只能称为 **consistent with format matching**，不能称为机制已经证明。

---

### 我建议现在做的最后一轮 artifact-only 修正

让 Claude不用 model、不做 gradient、不做 SFT，直接完成下面这一步。

先对全部 270,679 candidates 做 message-structure inventory，按 source 分别报告：

* `n_messages` 分布；
* exactly `user→assistant` 两条的比例；
* 多轮比例；
* system message 数；
* observation/function/tool 数；
* 非标准 role pattern 数。

这一步马上就能告诉我们当前手工 cache 到底危险不危险。

然后不要再手工重建 `[INST] ... [/INST]`。直接调用**与正式 SFT 相同的 pinned LlamaFactory dataset converter + `SupervisedDatasetProcessor` + `llama2` template + cutoff=2048 + 实际 train_on_prompt/mask_history/packing settings**，从真正 preprocessing 输出中逐条读取：

[
L_i = \operatorname{len}(\texttt{input_ids}_i)
]

和

[
R_i =
#{j:\texttt{labels}_{ij}\neq -100}.
]

这才是 authoritative cache。

最好再做一个 parity table：

* 全 pool current-manual vs actual processor mismatch rate；
* 按 source mismatch rate；
* sequence-length max/mean diff；
* label-position max/mean diff；
* draw0 五 subsets分别重新汇总。

如果全 pool真的都是单轮，可能会发现旧 cache几乎完全一致。那很好——我们就获得了证明，而不是假设。

如果有大量 multi-turn，新的数值可能明显变化，也正说明这次 HOLD 是必要的。

---

### LengthMatched control 我建议再加强一步

如果 exact preprocessing 以后，DSMC vs Random 的巨大 response/label-length差异仍然存在，我现在反而**不建议继续训练当前 sequence-only LengthMatched Random**。

我建议把它**替换**成：

> `Random-K-SeqLabelMatched`

仍然只是一个 secondary sensitivity control。

不是再加第七个 arm，所以总量仍保持：

[
36\text{ adapters}
]

而不是变成 42。

原因是现在真正需要 reviewer-resistant 地回答的问题已经不是：

> “DSMC 是否只是挑了长 input？”

而是：

> “DSMC 对 Random 的任何差异，是否只是来自它选择了完全不同的 instruction/response format——长 context + 极短 response？”

当前 draw0 sequence-only control虽然把 sequence tokens做到 DSMC 的 0.984×，但它仍然有 DSMC **7.14×** 的 label positions。

因此它只能排除：

> coarse sequence length

却完全不能排除：

> response length / supervision density / classification-vs-generation format.

而这恰好是文献已经表明会影响 SFT 的轴。([arXiv][5])

更强、也更值得花同样 6 个 adapter 的 control 是 coarse 2D matching：

[
(\text{sequence length},\text{label length}).
]

我会保持现在 sequence bins：

[
[0,256),[256,512),[512,1024),[1024,1536),[1536,2049]
]

再固定一组**事先定义的 log-scale label bins**，例如：

[
[0,4),[4,16),[16,64),[64,256),[256,2049].
]

然后每个 draw在 25 个二维 cells中，按 DSMC 的 cell counts随机抽候选，seed仍然：

[
7000+d.
]

先只做 feasibility：

> 每个 DSMC cell在 candidate pool里是否都有足够候选？

如果全部够，就冻结，不再修改 bins。

如果某 cell不够，停止带回来，不要看 downstream accuracy后再调 bins。

这样得到的是一个非常清晰的 secondary control：

> 一个 Random subset，在 fixed K 下粗粒度匹配 DSMC 的 input-context length 和 response/loss-bearing length，但完全不使用 query-gradient geometry。

这比当前 sequence-only control信息量大得多。

它还会产生非常清楚的解释：

* DSMC > plain Random 且 > SeqLabelMatched Random
  → 不能由粗粒度 input/output length format 单独解释；
* DSMC > Random，但 ≈ SeqLabelMatched
  → DSMC 的收益很可能主要来自 format matching，而不能归因于二阶 gradient geometry；
* DSMC ≤ 两种 Random
  → target-awareness negative result更强；
* SeqLabelMatched本身 > plain Random
  → “target-like format composition”本身就是重要机制；
* SeqLabelMatched < Random
  → 过度匹配短-response BBH format反而可能导致 specialization / negative transfer。

注意，这个 control **不能叫 source-controlled**。它很可能也会改变 flan/cot/oasst1比例，因为 format和source天然相关；只需要把 source composition作为 diagnostic一起报告即可，不再继续做 source-matched Random。

---

### 为什么我不建议现在直接启动两 adapter

因为现在是最后一个完全干净的时间点：

* BBH base reference看过了，但它不能用于 method selection；
* **没有任何 BBH SFT adapter accuracy**；
* secondary Random control还没有训练；
* 这个问题是由工程审计发现的，不是因为 DSMC结果不好看才产生。

所以现在把 secondary control从 sequence-only 改成更有解释力的 Seq+Label matched，是很干净的。

一旦 draw0 DSMC / Random adapter accuracy出来，再调整这个 control，即使工程理由正确，论文叙事上也更容易被质疑。

---

### 修完后就真的启动，不再加东西

我建议明确立一个 stop rule：

> 这次 authoritative preprocessing audit + secondary 2D control冻结之后，不再增加任何新的 matched Random、source control、LR control、epoch control 或方法变体。

然后：

1. authoritative LlamaFactory preprocessing cache；
2. 重算 draw0 五方法 sequence/label diagnostics；
3. 生成 draw0 `Random-K-SeqLabelMatched`，验证 K/hash/25-cell counts；
4. 更新 run plan，仍为 36 adapters；
5. 运行 draw0 DSMC seed42 + Random seed42 两-adapter engineering canary；
6. **不要从它们的 accuracy 做任何 decision**；
7. engineering green → 提取 draw1/2 target gradients → 生成所有 primary subsets + 两个 SeqLabelMatched subsets；
8. 直接跑剩余34 adapters；
9. 36/36结束后一次性解封 accuracy并分析。

---

我会直接这样回复 Claude：

> HOLD the two-adapter engineering canary for one final artifact-only correction. The newly built “exact LlamaFactory” cache is not yet demonstrably exact.
>
> `build_candidate_length_cache.py` currently concatenates all user messages into one string and all assistant messages into another, then manually constructs one `[INST] user [/INST] assistant` pair. Pinned LlamaFactory v0.9.3 instead encodes messages in role/order and supports multi-turn/system/observation/function conversations. Therefore the current cache is exact only if every one of the 270,679 candidate rows is provably a single user→assistant turn with no additional roles/system state.
>
> First emit a full candidate message-structure audit, overall and by source: number of messages/turns, role patterns, system/tool/observation/function presence, and fraction that is exactly one user + one assistant.
>
> Then rebuild the authoritative cache by invoking the **actual pinned LlamaFactory SFT preprocessing path**, using the same dataset converter, `SupervisedDatasetProcessor`, llama2 template, tokenizer, cutoff=2048, and frozen data settings used by downstream SFT. Read `len(input_ids)` and `count(labels != IGNORE_INDEX)` directly from the preprocessed examples; do not manually reconstruct the template. Compare the existing manual cache against this authoritative output overall, by source, and on all draw0 selected subsets.
>
> Tighten terminology: call `len(input_ids)` “post-cutoff sequence-token exposure,” not GPU cost; call non-ignored labels “loss-bearing label positions,” not supervised-signal amount.
>
> If the large DSMC-vs-Random response-length gap survives, **replace** the current sequence-only `Random-K-LengthMatched` secondary arm with one `Random-K-SeqLabelMatched` arm rather than adding another arm. Keep the total at 36 adapters. Use the existing fixed sequence bins and fixed log-scale label bins `[0,4), [4,16), [16,64), [64,256), [256,2049]`; match DSMC’s 2D histogram per draw at fixed K=2707 with seed `7000+draw_id`. First run an availability/feasibility gate on the 25 cells. If any cell cannot be matched, STOP and report rather than silently changing bins.
>
> Plain Random-K and all five primary comparisons remain untouched. The joint matched arm is secondary only; source composition is diagnostic, not matched. Preserve the existing sequence-only draw0 subset as an untrained diagnostic artifact, but remove it from the training run plan if the joint control is feasible.
>
> After the authoritative cache and the joint secondary control are frozen, that is the final protocol: no further controls or method changes. Then run the two-adapter engineering canary without using interim accuracy for decisions; if engineering passes, launch the remaining adapters through 36/36 before inspecting comparative BBH accuracy.

所以此刻我的判断不是“又发现一个要重做的大问题”，而是：

> **BBH 主实验已经定了；现在最后需要修的是 secondary control 的测量基础。**
>
> `47k vs 269k` 这个发现可能非常重要，但先用真正执行过的 LlamaFactory preprocessing把它坐实。坐实后，我更倾向用同样6个adapter预算做一个**input-length + response-length联合匹配 Random**，而不是只匹配 input length。完成后就停止审计扩张，正式开跑。

[1]: https://github.com/hiyouga/LLaMA-Factory/blob/v0.9.3/src/llamafactory/data/template.py "LlamaFactory/src/llamafactory/data/template.py at v0.9.3 · hiyouga/LlamaFactory · GitHub"
[2]: https://github.com/hiyouga/LlamaFactory/blob/main/data/README.md?utm_source=chatgpt.com "LlamaFactory/data/README.md at main · hiyouga/LlamaFactory · GitHub"
[3]: https://huggingface.co/docs/transformers/main/pad_truncation?utm_source=chatgpt.com "Padding and truncation · Hugging Face"
[4]: https://github.com/huggingface/transformers/blob/v4.50.0/src/transformers/loss/loss_utils.py "transformers/src/transformers/loss/loss_utils.py at v4.50.0 · huggingface/transformers · GitHub"
[5]: https://arxiv.org/abs/2402.06094?utm_source=chatgpt.com "Rethinking Data Selection for Supervised Fine-Tuning"
[6]: https://arxiv.org/abs/2410.09335?utm_source=chatgpt.com "Rethinking Data Selection at Scale: Random Selection is Almost All You Need"

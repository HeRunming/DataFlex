我重新看了 `93284f8`，也对照了 LLaMA-Factory、官方 BBH prompt/data 和 Llama 2 的实现。我的判断和 Claude 现在的判断有一个关键差别：

> **这版已经把真正的方法协议找回来了；两个“blocker”里，截断是真 P0，但 answer-flip demo 不应该继续作为 P0。**
>
> 所以现在不应该直接 launch，但也不用重新设计 BBH。最合理的是：**修 target-gradient cutoff，保留官方 few-shot demo 和 frozen draw，然后重新过 gate。**

### 1. P0-1 已经修好了

`bbh_execution_contract.json` 现在明确固定了：

* candidate = Adam-aware gradient；
* target/query = SGD gradient；
* projection dim = 8192；
* projection seed = 123；
* candidate cache 原样复用；
* DSMC = `select_moment_mmd.py --alpha 0`；
* LESS-style = `select_relevance_topk.py --order first`；
* First/Second-RR = 同一 `select_round_robin.py`；
* Random = frozen `torch.randperm` seed。

而且 candidate tensor-content hash 已经和过去主实验 artifact 对上。

这个部分现在我认为是 **PASS**。fresh Claude 上一轮最危险的上下文漂移已经被 machine-readable contract 消除了。

---

### 2. 2048 截断是真 blocker，而且现在证据非常充分

Claude 这次没有夸大。

audit 实测 192 条 query 中 7 条被 materially truncated，而且全部来自 `geometric_shapes`；target没有掉，但 query 本身和最后的 CoT cue 会掉。eval 侧最大只有 2596 context tokens，在它的 3072 input budget 下都完整。

这也符合 LLaMA-Factory 的实际长度分配逻辑：`infer_seqlen` 在 target 很短时优先保留 target，把剩余 budget给 source；source随后取 prefix，因此超长 prompt 的尾部会被裁掉。([GitHub][1])

而我们的 BBH prompt恰好是：

[
[\text{3 demos}] + [\text{actual query at the end}]
]

所以这是最坏的一种截断方式。

**我不建议接受并披露 7/192。** 这不是轻微 prompt mismatch，而是 7 个 target gradient事实上没有在对应 target question上计算。

也不建议只给 `geometric_shapes` 减少 shots。那会造成 task-dependent query protocol，同时不再对应固定的 3-shot BBH evaluation prompt。

### 我建议的处理：target-gradient extraction 单独改为 `cutoff_len=3072`

原因很干净。

Llama 2 原生 context limit是 4096。([GitHub][2]) 当前 lm-eval BBH 又给 generation预留 1024，因此有效 input ceiling正好约为：

[
4096-1024=3072.
]

现有 192 个 target context最大只有约 2596，所以 3072可以完整容纳所有实际 eval context。

这意味着我们可以把协议定义成：

[
\boxed{\text{BBH target-gradient cutoff}=3072}
]

但同时保持：

[
\boxed{\text{selected-data SFT cutoff}=2048}
]

candidate cache也完全不动。

这不是 accuracy-driven tuning。我们甚至还没有看到 BBH accuracy；这是根据 input-integrity audit 在 compute 前修复一个确定的数据处理 bug。因此论文上完全可以解释：

> BBH target-gradient extraction uses a 3072-token cutoff to preserve the complete pinned 3-shot evaluation context; the downstream SFT recipe remains frozen at 2048.

它确实不再和 MMLU 的 query-gradient cutoff完全相同，但我认为这比“为了 recipe parity而故意算 7 个错误 query gradients”科学上强得多。

---

### 3. 我不同意把 answer-flip demo 继续当作 launch blocker

这一点是这次 review 里我认为最重要的修正。

Claude找到的那个：

> demo: “at least one person” → Yes
> query: “more than one person” → No

是真的。

但它**不是我们这次 split 造成的污染**。

我直接检查了 BBH 官方 repo。官方 `causal_judgement` CoT prompt 本来就把那个 “at least one person” / answer Yes 的例子作为 few-shot demonstration。([GitHub][3])

而官方 BBH `causal_judgement.json` 本来就包含几乎相同的 “more than one person” / answer No 的 benchmark item。([GitHub][4])

换句话说：

> **这个 minimal pair 是官方 BBH CoT evaluation protocol 自带的结构。**

更重要的是，在我们自己的实验里，`causal_judgement::128` 是一个 **query-draw item**，而 query reservoir 与 5,209 条最终 held-out evaluation严格 disjoint。

所以它不是：

> final test answer leaked into prompt.

甚至不应该叫 “anti-leakage”。

它更准确是：

> **an official few-shot/query near-neighbor with an opposite label.**

这可能影响那个 query gradient，这是事实。但那恰恰是**固定 BBH prompt distribution 的一部分**。所有 target-aware methods都面对同一个 query context，而且正式 evaluation也使用同一官方 demonstrations。

因此三个选项里：

* swap demonstration：不建议，会人为偏离官方 BBH CoT protocol；
* exclude `causal_judgement::128`：不建议，相当于看完 prompt similarity以后修改 frozen random draw；
* **保留 + 披露：最干净。**

尤其不能因为它是一个困难 minimal pair就删除它。那反而容易受到 reviewer 的 post-hoc dataset editing 质疑。

我会把 `bbh_fewshot_leakage_audit` 的最终状态改成类似：

> `PASS_WITH_DISCLOSURE` / `WARN`

而不是 `REVIEW` blocking launch。

并把措辞从：

> “answer-flip leakage / anti-leakage”

改成：

> “official BBH few-shot/query near-duplicate minimal pairs; zero exact identity with evaluation/query items; disclosed as a prompt-structure characteristic.”

官方 BBH 本身就是 23 个 challenging tasks，并单独发布了这些 CoT prompts，所以保持其 fixed prompt structure 对这轮 external validation很重要。([GitHub][5])

---

### 4. 我还发现两个很小但应该顺手补掉的 provenance 点

第一个，execution contract里 warm-up checkpoint同时记录了 adapter hash 和 optimizer hash，但从 commit diff看，显式 `*_matches` flag只有 adapter那个。

建议变成两个都 fail-loud：

```text
adapter_sha256_matches = true
optimizer_sha256_matches = true
```

launch manifest两个都必须要求 true。

第二个，这次 truncation问题已经证明 LLaMA-Factory preprocessing code 本身属于 load-bearing provenance。现在你们 pin了 lm-eval Python runtime和 tokenizer，这是很好的一步；我还会把**实际安装版本**的：

```text
llamafactory/data/processor/processor_utils.py
llamafactory/data/processor/supervised.py
llamafactory/data/template.py
```

也做 SHA pin。

不是为了继续找问题，而是因为最终论文里如果有人问“为什么你确定 source tail被截”，你们可以直接对应到真正执行过的代码，而不是 2026 年之后的 upstream `main`。

---

## 所以现在具体怎么走

我建议现在给 Claude 一个很明确的 decision，不需要再让它给你三个选项。

1. **接受 official BBH demo/query minimal pairs**，不换 demo、不删 query。把 leakage gate降为 non-blocking disclosure。

2. **把 BBH target-gradient extraction cutoff从 2048 改成 3072，仅限 target gradients。** Candidate cache不重算，SFT cutoff仍然2048，其他 method/config完全不变。

3. 重新跑纯 artifact/token gate，必须得到：

```text
192 / 192 records audited
0 materially truncated
0 query missing
0 CoT cue missing
0 supervised target truncated
eval side still 0 truncated
```

4. 更新 execution contract/prereg，明确：

   * target-grad cutoff = 3072；
   * SFT cutoff = 2048；
   * 这是 pre-compute validity correction；
   * BBH与MMLU的 query-gradient cutoff不再完全相同。

5. 加 optimizer hash assertion + LLaMA-Factory runtime SHA pin，然后生成 clean-head launch receipt。

**到这里我就会放行 selection canary。**

selection canary依然按原计划：

> no-SFT held-out BBH eval + draw0 的 64 target gradients + 5 selectors @ K=2707；不做 SFT。

draw0 本身就包含 `geometric_shapes` 长 prompt，所以这会是很好的实际 canary：不仅看 tensor `(64,8192)` / finite / nonzero，还可以确认修复后的 gradient extraction真正在完整 query上运行。

如果这一步干净，我甚至建议再做一个很便宜的 **2-adapter end-to-end engineering canary：draw0 DSMC seed42 + draw0 Random seed42**，主要确认 custom BBH eval、manifest、resume、aggregate，而不是看 accuracy。然后直接剩余28个。

---

你可以直接这样回复 Claude：

> Resolve the two decisions as follows.
>
> **Truncation:** change the BBH **target-gradient extraction only** to `cutoff_len=3072`. Keep the frozen candidate cache unchanged and keep downstream selected-data SFT at `cutoff_len=2048`. The choice of 3072 is protocol-derived, not accuracy-tuned: Llama-2 has a 4096-token context and the pinned BBH lm-eval configuration reserves 1024 generation tokens, while the measured maximum BBH evaluation context is 2596. Re-run the execution-level audit and require all 192/192 records to retain the complete query, CoT cue, and supervised target with zero material truncation. Record explicitly that BBH target-gradient cutoff differs from the old MMLU arm as a pre-compute validity correction.
>
> **Few-shot near-duplicate:** do **not** swap the demonstration and do **not** remove `causal_judgement::128`. The “at least one → Yes” demonstration and “more than one → No” item are both part of the official BBH CoT prompt/data construction; moreover the item is a query-reservoir record, not part of the disjoint held-out evaluation split. Reclassify this from a launch-blocking “leakage/anti-leakage” claim to a documented official few-shot/query near-neighbor minimal pair (`PASS_WITH_DISCLOSURE` or equivalent). Keep zero exact-identity as the true leakage hard gate.
>
> Also make the warm-up optimizer hash an explicit fail-loud assertion, pin the actually installed LlamaFactory preprocessing/template source files by SHA, update the execution contract/launch manifest, and emit the clean-head receipt. Still run no model compute until those artifact gates are green.
>
> Bring back the updated artifacts. If the token audit is 192/192 clean and all provenance gates pass, proceed with the pre-registered no-SFT + draw0 5-selector selection canary. No method changes and no accuracy-based decisions.

所以我这轮的最终状态是：

> **93284f8：review质量明显提升。P0-1 已关闭；P0-2 截断确认成立并需要修；所谓 blocker 2 应降级为 disclosure，而不是改 benchmark。**
>
> **当前 HOLD；把 target cutoff principled地改成3072并复查后，我预计就可以进入真正的 selection canary。**

[1]: https://github.com/hiyouga/LlamaFactory/blob/main/src/llamafactory/data/processor/processor_utils.py "LlamaFactory/src/llamafactory/data/processor/processor_utils.py at main · hiyouga/LlamaFactory · GitHub"
[2]: https://github.com/meta-llama/llama?utm_source=chatgpt.com "GitHub - meta-llama/llama: Inference code for Llama models · GitHub"
[3]: https://github.com/suzgunmirac/BIG-Bench-Hard/blob/main/cot-prompts/causal_judgement.txt "BIG-Bench-Hard/cot-prompts/causal_judgement.txt at main · suzgunmirac/BIG-Bench-Hard · GitHub"
[4]: https://github.com/suzgunmirac/BIG-Bench-Hard/blob/main/bbh/causal_judgement.json "BIG-Bench-Hard/bbh/causal_judgement.json at main · suzgunmirac/BIG-Bench-Hard · GitHub"
[5]: https://github.com/suzgunmirac/BIG-Bench-Hard?utm_source=chatgpt.com "GitHub - suzgunmirac/BIG-Bench-Hard: Challenging BIG-Bench Tasks and Whether Chain-of-Thought Can Solve Them · GitHub"

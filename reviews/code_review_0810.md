我这次按“新的 Claude 可能缺失上下文”这个风险重新从底层检查了 `bee7d29`，包括 prereg、run plan、launch manifest、prompt renderer/parity audit、custom lm-eval suite、selection scripts，以及我们之前真正跑过的 MMLU feature protocol。

结论是：

> **`bee7d29` 比上一版扎实很多，但我现在不批准 `GO_FOR_SELECTION_CANARY=true`。**
>
> 不是 BBH 设计要推翻，而是 fresh Claude 确实已经出现了上下文漂移；我找到了 **两个需要在任何模型 compute 前解决的 P0**，以及几项最好一起补掉的 P1。
>
> **BBH family、20/80 split、3 draws × 2 seeds、5 methods、K=2707 都不用改。**

### 先说已经确认做对的部分

23/27 accounting 现在是正确的。BBH 原始工作称其为 23 个 conceptual tasks；lm-eval v0.4.5 的 `bbh_cot_fewshot` operational group 则明确列出 27 个 subtasks，并以 `weight_by_size=true` 聚合。([GitHub][1])

现有 split 也无需重做。它原本就是按 27 个本地 subtasks 做 stratified 20/80，所以正好对应 primary micro metric；三个 draw 的 2/3/2 overlap 也符合独立从 1302-example reservoir 取 64 的预期。

selection randomness 的结构基本合理：Random `5000+d`，两个 RR 共用 `6000+d`；同一 method×draw subset 被两个 SFT seeds 重用。run plan 确实是 15 subsets → 30 adapters。

custom held-out lm-eval suite 也设计得不错：它保留 v0.4.5 的 3-shot CoT、greedy `generate_until`、1024 generation tokens、regex exact-match 等行为，只替换 held-out dataset source。

另外，fresh Claude 自己发现 Gate B 的 vacuous-pass 和 `--verify` 没有完整 pin 27 config/data files，这次修复是实质性的，值得保留。

---

## P0-1：fresh Claude 对“我们真正冻结的方法”理解错了

它现在总结 DSMC 是：

> “MMD ... over Adam-preconditioned ... gradients ... (`src/.../mmd_selector.py`)”

这个描述对即将运行的 BBH **不够准确，而且可能直接导致跑错实验**。

我们已经完成的主实验真正冻结的是：

[
\textbf{candidate gradients = Adam-aware}
]

[
\textbf{target/query gradients = SGD}
]

[
d=8192,\qquad \text{projection seed}=123
]

并且 candidate cache 已经固定为那个 270,679×8192 artifact。

实验里的 DSMC 也不是“随便调用 online `MMDSelector`”。真正 frozen endpoint 是：

```text
scripts/select_moment_mmd.py --alpha 0
```

对两个 unit-normalized projected caches 做

[
k(u,v)=\langle u,v\rangle^2
]

的 exact marginal greedy。

这个区别非常重要，因为通用 `MMDSelector` 默认：

```text
gradient_type = adam
target_gradient_type = same
```

如果 fresh Claude照自己总结去实现 BBH，它很可能把 target 也算成 Adam，**那就不是之前论文主线里的 DSMC 了。**

同样，BBH 的 `LESS-style TopK` 应明确调用：

```text
select_relevance_topk.py --order first
```

而不是突然切到 official trajectory LESS；First/Second-RR 应调用同一个 `select_round_robin.py`，只切 first/second representation。

但我检查目前的 BBH prereg 和 launch manifest，**没有看到这个 frozen feature/implementation contract 被写进去**。launch manifest甚至基本没有记录 Adam-candidate / SGD-target。

这是第一项真正的 launch blocker。

---

## P0-2：prompt parity 只检查了字符串，却漏掉了最危险的 execution-level truncation

这是我认为这一轮最重要的新发现。

现在 BBH query 使用完整 3-shot CoT context，然后整个 context作为一个 LlamaFactory `user` message，再接 bare final answer。

但是 frozen SFT/query-gradient recipe 是：

[
\texttt{cutoff_len}=2048.
]

而 query prompt 已经出现：

* 7,368 chars；
* 7,580 chars；
* 5,200+ chars；
* 5,100+ chars；

这样的长样本。

字符数不能直接证明超过 2048 tokens，但这是一个非常强的 warning signal。

LlamaFactory 的 SFT preprocessing 在超过 `cutoff_len` 时确实会截断样本，而且官方 repo 的 issue也专门提醒这种 truncation 对 reasoning/math 样本可能破坏语义。([GitHub][2])

问题更严重的地方是：BBH prompt结构是

[
[\text{3 CoT demos}];[\text{actual query at the end}],
]

所以如果本机那个版本的 preprocessing主要截 source tail，极端情况下可能出现：

> gradient实际上是在“few-shot demonstrations + 被截断的 query”上算，甚至实际 query 本身被切掉了一部分。

此时当前 Gate B 所谓：

> query prompt == lm-eval prompt byte-for-byte

仍然完全可以 PASS，因为它比较的是**截断之前的字符串**。

而 evaluation 端也有另一套长度逻辑。Llama 2 原生支持 4096 tokens。([GitHub][3]) lm-eval v0.4.5 的 `generate_until` 会给 generation reserve `max_gen_toks`；BBH 固定了 1024 tokens，因此对 4096-context model，evaluation input 上限大约是 3072，然后由 lm-eval在编码阶段 left-truncate context。

所以现在可能实际比较的是：

[
\text{query-gradient context} \le 2048
]

vs

[
\text{eval context} \le 3072,
]

并且两边 truncation 方向/算法还可能不同。

这比 `[INST]` wrapper 本身严重得多。

**在这个 audit 跑完以前，我不会允许 target-gradient extraction。**

---

## fresh Claude 还已经记错了过去的结果

这也证明你担心 empty-window context loss 是对的。

它说：

> “DSMC beats every targeted selector at both 5% (10/10) and 1% (9/10)”

这是错误的。

真正的 1% 结果是：

* First-RR：9/10；
* LESS-style：9/10；
* NICE：7/10；
* GIST：8/10；
* **Second-RR：只有 5/10，block-level 3/5**。

而且 Second-RR 的平均差只有 +0.17 pp，descriptive interval 跨零。

正确结论一直是：

> DSMC 在 1% 对 first-order/adapted targeted baselines仍有明显优势，但相对 Second-RR 的额外 MMD gain基本消失。

它还说：

> “forensic mechanism analysis was refuted”

也容易误导。

被 refute 的是我们曾考虑的：

> DSMC 因追逐 skewed query 而牺牲 balanced geometry

这个**特定机制假设**。

forensic analysis本身反而提供了重要结果：DSMC 成功优化 D2，但更好的 D2 不足以保证 downstream utility。

我建议不要让 fresh Claude继续依靠自己的 memory summary。

---

## `[INST]` wrapper 我怎么判断？

它正确发现：

* query gradient：LlamaFactory `llama2` → `<s>[INST] ... [/INST]`
* lm-eval：裸 completion context

所以 token sequence不一样。

LlamaFactory 本身一般建议 training/inference使用相同 template，而 lm-eval也支持显式 chat-template evaluation。([GitHub][4])

但是**我现在不建议为了“修 parity”而改变它**。

原因是这轮 BBH 的目标是 external validation of the frozen selection/SFT pipeline：

* candidate gradients已经来自 llama2-template instruction data；
* SFT也是 llama2 template；
* MMLU主实验也是同样结构；
* 如果现在给 BBH evaluation突然加 chat template，又会偏离 pinned stock BBH CoT protocol。

所以当前 wrapper difference 可以保留为 limitation。

但前提是前面说的 **2048 truncation audit PASS**。如果实际 query被截残，那就不只是 wrapper caveat了。

更准确的论文语言应该是：

> query and evaluation are drawn from the same task distribution and share the same pre-wrapper CoT prompt context; their executed token sequences are not identical.

不要继续称“query-aligned”时让人误解为 token-level alignment。

---

## 还有几项 P1，最好在 canary 前一次补完

首先，**hard-coded CoT few-shot exemplars 还没有做 leakage check**。

每个 subtask有3个固定 CoT examples；这 81 个 demonstrations 应该与：

* full 6511；
* 1302 reservoir；
* 5209 heldout；
* 192 draw memberships

做 normalized exact + fuzzy match。

现在 Gate C只验证 reservoir/heldout彼此 disjoint，没有验证 demonstrations是不是来自这些 evaluation items。BBH 官方确实提供单独的 CoT prompt artifacts，所以这种 audit很便宜。([GitHub][1])

其次，LSH 的“95.1% / 99.99% recall”建议再收紧措辞。代码确实变成了 64 perms / 32 bands / 2 rows。 理想 MinHash假设下：

[
1-(1-s^2)^{32}
]

在 (s=.3) 和 (s=.5) 分别约为 95.1% 和 99.99%。

但是手写实现使用 NumPy `uint64` 做

```python
(a*h+b) % (2^61-1)
```

中间乘法会发生 uint64 wraparound，所以不能把标准 MinHash理论 recall当成严格保证。

这不需要阻塞实验，因为：

* normalized exact = 0；
* long n-gram几乎为零且人工为 false positives；
* 高敏感 approximate screen又为零。

只需要把措辞改成：

> nominal LSH detection probability under the ideal MinHash model

或者改用标准实现/安全 modular arithmetic。

第三，当前“exactly query realization × training stochasticity with no third random axis”也过强。

Random-K 的 seed还是：

[
5000+d,
]

也就是说三个 query blocks同时绑定三个不同 Random-selection realizations。RR query order同理。

这套设计**可以保留，而且我认为应该保留**；三个 Random realizations比只用一个 Random subset更健康。

但 statistical language应该是：

> three draw/selection-realization blocks, crossed with two SFT seeds.

对于 targeted method，block变化主要来自 query realization；对于 Random，block变化来自 Random subset realization。不能说所有 block spread都是纯 query variance。

第四，当前 launch manifest本身还是写：

```text
commit_at_emit = c2d16a5...
launch_commit_pending
```

而现在 HEAD 是 `bee7d29`。

所以这个文件还不能真的叫 final launch record。正式 canary前应该生成一个 clean-tree launch receipt，记录真正的执行 HEAD，不必尝试做会自引用的“manifest contains its own commit”。

第五，lm-eval pin现在很好地 pin了 version/config/data，但没有完整 pin Python execution code。它是 wheel install，所以没有 Git SHA。

至少再记录：

* `pip freeze`；
* `lm_eval/api/task.py` SHA；
* `lm_eval/evaluator.py` SHA；
* `lm_eval/models/huggingface.py` SHA；
* relevant filter implementation SHA；
* Transformers + tokenizer hashes/version。

因为 prompt parity依赖 `Task.fewshot_context()`，generation truncation则依赖 HFLM Python code。这个改动成本接近零。

另外，launch manifest里：

> optionally (D_2(S,P_{\rm heldout}))

最好直接删掉 optional。

如果没有预先决定就不要在看结果后选择做不做。当前 external experiment只需要

[
D_2(S,Q_d)
]

就足够回答 surrogate/outcome relation。

---

# 我建议现在怎么做

所以这次不要回复 fresh Claude “go ahead”。

让它先完成下面这一个最终 checkpoint。注意：前两项是 P0，后面基本都是便宜修补。

1. **创建一个 authoritative BBH execution contract**，不要再依赖 Claude memory。里面明确写死 candidate cache hash、Adam-candidate/SGD-target、projection seed 123、dim 8192，以及五个 method对应的实际脚本/参数。特别规定 DSMC 必须是 `select_moment_mmd.py --alpha 0`，不能自行改用 generic online `MMDSelector`。

2. **新增 execution-level token audit**。对全部 192 query records，用实际本机 Llama-2 tokenizer和实际 LlamaFactory preprocessing跑一遍，记录 wrapper后完整 token length、`cutoff_len=2048` 后实际 input IDs/labels长度、被截多少 token、最终 query文本是否完整保留、`A: Let's think step by step.` cue是否保留、assistant target是否保留。然后用实际 pinned lm-eval HFLM encoding计算 evaluation-side有效 context和 truncation。这个 audit只看 tokens，不加载 accuracy。

   如果 192/192 均无有意义 truncation，可以继续。

   如果任何样本把**当前 query或 target截掉**，立刻 HOLD，不要自动改 cutoff/few-shot数；把统计带回来，我们再一次性决定协议修正。

3. 审计 81 个 hard-coded CoT demos 与 full/query/eval examples之间的 exact/fuzzy overlap，并 fail-loud；同时把 wrapper wording改成“pre-wrapper context aligned”，不要称 executable tokens byte-identical。

4. 把 fresh Claude 的 historical context修正：1% Second-RR=5/10 cells、3/5 blocks；forensics refute的是 skew-capacity mechanism，而不是 forensic result本身。Random/RR selection randomness也按 draw-block nuisance正确表述。

5. 将 LSH recall改成 nominal wording，补 pin Python runtime files/pip environment，删掉 optional (D_2(S,P_{\rm heldout}))，并生成真正指向新 clean HEAD 的 canary launch receipt。

完成后再回来 review。**仍然不运行 no-SFT evaluation，不提取 gradients。**

---

你可以直接把下面这段发给 fresh Claude：

> Hold the selection canary. `bee7d29` is close, but the fresh-window context has drifted in two load-bearing places.
>
> First, restore the exact frozen feature/selector contract from the completed MMLU experiments: candidate gradients are Adam-aware, target/query gradients are SGD, projection dim=8192 and seed=123, with the existing candidate cache/hash reused. DSMC for this experiment is the offline `scripts/select_moment_mmd.py --alpha 0`, not the generic online `MMDSelector` (whose default target gradient type can be `same`). First/Second-RR use `select_round_robin.py`; LESS-style TopK uses `select_relevance_topk.py --order first`; Random uses the frozen per-draw seed. Put this mapping and all cache/checkpoint hashes into one authoritative BBH execution-contract artifact.
>
> Second, add a pre-model tokenization/truncation gate over all 192 query records. String-level parity is insufficient because target-gradient extraction uses the LlamaFactory `llama2` wrapper with `cutoff_len=2048`, while pinned lm-eval BBH generation reserves 1024 tokens within Llama-2's 4096-token context. Using the exact installed tokenizer and LlamaFactory preprocessing, record full and post-cutoff token lengths, truncated token counts, and whether the actual query, CoT cue, and supervised target survive intact. Separately record the exact lm-eval-side encoded/truncated context. If any query or target is materially truncated, STOP and report the distribution; do not silently alter cutoff, few-shot count, or prompts.
>
> Also audit all hard-coded BBH few-shot demonstration inputs against the 6,511 raw examples, the query reservoir/draws, and held-out eval for exact/normalized/fuzzy overlap. Tighten the LSH recall language to nominal detection probability unless the MinHash implementation is made formally safe.
>
> Correct the inherited-context summary before further work: at 1%, DSMC does **not** beat every targeted selector 9/10—Second-RR is 5/10 cells and 3/5 blocks; NICE is 7/10 and GIST 8/10. The refuted mechanism was the skew-capacity explanation, not the forensic analysis itself. Also describe the three Random seeds as selection realizations coupled to the three draw blocks rather than claiming there is literally no third randomness source.
>
> Finally pin the relevant installed lm-eval Python source files/environment, drop the optional `D2(S,P_heldout)` diagnostic unless it is explicitly preregistered, and emit a clean-head canary launch receipt. Still run **no evaluation, gradients, selection, or SFT** until this checkpoint is reviewed.

我的当前状态判断是：

> **科学设计：PASS。split：PASS。lm-eval custom suite：基本 PASS。现有 parity audit：字符串层 PASS。真正执行协议：尚未 PASS。**
>
> **最需要查的不是 `[INST]` 本身，而是 3-shot CoT prompt经过 LlamaFactory `cutoff_len=2048` 后，模型实际拿来求 query gradient 的到底还剩什么。**

这个问题一旦通过，我们就可以很放心地放行 no-SFT + draw0 五 selector canary。

[1]: https://github.com/suzgunmirac/BIG-Bench-Hard?utm_source=chatgpt.com "GitHub - suzgunmirac/BIG-Bench-Hard: Challenging BIG-Bench Tasks and Whether Chain-of-Thought Can Solve Them · GitHub"
[2]: https://github.com/hiyouga/LLaMA-Factory/issues/3995?utm_source=chatgpt.com "Feature suggestion: cutoff_len could optionally drop too long examples from dataset. · Issue #3995 · hiyouga/LlamaFactory"
[3]: https://github.com/meta-llama/llama?utm_source=chatgpt.com "GitHub - meta-llama/llama: Inference code for Llama models · GitHub"
[4]: https://github.com/hiyouga/LlamaFactory?utm_source=chatgpt.com "GitHub - hiyouga/LlamaFactory: Unified Efficient Fine-Tuning of 100+ LLMs & VLMs (ACL 2024) · GitHub"

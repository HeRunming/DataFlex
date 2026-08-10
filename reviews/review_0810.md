我又重新把 `0c03b4a + b88e737` 按“准备真正启动实验”的标准过了一遍。结论是：

**方法和数据层面已经没有新的 P0。可以进入 selection canary。**
不过我会在启动前再修 **两个很小的 provenance/P1 文案问题**；它们不需要重新设计、不需要再等一轮 review，修完即可直接跑 canary。

这次 Claude 对上两轮问题的修复基本是正确的。BBH target-gradient 输入现在确实指向被冻结的 rendered query JSONL，并会检查 file hash、64 行、ordered IDs、非 symlink，而且生成出的 `select_bbhx_draw0.yaml` 确实是 `cutoff_len=3072`、target-gradient `lora_dropout=0.1`。  execution contract 也清楚地区分了 Adam candidate / SGD target、target-grad dropout 0.1 / SFT dropout 0.05，并固定了 selector 实现。

3072 的 truncation gate 现在也真正通过了：192/192、0 material truncation、最大 gradient-side length 2581；Llama 2 官方 context 是 4096，所以这个 pre-compute correction 是合理的。 ([GitHub][1])

CoT 重复 trigger 的修复也应该保留。lm-eval 后来的官方 release 明确把 “removed redundant `Let's think step by step` from `bbh_cot_fewshot`” 列为 BBH bugfix，因此 Claude 不是自行创造了一个更有利的 prompt，而是在 compute 前修复一个 upstream 已知错误。([GitHub][2]) 当前 parity audit 对 causal_judgement 等任务已经显示 raw stock 不同、但去掉 duplicate cue 后一致，并且 demos 与官方 BBH CoT prompt 匹配。 residual 的那个 target 前空格也不用继续折腾，lm-eval task API 的 `target_delimiter` 默认本来就是一个空格。([GitHub][3])

但我确实找到了两个小问题。

第一，**launch manifest 和 prereg 还残留一句现在已经是假的旧描述**：

```text
only_difference_vs_stock:
    "dataset source (audited byte-for-byte: gate A)"
```

prereg 的 artifact index 也还说：

> frozen custom held-out suite (dataset source is the only change)

现在显然不再是 “only dataset source differs”，因为你们还对 v0.4.5 的 few-shot samples做了 upstream/official-validated CoT cue de-duplication。`bbh_eval_pin_manifest` 自己已经诚实记录了 `fewshot_samples_sha256_upstream_v045`、新的 hash、以及 `n_demos_cot_cue_deduped=3`，所以底层 provenance 是好的，只是上层描述没同步。

建议改成：

> Custom held-out suite changes (1) dataset source to the frozen 5,209-example held-out split and (2) applies the official/upstream-corrected removal of the redundant CoT trigger present in installed lm-eval v0.4.5; generation, filtering, metrics, num_fewshot, sampler and all other task semantics remain pinned.

这样就完全准确。

第二，**receipt 的“executing head”语义需要在真正运行时写清楚。** `b88e737` 这个 commit 本身只是添加 receipt，而 receipt 中：

```text
executing_head = 0c03b4a...
```

这在“receipt记录它所批准的 code snapshot”语义下没问题；但如果 Claude 实际从当前 HEAD `b88e737` 启动，那么严格意义上 runtime HEAD 就不是 `0c03b4a`。

不用再玩自引用 commit。最简单的是 canary log 开头明确记录：

```text
approved_code_snapshot = 0c03b4a...
runtime_head           = b88e737...
```

然后 assert：

> `0c03b4a..b88e737` 唯一差异就是 `bbh_canary_launch_receipt.json`。

或者直接 detached checkout `0c03b4a` 运行。两种都可以。

这只是 provenance clean-up，不是科学 blocker。

---

我建议**现在不要再无限找潜在问题**。下一步正应该让 canary 去验证那些只有真正执行才能回答的问题。

这轮 canary 应该严格限制在：

1. no-SFT/base model → frozen 5,209-example BBH heldout；
2. `bbhx_draw0` → 64 target gradients；
3. 五个 selectors全部跑到 (K=2707)；
4. 可导出五个 subsets、跑 diagnostics；
5. **不跑任何 SFT。**

其中 target-gradient extraction 是最关键的执行 gate。Claude 自己已经在 contract里预注册了一项非常好的检查：实际记录 loaded PEFT dropout 和 `model.training`。

我会把 canary pass criteria写得更明确一点：

* target YAML readback = cutoff 3072；
* exact target JSONL hash = frozen draw0 prompt file；
* candidate symlink resolves to frozen candidate cache，并验证 tensor-content hash；
* target tensor shape `(64,8192)`；
* dtype/finite/nonzero全部合法；
* 64 rows 对应 ordered target IDs；
* 实际 loaded adapter/checkpoint hashes匹配；
* 实际 `model.training` 和 LoRA dropout值进入 manifest。

**如果 dropout active**，按照已经预注册的规则，从干净 target cache把 draw0完整提取第二遍：

[
H(G_T^{(1)}) \stackrel{?}= H(G_T^{(2)}).
]

如果 hash一致，继续。

如果 hash不一致，**停止在这里**，不要自行设置 eval mode、dropout=0、换 seed或挑一份 cache继续。把：

* tensor cosine；
* row-wise cosine；
* norm differences；
* DSMC selection Jaccard

带回来再决定。这个问题正适合用 canary发现，不适合在 launch 前继续猜。

五个 selection 应要求：

[
|\mathcal S_m|=2707
]

全部 unique、range合法，并且重复执行 selector后 hash一致。First-RR/Second-RR 必须确认共用 `rr_perm_seed=6000`；Random必须确认 `seed=5000`。这和当前 execution contract一致。

---

还有一个我建议这次 canary **一定报告、但不要据此调算法** 的指标：DSMC 和 Random 的 post-SFT-template length/token exposure。

因为 BBH 主实验没有再带 `Random-K-LengthMatched`。这不是现在需要新增第六个 baseline 的理由——MMLU 已经有 length-matched control——但 BBH selection可能产生新的 length distribution。

所以 draw0 selection-only canary报告：

* 每方法总 post-template tokens；
* truncated-at-2048 tokens；
* length histogram；
* DSMC vs Random token ratio；
* source composition；
* pairwise Jaccard。

**不要根据这些信息改变 DSMC/LESS/RR。**

如果 DSMC 和 Random 的 token exposure 差异很小，30-adapter设计原样继续。

如果出现非常大的差异，我们可以在任何 BBH downstream accuracy出现以前，决定是否补一个 Random-LengthMatched control。因此这仍然不是 outcome-driven tuning。

---

所以我现在的明确决策是：

> **可以启动 selection-only canary。**
>
> 启动前只修两处小 provenance：
>
> 1. 删除“dataset source is the only difference vs stock”这个已经错误的描述，明确记录 CoT de-dup patch；
> 2. 明确 `approved_code_snapshot=0c03b4a` 与实际 runtime HEAD/receipt commit 的关系。
>
> **不用再回来等我 review 这两个文本修补；修完直接跑 canary。**

可以直接回复 Claude：

> The two previous execution blockers are resolved correctly. Proceed to the selection-only canary after two small provenance cleanups; no further methodology review is needed before the canary.
>
> First, fix the stale claim in both the launch-manifest generator/output and the prereg artifact index that the custom BBH suite differs from stock “only by dataset source.” It now differs in exactly two preregistered ways: (1) the frozen held-out dataset source and (2) removal of the redundant CoT trigger present in installed lm-eval v0.4.5, validated against the official BBH CoT prompts / upstream fix. Generation settings, filtering, metric, num_fewshot, sampler, and other task semantics remain pinned.
>
> Second, make launch-head semantics explicit. The receipt commit `b88e737` approves code snapshot `0c03b4a`; if runtime HEAD is `b88e737`, log both `approved_code_snapshot` and `runtime_head` and assert that the only diff is the receipt artifact, or execute from a detached clean checkout of `0c03b4a`.
>
> Then run exactly the approved canary: one no-SFT held-out BBH evaluation, draw0 target-gradient extraction, and all five selectors at K=2707. No SFT.
>
> For target extraction, validate the exact query JSONL hash/ordered IDs, emitted cutoff=3072, candidate cache hash, warm-up hashes, `(64,8192)` finite/nonzero tensor, and log the actually loaded PEFT dropout plus `model.training`. If dropout is active, repeat draw0 extraction from a clean cache and require reproducibility; if the projected-gradient hashes differ, STOP and report the tensor/selection stability diagnostics rather than silently changing the extraction recipe.
>
> For selection, require exactly 2707 unique in-range indices for all five methods, deterministic rerun hashes, RR seed/order correctness, Random seed correctness, and report Jaccard, source composition, and post-template token/length diagnostics. Do not use any geometry/overlap/base accuracy to alter a method.
>
> Push the canary evidence and stop before SFT. If the engineering canary is green, the next step is the two-adapter end-to-end canary (draw0 DSMC seed42 + draw0 Random seed42), then the remaining 28.

简化成一句话就是：

**现在已经不需要继续改实验设计；修两个 provenance 小尾巴后，正式放行 no-SFT + draw0 五方法 selection canary。**

[1]: https://github.com/meta-llama/llama?utm_source=chatgpt.com "GitHub - meta-llama/llama: Inference code for Llama models · GitHub"
[2]: https://github.com/EleutherAI/lm-evaluation-harness/releases?utm_source=chatgpt.com "Releases · EleutherAI/lm-evaluation-harness · GitHub"
[3]: https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/task_guide.md?utm_source=chatgpt.com "lm-evaluation-harness/docs/task_guide.md at main · EleutherAI/lm-evaluation-harness · GitHub"

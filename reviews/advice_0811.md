这次我看完 `f52d6a5 + 047960b` 后，判断终于可以明确一些了：

**可以启动 2-adapter engineering canary。现在不应该再修改科学协议、匹配 bins、selector 或新增 control。**
这一轮把最后一个真正有科学解释风险的问题——SFT 实际看到的 sequence/label 长度——基本处理干净了。

这次 authoritative cache 我认为已经足够可信。它不再手工拼 `[INST]`，而是直接调用 pinned LLaMA-Factory v0.9.3 的 `SupervisedDatasetProcessor._encode_data_example`，并固定 `cutoff_len=2048, train_on_prompt=False, mask_history=False`。 这与官方实现的语义吻合：LLaMA-Factory 的 multiturn template 按顺序把消息编码成多个 user/assistant pair，而官方数据说明也明确说 SFT 中 history 里的 assistant responses 会被学习。([GitHub][1])

因此 24,628 个 multi-turn OASST1 样本确实是上一版手工 flatten 会处理错误的部分；现在 single-turn 0% mismatch、multi-turn 100% mismatch 的模式非常符合预期。更关键的是，修正后核心现象几乎没变：

[
\frac{\text{DSMC sequence positions}}{\text{Random}}
=1.505,
]

而

[
\frac{\text{DSMC loss-bearing label positions}}{\text{Random}}
=0.1764.
]

现在“loss-bearing label positions”这个名称也是准确的。Transformers v4.50.0 的 causal-LM loss对 `-100` label执行 ignore，然后在剩余 token上计算 cross-entropy；所以把它解释成“参与 loss 的位置数”可以，但不能把 5× positions说成5×“学习信号”。([GitHub][2])

所以这个结果本身可以保留为一个很有意思的 descriptive finding：

> DSMC 在 BBH targeting 下偏向较长 context、较短 response 的训练样本；这与 BBH 的短答案格式一致，但目前只说明 **consistent with format matching**，不是机制证明。

这一层现在很稳。

SeqLabelMatched control 的决定我也支持。它现在匹配的是预先固定的

[
5\text{ sequence bins}\times5\text{ label bins}
]

二维 histogram，并且 availability gate先于 sampling运行，21/25 occupied cells都足够，seed固定为 `7000+d`。 这比之前只匹配 sequence length强得多，而且没有增加总 adapter数：仍然36，primary五方法不变，SeqLabelMatched只是 secondary sensitivity。

不过论文措辞上有一个小地方要特别注意：

**不要说它“精确匹配了 sequence 和 label token totals”。**

它精确匹配的是**粗粒度二维 bin histogram**。draw0的实际总量仍然是：

[
\text{sequence ratio}=0.9760,
]

[
\text{label-position ratio}=1.1515.
]

也就是 sequence少约2.4%，label positions多约15.2%。

这没有必要继续修。相较 plain Random 的 label ratio 5.67×，1.15×已经把 format axis控制得非常大幅度了。为了再从1.15调到1.02去重新设计 bins或挑随机 seed，反而会破坏现在很漂亮的 preregistration。

所以请坚持：

> **coarse joint sequence/label-length matched Random**

而不是：

> exact token-budget matched Random.

prereg里的 STOP RULE 是对的：从这里开始不再新增 matched Random、source control、LR/epoch control或方法变体。

现在只剩三项很小的 provenance cleanup，我建议 Claude修完就**直接启动，不用再回来等我们 review**。

第一，prereg还有两个旧数字没同步。现在上面仍有：

> “all 30 adapters are reported as deltas against it”

以及 artifact index里的：

> “15-subset invariant”

但现在正式 design 已经是：

[
36\text{ adapters},\qquad18\text{ planned subsets}.
]

这里改成36和18即可。严格说 draw1/2的 SeqLabelMatched subsets尚未生成，所以最好写：

> 18 planned/frozen-by-rule subsets

不要暗示18个 subset artifact现在已经全部存在。

第二，`047960b` 是一个 **selection-canary launch receipt**，它批准的是 `f52d6a5` 的 selection/artifact状态。 接下来要运行的是第一次 SFT，所以我建议再生成一个单独：

```text
bbh_sft_canary_launch_receipt.json
```

明确只批准两个 cells：

```text
bbhx_draw0_dsmc_seed42
bbhx_draw0_randk_seed42
```

并 pin：

* 两个 selection subset hashes；
* resolved SFT recipe；
* base BBH result hash；
* held-out eval-suite hash；
* runtime HEAD；
* tree clean；
* “engineering only, accuracy cannot alter protocol”。

这样后面 reviewer看 provenance会非常清楚：selection canary 和 SFT canary 是两个不同 gate。

第三，在真正训练之前做一个几十秒的 fail-loud dry check：确认 draw0 六个 dataset keys，包括新加入的

```text
bbhx_draw0_randk_seqlabelmatch
```

都能由实际 SFT loader resolve，且各自恰好2707条。因为两-adapter canary只训练 DSMC+Random，它本身不会覆盖新 secondary arm 的 dataset-registration路径。这个 check不需要训练第三个 adapter。

除此之外，我没有看到值得继续 HOLD 的科学问题。

接下来执行顺序我建议就冻结为：

1. 修上面三个小 provenance / loader check；
2. 运行 `draw0 DSMC seed42 + draw0 Random seed42` 两个 engineering adapters；
3. eval两者，但**不在摘要中打印二者的 accuracy comparison**；raw result照常保存和hash；
4. canary只判断：

   * train manifest完整；
   * 实际 subset hash正确；
   * frozen SFT recipe正确；
   * optimizer steps/epochs符合预期；
   * eval 27/27 subtasks、5209 examples；
   * result/manifest hashes完整；
   * resume能够正确 validated-skip；
   * aggregate pipeline接受36-cell schema；
5. 如果这些全部 PASS，**不要再来问“结果好不好要不要改实验”**，直接继续：

   * draw1、draw2 target gradients；
   * 第一份成功 extraction直接冻结hash；
   * 五个 primary selections；
   * SeqLabelMatched feasibility + selection；
   * 全部 subset hash验证；
   * 剩余34 adapters；
6. 直到 **36/36 train + 36/36 eval** 完成后，才统一解封 accuracy并分析。

这里我会特别强调一点：两-adapter canary的 DSMC/Random accuracy即便差得非常大，也**不构成停止条件**。它现在已经只能用于 engineering。否则经历这么多 pre-accuracy prereg之后，最后还是被两个 interim numbers影响，就前功尽弃了。

如果 canary工程失败，则只允许修**基础设施错误**：path、manifest、offline eval、resume、disk、hash bookkeeping等；不能改 method、SFT hyperparameter、prompt、subset或eval定义。

最后再说一个我觉得这次 audit意外带来的论文价值。现在你们 external experiment不只是“再跑一个 BBH”了，而是实际上能够区分三个东西：

[
\text{gradient geometry}
]

[
\text{instruction format / response length}
]

[
\text{plain random coverage}.
]

如果最终 SeqLabelMatched Random追上 DSMC，那么论文会得到一个非常有意思的机制结果：target-aware selector的收益可能相当一部分来自**隐式选择 target-like instruction format**，不一定来自 gradient geometry本身。

如果 DSMC仍明显超过 SeqLabelMatched，那反过来就说明粗粒度 format matching不足以解释 DSMC。

如果 Random依然最好，那你们原来的负面故事就更强。

三个结果都可解释，所以现在已经到了应该**停止设计、开始执行**的时候。

可以直接让 Claude：

> Patch the three remaining provenance items and then launch the two-adapter engineering canary without another methodology review.
>
> 1. Update stale prereg wording from 30→36 adapters and 15→18 planned/frozen-by-rule subsets.
> 2. Emit a dedicated `bbh_sft_canary_launch_receipt.json` for exactly draw0 DSMC seed42 and draw0 Random-K seed42, pinning their subset hashes, resolved frozen SFT recipe, held-out eval pin/base-reference hash, clean runtime HEAD, and the rule that interim accuracy cannot affect the protocol.
> 3. Before training, run a fail-loud loader-only check that all six draw0 dataset keys—including `bbhx_draw0_randk_seqlabelmatch`—resolve through the actual SFT data path to exactly K=2707 examples.
>
> Then run only the two engineering adapters and their evals. Store and hash raw accuracy outputs but do not compare or use them for decisions. Report only `ENGINEERING_PASS/FAIL` plus train/eval/provenance checks. If green, immediately proceed to draw1/2 target extraction, all remaining selections including the frozen SeqLabelMatched secondary controls, and the remaining 34 adapters. Do not reopen method/control design unless there is a genuine engineering or provenance failure.

**所以现在的答案是：可以开跑。** 先做三个非常小的 provenance/loader修补，然后2-adapter engineering canary；通过后直接剩余34，不再修改科学协议。

[1]: https://github.com/hiyouga/LLaMA-Factory/blob/v0.9.3/src/llamafactory/data/template.py "LlamaFactory/src/llamafactory/data/template.py at v0.9.3 · hiyouga/LlamaFactory · GitHub"
[2]: https://github.com/huggingface/transformers/blob/v4.50.0/src/transformers/loss/loss_utils.py "transformers/src/transformers/loss/loss_utils.py at v4.50.0 · huggingface/transformers · GitHub"

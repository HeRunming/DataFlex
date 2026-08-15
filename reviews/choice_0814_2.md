对，我现在会拍板：**用 Llama-3.2-3B 再做这一组。** 不再等 Qwen，也不再换第三个候选模型。

`7e07c31` 的选择过程是干净的：模型选择完全基于环境兼容性，没有看任何 BBH accuracy；Llama-3.2-3B 已经在冻结的 Transformers 4.50.0 / LLaMA-Factory 0.9.3 环境中完成了 config、bf16 load、tokenizer、template 和 preprocessing canary，而 Qwen3 需要改变软件栈。 Meta 官方模型卡也说明 Llama-3.2-3B 是 3.21B 的 base pretrained model，128k context，并支持 Transformers ≥4.43，因此 4.50.0 本身在官方支持范围内。([Hugging Face][1]) ModelScope 自己的 supported-model 映射也把 `LLM-Research/Llama-3.2-3B` 对应到 `meta-llama/Llama-3.2-3B`，并注明 Transformers ≥4.43。([GitHub][2])

更重要的是，这不是为了“找一个 DSMC 能赢的模型”。直接邻近的 *Critical Look* 本身主要用 Llama-2-7B，又在 additional-model experiments 里用了 Llama-3.2-3B；论文明确报告 BBH 上 LESS+RR 在 Llama-3.2-3B 和 Qwen3-4B-Base 上表现最好，并指出 newer models 的 downstream trends 更不一致。([arXiv][3]) 因此用 Llama-3.2-3B 回答你们现在最大的 reviewer 问题——“这是不是 Llama-2-7B 特例？”——非常自然。

但我会纠正 Claude 的一句 framing：

> 不要把它叫做“只有 model axis 改变”。

因为 D2c 已经证明 **serialization 是 load-bearing 的**。Llama-2 用 `llama2` wrapper，Llama-3.2 合理地会用 `llama3` wrapper；同时 tokenizer 也从 32k 变成约 128k。 因此这是一个：

> **second model-stack confirmation**

而不是严格的 architecture-only causal ablation。

这并不削弱实验价值。相反，它是在第二套现实模型/tokenizer/template stack上测试中心现象。只是论文不能说“we isolate model architecture while holding everything else identical”。

我赞成 Claude 提出的完整 24-adapter 设计：

[
3\text{ query draws}
\times
2\text{ SFT seeds}
\times
4\text{ methods}
=24.
]

方法就固定为 DSMC、First-RR、Second-RR、Random-K，再加一个共享 no-SFT reference。不要 LESS、GIST、NICE、SeqLabelMatched，也不要新方法。第二模型的目的是确认中心现象，不是重新做完整 benchmark。

正式 prereg 里我会写死以下内容：

* **BBH split、三个64-query draws、(K=2707)、SFT seeds `{42,1}` 全部复用。**
* **Random-K 复用完全相同的 candidate indices。** 这一点非常好，因为它给跨模型提供一个真正固定的数据基线。
* DSMC / First-RR / Second-RR 必须由 Llama-3.2 **重新选择**。
* 重新训练 Llama-3.2 自己的 warm-up checkpoint，并 hash adapter + optimizer。
* 重新提取全部 270,679 candidate gradients；不能复用 Llama-2 gradient cache。
* feature contract仍为 candidate=Adam-aware、target=SGD、projection dim=8192、seed=123。
* 使用与 Llama-2 完全相同的 warm-up data、warm-up recipe和 gradient protocol；只有模型相关的参数空间/tokenizer/template自然变化。
* SFT recipe保持原样：4 epochs、同 LR、batch/accum、LoRA recipe，不针对 3B重新调参。
* held-out evaluation继续用完全同一 frozen bare BBH suite。

这里还要预注册一个 **Llama-3.2 token gate**。不能因为它支持128k context就把 cutoff放大。保持：

[
\text{target-grad cutoff}=3072,\qquad
\text{SFT cutoff}=2048
]

和现有 BBH protocol一致，然后用 Llama-3.2 tokenizer重新验证192/192 query没有 material truncation。Meta模型支持更长context不是改变当前实验budget的理由。([Hugging Face][1])

### 第二模型最关键的不是“谁赢”，而是提前规定如何解释

我建议在 prereg 里直接放四种情况：

**A. 最强 replication**

[
D_2(\mathrm{DSMC}) < D_2(\mathrm{Random})
]

同时

[
Acc(\mathrm{DSMC}) < Acc(\mathrm{Random})
]

并且 operational query surrogate改善。

这会非常有力地支持：

> better target alignment is not sufficient for downstream utility across two model generations.

**B. DSMC geometry最好，但 downstream≈Random**

仍然支持核心命题，因为“better matching ⇒ better utility”依然失败；只是负迁移没有 Llama-2 那么严重。

**C. DSMC 在 Llama-3.2 上真的超过 Random**

也绝对不要把它看作实验失败。

这时结论变成：

> target matching can help, but its utility is **model-dependent** rather than reliable.

这其实与 Critical Look 的跨模型结果非常一致。([arXiv][3])

**D. DSMC 连 (D_2) 都不再最低**

那会削弱“DSMC跨模型稳定优化geometry”的说法，但仍然说明 MMLU/Llama-2 上的方法优势没有泛化。照实报告即可。

无论哪个结果，**都不追加第三模型或调超参。**

### diagnostics 也现在一次冻结

第二模型完成后只分析已经有理论动机的四项：

[
D_2(S,Q_d)
]

operational wrapped query CE，

same-query CoT EM，

bare-context CE。

held-out BBH micro EM仍是 primary outcome。

其中 bare CE现在已经明确是 serialization sensitivity diagnostic，不能再把它变成新的主判据。

另外，我会给 ModelScope provenance 再加一个小要求。虽然 ModelScope的官方支持表明确把这个镜像映射到 Meta 的 Llama-3.2-3B，([GitHub][2]) 但在无法访问 gated HF weights逐字节比对的情况下，论文最好不要声称：

> “we verified the ModelScope checkpoint is bit-identical to Meta's gated HF checkpoint.”

除非真的验证了。

应该记录：

* ModelScope repo ID；
* snapshot/revision；
* 全部 `safetensors` SHA256；
* config/tokenizer hashes；
* license/model card；
* 参数数量；
* tokenizer vocab；
* `model_type`.

论文中称：

> “Llama-3.2-3B checkpoint obtained from the ModelScope `LLM-Research` mirror”

即可。Meta官方模型卡和ModelScope mirror metadata在模型规格上是一致的。([Hugging Face][1])

### 接下来具体做什么

我会让 Claude **直接写 prereg，但还不要立即长跑**。完成 prereg 后先做一个短 engineering gate：

1. pin ModelScope snapshot和所有weight/tokenizer hashes；
2. 生成 Llama-3.2 warm-up recipe，并验证一次小 canary；
3. target prompt token audit 192/192；
4. 跑很小的 candidate-gradient canary，比如固定几十条，确认 Adam-preconditioning、8192 projection、finite/nonzero、deterministic behavior；
5. 跑 draw0 target-gradient canary；
6. 确认 DSMC/First-RR/Second-RR 三个 selector能到 K=2707；
7. **不看任何 BBH accuracy**；
8. engineering green 后直接生成完整 candidate datastore → 24 adapters。

不需要再回来讨论“选不选 Llama-3.2”。这个决定现在已经可以冻结。

你可以直接给 Claude：

> Proceed with **Llama-3.2-3B** as the second-model confirmation and freeze that choice now. Do not revisit Qwen3 or consider a third model.
>
> Frame this correctly as a **second model-stack confirmation**, not a pure architecture-only ablation: the model, tokenizer and model-appropriate LlamaFactory serialization change together. This matters because D2c showed serialization is load-bearing.
>
> Write and commit the preregistration before substantive compute:
>
> * same frozen BBH held-out split and the same three M=64 query draws;
> * K=2707;
> * SFT seeds {42,1}, fully crossed;
> * methods = DSMC, First-RR, Second-RR, Random-K;
> * shared no-SFT reference;
> * 24 adapters total;
> * Random-K reuses the exact Llama-2 candidate indices;
> * rebuild Llama-3.2 warm-up, candidate gradients, target gradients and all target-aware selections from scratch;
> * candidate=Adam-aware, target=SGD, projection dim=8192, seed=123;
> * keep target-gradient cutoff=3072 and SFT cutoff=2048; rerun the full 192-query truncation audit with the Llama-3.2 tokenizer;
> * keep the frozen SFT hyperparameters rather than tuning for the 3B model;
> * primary outcome = held-out BBH micro EM;
> * pre-register D2, operational wrapped query CE, same-query CoT EM and bare-context CE as diagnostics.
>
> Pin the exact ModelScope repository/snapshot and SHA256 of all weight/config/tokenizer artifacts. Do not claim bit-equivalence to the gated Meta HuggingFace checkpoint unless independently verified.
>
> Pre-register all result interpretations before training: replication, attenuation/tie, reversal in favor of DSMC, or failure of DSMC to minimize D2 are all reportable outcomes and none may trigger tuning.
>
> After the prereg, run only engineering canaries. If they pass, launch the full model-specific gradient datastore and 24-adapter experiment without further scientific changes.

所以答案就是：**是，用 Llama-3.2-3B 再来这一组，而且这应该是投稿 ICLR 前最后一组大实验。** 做完无论结果如何都停。

[1]: https://huggingface.co/meta-llama/Llama-3.2-3B?utm_source=chatgpt.com "meta-llama/Llama-3.2-3B · Hugging Face"
[2]: https://github.com/modelscope/ms-swift/blob/main/docs/source/Instruction/Supported-models-and-datasets.md?utm_source=chatgpt.com "ms-swift/docs/source/Instruction/Supported-models-and-datasets.md at main · modelscope/ms-swift · GitHub"
[3]: https://arxiv.org/abs/2602.14696 "A Critical Look at Targeted Instruction Selection: Disentangling What Matters (and What Doesn't)"

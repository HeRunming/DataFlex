可以启动 engineering canaries 了。`0c4c0d8` 这版 prereg 的核心设计我认为是合格的，而且现在继续停着找新问题的收益已经很低。

我核对后，Llama-3.2-3B 这个选择本身很合理：官方模型卡说明 Transformers ≥4.43 即可运行，所以你们冻结的 4.50.0 在支持范围内；它又确实是 3.21B、与 Llama-2-7B 在规模、tokenizer 和模型代际上都有实质差异。([Hugging Face][1]) 邻近的 targeted-instruction-selection 工作也确实公开了 Llama-3.2-3B 对应的数据/实验资产，并强调需要跨 models/tasks/budgets验证 selector 行为。([Hugging Face][2])

你们当前 frozen prereg 里最关键的几个决定也都是对的：不复用 Llama-2 的 target-aware subsets，而是重新做 warm-up、270,679 candidate gradients、三套 target gradients 和 DSMC/RR selections；Random 则固定使用相同 indices，从而给两个 model stacks 一个真正共同的数据基准。

还有一个我很喜欢的细节：Llama-3 tokenizer 下192条 query最大只有约2007 tokens。这样虽然 protocol继续写 target-gradient cutoff=3072，但实际上这些 query 在2048都已经能完整装下。因此这次不会再出现 Llama-2 那种“3072修复本身改变 query”的争议；3072只是保持 frozen BBH extraction contract，而不是在第二模型上获得额外信息。这反而让跨 stack 解释更干净。

不过在开始长时间 datastore 之前，我建议 canary明确验证下面几个东西。不是新实验设计，只是工程 pass criteria：

* warm-up必须使用和 Llama-2 arm完全相同的 warm-up raw data、训练轮数/step逻辑和 optimizer protocol；只改变模型/tokenizer/template。warm-up adapter和 `optimizer.pt` 都立即 hash，因为后续 Adam-aware candidate gradients依赖它。
* candidate-gradient小 canary最好同时验证两次独立提取的 row cosine / hash稳定性。Llama-2已经出现过极小GPU数值 nondeterminism，所以这里不要重新要求bit-exact；如果差异只是同量级小浮点扰动，可以记录并继续，若明显更大则停。
* 明确核对 LoRA target modules。Llama-3.2虽然仍是 `model_type=llama`，但不要仅因为模型能load就假设 Llama-2 的 module matching一定选中了完全预期的层；打印 trainable parameter names/count，与预注册 recipe一起存档。
* target-gradient canary要验证 candidate=Adam、target=SGD、projection dim=8192、seed=123，这些才是第二模型上最容易因为默认参数悄悄漂移的核心 contract。
* Random subset必须 byte-for-byte / index-for-index 等于 Llama-2 BBH Random；DSMC/First-RR/Second-RR则必须是 Llama-3.2自己的 selections。

我不会要求它在这些 canaries里跑 BBH accuracy。现在 model choice 和 protocol都已经冻结，任何 base accuracy都没有能力改变后续决策。

还有一个小的 framing 修正建议提前写进 prereg/results template：第二模型实验的 primary问题不要定义成“DSMC是否再次输Random”，而应定义成：

[
\text{Does better target alignment reliably imply better downstream utility across model stacks?}
]

于是四种预注册结果都很好解释：

如果 Llama-3.2再次出现 DSMC最低 (D_2) 但Random downstream最好，这是最强cross-stack replication。

如果DSM C最低 (D_2) 但和Random打平，也仍然支持“better matching不是充分条件”。

如果DSM C胜Random，论文就变成“target alignment的utility具有model dependence”，并不推翻Llama-2结果。

如果DSM C不再最低 (D_2)，则说明连second-moment geometry ranking本身也有model dependence。

无论哪一种，**都不要产生第三模型、LR sweep或新selector。**

我唯一会提醒 Claude不要犯的一个统计错误，是第二模型完成以后仍然按：

[
3\ \text{query draws}
]

作为主要selection-level blocks，先在两个training seeds内平均。24个adapter cells不是24个独立科学replicates。两个seeds用来显示SFT stochasticity，不要突然把 (n=6) 写成六个独立draws。

另外，ModelScope provenance继续按现在的方式处理是对的。没有逐字节比过Meta gated权重，就只写“Llama-3.2-3B checkpoint obtained from the ModelScope LLM-Research mirror”，不要声称和HF官方checkpoint bit-identical。你们已经把artifact hashes冻结，这对复现已经足够。

所以现在给Claude的指令可以很短：

> Proceed with the frozen Llama-3.2-3B engineering canaries now. Do not reopen the model choice or scientific protocol.
>
> Gate 1: train the model-specific warm-up with the frozen raw warm-up data/recipe; pin adapter and optimizer hashes and verify resolved steps/LoRA trainable modules.
>
> Gate 2: extract a fixed small candidate-gradient sample twice; verify candidate=Adam-aware, projection dim=8192/seed123, finite/nonzero gradients, expected shapes and selection-level numerical stability. Do not require bitwise identity if residual differences are only floating-point scale comparable to the Llama-2 canary.
>
> Gate 3: extract draw0 target gradients with target=SGD and verify the exact query IDs/prompts, token integrity, shape `(64,8192)`, finite/nonzero values, and frozen checkpoint/projection hashes.
>
> Gate 4: run DSMC, First-RR and Second-RR to K=2707 and verify deterministic selector hashes; verify Random-K uses exactly the frozen Llama-2 Random indices.
>
> Inspect no BBH accuracy during these gates. If all engineering gates pass, immediately launch the full Llama-3.2 candidate datastore, all three target caches, model-specific selections and the frozen 24-adapter run. No further scientific changes regardless of outcome.

所以答案是：**现在开 engineering canaries。Canary 绿了以后直接跑完整24个，不需要再回来重新讨论实验设计。**

同时论文可以并行开始写，不要等这组结果。第二模型现在是 confirmation，不再是决定论文有没有故事的前置条件。

[1]: https://huggingface.co/meta-llama/Llama-3.2-3B?utm_source=chatgpt.com "meta-llama/Llama-3.2-3B · Hugging Face"
[2]: https://huggingface.co/collections/Harvard-DCML/targeted-instruction-selection?utm_source=chatgpt.com "Targeted Instruction Selection - a Harvard-DCML Collection"

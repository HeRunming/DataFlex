我检查了 `a0510c2`，这一步总体是好的：Phase C 的 no-SFT reference 可以正式冻结，0.396429 的 micro aggregate 与 27 个 subtask 的按样本数复算完全一致，5,209 条也和 frozen held-out split 对上。 这个 reference 后面只能作为一条共享基线，不能当成 6 个 replicate。

现在真正需要决策的是 Phase A。我倾向于：**这不是继续修改 gradient recipe 的理由，但也不建议立刻随手选 run1 然后往下跑。先做最后一个非常便宜的、selection-only 的双缓存稳定性检查，再冻结 run1。**

原因是现在已经知道：

[
G_T^{(1)}\neq G_T^{(2)}
]

但差异非常小：54/64 行 bit-identical，其余 10 行 row cosine 最低 0.999915，DSMC 最终只换了 (1/2707) 条，Jaccard 0.999261。 从“这个数值噪声会不会破坏方法结论”的角度看，这已经是很强的稳定性证据。

PyTorch 官方也明确说，即使固定 seed，完全的 bitwise reproducibility 并不能普遍保证；可以通过 deterministic algorithms 等机制降低 nondeterminism，但这不等同于默认 GPU pipeline 必须 bit-exact。([PyTorch Docs][1]) NVIDIA 的 CUDA 文档也明确指出，并行浮点 reduction 中运算顺序变化会因浮点非结合性产生微小数值差异，因此工程上通常应该按容差比较，而不是默认逐 bit 相同。([NVIDIA Docs][2])

不过我不同意 Claude 现在那句特别肯定的：

> cause = floating-point non-determinism, not dropout

这需要收紧。

你们的报告自己写着 `model.train()`，并且 target-gradient LoRA dropout 是 0.1。 PyTorch 的 Dropout 在 training mode 下确实会采样 Bernoulli mask。([PyTorch Docs][3]) 但如果两个 run 的 RNG state/seed 被同样初始化，dropout mask 本身也可能重复，因此“54 行完全一样”并不能单独证明 dropout 完全无关；反过来，它确实说明这里不像是**未受控、完全不同的 dropout masks**在主导 run-to-run 差异。

所以论文/manifest最好改成：

> The two extractions are not bit-identical. The observed perturbations are small and most consistent with low-level numerical nondeterminism under the fixed extraction pipeline; active LoRA dropout is present, but the exact kernel-level source of the residual differences was not isolated.

不要花时间继续追“到底是哪一个 CUDA kernel”。

接下来我建议先用 run1/run2 **分别**跑四个 target-aware selectors，不需要 Random：

* DSMC
* Second-RR
* First-RR
* LESS-style TopK

这不是 Phase B 正式结果，而是 Phase A 的 numerical-sensitivity closure。对每种方法报告：

[
|\mathcal S^{(1)}\cap\mathcal S^{(2)}|,
\quad
\text{Jaccard},
\quad
K-|\cap|.
]

尤其 RR 很重要，因为 greedy round-robin 理论上可能对很小的 nearest-neighbor perturbation产生级联，而现在只检查了 DSMC。

我建议**现在就预先写死 engineering interpretation**，避免看到结果再决定：

* 如果所有 target-aware selectors 都只发生很小的 subset perturbation，比如 replacement 不超过 (1%) 的 (K)（27 条）且没有异常级联，就认为 target extraction 在 selection level 稳定；
* 如果任何一个 selector出现几十/几百条级别的变化，尤其 RR大面积 cascade，就 HOLD，再考虑 deterministic extraction；
* 这个阈值只用于判断数值稳定性，不用于方法优劣或 downstream claim。

我预计实际会远低于这个门槛。

如果这个检查通过，**就冻结 run1 `10c92b25...` 为 canonical cache**。理由必须写成：

> first successful extraction from the approved canary, frozen before downstream accuracy was observed.

而不是“因为 run1 看起来更好”。现在两份 cache都没经过 SFT/eval，因此这种 canonicalization是干净的。run2永久保留作为 numerical sensitivity artifact。

我不建议现在改成 `torch.use_deterministic_algorithms(True)`、math SDPA、dropout=0 或 eval mode来生成正式 BBH gradients。PyTorch确实提供 deterministic mode，并且新文档还特别说明 Flash/Efficient/CuDNN attention backward 可能默认 nondeterministic。([PyTorch Docs][4]) 但那样会改变已经冻结并与 MMLU 对齐的 feature-extraction implementation。眼下 selection instability只有 1/2707，修改 recipe 的科学成本大于收益。

还有一个很小的 provenance 修补：Phase C report目前记录了 results JSON 的绝对路径和完整复算值，但从我看到的字段里没有记录那个 authoritative results JSON 自身的 SHA256。 建议顺手加上。它不阻塞 Phase B，只是这个共享 base reference以后会很重要，最好现在就 immutable。

成本修正我也接受，而且不影响实验设计。严格说从现在开始 base eval已经做完，所以剩余预算不是约 65 GPU-h，而大约是：

[
30\times111.4/60 + 7.5 \approx 63.2
]

GPU-hours，再加少量 target extraction/selection overhead。8 张 H20 理想并行时大约 8 小时量级，实际考虑调度/I/O，按 9–12 小时 wall-clock规划会更稳妥。这个只是工程计划，不需要为了省算力改 batch size或评测协议。

所以接下来流程我建议这样冻结：

1. 双缓存跑 draw0 的四个 target-aware selectors，只看 subset stability，不看任何 accuracy。
2. 若稳定，冻结 run1 target cache/hash；把 run2标成 sensitivity replicate。
3. 正式执行 Phase B：run1 canonical cache 上的五方法 (K=2707)，重复 selector一次验证 deterministic selection hash，同时输出 source/token/Jaccard diagnostics。
4. Phase B 全绿后，跑之前预注册的两个 end-to-end adapters：draw0 DSMC seed42 + Random seed42。
5. 这两个 adapter 的 eval **只检查工程完整性**：训练 steps、manifest、eval 27/27 subtasks、5209 rows、aggregate存在、resume/hash正确。不要根据这两个 interim accuracies停止、调参或改方法。
6. 工程 canary 一旦通过，直接启动剩余 28 adapters，直到完整 30/30，再统一解封 accuracy 做分析。

甚至可以让 Claude在 2-adapter canary时“不把 DSMC/Random accuracy comparison打印到总结里”，只给 `ENGINEERING_PASS=true/false`，然后自动进入剩余28个。这样最干净。

你可以直接给 Claude：

> Before choosing a canonical draw0 target cache, close the numerical-stability question with one cheap selection-only sensitivity check. Run DSMC, Second-RR, First-RR, and LESS-style TopK on both retained target-gradient caches, using identical frozen seeds/settings, and report intersection size, Jaccard, and number of replaced examples at K=2707. Random-K does not need this check because it is target-independent.
>
> Predefine the engineering rule before inspecting those outputs: if every target-aware selector changes by at most 1% of K (≤27 replacements) with no pathological RR cascade, treat the extraction as selection-level stable. If any selector exceeds that, STOP and report it; do not silently change deterministic settings, dropout, eval/train mode, or the feature recipe.
>
> Also tighten the causal wording. Do not claim the residual is definitively “not dropout.” State that the two extractions are not bit-identical; the observed small, sparse run-level perturbations are most consistent with low-level numerical nondeterminism under the fixed pipeline, while active LoRA dropout is present and the exact kernel-level source was not isolated.
>
> If the four-method stability gate passes, freeze run1 (`10c92b25...`) as the canonical draw0 cache because it is the **first successful approved-canary extraction**, not because of any downstream outcome. Retain run2 and its hash as the numerical-sensitivity artifact. Add the authoritative Phase-C base-eval results JSON SHA256 to the canary provenance.
>
> Then run formal Phase B on the canonical run1 cache: all five selectors to K=2707, exact size/uniqueness/range checks, repeat-run selector hashes, frozen RR/Random seeds, source/Jaccard/post-template token diagnostics. No SFT yet.
>
> If Phase B passes, run the two pre-registered engineering adapters (draw0 DSMC seed42 and Random seed42). Validate training/eval manifests, step counts, 27 subtasks / 5209 examples, hashes and resume behavior, but make no decision from their interim accuracies. If engineering passes, launch the remaining 28 without changing the protocol.

一句话结论：**不用修方法，也不用追求 bit-exact gradient；先确认这种 (10^{-4}) 数值扰动对四个 target-aware selectors 都只是边界级变化。只要这个小 gate 通过，就冻结 run1并继续正式 Phase B → 2-adapter canary → 剩余28。**

[1]: https://docs.pytorch.org/docs/stable/notes/randomness?utm_source=chatgpt.com "Reproducibility — PyTorch 2.11 documentation"
[2]: https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html?utm_source=chatgpt.com "CUDA Best Practices Guide — CUDA C++ Best Practices Guide 13.3 documentation"
[3]: https://docs.pytorch.org/docs/main/generated/torch.nn.functional.dropout.html?utm_source=chatgpt.com "torch.nn.functional.dropout — PyTorch main documentation"
[4]: https://docs.pytorch.org/docs/stable/notes/randomness.html?highlight=reproducibility&utm_source=chatgpt.com "Reproducibility — PyTorch 2.13 documentation"

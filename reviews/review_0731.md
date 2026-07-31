建议让 Claude **继续接入 NICE 和 Random-K-LengthMatched，并完成真正的八方法 selection-only dry run**。可以同时编写 `train/eval/aggregate` 阶段，但暂时不要启动任何 SFT。

当前六方法结果是一个不错的局部工程检查，但“dry run complete and clean”还稍微过早。

已通过的部分包括：

* 复用共享 candidate gradient cache 的思路正确，避免了无意义的 270k candidate 重算；
* 64 个 target gradients 成功生成；
* 六个 selection 都得到 13,533 个唯一合法索引；
* Random-K 与其他集合的 Jaccard 约 (0.024!-!0.026)，符合两个独立 5% 子集的期望水平；
* GIST、RR、DSMC、LESS 没有意外退化成相同 selector。

但 Jaccard 只能证明集合有差异，不能验证方法实现正确，也不能支持下游效果结论。尤其是 DSMC–Second-RR 的 0.354 只比 DSMC–LESS 的 0.339 高 0.015，因此可以描述为“二阶方法 overlap 略高”，不宜称为强聚类证据。最新 targeted-selection 系统研究也强调，representation 和 selector 的价值最终必须通过受控 downstream comparison 判断，Random 有时还能匹配或超过复杂选择方法。([arXiv][1])

更重要的是，当前 driver 的实际能力和注释并不一致：

* 注释写“all 8 selections”，实际只运行 6 个方法；
* 注释声称输出 length distribution、subject distribution、runtime 和 memory，实际诊断文件只有 size 和 Jaccard；
* 注释写支持 `--phases`，但脚本实际上读取环境变量 `PHASES` 和 `DRAW`，没有解析命令行参数；
* `eval`、`aggregate` 尚未实现，`train` 仍是 stub；
* target gradient 和 selection 的 input/output hash manifest 尚未真正出现在 dry-run 报告里。

NICE 也不能留到正式 SFT 时才第一次执行。NICE 的核心是用针对实际非可微任务指标的 policy gradient 替换普通 NTP validation gradient，是独立且带随机生成过程的 baseline；其正确性不能从其他六个 selector 的运行推断出来。([Proceedings of Machine Learning Research][2]) 当前 `nice_select.py` 的说明还写着 MMLU 使用 “gold-token probability”，而实际代码路径定义的是 generation exact-match reward，这个定义冲突必须在运行前解决。

建议直接回复 Claude：

> Wire NICE and Random-K-LengthMatched next, and complete the full eight-method selection-only dry run on `stem80_draw0`. You may implement the train/eval/aggregate phases in parallel, but do not execute any SFT or evaluation yet.
>
> Before rerunning, please patch the current driver:
>
> 1. Make its interface real and unambiguous: either implement `--draw` / `--phases` argument parsing or document that `DRAW` / `PHASES` are environment variables. Reject unknown phases and empty phase lists.
> 2. Add a preflight manifest that validates and records the driver commit, target JSONL byte hash, ordered target-ID hash, candidate-cache resolved path and SHA256, checkpoint hashes, projection seed/dimension, gradient types, target-cache hash/shape/dtype/finiteness/norm statistics, method parameters, and output selection hashes.
> 3. When reusing the candidate cache through a symlink, verify that `readlink -f` resolves to the frozen cache and that its hash and shape match the manifest. Do not merely accept any existing file at that path.
> 4. Validate an existing target cache before skipping `gengrad`; require shape `(64,8192)`, finite nonzero rows, expected target-file hash, checkpoint, projector seed, and ordered-ID hash.
> 5. Wire NICE as a separately labelled adaptation. Resolve the current documentation/code mismatch before execution: the script says “MMLU gold-token probability” but currently implements sampled exact-match reward. Use one frozen definition, preferably label the present task-metric version `NICE-MMLU-EM`, and record MC count, temperature, top-p/top-k, generation seed, max-new-tokens, reward histogram, zero-signal target count, and number of retained policy-gradient rows. Do not change these after viewing its selected subset.
> 6. Implement Random-K-LengthMatched at fixed `K=13,533` using actual post-template tokenizer lengths with the Llama-2 tokenizer and `cutoff_len=2048`, not character count. Match DSMC’s exact bucket counts, report total-token difference and per-bucket differences, and fail if the match is infeasible.
> 7. Produce the diagnostics originally promised: all-eight selection hashes and sizes, pairwise Jaccard, post-tokenization length histograms, total effective tokens, selected-data subject/domain composition, runtime, and peak CPU/GPU memory. Treat these only as implementation sanity checks, not as a basis for method tuning.
> 8. Run the same selection commands twice or rerun through resume mode and verify identical output hashes. NICE must also be deterministic under its frozen sampling seed.
> 9. Avoid concurrent mutation races in `dataset_info.json` and `components_draws.yaml`: either perform setup for all pilot draws serially before parallel work, or generate immutable per-draw configs without concurrent read-modify-write.
>
> After the complete eight-method dry run passes, push the driver, manifests, NICE reward diagnostics, length-match diagnostics, full overlap matrix, runtime/memory report, and a concise pass/fail checklist. Then hold before SFT.

关于 Claude 提到的计算量：四个 target draws、八种方法表面上是 32 个 method–draw combinations；按照冻结协议，Random-K 在相同 draw index 的两个方向之间复用同一 subset、training seed 和 adapter，因此实际应是：

[
7\times4+2=30
]

个不同的 SFT adapters，而不是 32 个。正式启动前，driver 的 run plan 应明确列出这 30 个唯一 run，避免重复训练 Random-K。

最终结论：

> **六方法 dry run 局部通过；现在接入 NICE 和 LengthMatched Random，完成八方法 dry run。允许继续构建后续 driver 阶段，但不批准 SFT。**

[1]: https://arxiv.org/abs/2602.14696 "https://arxiv.org/abs/2602.14696"
[2]: https://proceedings.mlr.press/v267/wang25bm.html "https://proceedings.mlr.press/v267/wang25bm.html"

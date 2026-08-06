现在可以让 Claude **继续构建完整 1% pipeline**，不需要再停下来复审 5% 文案。`94743ce` 对 5% 结论的修正是正确的，nested-prefix 设计方向也正确。

最新 controlled study确实指出，targeted instruction selection 经常遗漏未微调模型的参考结果；同时，gradient representation 与 round-robin 的优势通常在较低预算更明显、在较大预算下减弱，但没有任何方法跨条件普遍领先。因而 1% 应被表述为预算交互检验，而不是“预计一定能击败 Random”的补救实验。([arXiv][1])

不过在 Claude 开始实现前，建议把以下几点一起写进回复。

## Nested-prefix 还需要一次廉价审计

当前脚本直接执行：

```python
idx1 = idx5[:2707]
```

并假定 `step_1.json` 的 `indices` 顺序分别是 greedy/RR append order、score descending order 或随机排列顺序。脚本只检查长度和唯一性，没有验证排序语义。

在冻结前，应对 `stem80_draw0` 的七个 prefixable methods 做一次直接对照：

[
\text{direct selection at }K=2707
=================================

\text{prefix}_{2707}(\text{frozen }K=13533\text{ ordering}).
]

包括：

* DSMC；
* LESS；
* First-RR；
* Second-RR；
* GIST；
* NICE；
* Random-K。

DSMC 和 Random 已经验证，再把其他五个补齐即可。若全部 bit-identical，就能确认保存下来的 index order 确实具有 prefix 语义，而不是只验证了理论上方法应当 prefix-monotone。

每个 1% selection manifest 还应记录：

* 5% `step_1.json` 的 byte SHA256；
* 5% ordered-indices SHA256；
* 1% prefix SHA256；
* `ordering_semantics`；
* `selection_budget=2707`；
* `parent_budget=13533`。

## Base reference 的名称要准确

这里不是严格意义上的 zero-shot evaluation，因为 lm-eval 仍使用 5-shot demonstrations。应称为：

> **base-model / no-selected-SFT reference, evaluated with the same 5-shot protocol**

只需评测一次未经过 selected-data SFT 的 Llama-2-7B，然后在表中作为参考线展示：

[
\Delta_{\text{abs}}
===================

\text{method score}-\text{base-model score}.
]

不要把同一个 base result 复制成十个“独立 replicates”，也不要计入 win counts 或 bootstrap。它是一个共同参照，不依赖 target draw。

这会区分：

* DSMC 在 1% 下是否真正提高基础模型；
* DSMC 与 Random 是否都退化，只是 DSMC 退化更少；
* 某些 selectors 是否产生负迁移。

## 固定 4 epochs 的含义也要预注册

1% 继续使用 4 epochs 是合理的，但这意味着它同时是：

* 数据预算更小；
* 实际训练 steps/token exposure 更小。

因此论文研究的是 **data/compute-budget interaction**，不是 equal-training-steps 下的纯 subset-size interaction。

现在应冻结：

* 两个预算都固定 4 epochs；
* 不根据 1% 结果再补 equal-step training；
* canary 中记录实际 optimizer steps、有效 tokens 和 wall-clock。

预计约 84–85 steps，但以 trainer manifest 的实际值为准。不要为了刚好得到某个 step 数改变 epochs。

## Canary 建议覆盖三条路径

原计划的两个 adapter 可以保留：

1. `stem80_draw0 × DSMC-1pct`；
2. shared `Random-K drawidx0-1pct`。

同时建议增加第三个很便宜的 canary：

3. `stem80_draw0 × Random-K-LengthMatched-1pct`。

原因是前两个都是 5% ordering 的直接 prefix，而 LengthMatched Random 是 1% 唯一重新构造的 selector。三-adapter canary 才同时覆盖：

* deterministic prefix path；
* shared-adapter prefix path；
* newly rebuilt length-matching path。

验证后仍应检查：

* DSMC 与 Random-K 的 direct-vs-prefix equality；
* LengthMatched Random 精确 (K=2707)；
* bucket counts 与 1% DSMC 一致；
* 总有效 token 差异；
* 约 84–85 training steps；
* train/eval manifest；
* resume validated-skip；
* partial aggregate；
* base-model reference 正确进入表格但不伪装成 replicate。

## 给 Claude 的回复

可以直接发送：

> Proceed with building the complete budget-specific 1% pipeline now; no further review of the corrected 5% writeup is needed.
>
> Before freezing the pre-SFT snapshot, add one ordering audit: on `stem80_draw0`, run every prefixable selector directly at `K=2707` and require bit-identical equality with the first 2707 entries of its frozen 5% ordering. DSMC and Random-K are already checked; add LESS, First-RR, Second-RR, GIST, and NICE. Record the parent 5% selection-file hash, ordered-index hash, prefix hash, and ordering semantics in each 1% manifest.
>
> Add one base-model/no-selected-SFT reference evaluated under the same `mmlu_stem,mmlu_humanities`, 5-shot protocol. Do not call it zero-shot, duplicate it as independent draw observations, or include it in win counts; use it only as a common absolute-performance reference.
>
> Keep four epochs frozen at both budgets and explicitly define this as a data/compute-budget interaction, not an equal-step experiment. Record actual optimizer steps, effective tokens, and wall-clock.
>
> Build fully separate 1% namespaces and make every validator obtain `K=2707` from the run plan—no hard-coded 13,533. Export all 80 subsets, generate the 75-adapter run plan and 10-draw master manifest, validate all hashes, and commit a clean launch snapshot.
>
> Then run a three-adapter engineering canary:
>
> 1. DSMC draw0;
> 2. shared Random-K drawidx0;
> 3. draw0 Random-K-LengthMatched.
>
> Include the base-model evaluation in the canary report. Verify manifests, expected roughly 84–85 training steps, exact length buckets, authoritative eval files, partial aggregation, and validated resume. Do not use canary accuracy to alter any method. If all engineering checks pass, continue the remaining 72 adapters automatically and report both 1% and 5% regardless of outcome.

所以当前执行顺序是：

1. 构建 1% pipeline；
2. 补齐 all-method prefix equality audit；
3. 运行一次 no-SFT base evaluation；
4. 冻结并提交完整 pre-SFT snapshot；
5. 跑三-adapter canary；
6. 工程通过后自动跑剩余 72 个；
7. 最终分析五个 direction-averaged blocks 的：

[
[\mathrm{DSMC}-\mathrm{Random}]_{1%}
------------------------------------

[\mathrm{DSMC}-\mathrm{Random}]_{5%}.
]

现在不需要继续讨论方法或调参，直接进入这套工程流程。

[1]: https://arxiv.org/abs/2602.14696?utm_source=chatgpt.com "A Critical Look at Targeted Instruction Selection: Disentangling What Matters (and What Doesn't)"

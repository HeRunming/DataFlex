Claude 这次做得很快，但我的复审结论是：

> **true-RR 可以立刻继续实现；target-draw protocol 暂时不要冻结；GIST 的“数值 review 已通过”需要重新打开。**

原因不是当前 `select_gist.py` 数学写错了，而是我查到 **GIST 已经有官方代码仓库**，Claude 所说“没有 official repo”已经不成立。论文 PDF 也直接给出了官方仓库，仓库目前包含完整梯度提取和 matching 实现。([arXiv][1])

## 1. 当前 GIST 实现：内部自洽，但还不能叫“精确复现 GIST”

当前代码做的是：

[
\text{unit-normalized 8192D JL gradients}
\rightarrow \text{SVD}
\rightarrow 95%\text{ EVR}
\rightarrow \text{projected cosine}
\rightarrow \max_j
\rightarrow \text{top-}K.
]

代码本身的 SVD、rank 选择、投影 cosine 和 top-(K) 都是正确实现。

orthonormality、basis rotation invariance 等检查也有价值。

但它和论文、官方代码之间存在实质差异。

### 论文使用 raw LoRA gradients 构造子空间

论文首先在原始 LoRA 参数空间计算候选与 target 的梯度，再对 raw target-gradient matrix 做 SVD；只有在投影进入目标子空间后，才通过 cosine 消除样本梯度幅度的影响。论文也确实提出用约 95% explained variance 决定 rank，并在 Eq. 17 中对 target examples 取最大相似度。([arXiv][1])

你们当前在 SVD **之前**逐样本归一化 target gradients：

```python
Tg = Tg / Tg.norm(...)
U, S, _ = svd(Tg.T)
```

这不是无害变化。对候选梯度来说，投影前乘一个标量最终会被 cosine 抵消；但是对 target matrix 的每一列分别缩放，会改变 SVD 的主子空间：

[
\operatorname{span}*{\rm principal}
\left[
\frac{g_1}{|g_1|},\ldots,
\frac{g_M}{|g_M|}
\right]
\neq
\operatorname{span}*{\rm principal}
[g_1,\ldots,g_M]
]

一般并不相同。

因此当前方法更准确的名字应是：

> **GIST-JL-Norm** 或 **projected normalized GIST adaptation**

而不是“follows the paper exactly”。

### 官方实现也和当前代码不同

官方梯度代码直接计算 raw LoRA gradients，通过 target Gram matrix 做 eigendecomposition，构造低秩投影矩阵 (P)，再将每个候选的 raw gradient 投影到低维空间。

官方 matching 代码还会：

1. 按 MMLU subtask 对 validation gradients 分组；

2. 先在每个 subtask 内平均 raw gradients；

3. 再对 task vectors 做归一化；

4. 最后对 task vectors 取 maximum relevance。

论文正文写的是对每个 target example 取 max，而官方代码是对每个 task-average vector 取 max，二者自身也有一点 paper–code discrepancy。论文的 Eq. 17 明确是逐 target example 最大值。([arXiv][1])

所以接下来不能简单说“我们的实现已验证正确”；需要决定：

* 复现论文公式；
* 复现官方代码；
* 还是使用你们的 projected normalized adaptation。

## 2. GIST 下一步应该怎么处理

我建议保留当前代码，但将它重命名并降级成 adaptation，然后补一次官方对齐 gate。

最低成本方案是：

### A. 保留现有版本

改名为：

```text
GIST-JL-Norm
```

metadata 明确写：

```json
{
  "raw_or_projected": "JL_projected_8192",
  "normalize_before_svd": true,
  "aggregation": "max_per_example",
  "rank_rule": "95pct_evr"
}
```

### B. 实现 paper-faithful GIST

按照论文和官方梯度代码：

* raw target LoRA gradients；
* raw target matrix 上的 SVD/Gram eigendecomposition；
* 95% EVR rank；
* raw candidate gradient 在线乘低秩 projector；
* projected cosine；
* Eq. 17 的 per-example max。

论文对 MMLU 使用 (r=150) 是因为完整 target set 有 285 个样本；你们的 target draw 只有 64 或 80，因此继续使用预注册的 95% EVR 比硬设 150 更合理。([arXiv][1])

### C. 在现有 STEM80 上做一次对齐测试

只做 selection-only：

* paper-faithful raw GIST；
* 当前 GIST-JL-Norm。

比较：

* rank；
* score Spearman correlation；
* top-(K) Jaccard；
* subject/length distribution；
* wall-clock 和 storage。

判断规则：

* Jaccard 高，例如 (>0.8)，score correlation 也高：可以在主 pilot 中使用便宜版本，但必须注明 adaptation；
* 差异很大：主 baseline 必须使用 raw paper-faithful GIST；
* 不要用“和 DSMC overlap 低”证明正确。低 overlap 只能证明它不同，不能证明它是 GIST。

另外，validation 文档里的这句话也需要修改：

> `rank(G_val)=80` means all targets independent and (r=62) genuinely filters noise.

数值满秩只说明矩阵在当前 tolerance 下没有精确线性依赖，不代表样本在统计意义上“独立”；截断低能奇异方向也不自动证明那些方向是噪声。更准确的说法是：

> (G_{\rm val}) is numerically full row rank, while the leading 62 components explain 95% of its spectral energy.

## 3. target-draw protocol：框架很好，但还不能冻结

数据角色分配是正确的：

* validation → target draws；
* dev → 5-shot demonstrations；
* test → evaluation。

标准 `cais/mmlu` 的整体 split 是 dev 285、validation 1,531、test 14,042，因此 protocol 中写的 “18,738 test items” 是错误的。应该删除硬编码数字，改为从当前 lm-eval 版本实际输出中记录 STEM 和 Humanities 各自的 evaluation item count。([Hugging Face][2])

### 我建议把主 target size 从 80 改成 64

现在 (n_T=80) 时，两个方向之间必须共享一部分 STEM validation examples，所以只有“方向内不重叠”，十个 draws 并不全局独立。

若改成：

[
n_T=64,\qquad
51\text{ majority}+13\text{ minority},
]

五个 STEM-majority draws 和五个 HUM-majority draws总共分别需要：

[
5(51+13)=320
]

个 STEM 和 320 个 Humanities examples。

你们已有：

[
335\text{ STEM},\qquad 518\text{ Humanities}.
]

于是 **十个 draws 可以全局完全不重叠**。

这有几个优势：

* target draw 真正成为近似独立统计单位；
* direction interaction 的分析更干净；
* (n=64) 本来就在计划的 target-size axis 中；
* 现有 (n=80) 结果仍可作为 preliminary/mechanism experiments。

51/13 对应 79.7%/20.3%，与 80/20 的差异可以忽略，但要在 protocol 中精确写明，而不是称作严格 64/16。

如果坚持 (n=80)，也可以，但必须：

* 分方向分别分析；
* 不把十个 draws 当成十个相互独立样本；
* 报告跨方向 overlap；
* 对 combined CI 保持非常克制。

我更推荐现在切换到 (n=64)。

### 抽样流程也需要改一处

protocol 同时写了：

* 每个 draw 有独立 seed；
* 将一个 shuffled reservoir partition 为五个互斥 blocks。

这两种描述容易冲突。要保证严格不重叠，应当**一次性联合生成全部十个 draws**：

1. 一个 master seed；
2. 分别 shuffle STEM/HUM reservoir 一次；
3. 一次性分配全部 block；
4. 最后为每个 draw 写出 ID、subject composition 和 hash。

不要让每个 draw 各自独立调用随机采样，然后再依靠后处理避免重复。

## 4. pilot 方法数量现在写错了

protocol 说“7 method-rows”，但第三项写的是：

> LESS (+ true First-RR)

LESS 和 First-RR 是两个不同 selector：

* LESS：通常是对 target relevance 聚合后 top-(K)；
* First-RR：逐 query 轮流选择当前最相似的未选 candidate。

因此若两者都跑，实际是 8 个方法：

1. DSMC；
2. Second-RR；
3. LESS；
4. First-RR；
5. GIST；
6. NICE；
7. Random-K；
8. Random-K-LengthMatched。

我建议保留 8 个。First-RR 和 Second-RR 是 selector attribution 的重要对照，不应该被一个括号隐藏掉。

## 5. Random baseline 还要再明确随机性

当前 protocol 说 Random-K adapter 可以跨 target draws 复用。

完全复用同一个 random subset 会低估 Random selection 的方差。更好的安排是：

* 每个 draw index (d) 使用不同的 Random subset seed；
* 相同 draw index 的 STEM/HUM 两个方向可复用同一个 Random-K adapter，因为 training seed 也相同；
* 不同 draw index 不复用。

例如：

```text
draw 0: random subset seed 2000
draw 1: random subset seed 2001
...
draw 4: random subset seed 2004
```

这样只需要 5 个 Random-K adapters，而不是 10 个，同时可以估计 random-subset variation。

training seeds 也建议使用五个不同值：

```text
42, 1, 2, 3, 4
```

而不是：

```text
42, 1, 2, 42, 1
```

配对设计会控制 training-seed nuisance，没有必要重复同一个 seed。

“代表性 draw”也必须预先指定，比如固定 draw 0；不要看完 downstream 结果后再选择“代表性”draw 补 seeds，否则容易产生 post-hoc selection。

## 6. True-RR 现在要不要继续做

**要，立即继续实现。**

它与 target protocol 和 GIST fidelity 修复可以并行，没有必要停。

实现完成后至少需要检查：

* First-RR similarity：
  [
  x^\top t;
  ]
* Second-RR similarity：
  [
  (x^\top t)^2;
  ]
* 每个 target query 每轮最多选一个未选 candidate；
* target query order 固定并写入 metadata；
* 无重复 indices；
* budget 超过 valid candidates 时 fail loud；
* 相同输入与 seed 完全确定性；
* chunked similarity 与完整矩阵实现，在小 synthetic case 上输出完全相同；
* 若 target order 被打乱，记录其对 selection Jaccard 的敏感性。

Round-robin 对 target 顺序存在路径依赖。建议每个 draw 使用由 draw seed 决定的 target permutation，并在所有 First-RR/Second-RR 方法中共享该顺序。

## 7. 建议的执行顺序

现在不要启动 SFT pilot。按下面顺序推进：

1. **Claude 立即实现 true First-RR/Second-RR。**
2. **重新打开 GIST gate**：对照官方 repo，增加 raw paper-faithful GIST；当前版本改名为 GIST-JL-Norm。
3. **修改 protocol**：

   * test count 修正；
   * 推荐 (n_T=64)，十个 draws 全局不重叠；
   * 联合生成 draws；
   * 明确 8 个方法；
   * 五个不同 training seeds；
   * Random subset seeds 随 draw 变化。
4. 只生成 target JSON/meta/overlap matrix，不计算 gradients，先 review。
5. 对一个 draw 做全方法 **selection-only dry run**：

   * 每个方法恰好 (K) 个唯一 indices；
   * selection overlap；
   * token/length histogram；
   * subject composition；
   * runtime/memory；
   * manifest hashes。
6. dry run 通过后，再跑：
   [
   2\text{ directions}\times2\text{ draws}
   ]
   的 SFT pilot。
7. pilot 只负责检查方法排序和 pipeline，不作统计显著性结论；通过后扩展到五个 draws。

可以直接回复 Claude：

> Build the true First-RR and Second-RR selectors now, but do not freeze or launch the pilot yet. Re-open the GIST gate: an official GIST repository now exists, and the current implementation is a normalized 8192-D JL adaptation rather than an exact paper/code reproduction. Rename it GIST-JL-Norm, implement a raw-gradient paper-faithful GIST baseline, and compare the two on STEM80 selection-only. Revise the protocol to use globally disjoint n=64 draws if feasible, fix the MMLU test count, define eight method rows explicitly, use five distinct training seeds, and vary the Random-K subset seed by draw. Generate the target draw manifests and overlap matrix first and bring them back for review before any gradients or SFT.

最终状态是：

> **true-RR：继续。GIST：暂未真正过 gate。Protocol：接近可用，但需要一轮修订后再冻结。Pilot：现在不要启动。**

[1]: https://arxiv.org/pdf/2602.18584 "GIST: Targeted Data Selection for Instruction Tuning via Coupled Optimization Geometry"
[2]: https://huggingface.co/datasets/cais/mmlu?utm_source=chatgpt.com "cais/mmlu · Datasets at Hugging Face"

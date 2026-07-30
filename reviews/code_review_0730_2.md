我复核了 `0bcf999`。结论是：

> **这版已经非常接近可冻结，但现在仍不应批准“freeze + generate draws”。有一个真实的分配算法错误需要先修；GIST 的代数判断正确，但“fidelity 已彻底解决”的措辞仍然过强。**

true-RR 可以视为通过，协议的总体设计也成立。

## GIST：这次代数修正是正确的

Claude 对

[
P=G^\top U_k\Sigma_k^{-1}=V_k
]

的修正是对的。若

[
G=U\Sigma V^\top,
]

则

[
G^\top U_k\Sigma_k^{-1}
=======================

# V\Sigma U^\top U_k\Sigma_k^{-1}

V_k.
]

因此官方 Gram–eigendecomposition–whitening 形式，本质上是在求 target gradient matrix 的 top-(k) 右奇异子空间；这不是 cosine 抵消 whitening，而是线性代数恒等式。仓库文档现在对此表述正确。

在 (k=M)、target matrix 满行秩，且每个 target gradient 只乘正标量的条件下，行空间不变，因此 raw norm 与 unit norm 确实给出同一个 target row space。投影基最多发生正交旋转，projected cosine 不变。因此在同一个 8192-D feature space 内，不必仅为恢复 target norm 重新计算梯度。

但这个结论需要在每个新 draw 上验证：

* numerical rank 是否等于 (M=64)；
* 最小非零奇异值和 condition number；
* (|P^\top P-I|)；
* normalized/rescaled sanity check。

否则当 target matrix 近似秩亏时，Gram 矩阵会平方 condition number，代码中的 (10^{-6}) 正则也可能使“严格等价”变成数值近似。

## GIST fidelity 还没有完全解决

Claude 说“唯一剩余差距只是 raw LoRA space 与 8192-D JL space”，这仍不完全准确。

当前 baseline 至少还有两个 adaptation：

1. raw LoRA gradient space → 8192-D JL/TRAK projected space；
2. 官方 GIST 的 candidate 与 target 使用同一种 raw-gradient geometry，而当前实验沿用的是 LESS-aligned 的 **Adam-preconditioned candidate / SGD target** 协议。

第二点不仅是梯度长度缩放，而可能改变梯度方向，所以不会被 cosine 自动抵消。当前比较对所有方法使用相同 cache，作为 controlled comparison 是合理的；但它不应叫 byte-faithful 或 exact GIST。

建议正式命名为：

> **GIST-SharedProj**
> GIST scoring algorithm on the shared LESS-aligned projected-gradient representation.

不要写：

> faithful GIST / exact official GIST.

官方 GIST 的核心是从 validation gradients 中通过 spectral filtering 恢复低维 task subspace，并在其中评估 candidate alignment；论文还将低存储、低计算作为主要贡献。你当前版本只复现了 scoring algorithm，不复现其完整 gradient representation 与效率路径。([arXiv][1])

### 还有一个 rank 问题

新 target size 是 (M=64)，官方默认 `target_dim=150` 会变成：

[
k=\min(150,64)=64=M.
]

这意味着 main GIST baseline 使用完整 target row span，**没有任何 spectral truncation**。但 spectral filtering 正是 GIST 方法故事的核心组成部分。([arXiv][1])

这不是实现错误，却可能形成一个偏弱或失真的 baseline。建议在新 target draws 之前，用已经完成的旧 STEM80/HUM80 作为 development settings，比较：

[
\text{GIST-SharedProj-full}: k=M,
]

和

[
\text{GIST-SharedProj-EVR}: k=\min{r:\mathrm{EVR}(r)\ge0.95}.
]

每个方向各跑一个 seed，共 4 次 SFT。按两个方向的平均 balanced accuracy 选择一个**全局 rank rule**，然后冻结，不在新 draws 上再调。这样既保留官方 default，也避免故意用一个关闭 spectral filtering 的弱 baseline。

## 真正的 blocker：subject allocation 表没有分配满 320 条

协议要求每个 domain 在十个 draw 中一共消耗：

[
5\times51+5\times13=320.
]

但 feasibility JSON 使用：

```text
need = round(w_s × 320)
```

对每个 subject 独立取整，导致 STEM：

```json
"need_total": 318
```

而不是 320。

随后两个短缺 subject 又少 3 条：

* college chemistry：短 2；
* high-school computer science：短 1。

如果只做文档所说的“cap and redistribute shortfall”，总数会从 318 降到 315，再补回 3，最终仍只有 318；还缺两条。当前 feasibility table 因而不能作为 draw generator 的准确规格。

这是冻结前必须修复的问题。

### 应使用联合整数分配，而不是逐 subject round

最干净的定义是对每个 domain 求一个整数矩阵：

[
a_{b,s}\in\mathbb Z_{\ge0},
]

其中 (b) 是十个 domain blocks——五个大小 51、五个大小 13，(s) 是 subject。约束：

[
\sum_s a_{b,s}=B_b,
]

[
\sum_b a_{b,s}\leq \operatorname{valCount}_s,
]

同时最小化：

[
\sum_{b,s}
\left(a_{b,s}-B_bw_s\right)^2
]

或绝对偏差。

这可以用 min-cost flow、MILP，或者确定性的 capped largest-remainder + repair 实现。必须满足：

* 每个 block 精确为 51 或 13；
* 每个 domain 总数精确为 320；
* subject 不超过 validation reservoir；
* 十个 draw 的 example IDs 全局不重复；
* deterministic tie-breaking；
* 输出期望 quota、实际 quota、绝对偏差和总变差距离。

lm-eval 默认的 MMLU group aggregation确实是按 subtask 文档数量加权的 micro-average，所以用 test-doc proportions 作为 domain 内 target weights 是合理的。([GitHub][2]) MMLU 官方数据卡也确认标准 split 总数为 dev 285、validation 1,531、test 14,042。([Hugging Face][3])

## Protocol 还有几个小修正

第一，开头仍写：

> “independent skewed target sets”

后面已经正确改成 “globally non-overlapping replicate units”。开头也应同步修改。

第二，`Random-K` 定义在第 8 节重复了两次。 删除一份即可。

第三，RR 的 `perm_seed` 还应正式进入 protocol：

* 每个 draw 使用固定的、由 draw ID 推导的 permutation seed；
* First-RR 与 Second-RR 在同一 draw 中使用完全相同的 target query order；
* metadata 记录完整 order。

当前 true-RR 代码的 top-(K)-per-query 内存优化是精确的，而不是近似：当全局只需选 (K) 个样本时，某个 query 不可能在全局尚未选满 (K) 前就需要访问自己第 (K+1) 名的 candidate。实现已由 (O(MN)) 降到 (O(MK))。

第四，GIST 文档中的 “faithful” 应改成 `algorithm-faithful on shared projected features`，并明确不报告官方论文的 storage/compute improvement 作为本实现的测量结果。

## 统计方案现在是合理的

每方向 (n=5) 的 exact two-sided sign test 最小可能值确实是：

[
2/2^5=0.0625.
]

所以把：

* per-draw paired differences；
* mean；
* median；
* win count；

作为主证据，把 bootstrap interval 仅作为 descriptive summary，是正确处理。

由于十个 draws 是一次有限 reservoir partition 产生的负相关 replicate，普通 bootstrap 也不能被理解为严格的独立同分布置信区间；当前“descriptive only”的限定已经足够。

## 现在应该怎么做

我的明确决定是：

> **暂不批准 freeze；再做一个很小的 protocol patch，然后即可 freeze + generate artifacts。**

让 Claude 完成：

1. 用联合整数分配重做 subject allocation，确保 STEM/HUM 都精确合计 320；
2. 输出完整 (10\times\text{subjects}) allocation plan，而不只是一张全局 `round(w×320)` 表；
3. 修复开头的 “independent”、重复 Random-K、RR permutation seed；
4. 将 GIST 主 baseline 改名为 `GIST-SharedProj`；
5. 在旧 STEM80/HUM80 上完成 (k=M) 与 EVR95 的 rank-rule development gate，冻结一个全局规则。

完成后可以直接：

* 将 protocol 状态改为 `FROZEN`；
* 生成 10 个 draw JSON、meta、allocation matrix 和 overlap matrix；
* **仍然不要生成 gradients 或运行 SFT**；
* 把这些 artifacts 带回来进行最后一次数据级 review。

可以直接回复 Claude：

> Hold the freeze for one final patch. The current STEM feasibility table is not an exact allocation: independently rounding `w_s × 320` sums to 318, so cap-and-redistribute still leaves two unassigned slots. Replace it with a deterministic joint integer allocation over the ten 51/13 blocks, enforcing exact block totals, exact domain total 320, subject-cap constraints, and minimum deviation from lm-eval micro weights. Also fix the remaining “independent” wording, duplicate Random-K paragraph, and preregister a shared per-draw RR permutation seed. Rename the main GIST row to GIST-SharedProj, since it is algorithm-faithful but uses JL-projected Adam-candidate/SGD-target features. Before freezing, use the existing STEM80/HUM80 settings to choose one global GIST rank rule between k=M and EVR95. After those changes, freeze and generate only the draw JSON/meta/allocation/overlap artifacts for review—no gradients or SFT yet.

**Review 状态：true-RR 通过；统计设计通过；GIST 数学修正通过但命名与 rank rule 待定；subject allocation 仍是唯一硬 blocker。**

[1]: https://arxiv.org/abs/2602.18584?utm_source=chatgpt.com "GIST: Targeted Data Selection for Instruction Tuning via Coupled Optimization Geometry"
[2]: https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/task_guide.md?utm_source=chatgpt.com "lm-evaluation-harness/docs/task_guide.md at main · EleutherAI/lm-evaluation-harness · GitHub"
[3]: https://huggingface.co/datasets/cais/mmlu/blob/main/dataset_infos.json?utm_source=chatgpt.com "dataset_infos.json · cais/mmlu at main"

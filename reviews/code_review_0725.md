结论：**Moment-MMD 的核心 kernel 和贪心公式是正确的，但当前实现还不能算“正确无误”**。我确认了 3 个需要先修的高优先级问题，以及若干实验接口风险。

我做的是源码静态审查，并按源码公式构造了小规模数值测试；对 (\alpha\in{0,0.25,0.5,1})，每一步选择都与直接枚举“加入哪个候选后 MMD 最小”一致。但这不是在你的 H100/DataFlex 环境里跑完整 selection→SFT→eval；当前提交也没有 GitHub Actions 运行记录。

## 核心算法正确的部分

你定义的 kernel 是

[
k_\alpha(u,v)
=============

\alpha\frac{1+\langle u,v\rangle}{2}
+
(1-\alpha)\langle u,v\rangle^2,
\qquad 0\le\alpha\le1.
]

源码实现与文档一致。

在单位化梯度上：

* 常数核、线性核和二次多项式核都是 PSD；
* 非负权重组合仍然是 PSD；
* (k_\alpha(u,u)=1)；
* kernel 值非负，因此也适合你后续的 submodular 分析。

而且它有一个非常干净的精确解释：

[
\operatorname{MMD}*{k*\alpha}^2(S,T)
====================================

\frac{\alpha}{2}
\left|
\mathbb E_S[u]-\mathbb E_T[u]
\right|_2^2
+
(1-\alpha)
\left|
\mathbb E_S[uu^\top]-\mathbb E_T[uu^\top]
\right|_F^2.
]

注意第一阶项前面是 (\alpha/2)，不是 (\alpha)。论文理论和 alpha 解释里最好明确写出这个比例。

贪心 score 也正确。代码维护

[
r_T(x)=\frac1{|T|}\sum_t k(x,t),
\qquad
r_S(x)=\sum_{s\in S}k(x,s),
]

并选择

[
r_T(x)-\frac{r_S(x)+k(x,x)/2}{|S|+1}.
]

这确实最小化当前前缀 (S\cup{x}) 的 biased empirical MMD。

第一步没有显式减去 (k(x,x)/2)，但因为所有有效向量都重新单位化，(k(x,x)=1) 对所有候选都是常数，所以不影响 argmax。这一点是正确的。

---

## 必须先修的三个问题

### 1. 有效候选不足时会重复选择索引 0

代码先令

```python
k = min(args.num_select, N)
```

但之后又排除了零梯度候选，并没有把 (k) 限制到有效候选数量。

当所有有效候选都已经选完时，`scores` 会全部变为 `-inf`，随后：

```python
b = int(torch.argmax(scores).item())
```

会再次返回索引 0，于是输出重复索引。

我用“1 个有效向量 + 2 个零向量、选择 3 个”的测试复现了：

```text
[0, 0, 0]
```

这属于真实的静默错误。建议不要悄悄缩小预算，而是直接失败：

```python
valid_count = int(avail.sum().item())
if args.num_select <= 0:
    raise ValueError("--num_select must be positive")
if args.num_select > valid_count:
    raise ValueError(
        f"Requested {args.num_select} samples, but only "
        f"{valid_count}/{N} candidates have valid gradients."
    )
k = args.num_select
```

同时在完成后 assert：

```python
assert len(selected) == len(set(selected)) == k
```

### 2. target cache 缺少有限性和零向量检查

当前代码直接加载、单位化 target gradients，然后计算矩阵和均值。

以下情况都会产生问题：

* target 为空：`mean` 得到 NaN；
* target 中含 NaN/Inf：全部候选的 relevance 可能变成 NaN；
* target 中含零向量：二阶 target relevance 被缩小，改变 relevance 与 redundancy 的相对权重；
* candidate 中含 NaN：当前会被 `avail` 间接排除，但日志会错误地把它当作“zero row”。

旧的 MMDSelector 已经意识到这个风险，甚至在 target 零向量超过 50% 时直接抛错，因为这会让方法退化成纯 diversity selection。 新脚本应至少保持同等防护。

建议先检查再归一化：

```python
if X.ndim != 2 or Tg.ndim != 2:
    raise ValueError("Gradient caches must be rank-2 tensors")
if X.shape[1] != Tg.shape[1]:
    raise ValueError(
        f"Feature dimension mismatch: {X.shape[1]} vs {Tg.shape[1]}"
    )
if X.shape[0] == 0 or Tg.shape[0] == 0:
    raise ValueError("Candidate and target caches must be non-empty")
if not torch.isfinite(X).all():
    raise ValueError("Candidate cache contains NaN/Inf")
if not torch.isfinite(Tg).all():
    raise ValueError("Target cache contains NaN/Inf")

x_norm = X.norm(dim=1)
t_norm = Tg.norm(dim=1)

if (t_norm <= 1e-12).any():
    raise ValueError(
        f"Target cache contains {(t_norm <= 1e-12).sum().item()} zero rows"
    )
```

对于 paper experiment，我倾向于 target 出现任何零向量都 fail loudly，而不是自动删除。

### 3. 使用 candidate subsample cache 时，输出索引会映射错误

脚本直接把梯度矩阵的行号写入 `step_1.json`。

如果 cache 是通过 `candidate_subsample` 生成的，那么这些只是 subsample 内部的局部索引，并不是原 candidate dataset 的全局索引。原来的 MMDSelector 会读取 `subsample_indices.pt` 并映射回全局索引。

当前实验似乎使用的是完整 270k cache，所以现有 alpha sweep 大概率不受影响。但脚本作为通用工具时会静默选错数据。

建议增加：

```python
ap.add_argument("--subsample_indices", default=None)
```

输出前：

```python
output_indices = selected

if args.subsample_indices is not None:
    mapping = torch.load(args.subsample_indices, map_location="cpu")
    mapping = (
        mapping.tolist()
        if torch.is_tensor(mapping)
        else list(mapping)
    )
    if len(mapping) != N:
        raise ValueError(
            f"Subsample map has {len(mapping)} entries, expected {N}"
        )
    output_indices = [int(mapping[i]) for i in selected]
```

---

## 另外四个需要处理的风险

第一，`alpha` 没有检查。当前任何浮点数都会被接受。 当 (\alpha\notin[0,1]) 时，“凸组合、非负 kernel、Moment-MMD”这些性质都不再成立。应验证：

```python
if not np.isfinite(a) or not 0.0 <= a <= 1.0:
    raise ValueError("--alpha must be finite and in [0, 1]")
```

第二，`--seed` 完全没有使用，也没有写入输出 metadata。 当前算法是确定性的，完全可以删除这个参数；若想处理完全相同 score 的 tie，可以使用 seeded permutation，但需要把 permutation 映射回原索引。

第三，脚本没有验证两个 cache 是否真的兼容。只要维度相同，即使 candidate 和 target 使用了不同：

* projection seed；
* checkpoint；
* LoRA 参数集合；
* tokenizer/template；
* gradient preconditioner；
* projection matrix；

代码仍然会运行，但点积可能没有意义。至少应为每个 cache 保存 sidecar metadata，并验证 `proj_dim`、projection seed、checkpoint ID 和 feature definition。最终 `step_1.json` 里也应记录输入路径、SHA/hash、shape 和 alpha。

第四，当前 alpha sweep 并不是完整的可复现 pipeline。selector 只生成 `step_1.json`， 训练 driver 却直接假定 `moment_a*.jsonl` 已经存在并开始 SFT。 `dataset_info.json` 也指向这些手工生成的 `.jsonl` 文件。

现有 exporter 输出的是 `selected_subset.json`，不是这些 `.jsonl` 路径。

建议把下面三步都写进同一个 driver：

```text
select_moment_mmd.py
→ 映射并导出 selected subset
→ 注册/生成 dataset config
→ SFT + eval
```

这样 clean checkout 后才能一条命令复现。

---

## 方法表述上需要特别小心

### (\alpha=1) 并不等于 LESS

源码文档写的是 “LESS-like, but MMD”，这是准确的。

因为即使 (\alpha=1)，代码仍然包含 selected-set redundancy 项 `ksum`。 LESS 是对每个候选独立计算与 target mean gradient 的相似度再 top-k；Moment-MMD 的 (\alpha=1) 是 linear-kernel herding / first-moment coreset。

因此论文里建议写：

> (\alpha=1) recovers first-order gradient-mean matching with coreset repulsion.

不要写：

> (\alpha=1) exactly recovers LESS.

### (\alpha=0) 可以视为 GradCov endpoint

在相同、单位化 cache 和相同 prefix greedy 下，(\alpha=0) 确实退化为

[
k(u,v)=\langle u,v\rangle^2.
]

所以你观察到当前 cache 上 100% selection overlap 是合理的。但因为新脚本会再次单位化并把 self-kernel 硬编码为 1，而原实现按实际范数计算，最好表述为：

> exactly equivalent under unit-normalized features; 100% overlap on our caches.

---

## 训练 driver 的一个协议检查

共享 SFT config 的说明和默认值是 LoRA alpha 256、训练 3 epochs。

Moment sweep 则覆盖为 LoRA alpha 512、训练 4 epochs。

这不一定是错误——很可能是在对齐你之前的 skew 实验——但必须确认 LESS、NICE、TSDS、GradCov 的 skew baselines 也使用完全相同的覆盖参数。否则新的 alpha sweep 不能直接和旧表比较。

另外 shell 使用 `set -uo pipefail` 而不是 `set -euo pipefail`，训练失败后仍会继续，最后依旧打印 “COMPLETE”。   建议记录失败 alpha 并最终返回非零退出码。

## 建议先补的最小测试集

在正式跑更多昂贵实验前，先加入 `tests/test_moment_mmd.py`：

1. Gram matrix 在 (\alpha\in[0,1]) 时 PSD，且 (k(u,u)=1)。
2. 数值验证 MMD 等于“一阶均值差 + 二阶矩差”的闭式表达。
3. 小规模下逐步 greedy 与直接重算 MMD 的 argmin 完全一致。
4. (\alpha=0) 与 GradCov 选择一致。
5. (\alpha=1) 与 linear-MMD 一致，但明确不要求与 LESS top-k 一致。
6. NaN、Inf、空 target、零 target、负预算、超出有效候选数时必须抛错。
7. subsample local-to-global mapping 正确。
8. CPU/GPU 输出一致，允许浮点 tie 时使用 objective equality 而非死比较索引。

综合来看：**算法核心可以继续跑，但建议先修“有效候选耗尽、target cache 验证、subsample 映射”这三个问题，再把新的实验结果用于论文。**

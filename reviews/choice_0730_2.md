我复核了 `ea5c54c`。整体推进是对的：true-RR 已经是实质性的 selector，protocol 也比上一版严谨很多。但两个界面里，我建议这样选：

> **GIST fidelity：选 4“Type something”——做官方 raw-gradient GIST，但流式处理 candidate，不保存 270k raw cache。**
> **Freeze protocol：选 2“Hold, review first”——再改三处后冻结。**

## 一、GIST：Claude 的新解释还有一个数学误区

当前 `select_gist_faithful.py` 对官方流程的代码实现基本合理：

[
K=GG^\top,\qquad
P=G^\top U_k\Sigma_k^{-1},
]

再将 target 和 candidate 投影到 (P) 后做 cosine/max/top-(K)。 官方仓库也确实先用 raw LoRA gradients 构造 Gram 矩阵和低维投影，再流式投影训练样本，而不是保存完整 raw candidate cache。 GIST 的公开论文将方法描述为从 validation gradients 中恢复低维任务子空间，然后根据 candidate 与 target 方向的对齐评分。([arXiv][1])

但下面这句话不准确：

> whitening 在 normalized caches 上才是 no-op，因为 cosine 抵消了 (S^{-1})。

实际上，设

[
G=U\Sigma V^\top,
]

那么对任意输入矩阵——无论是否逐样本归一化——都有

[
P
=

# G^\top U_k\Sigma_k^{-1}

V_k.
]

所以官方的 Gram–eigendecomposition–(S^{-1}) 构造，与直接取 target matrix 的 top-(k) 右奇异向量，**本来就严格等价**。不是 cosine 后来抵消了 whitening，也不是 normalized cache 才导致等价。

这正好解释了为什么：

[
\text{faithful}(r=62)
\equiv
\text{JL-Norm}(r=62)
]

得到 Jaccard 1.0。这是应当出现的代数恒等关系，不是一个特殊的经验发现。

真正会改变结果的是：

1. raw target gradients 与逐样本 normalized target gradients产生的截断 top-(k) 子空间不同；
2. raw LoRA 空间与 8192-D JL 空间不同；
3. rank (k) 不同；
4. paper/code 的 target aggregation 细节不同。

## 二、为什么不建议选 1：只重提取 raw target

这里有一个更关键的点。

新 protocol 使用 (n_T=64)，官方默认 `target_dim=150` 会被截断为：

[
k=\min(150,64)=64.
]

如果 target matrix 数值满秩，那么 (k=M=64) 是完整 target row space，不再做谱截断。逐行乘以非零 norm：

[
G_{\mathrm{raw}}
\longrightarrow
D^{-1}G_{\mathrm{raw}}
]

不会改变 row space；每个 target 的尺度又会被最终 cosine 消掉。

因此，在相同 8192-D JL 空间、(k=64) 的条件下：

> **仅重新提取未归一化 target cache，大概率不会改变 GIST selection。**

只有使用 (k<64)，例如 95% EVR rank 时，target norm 才会通过改变谱排序影响 top-(k) 子空间。但这又不再是官方固定 `target_dim=150` 在 (M=64) 下的直接设定。

所以 option 1 既不 byte-faithful，也很可能没有实际信息增量。

## 三、推荐的 GIST 选择：官方 raw-gradient 流式版本

最严谨的方案在概念上接近 option 2，但不要按界面描述“再保存一个巨大的 raw candidate cache”。

应当：

1. 提取 64 个 raw target LoRA gradients；
2. 按官方代码构造 raw-space (P)；
3. 流式遍历 270k candidates：

   * 对一个 candidate 计算 raw LoRA gradient；
   * 立即计算
     [
     z_i=g_iP;
     ]
   * 保存低维 (z_i) 或直接保存 score；
   * 释放 raw gradient；
4. 不将 (270k\times D_{\text{LoRA}}) raw matrix 落盘。

这正是官方实现强调的低存储路径。官方代码通过 target gradients 构造 (D\times k) 投影矩阵，然后逐个处理 training gradients，只保存低维表示。 论文也将低存储和低计算作为 GIST 的主要卖点之一。([arXiv][1])

因此在界面输入：

> Run the byte-faithful raw-gradient GIST, but implement it in the official streaming form: extract raw target gradients and build the raw-space projector P; then stream the 270k candidates through P and retain only k-dimensional projected features or final scores. Do not materialize or save a full raw candidate cache. Use the official fixed target_dim setting, with k=min(150,M), and keep the 95%-EVR variant only as a clearly labelled adaptation/ablation.

若必须选 1–3 中的一个，选 **2 Target + candidate (exact)**，但附加说明“candidate 必须流式，不保存 raw cache”。

主 pilot 中建议报告：

* `GIST-official-raw`：主 GIST baseline；
* `GIST-JL-Norm`：只作为 appendix 中的共享-cache adaptation。

这样 reviewer 无法质疑你故意用一个不忠实版本削弱最接近的 baseline。

## 四、true-RR：算法正确，但建议先做一个内存优化

当前 RR 的定义是正确的：

* 逐 target query 循环；
* 每次选择该 query 最相似的尚未选择 candidate；
* First-RR 使用 (x^\top t)；
* Second-RR 使用 ((x^\top t)^2)。

但代码为每个 query 保存全部 (N) 个 candidate 的 Python list：

```python
ranked.append(torch.argsort(sim, descending=True).tolist())
```

在 (M=64,N\approx270k) 时，大约是 1730 万个 Python integers，内存可能达到数百 MB，甚至接近 1 GB。

每个 query 实际只需要自己的 top-(K) candidates：

```python
ranked_j = torch.topk(sim, k=K, largest=True, sorted=True).indices.cpu().tolist()
```

这是**精确的**，不只是近似。因为在总共还没选择满 (K) 条时，不可能出现某个 query 的 top-(K) candidates 全部已被选中的情况；若它们全被选中，全局 selected count 已经至少是 (K)。

因此可以把存储从：

[
O(MN)
]

降到：

[
O(MK),
]

当前约减少 20 倍。开 pilot 前值得修。

## 五、Protocol：接近冻结，但先改三处

### 1. “globally disjoint”不等于“independent”

十个 draw 确实全局无重叠，这很好。

但它们来自同一个有限 reservoir，并通过一次 partition 联合生成，因此是负相关的，不是概率意义上的独立样本。一个 draw 使用了某个题目，会约束其他 draw 能用什么题目。

把：

> genuinely independent statistical unit

改为：

> globally non-overlapping replicate unit

统计单位仍然可以是 target draw，但不要声称严格独立。

### 2. 必须定义 category 内的 subject 分布

当前协议写：

> Subjects balanced round-robin within each block.

但 lm-eval 的 MMLU group score默认是按各 subtask 文档数量加权的 micro-average，而不是每个 subject 等权。([GitHub][2])

因此你需要明确潜在分布 (P^\star) 的完整定义：

* STEM/Humanities 之间 50/50；
* STEM 内各 subject 如何分配；
* Humanities 内各 subject 如何分配。

我建议在每个大类内部，按当前 evaluation split 的 subject document proportions 分配 target quota，然后用 constrained rounding 保证总数为 51/13，并检查 validation reservoir 是否足够。

若你希望每个 subject 等权，也可以，但那应明确写成：

> (P^\star) is balanced across the two domains and uniform over subjects within each domain.

同时 evaluation 也应额外报告一个 subject-macro 指标，否则 sampling distribution 与 primary metric 不完全一致。

更自然的是：

* 主指标继续用当前 lm-eval micro-average；
* target 的 category 内 subject composition 尽量匹配 evaluation micro weights；
* 附录报告 subject-macro robustness。

在冻结前，让 Claude生成一张 allocation feasibility table，确认每个 subject 的 validation 数量足以支持十个全局无重叠 draws。

### 3. 弱化 n=5 的 CI 表述

protocol 计划每方向只有 5 个 draw，并做 cluster bootstrap。

5 个 cluster 的 bootstrap CI 会非常粗糙，不适合承担“统计显著性”结论。建议预注册：

* 每个 draw 的完整 paired difference；
* mean、median；
* win count；
* descriptive bootstrap interval；
* 不根据 (p<0.05) 作主判断。

每方向 5 个 paired draws 的 exact sign-flip 检验总共只有 (2^5=32) 种符号组合；即使五个差值全部同号，双侧最小 (p) 也约为 0.0625。因此当前设计天然更适合做 effect-consistency 证据，而不是显著性宣称。

## 六、Protocol 界面怎么选

现在选择：

> **2. Hold, review first**

并回复 Claude：

> Hold before freezing. The protocol is close, but revise three points first: replace “independent” with “globally non-overlapping replicate units”; explicitly define and preregister the within-STEM/within-Humanities subject distribution, preferably matching lm-eval’s micro-averaged evaluation weights, and provide a per-subject feasibility/allocation table; treat the n=5 clustered bootstrap as descriptive and make paired differences, median, and win counts the primary stability evidence. Also specify that the main GIST baseline is the official raw-gradient streaming implementation with fixed k=min(150,M), while EVR and JL-normalized versions are labelled adaptations. Then freeze and generate only the JSON/meta/overlap artifacts for review.

这些改完后，可以直接 freeze + generate draws，不需要再进行第三轮大改。

## 最终决策

GIST fidelity：

> **选择 4，自定义“option 2 的流式精确实现”**。不要只提取 raw target，也不要把 normalized-cache 版本冒充正式 GIST。

Freeze protocol：

> **选择 2，先 hold**。修正“独立性”、subject sampling distribution 和 n=5 统计措辞后，再冻结并生成 draws。

当前项目状态已经很接近正式 pilot：DSMC 已冻结，Moment-MMD 分支已关闭，2×2 attribution 已完成，true-RR 已有正确实现。剩余真正的 launch blockers 只有：

1. official raw GIST streaming；
2. subject-level target allocation protocol；
3. draw manifests 的最终复核。

[1]: https://arxiv.org/abs/2602.18584?utm_source=chatgpt.com "GIST: Targeted Data Selection for Instruction Tuning via Coupled Optimization Geometry"
[2]: https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/task_guide.md?utm_source=chatgpt.com "lm-evaluation-harness/docs/task_guide.md at main · EleutherAI/lm-evaluation-harness · GitHub"

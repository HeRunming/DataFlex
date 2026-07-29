这两个问题应该分别这样选：

> **Decision 1：不要选现有 1–3，选 “Type something”，改用 MMLU `validation` split 作为 target-draw reservoir。**
> **Decision 2：方向上选 1“立即跑 gate”，但当前代码先修一个计算复杂度问题，并把 RR 的命名改掉，再启动。**

## Decision 1：Target draws 用 MMLU validation，不要动 test

Claude 漏掉了 MMLU 本身还有一个独立的 `validation` split。

公开的 `cais/mmlu` 数据结构是：

* `dev`：285 条，每个 subject 固定 5 条，主要用于 few-shot demonstrations；
* `validation`：1,531 条；
* `test`：14,042 条；
* `auxiliary_train`：99,842 条。([Hugging Face][1])

原始 MMLU 论文也明确说，数据被划分为 few-shot development、validation 和 test；dev 是每个 subject 固定 5 个样例。([ResearchGate][2])

所以最干净的协议是：

* **validation：用于产生多个 target/query draws；**
* **dev：继续作为 5-shot evaluation demonstrations；**
* **test：完全不触碰，继续作为官方 evaluation。**

这比 Claude 给出的三个方案都更好：

### 为什么不选 1：Partition test

即使 selection target 与 held-out evaluation test subset 严格不重叠，整个实验已经使用了官方 test pool 来设计选择策略。它会：

* 失去与此前完整 MMLU test 结果的直接可比性；
* 增加 reviewer 对 test-set adaptation 的疑虑；
* 需要自定义 evaluation subset；
* 让论文解释复杂很多。

除非 validation 真的不够，否则没有必要。

### 为什么不选 2：Overlapping dev draws

大量重叠会让 5 个 draw 并不是 5 个近似独立的统计单位。你可以披露 overlap，但有效样本量会明显小于 5，cluster bootstrap 也会显得证据偏弱。

### 为什么不选 3：立刻改成 (n=32)

(n=32) 本身值得实验，但它同时改变了：

* target draw；
* target size；
* gradient-estimation noise。

这样第一轮 external-validity 结果无法判断差异来自“独立 draw”还是“target 更小”。应先在原来的 (n=80) 上保持连续性，再把 (n=32) 作为 target-size axis。

### 具体让 Claude 怎么做

选 “Type something”，回复：

> Use the MMLU validation split as the target-draw reservoir. Keep dev exclusively for the fixed 5-shot demonstrations and keep the full test split untouched for evaluation. First count the exact STEM/Humanities validation examples under the same subject mapping used by lm-eval. Construct five disjoint draws within each skew direction if counts allow; overlap between opposite skew-direction families may be allowed but must be recorded. Bring back the exact counts, draw IDs, subject compositions, and overlap matrix before generating gradients.

这里有一个细节：5 个 STEM-majority draws 和 5 个 HUM-majority draws 如果要求 **10 组之间全局完全不重叠**，可能仍然受到某一类别数量限制。因此预注册成：

* STEM80 的 5 个 draws 在本组内部互不重叠；
* HUM80 的 5 个 draws 在本组内部互不重叠；
* 两个方向之间允许部分 overlap；
* 完整报告 (10\times10) overlap matrix。

统计比较主要在同一方向内进行，这样是合理的。

`auxiliary_train` 不推荐作为 target reservoir，因为它是由 ARC、OpenBookQA、RACE 等其他来源汇成的辅助训练题，不等同于 MMLU 的同分布 validation questions。([Hugging Face][3])

---

## Decision 2：Gate 应该立即跑，但当前代码不能直接按 “RR” 名义运行

选择界面上应该选：

> **1. Launch gate now**

它和 target-draw protocol 没有依赖关系，而且确实可能改变论文 attribution story。最新 targeted-selection 系统研究也明确要求分离 representation 与 selection algorithm，并发现 greedy round-robin 在低预算下经常较强。([arXiv][4])

但 `aaf6747` 中当前实现有两个问题。

### 问题一：它不是 Round-Robin

代码明确做的是：

```python
rel = ...
sel = torch.topk(rel, K).indices
```

即对所有 target 求平均 relevance 后直接 top-(K)。

真正的 greedy RR 定义是：

> 对每一个 query sample，选与该 query 最相似的尚未选择 candidate；然后在全部 query 上循环，直到预算耗尽。([arXiv][4])

所以当前四格实际上是：

| Representation | Relevance Top-K | MMD coreset |
| -------------- | --------------: | ----------: |
| First-order    |      First-TopK |  Linear-MMD |
| Second-order   |     Second-TopK |        DSMC |

这是一个**有效而且干净的 attribution gate**，能回答：

> DSMC 的收益来自 second-order representation，还是来自 MMD 的 repulsion/diversity？

但它不能叫 First-RR / Second-RR。

我前面使用过“relevance/RR”的宽泛写法，这里需要正式纠正：**Top-K 和 RR 是不同 selector。**

建议：

* `First-RR` → `First-TopK`；
* `Second-RR` → `Second-TopK`；
* 文件名可以保留 `select_relevance_topk.py`，这个名字本身是正确的；
* 后续 external-validity baseline 再实现真正的 First-RR 和 Second-RR。

### 问题二：Second-TopK 当前实现计算量极不合理

现在代码计算：

```python
MT = (Tg.T @ Tg) / M       # 8192 × 8192
rel = ((X @ MT) * X).sum(1)
```

数学上没错，但代价是：

[
O(ND^2),
]

其中：

[
N\approx270,679,\qquad D=8192.
]

`X @ MT` 本身就是极其巨大的矩阵乘法，并产生约 (270679\times8192) 的中间 tensor，FP32 大约 8.9 GB；计算量也远远没有必要。仅仅“编译通过”不代表它适合运行。

利用恒等式：

[
x^\top
\left(\frac1M\sum_t t t^\top\right)x
====================================

\frac1M\sum_t(x^\top t)^2,
]

应改成：

```python
XT = X @ Tg.T
rel = XT.square().mean(dim=1)
del XT
```

复杂度变为：

[
O(NDM),
]

其中 (M=80)，比 (O(ND^2)) 小约 (8192/80\approx102) 倍，中间矩阵只有 (270679\times80)，约 87 MB FP32。

First-order 也可以统一写成：

```python
XT = X @ Tg.T

if args.order == "first":
    rel = XT.mean(dim=1)
else:
    rel = XT.square().mean(dim=1)
```

这与当前数学定义完全等价。

还应顺手补上：

```python
if X.shape[0] == 0 or Tg.shape[0] == 0:
    raise ValueError("candidate and target caches must be non-empty")
```

## Gate 跑完怎么解释

修完后立即跑现有 STEM80/HUM80、seed 42 的 4 个新增 SFT 是合理的。

实际四格应命名为：

[
\begin{array}{c|cc}
& \text{Top-K relevance} & \text{MMD coreset}\
\hline
\text{first-order}
& \text{First-TopK}
& \text{Linear-MMD}\
\text{second-order}
& \text{Second-TopK}
& \text{DSMC}
\end{array}
]

结果解释：

* **Second-TopK 与 DSMC 都优于 first-order 两格**
  → 核心贡献主要来自 second-order representation。

* **DSMC 明显优于 Second-TopK，而 first-order 两格接近**
  → MMD coreset diversity 是关键，second-order representation 与 selector 存在 interaction。

* **Second-TopK 优于 DSMC**
  → 当前 headline method 可能需要改为 second-order relevance selector，不能继续大规模跑 DSMC。

* **Linear-MMD 优于 First-TopK，同时 DSMC 优于 Second-TopK**
  → MMD diversity 对两种 representation 都有一致收益。

这个 gate 后面仍需真正的 RR baseline。最新论文的 RR 是逐 query 循环选择，而不是 mean-relevance top-k；它在低预算下表现较强，因此主论文不能只用 TopK 冒充 RR。([arXiv][4])

## 最终选择

对 Claude 的两个界面分别选择：

1. **Target draws：选 “Type something”**，要求改用 MMLU `validation` split，并先返回精确类别计数和 overlap 方案。
2. **2×2 gate：选 “Launch gate now”**，但先把 second-order relevance 改为 `((X @ Tg.T)**2).mean(1)`，并将 First-RR/Second-RR 改名为 First-TopK/Second-TopK。

可以直接回复 Claude：

> For target draws, use MMLU validation rather than dev or test: dev remains the fixed 5-shot source, validation becomes the query-draw reservoir, and the full test split remains untouched. Count the exact validation examples by the same STEM/Humanities mapping and draft five within-direction disjoint draws with a full overlap matrix.
>
> Launch the 2×2 gate after two fixes: rename the relevance cells First-TopK/Second-TopK because the current implementation is not greedy round-robin, and compute second-order relevance as `(X @ Tg.T).square().mean(1)` instead of materializing the 8192×8192 moment matrix. Then run the four new SFTs now.

[1]: https://huggingface.co/datasets/cais/mmlu/blob/main/dataset_infos.json "https://huggingface.co/datasets/cais/mmlu/blob/main/dataset_infos.json"
[2]: https://www.researchgate.net/publication/344159828_Measuring_Massive_Multitask_Language_Understanding "https://www.researchgate.net/publication/344159828_Measuring_Massive_Multitask_Language_Understanding"
[3]: https://huggingface.co/datasets/tnguyen1114/mmlu/blob/main/README.md "https://huggingface.co/datasets/tnguyen1114/mmlu/blob/main/README.md"
[4]: https://arxiv.org/abs/2602.14696 "https://arxiv.org/abs/2602.14696"

现在的两个选择，我建议：

> **Rank gate：选 2，固定 (k=M)，跳过下游训练 gate。**
> **Protocol freeze：选 3“Type something”，做一个很小的文档清理后直接冻结并生成 artifacts。**

这次 allocation 的硬错误已经修好；真正不应该执行的是当前 `run_gist_rankrule_gate.sh`。

## 为什么不要运行 GIST rank gate

脚本用旧 STEM80/HUM80 target 训练四个 adapter，然后仍然在：

```text
mmlu_stem,mmlu_humanities
```

这套官方 test evaluation 上比较 balanced accuracy，并按测试表现选择 rank rule。

而后续 pilot 仍然使用同一批 MMLU test items 作为最终评测。即使 target draws 换成了 validation split，**用 test accuracy 选择 GIST rank 仍然属于 test-set hyperparameter tuning**。这会给 reviewer 留下一个很明显的问题：

> 为什么 GIST 的超参数是在最终测试集上挑的，而 DSMC 已经冻结？

此外，这个 gate 只有：

* 两个旧 target sets；
* 一个 training seed；
* 四次 SFT。

它很容易根据约 0.5–1 个点的训练波动选错规则。

GIST 论文确实依赖 SVD spectral filtering，并报告在其完整 MMLU validation-gradient setting 中，rank 150 大约捕获 95% 的谱能量。([arXiv][1]) 但在你现在的 (M=64) setting 中，官方 `target_dim=150` 自然截断为

[
k=\min(150,64)=64=M.
]

因此最可辩护的主 baseline 是：

> **GIST-SharedProj-Full，固定 (k=M)，因为这是官方固定 target dimension 在当前 target size 下的直接结果。**

EVR95 可以作为预先声明的 sensitivity/appendix variant，但不要根据 MMLU test performance 从两者中挑赢家。

所以 Rank gate 界面选择：

### 2. Default to (k=M), skip gate

并补一句：

> Keep EVR95 as a predeclared sensitivity ablation; do not use downstream MMLU test accuracy to choose the main rank rule.

这样既避免测试集调参，也避免为了 baseline 继续消耗 5 小时。

## Allocation 硬 blocker 已修复

新脚本现在确实保证：

[
\sum_s a_{b,s}=51\ \text{或}\ 13,
]

每个 domain 总数：

[
5\times51+5\times13=320,
]

并满足：

[
\sum_b a_{b,s}\leq \mathrm{cap}_s.
]

代码先对每个 block 做 largest-remainder rounding，再把 over-cap subject 的单位转移给仍有容量的 subject。 最终还显式断言两个 domain 都等于 320、row sums 正确且 caps 满足。

生成的计划也显示：

* STEM total = 320；

* Humanities total = 320；

* STEM column-distribution TVD 约 0.0167；

* 所有 caps 和 block sums 均通过。

这部分已经可以用于 draw generation。

## 冻结前还要清理三处小问题

当前 protocol 第 3b 节仍保留了旧叙述：

> STEM `need 318`，然后 cap-and-redistribute。

但新方案已经是 joint allocation、精确 320。这段必须更新，否则 protocol 和实际计划互相矛盾。

建议改为：

> The deterministic joint allocator assigns exactly 320 STEM and 320 Humanities slots across the ten blocks. All block totals and reservoir caps are satisfied. The realized aggregate subject distribution differs from the ideal micro-weighted distribution by TVD = … .

同时：

1. 将旧的 `subject_allocation_feasibility.json` 删除，或明确改名为 `superseded_subject_allocation_feasibility.json`；
2. protocol 只引用新的 `subject_allocation_plan.json`；
3. 将分配脚本里的“minimize deviation”改为“deterministically keeps deviation small”。

因为当前 largest-remainder + greedy cap repair 没有求解或证明全局最优，只是一个合理、确定性的低偏差 heuristic。

这不影响当前 allocation 的可用性，只影响措辞的准确性。

## Subject weighting 方案可以通过

protocol 将潜在分布定义为：

* STEM/Humanities 两个 domain 各占 50%；
* domain 内 subject 按 test document 数量加权。

这与 lm-eval 的 group metric 一致：`weight_by_size=true` 时，对子任务按文档数量进行 micro-average；MMLU 默认也是将各 subject 的所有文档合并后计算总体 accuracy。([GitHub][2])

因此这个抽样目标定义是合理的。subject-macro 作为补充指标也应该保留。

## Protocol freeze 怎么选

不要选界面中的 1，因为它写的是：

> After the rank gate resolves...

而 rank gate 不应该运行。

也不需要再进行一次完整 protocol review。选择：

### 3. Type something

填入：

> Skip the downstream GIST rank-rule gate and freeze the main GIST-SharedProj baseline at the official default `k=min(150,M)=M`; keep EVR95 only as a predeclared sensitivity ablation, not a test-selected alternative. Before freezing, update §3b to replace the stale “need=318” feasibility text with the exact joint-allocation result (320 per domain), remove or mark the old feasibility JSON as superseded, and soften “minimizes deviation” to “deterministically keeps deviation small.” Then mark the protocol FROZEN and generate only the 10 draw JSON/meta files, the allocation matrix, reservoir counts, and overlap matrix. Do not generate gradients, selections, or run SFT until those artifacts are reviewed.

## 生成 artifacts 后需要检查什么

下一次 review 只需检查数据层，不需要再讨论方法方向：

* 10 个 JSONL 是否各有 64 条；
* 每组是否精确为 51/13；
* 所有 640 个 target IDs 是否全局唯一；
* subject counts 是否与 allocation matrix 完全一致；
* validation/dev/test 是否严格分离；
* prompt/template 是否与旧 target 格式相同；
* 每个 meta 中 seed、IDs、hash 和 training seed 是否完整；
* overlap matrix 非对角元素是否全部为 0；
* draw (d) 的 RR seed 是否为 (3000+d)，First/Second-RR 是否共享顺序。

最终选择：

* **Rank gate：2，跳过，主 GIST 固定 (k=M)。**
* **Protocol freeze：3，自定义“清理三处旧内容后立即冻结并生成 artifacts”。**

当前已经没有算法级 launch blocker；剩下只是 protocol 与 artifact 的一致性清理。

[1]: https://arxiv.org/abs/2602.18584 "GIST: Targeted Data Selection for Instruction Tuning via Coupled Optimization Geometry"
[2]: https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/task_guide.md?utm_source=chatgpt.com "lm-evaluation-harness/docs/task_guide.md at main · EleutherAI/lm-evaluation-harness · GitHub"

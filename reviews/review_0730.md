我复核了 `8e6ad2e`。结论是：

> **核心数据 artifacts 基本通过，但属于“条件通过”。暂时不要开始 target-gradient extraction；先修正两处 provenance 问题和一处错误声明。Pilot driver 可以同时开始搭建，但不能启动。**

## 已经确认正确的部分

生成脚本确实从 validation split 读取数据，按 subject 维护全局游标，每个样本只会被切片一次；每个 draw 强制检查 64 条、51/13 domain 比例和 draw 内 ID 唯一性。最后还断言 overlap matrix 非对角元素全为零。

仓库中的 overlap matrix 也确实为：

[
J(D_i,D_j)=0,\qquad i\ne j,
]

即 10 个 draw、640 个 target examples 全局没有 ID 重叠。

数据 split 的角色设计也是干净的。公开数据卡确认 `hails/mmlu_no_train` 是 `cais/mmlu` 去掉 `auxiliary_train` 后的副本；标准 MMLU 将 dev 用作每个 subtask 的 5-shot examples，并将 validation 与 test 分开。([Hugging Face][1])

按 test document count 设置 domain 内 subject 权重也与 lm-eval 的 MMLU group metric 对齐：`weight_by_size=true` 对各 subtask 做 micro-average，等价于拼接该 group 的全部 evaluation documents 后计算 accuracy。([GitHub][2])

因此，以下核心设计都已通过：

* validation → target；
* dev → evaluation demonstrations；
* test →最终评测；
* 每个 draw 精确 51/13；
* 十个 draw 全局不重叠；
* paired training seeds 和 RR seeds 已记录；
* allocation 的总量、block size 和 subject caps 均满足。

## 需要修正一：“所有 draw subject composition 相同”是错误的

Claude 的这句话：

> all 5 draws in a direction share the same subject composition

与 artifacts 不符。

例如 `stem80_draw0` 中：

* anatomy：1；
* college chemistry：0；
* high-school biology：7；
* high-school statistics：5。

而 `stem80_draw1` 中：

* anatomy：2；
* college chemistry：2；
* high-school biology：5；
* high-school statistics：3。

所以 draws 不只是“同一 subject quota 下替换不同 validation questions”，还包含一定的 **subject-composition variation**。

生成脚本的 docstring 也错误地说五个 majority blocks 和五个 minority blocks分别共享相同 subject counts。

这不是简单的实现疏忽。由于十个 draws 要全局不重叠，且部分 STEM subjects 的 validation reservoir 很小，要让五个 draw 完全重复同一套 subject quota，一些 subject 的总消耗必须是 5 的倍数。根据 allocation plan 中的 caps，将每个 subject 容量向下约束为 5 的倍数后，STEM 最多只能提供 300 条，而协议需要 320 条，因此严格相同的 subject composition 实际上不可行。

正确处理不是重新声称它们相同，而是：

> All draws have the same 51/13 domain skew and jointly follow the frozen micro-weighted allocation, but their exact subject compositions vary slightly because of integer and finite-reservoir constraints.

建议生成并提交：

```text
subject_composition_matrix.csv
subject_composition_tvd.csv
```

其中报告：

* 每个 draw 相对理想 (P^\star) 的 subject-level TVD；
* 同方向 draws 之间的 pairwise subject-composition TVD；
* 每个 subject 在十个 draws 中的总使用量。

这样 reviewer 能清楚看出 target variation 有多少来自题目本身，有多少来自 subject mix。

无需因此废弃现有 JSONL；但必须修正文档和分析解释。

## 需要修正二：meta 没有协议要求的 target data hash

冻结协议要求记录：

> target data sha256

但当前 meta 实际只有：

```json
"target_ids_sha256": "..."
```

而且这个 hash 是对：

```python
sha(sorted(ids))
```

计算的。

这有两个问题：

1. 它只验证 ID 集合，不验证 question、choices、answer 或 prompt 内容；
2. 排序后再 hash，无法检测行顺序变化。

第二点对 RR 尤其重要。RR 使用 `perm_seed` 产生的是**行位置排列**；同一 ID 集合若重新排序，虽然 `target_ids_sha256` 不变，实际 query visiting order 会改变。

每个 meta 至少补充：

```json
{
  "source_dataset": "hails/mmlu_no_train",
  "source_split": "validation",
  "source_snapshot_or_revision": "...",
  "generator_commit": "8e6ad2e...",
  "allocation_plan_sha256": "...",
  "target_file_sha256": "...",
  "ordered_target_ids_sha256": "...",
  "unordered_target_ids_sha256": "...",
  "prompt_schema": "mmlu_single_example_supervised_v1"
}
```

`target_file_sha256` 应直接对 JSONL bytes 计算；`ordered_target_ids_sha256` 应保持当前行顺序。`draws_index.json` 也应汇总每个文件的 hash，而不只是记录总 unique count。

由于 `hails/mmlu_no_train` 是 `cais/mmlu` 的无 auxiliary-train 副本，数据来源本身没有问题；关键是把实际使用的 snapshot/revision 固定下来。([Hugging Face][1])

## 需要修正三：这不是 “Hendrycks 5-shot template”

protocol 写道：

> the Hendrycks 5-shot template

但实际 target JSONL 每条只有：

* task description；
* 一个 validation question；
* 四个选项；
* 正确答案。

没有五个 dev demonstrations。

旧 target builder 也是同样的 single-example supervised format，因此当前新 draw 与过去实验是可比的。

这不是数据泄漏或代码错误，而是文档错误和一个需要公开说明的 modeling choice。建议改成：

> Target gradients use the same single-example supervised MMLU format as the existing target caches. Evaluation separately uses five dev demonstrations through lm-eval.

同时在 meta 记录：

```json
"target_num_fewshot": 0,
"evaluation_num_fewshot": 5
```

不要在这一步突然把五个 demos 加进 target prompts，否则会改变此前所有 mechanism experiments 的 target-gradient definition。target prompt 与 evaluation-context 是否应完全一致，可以以后作为独立 ablation，而不是现在临时修改主协议。

## 一个小的状态文档问题

协议顶部仍写着：

> Next step is artifact generation only

但 artifacts 已经生成。

可更新为：

> Draw artifacts generated and pending data-level approval; no gradients, selection, or SFT have been run.

这不影响实验。

# 下一步怎么做

不用完全停住工程。合理顺序是：

1. 修正 “same subject composition” 和 “5-shot target template” 两处表述；
2. 为每个 JSONL 增加内容 hash、顺序 hash、dataset revision 和 allocation-plan hash；
3. 输出 subject-composition matrix 与 TVD diagnostics；
4. 重新运行 artifact validator，确认 JSONL 内容本身没有变化，或明确记录新的 hash；
5. 同时开始构建 pilot driver；
6. driver 构建完先做一个 `stem80_draw0` 的八方法 **selection-only dry run**；
7. dry run 通过后，再启动 2 directions × 2 draws 的 SFT pilot。

Pilot driver 可以现在搭建，但必须具备明确的阶段分离：

```text
generate_target_grads
select
export
train
eval
aggregate
```

并支持 resume、fail-fast 和 manifest validation。每个阶段启动前检查：

* target JSONL hash；
* target gradient hash；
* candidate cache hash；
* selection indices 数量、唯一性和范围；
* DSMC/GIST/RR/LESS/NICE 的参数；
* RR query order；
* Random subset seed；
* dataset registration；
* adapter 和 eval 输出完整性。

## 给 Claude 的明确回复

可以直接发：

> The core draw artifacts pass structural review, but do not launch gradients yet. Patch three artifact-level issues first:
>
> 1. Correct the claim that all draws share identical subject composition; they do not. Keep the current draws, but emit a per-draw subject-composition matrix, TVD-to-(P^\star), and pairwise composition TVD.
> 2. Add actual JSONL byte hashes and ordered-ID hashes to every meta and to the draw index, together with source dataset/revision, generator commit, and allocation-plan hash. The current sorted-ID hash does not validate content or RR-relevant row order.
> 3. Replace “Hendrycks 5-shot target template” with “single-example supervised MMLU target format”; record target_num_fewshot=0 and evaluation_num_fewshot=5.
>    You may build the pilot driver now in parallel, but do not run gradients or SFT. After the metadata patch, run one-draw, all-eight-method selection-only dry run and bring back its manifests, selection sizes/overlaps, length distributions, runtime, and memory before launching the 2×2-draw SFT pilot.

最终 review 状态：

> **数据内容与全局 disjointness 通过；allocation 通过；split/no-leakage 通过；metadata provenance 与 subject-composition 声明尚需小修。允许构建 driver，不批准 launch。**

[1]: https://huggingface.co/datasets/hails/mmlu_no_train?utm_source=chatgpt.com "hails/mmlu_no_train · Datasets at Hugging Face"
[2]: https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/task_guide.md?utm_source=chatgpt.com "lm-evaluation-harness/docs/task_guide.md at main · EleutherAI/lm-evaluation-harness · GitHub"

这版已经把大方向基本收拾干净了。我的结论是：

> **BBH 作为 external-validation family 可以继续，不需要重新选任务。**
> 但我暂时还不会批准直接跑 30 个 SFT。现在还剩 **4 个 pre-compute protocol/engineering 点**，其中两个是我认为比较重要的新发现。修完后就进入 BBH canary → 全量实验。

先说好的部分：`c2d16a5` 已经把 dropout/provenance typo 修正，MMLU lexical contamination screen 也升级成了 pool-wide approximate retrieval；BBH 的 20/80 held-out split、3 个独立 query draws、draw×seed crossed design、固定 (K=2707) 和缩减后的 baseline set，整体方向是对的。 三个 draw 的实际 overlap 2/3/2，也和从 1302 条 reservoir 中独立抽 64 条时的预期量级一致。

不过有几个地方建议现在修掉。

### 1. “BBH 23 tasks”需要改成“23 task families / 27 lm-eval subtasks”

这里我需要修正上一轮给你的说法。

BBH 原论文/官方仓库确实称它为 **23 个 challenging tasks**。([GitHub][1]) 但当前 `lm-eval` 的 `bbh_cot_fewshot` group 实际明确列出了 **27 个 evaluation subtasks**：logical deduction 的 3/5/7-object 和 tracking shuffled objects 的 3/5/7-object 都分别作为独立 subtask。更关键的是，这个 group 的 primary metric 是：

```yaml
aggregation: mean
weight_by_size: true
```

也就是按 27 subtasks 的样本数做 micro aggregation。 lm-eval 文档也明确说明 `weight_by_size=true` 是 size-weighted/micro aggregation。([GitHub][2])

所以 Claude 当前：

> “23 task-level scores saved”

是不准确的。

好消息是，**split 实现本身反而是对的**：它按 27 个本地 file tasks 分层做 20/80 split，然后 query draw 从 reservoir 按 example 采样。 这与 lm-eval 的 micro metric 对齐。

建议只修 metadata / prereg：

* BBH = 23 conceptual task families；
* operational evaluation = 27 lm-eval subtasks；
* primary 保存和报告 **27 subtask scores**；
* 23-family regrouping只能作为可选 secondary diagnostic；
* draw manifest 同时记录 `subtask_composition_27` 和 `conceptual_family_composition_23`。

**不要重新生成 split。**

---

### 2. MinHash/LSH contamination screen 仍不能支持现在这么强的“0 ≥0.3”表述

Claude 修掉了上一轮的 L3 scope bug，这很好；但当前 LSH 参数是：

* 64 permutations；
* 16 bands；
* 因而每 band 4 rows。

LSH 是 probabilistic candidate generation，不是 exhaustive comparison。

在这个 banding 下，一个真实 Jaccard similarity 为 (s) 的 pair 至少撞一个 band 的概率大约是：

[
P_{\rm detect}(s)=1-(1-s^4)^{16}.
]

所以大约：

[
P_{\rm detect}(0.3)\approx12.2%,
]

[
P_{\rm detect}(0.5)\approx64.4%.
]

换句话说：

> “LSH 没找到任何 (J\ge0.3)” ≠ “整个 pool 不存在 (J\ge0.3) pair”。

它仍然是一个有价值的 approximate lexical screen，但当前参数的 recall 不足以支持很强的“0 near duplicates”表述。

这个很好修，而且便宜。比如改成 64 perms / 32 bands / 2 rows，同样的近似碰撞概率变成：

[
P_{\rm detect}(0.3)\approx95.1%,\qquad
P_{\rm detect}(0.5)>99.99%.
]

然后所有 collisions 仍用 exact Jaccard 验证。

建议：

* 用高-recall LSH 配置重新扫一次 MMLU；
* 更重要的是对**最终 BBH held-out 5209 eval examples**也跑同样的 global lexical screen；
* 文中称作 `high-recall lexical near-duplicate screen`；
* semantic NN 继续作为 release-time limitation，没有必要因此阻塞这次实验。

因此 contamination **不是 launch blocker 的科学风险**，但最好在 launch snapshot 前把审计说法和参数做扎实。

---

### 3. “pin lm-eval”目前还只是文字承诺，没有真正 pin

prereg 写的是：

> pin the lm-eval commit and official `bbh_cot_fewshot` config

但当前文档没有给出具体 Git SHA。

这次尤其不能省，因为 `bbh_cot_fewshot` 的 prompt 最近真的改过：lm-eval 在 2025 年修掉了 few-shot prompt 中重复的 “Let's think step by step” 文本。([GitHub][3]) 当前模板已经是 version 4.0，固定 3-shot、greedy generation、`max_gen_toks=1024`。

所以正式实验前要保存：

* 本机实际 lm-eval Git SHA / package version；
* `_bbh_cot_fewshot.yaml` SHA；
* `_cot_fewshot_template_yaml` SHA；
* 27 个 per-subtask YAML hashes；
* `SaylorTwift/bbh`/本地 raw BBH 数据 hashes；
* few-shot sample hashes。

而且你们不能直接调用标准 `bbh_cot_fewshot` 做最终 eval，因为标准 config 的 test split 是完整 BBH，而你们要评的是自己 carve 出来的 5209 held-out subset。

下一步实际上需要做一个 **frozen custom held-out BBH lm-eval suite**，只替换 dataset split，保持 prompt/filter/generation/metric逻辑与 pinned config一致。

---

### 4. prereg 还缺几个会影响 selection realization 的随机种子

当前已经冻结 SFT seeds `{42,1}`，但还没有看到明确冻结：

* Random-K 的 selection seed；
* First-RR / Second-RR 的 query visitation permutation seed。

这是 launch 前必须补的。

建议简单预注册：

[
\text{random_seed}(d)=5000+d
]

[
\text{rr_perm_seed}(d)=6000+d
]

并规定：

* First-RR 和 Second-RR **共享同一个 RR query order**；
* 同一个 draw 的 Random subset在两个 SFT seeds之间完全相同；
* training seed只是 training axis，不能顺便改变 subset；
* 3 draws × 5 methods = 15 frozen subsets；
* 每个 subset再跑 2 training seeds = 30 adapters。

这样 crossed design 才真正解耦：

[
\text{query realization}
\times
\text{training stochasticity}.
]

否则如果 seed 42/1 连 Random subset也跟着变，你其实又混进了第三个 random-selection axis。

---

## Prompt alignment 还有一个需要写清楚的 limitation

Claude 现在说：

> query-gradient prompt aligned to eval template

这个方向是对的，但不能称为完全对齐。

当前 lm-eval BBH CoT prompt类似：

```text
Q: ...
A: Let's think step by step.
```

few-shot examples中的 target 是完整 CoT rationale + final answer。

但你们 BBH raw examples只有 final target，例如 `(C)`、`14`、`Yes`；prereg规定 query gradient 的 supervised continuation就是这个 raw final answer。

也就是说：

> **prompt context aligned，generation trajectory / supervision target 并未完全 aligned。**

这不是需要改实验的 blocker——BBH 并没有每个 test item 的 gold CoT rationale，硬造 teacher rationale反而会引入更大的 confound。

建议只把措辞改成：

> query gradients use the exact pinned CoT few-shot prompt prefix, but supervise only the provided final BBH target; therefore prompt context is aligned while reasoning-trace supervision is unavailable.

这比宣称“fixes MMLU prompt mismatch”更准确。

---

## 还有两个统计措辞的小修

prereg 写：

> separately the variance attributable to query realization vs SFT seed.

(3\times2) crossed design确实比原 MMLU强很多，但每个 draw×seed cell只有一次 observation，样本量也很小。不要声称精确“variance attributable”。

改成：

* average over seeds within each query draw；
* average over draws within each SFT seed；
* report query-draw spread；
* report seed sensitivity；
* optional descriptive two-way decomposition；
* 不做强 variance-component inference。

另外，`D2 to a balanced reference` 在 BBH prereg 中还没有明确定义。MMLU 的 balanced reference有 STEM/HUM 50/50 的明确含义；BBH 没有。

我建议直接改成：

[
D_2(S,Q_d)
]

和，如果愿意额外算：

[
D_2(S,P_{\rm heldout})
]

其中 (P_{\rm heldout}) 明确定义为 5209 held-out examples 的 lm-eval micro distribution。

如果提取 5209 条 target gradients太贵，就只保留 (D_2(S,Q_d))，不要为了延续 MMLU forensic 人为发明一个“balanced BBH reference”。

---

# 接下来具体做什么

我建议现在**仍然不跑 SFT**，但已经不需要再讨论 family 或方法。

让 Claude 完成一个最后的 pre-compute engineering checkpoint：

1. 修 23-family / 27-subtask 的 prereg 和 manifest；
2. 用 high-recall LSH 重跑 lexical screen，尤其是 final BBH heldout；
3. pin lm-eval exact commit + 29-ish config/hash artifacts；
4. 冻结 Random/RR selection seeds；
5. 构建 custom `bbh_external_cot_fewshot` held-out evaluation configs；
6. 构建 query prompt renderer；
7. 做 **prompt parity audit**：

   * 对 27 个 subtasks每个至少抽 1 条；
   * query-gradient prompt prefix与 pinned lm-eval构造的 prompt byte-for-byte一致；
   * few-shot examples、delimiters、description、`Let's think step by step` 全一致；
8. 生成最终 prereg/launch manifest，working tree clean。

lm-eval 官方本身提供 `write_out.py` 用来检查模型实际收到的 prompt，因此这个 parity gate 很自然。([GitHub][4])

完成这些以后，我会批准一个非常小的 **pre-SFT canary**：

* base/no-SFT 在 held-out BBH 上评一次；
* `draw0` 提取 target gradients；
* 五种 selector全部跑到 (K=2707)；
* 不训练；
* 检查 target cache、selection size、hash、determinism、RR order、Random seed、Jaccard/source/token diagnostics。

这个 selection-only canary 通过后，再直接跑：

[
3\text{ draws}\times
5\text{ methods}\times
2\text{ SFT seeds}
==================

30
]

个 adapters。

不用再做第二轮方法学选择。

可以直接给 Claude：

> The BBH family choice and the 20/80 + 3-draw × 2-seed design pass. Do not launch SFT yet; complete one final pre-compute checkpoint.
>
> 1. Correct the BBH task accounting. The original BBH paper describes 23 conceptual task families, but the pinned lm-eval `bbh_cot_fewshot` group operationally contains 27 subtasks and micro-aggregates them with `weight_by_size=true`. Keep the existing per-file 20/80 split, but record/report all 27 lm-eval subtask compositions and scores; 23-family regrouping may be secondary only.
> 2. Strengthen the MinHash screen. With 64 perms / 16 bands / 4 rows, LSH recall is too low to interpret “0 at Jaccard ≥0.3/0.5” as a strong pool-wide exclusion. Rerun a high-recall configuration (e.g. 64 perms / 32 bands / 2 rows, exact-Jaccard verification of all collisions), especially against the final 5,209-example BBH heldout set. Keep semantic NN as a disclosed release-time item if unavailable.
> 3. Actually pin lm-eval: record the exact installed Git SHA/version and hashes of the BBH group YAML, template YAML, all 27 subtask YAMLs, and hard-coded few-shot samples. Build a custom held-out BBH task suite that changes only the dataset split while preserving the pinned prompt, generation, filtering, and metric behavior.
> 4. Freeze selection randomness now: Random-K seed = `5000+draw_id`; RR permutation seed = `6000+draw_id`, shared by First-RR and Second-RR. The selected subset for a draw must be identical across the two SFT seeds.
> 5. Tighten prompt wording: query gradients use the exact CoT few-shot prompt prefix but supervise only the provided final BBH target; this is prompt-context alignment, not full reasoning-trace alignment.
> 6. Replace “variance attributable to query vs seed” with descriptive crossed-design summaries. Define the BBH D2 reference explicitly or drop the ambiguous “balanced reference.”
> 7. Run a 27-subtask prompt-parity audit against the pinned lm-eval request construction and commit the final split/config/prereg/launch manifest. Still no SFT.
>
> Bring that checkpoint back. If it passes, the next gate is one no-SFT heldout evaluation + draw0 five-selector selection-only canary; after that, launch the frozen 30-adapter experiment.

所以，现在不是重新思考研究方向，而是**把 BBH protocol 从“科学设计已定”推进到“可无歧义执行”**。最重要的新修正是 **23/27 task accounting、真正 pin lm-eval、以及 LSH recall**。修完这三点，这轮 external validation 就可以正式进入计算阶段。

[1]: https://github.com/suzgunmirac/BIG-Bench-Hard "https://github.com/suzgunmirac/BIG-Bench-Hard"
[2]: https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/task_guide.md "https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/task_guide.md"
[3]: https://github.com/EleutherAI/lm-evaluation-harness/releases "https://github.com/EleutherAI/lm-evaluation-harness/releases"
[4]: https://github.com/EleutherAI/lm-evaluation-harness "https://github.com/EleutherAI/lm-evaluation-harness"

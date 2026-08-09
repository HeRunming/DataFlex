这轮推进总体方向是对的，而且现在已经非常接近“可以冻结 external-validation protocol”的状态。`2b3f990` 没有偷偷跑 accuracy，也把旧结果的 provenance 补得比之前扎实很多。

不过我复核后发现了 **3 个需要先修的小问题，其中两个是实质性的文档/审计错误**。它们不推翻已有实验，但我不建议现在马上写 prereg 然后开跑。

第一，`resolved_run_provenance.json` 还不是完全正确。它写了 `lora.dropout=0.1`，但实际训练 YAML 是 `lora_dropout: 0.05`，而 `run_pilot_sft.sh` 并没有 override dropout，所以已完成主实验真正使用的应当是 **0.05**。   同一个 provenance 文件里还有：

```text
"effective_global_batch": 16
"effective_batch": 128
```

而真正计算是：

[
4\text{ per device}\times4\text{ accum}\times8\text{ GPUs}=128.
]

所以 `effective_global_batch:16` 也是明显 typo。

这说明 Codex/Claude 所说“A–G 全部 cleared”还差最后一个小 patch。好消息是：这些错误都只是**事后 summary metadata 错误，不影响已经训练出的 adapter 或结果**。实际 step counts 是从 `trainer_state.json` 恢复出来的，这部分证据很强。

第二，contamination audit 很有价值，但目前应该叫：

> **lexical contamination audit passed**

而不是完整的 contamination hard gate passed。

原因不是只有 L4 semantic NN 没做。还有一个更隐蔽的问题：当前 L3 fuzzy Jaccard **只在已经通过 L2 13-gram filter 的 7 个 suspects 上计算**。

这意味着一个候选样本完全可能：

* 没有连续 13-token/word overlap；
* 但具有很高的 5-shingle Jaccard / fuzzy similarity；

它根本不会进入 L3。

所以：

[
L3=0
]

不能解释成“整个 270k pool 没有 fuzzy lexical near-duplicates”，只能解释成：

> none of the seven 13-gram suspects also passes the fuzzy criterion.

已有 MMLU 结果因此不需要推翻——L1=0、L2只有7个且人工全是假阳性，本身已经是很好的证据。 但论文不能写成完整 decontamination。

我会让 Claude 在新实验开始前补：

* 全局 MinHash/LSH 或 TF-IDF/BM25 top-N retrieval → exact Jaccard verification；
* 有条件就补 BGE semantic NN；
* 如果 BGE 暂时没有，也不必因此阻塞 BBH 实验，但把 L4 保留为 paper-release checklist。

“这是 LESS 标准 pool，所以残余风险受到限制”这句话也建议删弱一点。沿用前人数据并不能数学上 bound contamination risk。

第三，external-family feasibility table 有两个事实要修正，而且会影响 family 决策。

### MMLU-Pro 实际上有官方 clean validation/test split

Claude 写：

> none of the three has an off-the-shelf clean query/test split.

这对 MMLU-Pro 不准确。

官方 MMLU-Pro 数据明确包含：

* validation：70；
* test：12,032。([Hugging Face][1])

所以它确实有 clean validation/test split。

但这并没有让 MMLU-Pro 成为我首选，因为 validation 只有 **70 条**。如果目标还是三个 (M=64) query draws，那三个 draws 几乎会完全重叠；如果把 (M) 改成 16/32，又同时改变了 query-size axis。如果从 test 中另外 carve query reservoir，又失去了官方 test split 的主要优势。

而且 MMLU-Pro 并不是特别“外部”的第二 family：官方数据卡说明 12,032 个 test questions 中有 **6,810 个来自原始 MMLU**，剩余部分主要来自 STEM 网站、TheoremQA 和 SciBench。([Hugging Face][1])

因此它虽然在“没有看过结果、低 cherry-picking risk”方面很好，却不够强地解决：

> 你们的现象是否只是 MMLU-family 特例？

### BBH 官方是 23 tasks，不是 27

官方 BBH 仓库和 lm-eval 文档都明确称其为 **23 个 challenging BIG-Bench tasks**。([GitHub][2]) ([GitHub][3])

所以 Claude 本地看到的：

> 27 task files

需要先解释清楚。可能是本地额外 variants/configs，不一定是错误，但最终 external experiment 应明确限定成**官方 23-task BBH suite**，不能不知不觉拿 27 个本地文件组成一个非标准 benchmark。

这一点在 prereg 前必须解决。

---

## 我现在会选 BBH，而不是 MMLU-Pro

经过这轮核对，我的推荐比上次更明确：

> **external confirmation family 选 BBH。**

理由不是因为我们觉得它容易赢，而是它最能解决目前论文真正的 generalization vulnerability。

BBH 与 MMLU 是明显不同的 reasoning benchmark，官方 suite 是 23 个具有多步推理性质的任务。([GitHub][2]) MMLU-Pro 虽然更新、更难，也有稳定的 lm-eval implementation，但它超过一半的问题继承自原始 MMLU，因此作为唯一 external family 的说服力略弱。([Hugging Face][1])

TyDiQA 我仍不选，因为：

* 官方 GoldP 主要提供 train/dev，而不是类似 MMLU 的自然 query/validation/test 三分法；([GitHub][4])
* 你们之前已经看到二阶方法在它上面表现相对 favorable，因此现在把它选成唯一 confirmation target 会产生不必要的 cherry-picking 观感。

近期那个 controlled targeted-selection study本身也强调结论需要跨 datasets/models/budgets验证，而且没有一个 selector 普遍占优；gradient + RR 通常只是低预算平均更强。([arXiv][5]) 这正是我们现在需要真正 external target 的原因。

---

# 我建议的 BBH protocol

不要现在直接跑 SFT。下一步先让 Claude 写 prereg + split artifacts 给我们 review。

我会冻结成：

[
3\text{ query draws}
\times
2\text{ SFT seeds}
\times
5\text{ methods}
================

30\text{ adapters}.
]

方法：

* DSMC；
* Second-RR；
* First-RR；
* LESS-style TopK；
* Random-K；
* 加一个共享 no-SFT reference。

保留 First-RR 很重要，因为近期系统研究发现 gradient representation + greedy RR 在低预算是很强的 controlled baseline。([arXiv][5])

预算：

[
K=2707
]

只跑一个预算。继续四 epochs。

### BBH 数据划分

不要从 6511 个 examples 中直接拿 3×64 后把剩下都叫 test。

更干净的是先一次性、按 task stratified 地划：

[
20%\rightarrow \text{query reservoir}
]

[
80%\rightarrow \text{fixed held-out evaluation}.
]

假设总计约 6511，则 query reservoir 大约有 1300 条。

然后三个 query draws 都从**同一个固定 reservoir 独立采样**：

[
|Q_d|=64.
]

draw 内不放回，draw 间允许自然 overlap，报告 overlap matrix。

在 1300 左右的 reservoir 中独立抽 64，两组随机 draw 的预期重叠只有约：

[
64^2/1300\approx3.2
]

条，远比 MMLU-Pro 的 70-example validation reservoir 干净。

这样：

* query 和 evaluation 完全 disjoint；
* 三个 draws是真正独立随机 realization；
* 不再强制全局 disjoint产生负相关；
* query draw 和 SFT seed也被 crossed design 解耦。

### 两个 SFT seeds 完全 crossed

例如：

[
seed\in{42,1}.
]

每一个 (Q_d) 都训练两个 seeds。

于是不会再出现：

[
Q_0\leftrightarrow42,\quad
Q_1\leftrightarrow1
]

这种 confounding。

最终可以分别估计：

* query realization variability；
* SFT seed variability；
* paired method differences。

这比现在 MMLU 的五个 coupled blocks 强很多。

---

## Prompt alignment 这次也一起修掉

这是一次很好的机会解决 MMLU 主实验中的另一个 limitation：

> target gradients 是 single-example supervised，而 evaluation 是 5-shot。

BBH 新实验应该让 query-gradient construction 与 eval 尽可能使用**同一个固定、版本 pin 住的 lm-eval prompt/template**。

lm-eval 当前正式支持 CoT BBH，并把 evaluation config 作为可复现实验定义的一部分。([GitHub][6])

所以建议：

* pin lm-eval commit；
* pin official `bbh_cot_fewshot` group/config；
* query gradient和 evaluation使用同样的 task instruction/few-shot context；
* gold continuation如何形成 loss也在 prereg 中写死；
* 不要在看到 base-model BBH accuracy 后再改 CoT/direct setting。

Primary metric直接采用 pinned lm-eval BBH group输出的官方 metric；同时保存23个 task-level scores。不要自己再发明一个更有利的 aggregate。

因为我们会 carve official BBH examples，所以论文中将它叫：

> **held-out BBH external-validation split**

而不要假装它是官方完整 BBH leaderboard score。

---

## 正式跑之前还有四个 cheap gates

我建议下一条给 Claude 的任务只做到这些：

1. 修 provenance：

   * dropout `0.1 → 0.05`；
   * 删除错误的 `effective_global_batch=16`，只保留权威的 128；
   * consolidated / audit / resolved JSON 三处统一。

2. 修 feasibility 文档：

   * MMLU-Pro = official validation 70 / test 12,032；
   * BBH official suite = 23 tasks；
   * 解释本地“27 files”究竟多出来哪四个。

3. 修 contamination：

   * 把 L3 从“L2 suspects only”改成真正 global approximate lexical retrieval；
   * 将当前结果称作 lexical screen，而不是 complete decontamination；
   * BBH split确定以后，对**最终 held-out BBH eval subset**重新跑 contamination。

4. 只生成 BBH prereg + split artifacts：

   * 23-task exact list；
   * reservoir/eval IDs；
   * 3 draw IDs；
   * task composition；
   * pairwise overlap；
   * all hashes；
   * 2 seeds；
   * K=2707；
   * 5 methods；
   * prompt/config commit；
   * primary/secondary analysis。

**仍然不要 gradient extraction / selection / SFT。**

我们 review split 以后才启动 compute。

---

你可以直接发给 Claude：

> Select BBH as the external-validation family, but do not launch compute yet. Before writing the final prereg, patch three audit issues.
>
> First, fix `resolved_run_provenance.json`: the executed training config used `lora_dropout=0.05` from `train_llama7b_lora.yaml` because the driver does not override dropout; the current provenance incorrectly says 0.1. Also remove/fix the contradictory `effective_global_batch=16`; the authoritative effective batch is 4×4×8=128. Propagate the correction to the artifact audit.
>
> Second, correct `external_family_feasibility.md`: MMLU-Pro does have an official validation/test split (70 validation / 12,032 test), although 70 validation examples are too small for three useful M=64 query draws. Also reconcile the local “27 BBH task files” with the official 23-task BBH suite and use exactly the official 23 tasks in the external experiment.
>
> Third, strengthen the contamination wording/check: the present L3 Jaccard test only runs on L2 13-gram suspects, so L3=0 is not a pool-wide fuzzy-overlap result. Add a global approximate lexical retrieval stage (e.g. MinHash/LSH or sparse top-N retrieval followed by exact shingle Jaccard). Semantic NN can remain a disclosed release-time item if the encoder is unavailable, but do not call the current audit complete semantic decontamination.
>
> Then draft and freeze a BBH preregistration only—no gradients/SFT yet:
>
> * official 23-task BBH suite;
> * one deterministic per-task-stratified 20% query-reservoir / 80% held-out-eval split;
> * three independently sampled M=64 query draws from the fixed reservoir, overlap allowed and reported;
> * fully crossed SFT seeds {42, 1} for every draw;
> * methods = DSMC, Second-RR, First-RR, LESS-style TopK, Random-K;
> * one shared no-SFT reference;
> * K=2707, 4 epochs, all other frozen SFT settings unchanged;
> * pin the lm-eval BBH CoT-fewshot task/config commit and align query-gradient prompting with the evaluation template as closely as the supervised target format permits;
> * primary metric = the pinned lm-eval BBH group metric, plus all task-level scores;
> * log query loss, D2, downstream score, subset/source diagnostics;
> * rerun candidate-pool contamination against the final held-out BBH evaluation split.
>
> Generate only the split/draw/prereg/provenance artifacts and bring them back for review before any compute.

所以当前阶段的判断是：

**现有 MMLU 主结果不用重做；artifact cleanup 还剩两个 metadata typo + contamination L3 范围问题。修完以后，选择 BBH 做真正的第二-family external validation，并且这次用 3 draws × 2 seeds 的 crossed design。**

这轮 BBH 实验如果做好，无论 DSMC 最后赢不赢 Random，对论文都会比再做 MMLU-Pro 或继续修 MMLU 更有价值。

[1]: https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro/blob/main/README.md "README.md · TIGER-Lab/MMLU-Pro at main"
[2]: https://github.com/suzgunmirac/BIG-Bench-Hard?utm_source=chatgpt.com "GitHub - suzgunmirac/BIG-Bench-Hard: Challenging BIG-Bench Tasks and Whether Chain-of-Thought Can Solve Them · GitHub"
[3]: https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/leaderboard/README.md?utm_source=chatgpt.com "lm-evaluation-harness/lm_eval/tasks/leaderboard/README.md at main · EleutherAI/lm-evaluation-harness · GitHub"
[4]: https://github.com/google-research-datasets/tydiqa?utm_source=chatgpt.com "GitHub - google-research-datasets/tydiqa: TyDi QA contains 200k human-annotated question-answer pairs in 11 Typologically Diverse languages, written without seeing the answer and without the use of translation, and is designed for the training and evaluation of automatic question answering systems. This repository provides evaluation code and a baseline system for the dataset. · GitHub"
[5]: https://arxiv.org/abs/2602.14696?utm_source=chatgpt.com "A Critical Look at Targeted Instruction Selection: Disentangling What Matters (and What Doesn't)"
[6]: https://github.com/EleutherAI/lm-evaluation-harness/blob/main/README.md?utm_source=chatgpt.com "lm-evaluation-harness/README.md at main · EleutherAI/lm-evaluation-harness · GitHub"

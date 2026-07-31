现在建议不要直接选 1，也不要接受“CUDA sampling 天生无法复现”这个结论。更稳妥的是：

> **选 4：先做一次严格确定性诊断；若仍无法复现，再退回方案 1，冻结一次预注册的随机 realization。**

必须按这个顺序，是因为当前只能确认“普通 seeding 不够”，还不能确认“严格 deterministic 配置也无效”。PyTorch 明确说明，单独设置 seed 并不能保证完全复现；可以通过 `torch.use_deterministic_algorithms(True)`、关闭 benchmark、固定 CUDA 算法等进一步限制非确定性，不支持确定性实现的操作会直接报错。([PyTorch 文档][1])

## 当前八方法 dry run 的结论

这轮整体已经通过了主要工程检查：

* 八个方法都恰好选择 13,533 个唯一索引；
* NICE 的 off-by-one 已修复；
* LengthMatched Random 的五个长度桶与 DSMC 完全一致；
* target/candidate/cache provenance 校验已加强；
* DSMC、Second-RR、GIST 重跑 selection hash 一致。

NICE 也已经被合理地标成 `NICE-MMLU-EM`，而不是声称完全复现官方所有设定。NICE 本身确实依靠 policy gradient 将非可微任务指标转为选择信号，因此 Monte Carlo rollout 是其方法的一部分，不能随意改为 greedy decoding。([Proceedings of Machine Learning Research][2])

## 为什么不直接选 3：Greedy decoding

这会将

[
a_i\sim\pi_\theta(\cdot\mid x)
]

改成单一的

[
a_i=\arg\max_a\pi_\theta(a\mid x),
]

MC rollout 不再估计策略下的期望 reward-gradient。它不只是减少随机性，而是改变了 NICE 的核心估计器。

因此方案 3 不合适。

## 为什么不选 2 作为主实验

让 NICE 每个 draw 跑多个 generation seeds 虽然能估计 NICE 自身的 selection variance，但会产生两个问题：

1. NICE 获得比其他 baseline 更多的重复次数；
2. target draw 不再是唯一主要重复单位，paired design 变复杂。

当前 pilot 的目标是比较方法在不同 target draws 上的稳定性，不是系统研究 NICE 的 MC variance。主矩阵没有必要因此翻倍或三倍。

NICE 的 policy-gradient estimator 本身是 Monte Carlo 估计，存在采样方差是正常现象。([arXiv][3]) 可以在主 pilot 后，只有当 NICE 与 DSMC 非常接近并成为 load-bearing baseline 时，再对预注册的 draw 0 补少量 NICE generation seeds。

## 先做严格确定性诊断

让 Claude 在同一台机器、同一 GPU、同一软件环境下，对 `stem80_draw0` 连续执行两次 NICE，但先加入：

```bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8
```

以及 Python 中：

```python
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

torch.use_deterministic_algorithms(True)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
```

还应强制使用 deterministic math attention，而不是 Flash/Efficient/CuDNN SDPA backend。新版 PyTorch 文档明确指出，不同 SDPA backend 的 backward determinism 不同；Math backend 在 deterministic mode 下可确定执行，而 fused backend 可能具有非确定性 backward。([PyTorch 文档][4])

NICE 既包含 generation forward，也包含每个 sampled continuation 的 LoRA gradient backward，因此 backward backend 同样重要。

诊断时比较的不是最终 Jaccard，而应从最上游逐层比较：

1. 每个 target 的 sampled token IDs；
2. 每个 rollout 的 reward；
3. 每个 target 的 projected policy gradient；
4. 完整 validation-gradient tensor；
5. candidate scores；
6. selected indices。

这样才能定位差异到底来自 generation、backward、JL projection 还是 top-(K) 边界。

## 如果严格模式成功

那就冻结 strict deterministic NICE：

* 固定 GPU 型号和设备编号；
* 固定 PyTorch、Transformers、CUDA、cuDNN 版本；
* 固定 deterministic flags；
* 固定每个 target 的 generation seed；
* 保存 sampled-token hash、reward hash、val-gradient hash 和 selection hash。

之后所有 draws 按相同设置运行。

## 如果严格模式报错或仍不一致

此时采用方案 1，但不要只缓存一个匿名 `.pt`。

应将一次预注册 stochastic realization 视为 NICE 的固定输入 artifact，保存：

```text
sampled_token_ids.jsonl
rollout_rewards.json
retained_target_ids.json
nice_target_policy_grads.pt
nice_target_policy_grads.meta.json
step_1.json
```

metadata 至少记录：

* target JSONL hash；
* ordered target IDs hash；
* base/adapter/checkpoint hash；
* generation seed formula；
* MC count、temperature、top-k、top-p；
* CUDA/PyTorch/Transformers 版本；
* sampled token IDs hash；
* reward vector hash；
* zero-signal target IDs；
* projected-gradient tensor hash；
* final selection hash。

这样“重新生成 rollout”不一定 bitwise identical，但：

> 给定被冻结的 stochastic rollout/gradient artifact，NICE scoring 与 selection 可以完全复现。

这对随机算法是合理的 artifact-level reproducibility。PyTorch 官方也明确说，跨 release、平台乃至 CPU/GPU 之间并不保证完全相同的结果，因此复现边界必须被明确限定。([PyTorch 文档][5])

## 当前 NICE 还有一个小细节

现在 64 个 target 中有 13 个 zero-signal targets，最后使用 51 条 policy-gradient rows。

跳过零向量不会改变 top-(K) 排序，因为：

[
\frac1{64}\sum_{t=1}^{64}\langle g_i,v_t\rangle
===============================================

\frac{51}{64}
\left(
\frac1{51}\sum_{t:v_t\ne0}\langle g_i,v_t\rangle
\right),
]

只是乘了一个所有 candidates 共享的正常数。不过应保存这 13 个 target 的 ID，而不只是保存数量，方便检查 zero-signal 是否集中在某些 subjects。

## 可以直接回复 Claude

> Choose option 4. Before concluding that NICE cannot be made bit-reproducible, run one bounded strict-determinism diagnostic on `stem80_draw0`: set `CUBLAS_WORKSPACE_CONFIG=:4096:8`, enable `torch.use_deterministic_algorithms(True)`, disable cuDNN benchmarking and TF32, force the deterministic math SDPA backend, and keep execution on one fixed GPU. Run NICE twice and compare, in order, sampled token IDs, rollout rewards, per-target projected gradients, the full validation-gradient tensor, candidate scores, and selected indices.
>
> If strict mode succeeds, freeze and use that deterministic configuration for all draws. If it raises on unsupported operations or still fails bitwise reproducibility, fall back to option 1: generate one preregistered stochastic NICE realization per draw and freeze it as an immutable artifact. Cache not only `val_grads.pt`, but also sampled token IDs, rollout rewards, retained/zero-signal target IDs, environment versions, all input/output hashes, and the final selection. Selection must then read only the frozen gradient artifact and be bit-reproducible.
>
> Do not switch to greedy decoding, because MC sampling is part of the NICE policy-gradient estimator. Do not add multiple NICE seeds to the main pilot; reserve a small seed-sensitivity study on preregistered draw 0 only if NICE becomes competitive enough for its stochastic variance to be load-bearing.
>
> No SFT yet. After this NICE reproducibility gate is resolved, freeze the eight selection hashes and bring back the final manifest before launching the 2-direction × 2-draw pilot.

最终判断：

> **八方法 dry run 基本通过；NICE 是最后一个 selection-level gate。选择 4，严格确定性诊断失败后再采用 1。不要选 2 或 3。**

[1]: https://docs.pytorch.org/docs/main/generated/torch.use_deterministic_algorithms.html?utm_source=chatgpt.com "torch.use_deterministic_algorithms — PyTorch main documentation"
[2]: https://proceedings.mlr.press/v267/wang25bm.html "NICE Data Selection for Instruction Tuning in LLMs with Non-differentiable Evaluation Metric"
[3]: https://arxiv.org/abs/1906.10652?utm_source=chatgpt.com "Monte Carlo Gradient Estimation in Machine Learning"
[4]: https://docs.pytorch.org/docs/stable/notes/randomness.html?highlight=reproducibility&utm_source=chatgpt.com "Reproducibility — PyTorch 2.13 documentation"
[5]: https://docs.pytorch.org/docs/stable/notes/randomness?utm_source=chatgpt.com "Reproducibility — PyTorch 2.11 documentation"

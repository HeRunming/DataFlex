现在可以正式判定：

> **selection-level gate 已通过。批准继续构建 `train/eval/aggregate`，但暂时不要一次性启动 30 个 SFT。**

NICE 的严格确定性方案是合理的：`torch.use_deterministic_algorithms(True)` 会在可用时强制确定性算法，并在没有确定性实现时抛错；配合进程启动前设置 `CUBLAS_WORKSPACE_CONFIG`、关闭 TF32、固定 Math SDPA，可以在同一软件与硬件环境中实现重复执行一致。PyTorch 同时明确说明，这种 bitwise reproducibility 不能外推到不同版本、平台或设备，因此论文中应写成“在冻结的 H20/PyTorch/CUDA 环境中可复现”，而不是普适确定性。([PyTorch 文档][1])

NICE 保留 Monte Carlo generation 和 policy-gradient estimator，而没有为了确定性改成 greedy decoding，也是正确选择；NICE 的核心贡献本来就是使用 policy gradient 将非可微任务指标转换成 selection signal。([Proceedings of Machine Learning Research][2])

仓库中两次运行得到相同 NICE validation-gradient 和 selection hashes，八种方法也都通过 exact-(K) 检查；冻结 manifest 已记录环境和八个 selection hashes。

不过 Claude 的一句话仍稍微超前：

> “Everything up to SFT is done.”

目前完整生成并冻结八方法 selection 的只有 `stem80_draw0`。其余三个 pilot draws：

* `stem80_draw1`
* `hum80_draw0`
* `hum80_draw1`

仍需生成 target gradients、八种 selections、exports 和 manifests。`train/eval/aggregate` 阶段也尚未实现。因此现在不能直接批准整批 30-adapter launch。

## 建议的下一步

让 Claude 同时完成两件事：

第一，构建完整的 `train/eval/aggregate` 阶段，但不立即运行 SFT。

第二，对剩余三个 pilot draws 运行所有 **pre-SFT phases**：

```text
setup → gengrad → select → export → diag
```

这不是新的调参实验，只是在冻结协议下生成剩余输入 artifacts。完成后应提交一个四 draw 的 master manifest，包括：

* 四个 target JSONL hashes；
* 四个 target-gradient hashes；
* 每个 draw 八个 selection hashes；
* 每个 subset JSONL hash；
* NICE val-gradient hash、zero-signal IDs 和 reward diagnostics；
* Random-K seed；
* RR permutation seed；
* LengthMatched Random 的精确 bucket counts；
* candidate cache、checkpoint、projection 和环境 hashes。

然后进行一次代码 review，再启动两个 planned adapters 作为端到端 canary：

1. `stem80_draw0 × DSMC`：测试普通 draw-specific headline 路径；
2. `draw0 × Random-K shared adapter`：测试跨 STEM/HUM direction 复用同一个 adapter 的路径。

这两个都属于已经预注册的 30 个 runs，不是额外实验。Canary 只检查：

* training config；
* dataset 注册和 hash；
* checkpoint 输出；
* eval 输出；
* shared-adapter mapping；
* aggregation；
* resume/fail-fast。

不能根据 canary 的 accuracy 修改方法、超参数或删除 baseline。两条 canary 工程上通过后，继续剩余 28 个 adapters。

## Train/eval/aggregate 阶段的关键要求

运行矩阵必须显式列出 32 个 method–draw cells 和 30 个唯一 adapter IDs。Random-K 的同一 draw index 在两个方向间复用 adapter，但 aggregation table 仍要保留两个 cells，并记录相同的 `shared_adapter_id`。当前 30 个唯一 adapter 的计算是正确的：

[
7\times4+2=30.
]

训练前应验证 subset hash，而不只是 dataset key。每个 adapter 固定：

[
\text{seed}\in{42,1},\qquad
\text{effective batch}=128,\qquad
\text{epochs}=4.
]

Eval 阶段要求每个唯一 adapter 只产生一个权威 `results_*.json`；存在多个结果文件时不能简单取字典序最后一个。应由 manifest 记录确切路径和 hash。

Aggregate 阶段至少输出：

[
\text{balanced}
===============

\frac{\text{STEM}+\text{HUM}}2,
]

以及 target-weighted：

[
0.797,\text{majority}+0.203,\text{minority},
]

并为每个 baseline 报告相对 DSMC 的 paired difference。Pilot 只有两个 draws/direction，因此现在只看 pipeline 与粗略排序，不做稳定性或显著性结论。

## 可以直接回复 Claude

> The NICE determinism gate passes, and the selection-level pipeline is now approved. Build the complete train/eval/aggregate phases now, but do not launch all 30 adapters yet.
>
> In parallel, run only the pre-SFT phases for the remaining three pilot draws (`stem80_draw1`, `hum80_draw0`, `hum80_draw1`): setup, target-gradient extraction, all eight selections, export, and diagnostics. Freeze a four-draw master manifest containing all target-gradient, selection, exported-subset, NICE-val-gradient, checkpoint, candidate-cache, seed, and environment hashes.
>
> Then push the complete driver, the explicit 32-cell/30-unique-adapter run plan, and the four-draw manifest for review. After that review, launch two planned canaries only:
>
> 1. `stem80_draw0 × DSMC`;
> 2. the shared draw-0 `Random-K` adapter.
>
> Train and evaluate those two, verify fail-fast/resume, exact configs, result-file provenance, shared-adapter reuse, and aggregation. Do not make any method or hyperparameter changes based on their accuracy. If both canaries pass the engineering checks, continue the remaining 28 adapters without another methodological decision.
>
> Keep all NICE strict-determinism settings frozen and scope the reproducibility claim to the recorded H20/PyTorch/CUDA environment.

所以现在不是“hold everything”，也不是“直接跑一天”。正确状态是：

> **批准构建完整 driver，并批准生成剩余三个 draw 的 pre-SFT artifacts；全量 SFT 前先做四-draw manifest review 和两个 adapter canary。**

[1]: https://docs.pytorch.org/docs/stable/notes/randomness?utm_source=chatgpt.com "Reproducibility — PyTorch 2.11 documentation"
[2]: https://proceedings.mlr.press/v267/wang25bm.html?utm_source=chatgpt.com "NICE Data Selection for Instruction Tuning in LLMs with Non-differentiable Evaluation Metric"

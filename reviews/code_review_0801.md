我看完 `dddcf1a` 了。结论是：

> **四个 draw 的 pre-SFT artifacts 可以通过；方法层面已经没有 blocker。**
> **但不要立刻启动两个 canary，先修 4 个很小的执行层问题。修完即可直接跑 canary，不需要再做一轮方法学 review。**

32-cell/30-adapter 的映射是正确的：除 Random-K 外，7 个方法在 4 个 draw 上各自训练；Random-K 按 draw index 在两个方向间复用，因此是

[
7\times4+2=30.
]

run plan 中 `randk_drawidx0_seed42` 和 `randk_drawidx1_seed1` 的共享映射也正确。

四个 draw 的 target gradient、selection、subset、NICE 和缓存 provenance 都已进入 master manifest，整体足以支持进入训练阶段。

## 启动 canary 前需要修的地方

### 1. Register 阶段实际没有验证 subset hash

脚本注释写的是：

> validates subset hash + 13533 rows

但实现只检查了行数：

```python
n = sum(1 for _ in open(j))
assert n == K
```

没有将 JSONL SHA256 与 `pilot_4draw_master_manifest.json` 对照。

应把每个 adapter 的 `subset_sha256` 写入 `pilot_run_plan.json`，然后 register 时同时验证：

* 文件存在；
* 13,533 行；
* byte SHA256 一致。

对于共享 Random-K，还应在 plan builder 中显式断言 STEM/HUM 两份 subset hash 完全相同，而不是仅依赖“之前检查过”。

### 2. Resume 目前是 fail-open 的 provenance

训练阶段只要发现：

```bash
$out/adapter_model.safetensors
```

就直接跳过。它不会确认这个 adapter 是否来自：

* 当前 subset；
* 当前 seed；
* 当前训练配置；
* 当前代码 commit。

Eval 阶段也只要发现任意 `results_*.json` 就跳过。

这意味着旧的、部分完成的或使用其他配置生成的 artifact 可能被误认为有效。

每个 adapter 应有 `train_manifest.json`，至少记录：

* adapter ID；
* subset path/hash；
* dataset key；
* seed；
* base model；
* 完整 resolved training arguments；
* run plan hash；
* master manifest hash；
* git commit；
* adapter file hash。

跳过训练前必须验证该 manifest 和 adapter hash，而不只是检查文件存在。

### 3. Eval 和 aggregate 没有唯一权威结果文件

当前 eval 只要求输出目录里“至少存在一个” `results_*.json`；aggregate 则读取：

```python
sorted(fs)[-1]
```

lm-eval 官方接口允许 `--output_path` 是目录或 JSON 文件；当使用目录时，结果可能按运行生成多个文件。官方仓库也有重复使用相同 output path 时旧结果不被覆盖的报告。([GitHub][1])

因此不能靠文件名字典序挑“最后一个”。

推荐：

* 每次新 eval 前要求目标目录为空，或使用唯一 run 子目录；
* eval 完成后要求恰好生成一个 result JSON；
* 写 `eval_manifest.json`，记录准确路径、SHA256、adapter hash、lm-eval 版本、task 配置和 few-shot 数；
* resume 时只接受 manifest 中指定且 hash 匹配的文件；
* aggregate 只读取 `eval_manifest.json` 指向的结果。

同时把 `lm_eval`、`accelerate`、`peft` 版本加入环境 manifest。lm-eval 返回结果中本身也包含 task config、版本、few-shot 和样本数量等字段，适合一并保留。([GitHub][2])

### 4. Aggregate 的差值方向与冻结协议相反

冻结协议定义的是：

[
\Delta_d
========

## \text{score}_{\mathrm{DSMC},d}

\text{score}_{m,d}.
]

当前代码计算的是：

```python
r["balanced"] - dsmc["balanced"]
```

也就是：

[
\text{method}-\text{DSMC}.
]

两者必须统一。建议按冻结协议改成 `DSMC − method`，列名明确写：

```text
dsmc_minus_method_balanced
dsmc_minus_method_target_weighted
```

target-weighted 也不要使用近似的 `0.797/0.203`，直接使用：

[
\frac{51}{64},\qquad \frac{13}{64}.
]

此外 aggregate 需要两种模式：

* canary：允许部分结果，例如 `--allow-partial`；
* 正式 pilot：默认要求 32/32 cells 完整，缺一个就非零退出。

否则正式运行中途误调用 aggregate，也会生成一个看起来正常但不完整的 CSV。

## 一个很小的接口问题

`ADAPTERS` 过滤器只作用于 train/eval；register 仍然会注册全部 30 个 datasets。

两种处理都可以：

* 明确规定 register 永远一次性串行注册全部 datasets；
* 或让 register 也遵守 `ADAPTERS`。

前者更简单，但注释要说明，避免误以为 canary 是完全隔离的两 adapter 操作。

## 修完后直接运行两个 canary

推荐给 Claude 的回复：

> The four-draw pre-SFT artifacts and the 32-cell/30-adapter mapping pass review. There are no remaining methodological blockers, but patch four execution-provenance issues before launching the canaries:
>
> 1. Add each adapter’s expected subset SHA256 to the run plan and validate it during registration, not only the 13,533-row count. For shared Random-K, assert the two directional subset hashes are identical.
> 2. Add per-adapter train manifests containing the subset hash, seed, resolved training arguments, run-plan/master-manifest hashes, git commit, and adapter hash. A resume skip must validate the manifest and artifact hash rather than merely detect `adapter_model.safetensors`.
> 3. Make eval produce one authoritative result referenced by an `eval_manifest.json`. Do not skip on “any results file” or aggregate via `sorted(files)[-1]`. Record the adapter hash, result hash, lm-eval/accelerate/peft versions, tasks, few-shot setting, and exact result path.
> 4. Align aggregation with the frozen convention `DSMC − method`, use exact weights `51/64` and `13/64`, and add an explicit partial mode for the canary while requiring 32/32 cells by default for the full pilot.
>
> Also clarify whether registration intentionally registers all 30 datasets even when `ADAPTERS` is set.
>
> After this patch, launch exactly the two planned canaries:
>
> * `stem80_draw0_dsmc_seed42`
> * `randk_drawidx0_seed42`
>
> Run register/train/eval/partial-aggregate, then rerun train and eval once to verify that the resume paths validate and skip the exact frozen artifacts. Check engineering only: subset/config hashes, effective batch and epochs, adapter output, authoritative eval file, shared Random-K expansion into both draw-0 cells, fail-fast, and resume behavior. Do not use canary accuracy to modify any method or hyperparameter.
>
> If both pass, continue the remaining 28 adapters without another methodological decision.

## 接下来的执行顺序

1. 修上面四个小问题；
2. 跑 2-adapter canary；
3. 重新执行一次 train/eval，确认是“validated skip”；
4. 检查 Random-K 的一个 adapter 是否正确展开成两个 aggregation cells；
5. 工程通过后，直接跑剩余 28 个；
6. 30 个完成后再分析准确率。

所以目前状态不是“继续 review artifacts”，而是：

> **pre-SFT 已通过；补齐 execution provenance 后马上跑两个 canary，canary 通过即进入剩余 28 个正式 pilot runs。**

[1]: https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/interface.md?utm_source=chatgpt.com "lm-evaluation-harness/docs/interface.md at main · EleutherAI/lm-evaluation-harness · GitHub"
[2]: https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/python-api.md?utm_source=chatgpt.com "lm-evaluation-harness/docs/python-api.md at main · EleutherAI/lm-evaluation-harness · GitHub"

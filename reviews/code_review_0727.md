我审完了 `53dfeeb`。结论是：

> **实验设计方向基本正确，但不建议直接运行当前脚本。先修一个明确的 fail-open bug，并核验候选/目标梯度缓存的 checkpoint provenance。完成后可以启动 mirror。**

网页搜索当前不可用，因此我没有刷新 GIST 等 2026 年相关工作的公开版本；下面是基于 GitHub 仓库本身的代码审查。

## 总体评价

做得正确的地方有：

* 三个方法重新走同一套 offline greedy，而不是复用 online GradCov adapter，这避免了最明显的 selector-code-path confound。
* 三个方法共享固定 subset，训练 seed 使用 `{42,1,2}`，适合作 paired SFT comparison。
* 目标、候选、预算和评测任务与 STEM80 流程基本一致。
* summary 对统计结论的措辞已经比较克制，没有宣称显著性。

但有两个问题应当在开跑前解决。

# P0：脚本目前会“失败后继续运行”

脚本使用：

```bash
set -uo pipefail
```

没有 `-e`。

这会造成以下情况：

1. selection 失败后，脚本继续 export；
2. export Python block 失败后，shell 继续打印 `registration done`；
3. SFT 失败后只打印 `[FAIL sft]`，继续下一个实验；
4. eval 失败后只打印 `[FAIL eval]`；
5. 最后仍然打印：

```text
=== T_hum80 MIRROR COMPLETE ===
```

selection 部分尤其明显：

```bash
[ -f $gc/step_1.json ] || {
  ...
  python ... && log "[done]"
}
```

命令失败时，没有任何 `exit 1`。

export 中也使用了没有 `check=True` 的 `subprocess.run`，随后直接读取预期输出。 即使这个 Python block 异常退出，因为 shell 没有 `-e`，后面的 SFT 仍然会继续。

### 建议修法

不要只简单改成 `set -euo pipefail`，因为 `||` 和条件列表中的 `set -e` 行为容易令人误解。最好显式写 fail-fast function：

```bash
set -Eeuo pipefail
trap 'log "[FATAL] line=$LINENO command=$BASH_COMMAND"' ERR

run_selection() {
  local name=$1
  local output=$2
  shift 2

  if [[ -s "$output/step_1.json" ]]; then
    log "[skip select] $name"
    return
  fi

  log "SELECT $name"
  if ! "$@" >"$LOGD/sel_${name}.log" 2>&1; then
    log "[FAIL select] $name"
    exit 1
  fi

  [[ -s "$output/step_1.json" ]] || {
    log "[FAIL select] missing $output/step_1.json"
    exit 1
  }
}
```

SFT 和 eval 同样应在失败时立即退出，而不是只记录日志。

export 改成：

```python
with open(log_path, "w") as log_file:
    subprocess.run(
        command,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        check=True,
    )
```

并在训练开始前验证每个 subset：

```python
assert len(indices) == 13533
assert len(set(indices)) == 13533
assert min(indices) >= 0
assert max(indices) < num_candidates
assert len(exported_rows) == 13533
```

这是当前最明确的代码 bug。

# P0：必须核验两个梯度缓存是否来自同一个模型状态

mirror 脚本使用：

```text
candidate:
less_output/train/1/all_projected_grads.pt

target:
mmd_grad_cov_adam_hum80_output/target/1/all_projected_grads.pt
```

它们的 projection 配置看起来一致：

* candidate LESS：`proj_dim=8192`、`seed=123`、Adam gradient；
* HUM80 target selector：`proj_dim=8192`、`seed=123`、target 使用 SGD。

但是对应配置引用的 warmup checkpoint 路径不一样：

candidate LESS 配置引用：

```text
.../sft_results/random_selected/checkpoint-1692
```

HUM80 target 配置引用：

```text
.../sft_results/warmup_seed42/checkpoint-1692
```

这两个目录可能只是同一 checkpoint 的复制或别名，也可能不是。仓库代码本身无法证明它们相同。

如果不是同一模型状态，那么

[
X_{\text{candidate}}X_{\text{target}}^\top
]

会混入 checkpoint 差异，mirror 不再只是比较一阶和二阶 kernel。

### 开跑前必须执行

```bash
sha256sum \
  /jizhicfs/karonhe/dataflex_saves/sft_results/random_selected/checkpoint-1692/adapter_model.safetensors \
  /jizhicfs/karonhe/dataflex_saves/sft_results/warmup_seed42/checkpoint-1692/adapter_model.safetensors
```

还建议比较：

```bash
sha256sum \
  .../random_selected/checkpoint-1692/optimizer.pt \
  .../warmup_seed42/checkpoint-1692/optimizer.pt
```

以及 adapter config。

判断规则：

* hash 相同：可以继续；
* hash 不同：不要运行当前 mirror，应重新生成 candidate 或 target cache，使两者来自同一个 checkpoint；
* 文件名或格式不同：至少加载 state dict 后逐 tensor 比较。

Claude 所说“所有路径存在”不够，**这里需要验证内容一致性**。

## 一个非阻断、但必须记录的理论问题

当前是：

* candidate：Adam-preconditioned gradient；
* target：raw SGD gradient。

配置明确如此设置，以对齐 LESS。

为了和已经完成的 STEM80 实验保持可比，HUM80 mirror 可以继续使用这个协议。但论文中“匹配同一个 directional second moment”

[
\mathbb E_S[uu^\top]
\approx
\mathbb E_T[uu^\top]
]

的解释并不完全干净，因为 candidate 和 target 使用了不同的梯度变换。

所以：

* **本次 mirror 保持 Adam-candidate/SGD-target，不要临时改协议；**
* 后续主论文至少补一个对称版本：

  * Adam/Adam，或者
  * SGD/SGD；
* 将 Adam/SGD 保留为 LESS-aligned protocol。

否则 reviewer 很可能指出这不是严格意义上的同一 feature map 下的 moment matching。

# P1：现有缓存判断可能复用陈旧结果

当前只检查：

```bash
[ -f $gc/step_1.json ]
```

存在就直接跳过。

但它没有验证该文件对应的：

* target cache；
* candidate cache；
* budget；
* (\lambda/\alpha)；
* 当前 commit；
* checkpoint hash。

同样，导出的 `.jsonl` 一旦存在，就不会重新生成，即使 `step_1.json` 已经更新。

这会形成隐蔽的 stale-cache bug。

最低限度应验证 `step_1.json` 的 metadata；更稳妥的是使用带 hash 的目录：

```text
hmoment_gradcov_hum80_<target_hash>_<candidate_hash>_k13533
```

导出时也应保存：

```json
{
  "selection_commit": "53dfeeb",
  "candidate_grad_sha256": "...",
  "target_grad_sha256": "...",
  "candidate_data_sha256": "...",
  "num_selected": 13533,
  "indices_sha256": "..."
}
```

在这个 mirror 首次运行且目录确定为空的情况下，这个问题不一定触发，但最好现在就修，避免未来复现实验踩坑。

# P1：summary 中有一处数学措辞不准确

summary 写道：

> Every interior λ is a Pareto improvement in selection geometry.

但表格实际上显示：

* (D_1) 下降；
* (D_2) 略微上升；
* effective rank 在 (\lambda=0.005,0.01) 时还从 2398 降到了 2380、2385。

严格说，这不是 Pareto improvement，因为至少一个指标变差。

建议改成：

> Increasing (\lambda) traces a smooth trade-off: it substantially improves (D_1), incurs only a small degradation in (D_2), and generally increases effective rank for (\lambda\ge 0.02).

另外：

> any (\alpha>0) lets the first-order term hijack the ranking

也略强。 更准确的是：

> the tested coarse grid (\alpha\in{0.25,0.5,0.75,1}) was first-order dominated.

因为非常小的 (\alpha) 不一定立即完全主导。

# 是否保留 linear 的三个 seed

我不建议完全删掉 linear，但也不认为它值得直接跑三个 seed。

mirror 的核心统计问题是：

[
\lambda=0
\quad\text{vs}\quad
\lambda=0.02.
]

linear 只是解释 endpoint，并不是需要精确估计的主要 paired comparison。

更高效的设计是：

* GradCov：seed 42、1、2；
* joint (\lambda=.02)：seed 42、1、2；
* linear：先只跑 seed 42。

总共 **7 次 SFT，而不是 9 次**。

如果 HUM80 的 linear seed42 明显低于另外两个，例如低 1–2 个点，就没有必要补 seed1/2；如果它意外接近或超过 GradCov，再追加两个 seed。

这样保留了解释价值，同时减少两次大概率低信息量的训练。

若 11–12 小时算力成本完全不敏感，跑完整 9 次也没有设计错误，只是算力利用率不够高。

# 还建议补一个最终完整性检查

目前脚本没有自动汇总结果。建议结束时强制验证预期数量：

```bash
expected=7  # 或完整设计中的 9
actual=$(find "$EVAL/skew" \
  -path "*hum80_seed*" \
  -name "results_*.json" | wc -l)

[[ "$actual" -eq "$expected" ]] || {
  log "[FATAL] expected $expected eval outputs, found $actual"
  exit 1
}
```

然后自动输出一个 CSV/JSON：

```text
method,seed,stem,hum,balanced
```

不要依赖日志中的 “COMPLETE” 判断实验完成。

# 最终 launch 建议

**不要直接运行 commit `53dfeeb` 中的原脚本。**

先完成下面四项：

1. 将 selection、export、SFT、eval 改成 fail-fast；
2. 验证两个 checkpoint 的权重 hash 是否一致；
3. 验证 candidate/target cache 的 shape、projection seed、checkpoint provenance；
4. 修复 stale-cache 校验，至少确认本次三个输出目录和三个 subset 文件为空。

完成后，实验设计可以启动。我建议采用：

[
\text{GradCov}*{3\ seeds}
+
\text{joint}*{3\ seeds}
+
\text{linear}_{1\ seed},
]

而不是彻底删除 linear，也不必默认给 linear 三个 seed。

一句话 verdict：

> **设计通过，当前执行脚本需小改；checkpoint provenance 核验是 launch blocker，fail-open 是代码 blocker。**

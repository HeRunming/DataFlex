现在应选：

> **1. Launch 28 now。**

Canary 已经覆盖了正式运行最关键的两条路径：

* 普通 draw-specific adapter：`stem80_draw0_dsmc_seed42`
* 跨方向复用 adapter：`randk_drawidx0_seed42`

训练、评测、唯一结果文件、aggregation 展开以及 hash-validated resume 都通过。仓库中的 partial aggregate 也正确出现了 3 个 cell：一个 DSMC cell，以及同一个 Random-K adapter 映射出的 STEM/HUM 两个 cell。

因此继续 hold 不会增加方法学信息，只会推迟已经冻结的实验。DSMC 比 Random-K 高 0.43 pp 只是单 draw 的中间结果，不应被用于决定是否继续；正确的 gate 是工程是否通过，而这一点已经通过。

## 给 Claude 的回复

可以直接回复：

> Launch the remaining 28 adapters now. The canary engineering gate has passed, so do not make any further methodological or configuration changes.
>
> Before launching, explicitly clear the canary-only environment variables so no filter or partial aggregation leaks into the full run:
>
> ```bash
> env -u ADAPTERS -u AGG_ALLOW_PARTIAL \
>   PHASES=register,train,eval,aggregate \
>   bash experiments/less_aligned/run_pilot_sft.sh
> ```
>
> Include the cheap idempotent `register` phase again as a final hash-validated preflight. Run the command in a detached `tmux` session with a top-level log and outer `pipefail`, so a terminal disconnect does not terminate the experiment:
>
> ```bash
> tmux new -s targetdraw_pilot
> cd /jizhicfs/karonhe/DataFlex_fa
> set -o pipefail
> env -u ADAPTERS -u AGG_ALLOW_PARTIAL \
>   PHASES=register,train,eval,aggregate \
>   bash experiments/less_aligned/run_pilot_sft.sh \
>   2>&1 | tee /jizhicfs/karonhe/dataflex_saves/logs/pilot_full_0801.log
> ```
>
> Stop only on an engineering failure: manifest/hash mismatch, training or evaluation failure, invalid resume state, multiple/missing authoritative result files, or final aggregation below 32/32 cells. Do not stop or change anything based on interim accuracy or method ordering.
>
> When complete, report the frozen 32-cell table, the 30 train/eval manifest status summary, per-draw DSMC-minus-method differences for balanced and target-weighted scores, and a concise failure/retry log. Do not expand to five draws yet; bring the completed two-draw-per-direction pilot back for analysis first.

显式清除 `ADAPTERS` 和 `AGG_ALLOW_PARTIAL` 很重要，因为子进程会继承父 shell 中导出的环境变量；GNU `env -u` 正是用于从本次命令的环境中删除指定变量。([GNU][1]) `tmux` detach 后会让会话中的程序继续运行，适合这类持续约一天半的任务。([GitHub][2])

## 你接下来需要做什么

授权 Claude 启动后，**等最终结果即可**。预计连续运行约 36 小时，按当前时间计算，大致在 8 月 3 日上午前后完成，实际取决于 eval 时长和机器占用。

中间不需要：

* 查看某个方法的暂时 accuracy 后作选择；
* 删除表现差的 baseline；
* 调整 seed、epochs、rank、NICE 参数或预算；
* 提前扩展到 5 draws。

只有脚本因 provenance、训练、评测或 aggregation 错误停止时才需要介入。正常完成后，我们下一次要做的是分析 4 个 draw × 8 方法的 pilot 表，决定是否值得扩展到完整的每方向 5 draws，而不是继续调方法。

[1]: https://www.gnu.org/s/bash/manual/html_node/Command-Execution-Environment.html?utm_source=chatgpt.com "Command Execution Environment (Bash Reference Manual)"
[2]: https://github.com/tmux/tmux/wiki/Getting-Started?utm_source=chatgpt.com "Getting Started · tmux/tmux Wiki · GitHub"

这版比上一轮可靠很多，但我仔细查完后，**还不建议现在启动 selection canary**。`3072` 截断修复和 few-shot near-neighbor 的处理都做对了；不过我又发现了两个 launch 前应该解决的问题，其中一个是明确的执行 bug，另一个会影响这次 external validation 的科学解释。

先确认已经通过的部分。新的 token audit 确实是 192/192 全覆盖，`cutoff_len=3072` 后 query、CoT cue、assistant target 都完整，最大 gradient-side 总长度 2608；SFT cutoff 仍是 2048，candidate cache不动。 3072 这个选择也有合理外部依据：Llama 2 支持最长 4096 tokens，而你们 pinned BBH generation预留 1024，所以用 3072 作为“与 evaluation input ceiling 对齐”的 target-gradient cutoff是合理的 pre-compute integrity fix。([GitHub][1])

few-shot minimal pair降成 disclosure我也同意。官方 BBH 本身就是带固定 CoT prompts 的 23-task benchmark，保留官方 demo结构比看见 near-neighbor 后去删 draw 或换 demonstration 更干净。([GitHub][2]) 当前 artifact也确认 exact identity是 0，并把官方 near-neighbor保留下来。

但下面两个问题需要先处理。

**第一个是明确的 P0：当前 `setup_draw_target.py` 其实还不能直接生成 BBH target-gradient config。**

脚本仍硬编码：

```python
jsonl = f"{ROOT}/data/target_draws/{draw}.jsonl"
```

而你真正冻结的 BBH query data 在：

```text
data/bbh_external/query_prompts/bbh_query_draw0_prompts.jsonl
```

并且里面确实是完整 3-shot BBH query + final-answer supervised message。

run plan 的名字又是 `bbhx_draw0`。 我实际检查不到一个已提交的：

```text
data/target_draws/bbhx_draw0.jsonl
```

所以如果 fresh Claude现在按 execution contract 调：

```bash
setup_draw_target.py --draw bbhx_draw0 --cutoff_len 3072
```

它会找错输入路径，除非它在计算节点上临时做了一个没有进入 provenance 的 copy/symlink。

这个必须在 canary 前修。最好的做法不是复制文件，而是把 setup script参数化，例如：

```text
--target_jsonl data/bbh_external/query_prompts/bbh_query_draw0_prompts.jsonl
```

默认仍保留旧 MMLU 路径。BBH 调用时应 fail-loud 验证：

* exact file SHA == `bbh_query_prompt_manifest.json`；
* exactly 64 rows；
* ordered IDs hash一致；
* emitted dataset registration指向这个 exact file；
* emitted YAML `cutoff_len: 3072`。

否则前面做得很漂亮的 192-record audit，可能和真正 gradient extraction读的并不是同一个文件。

---

**第二个我会当作 scientific launch blocker：现在 pinned 的 lm-eval v0.4.5 BBH prompt含有一个后来被官方修复的已知 prompt bug。**

你们当前已经生成的 query prompt里，我能直接看到：

```text
A: Let's think step by step.
 Let's think step by step.
...
```

也就是每个 few-shot demonstration 的 CoT trigger重复了一次。

这不是我猜测 upstream 行为。lm-evaluation-harness 后来的官方 release notes明确记录了一个 BBH 修复：

> removed redundant “Let's think step by step” text from `bbh_cot_fewshot` prompts.

([GitHub][3])

这件事在你们这里尤其值得修，因为论文的核心比较是：

[
\text{target-aware selectors} \quad vs \quad \text{Random}.
]

Random 完全不使用 query prompt；DSMC、RR、LESS-style 都使用 query gradients。因此，一个已知有 bug 的 query prompt不是简单“所有方法共同受影响”——它可能专门改变 target-aware selection 的 gradient geometry，而 Random完全不受影响。

现在又**一条 BBH accuracy都还没有看过**，所以这是修它的最佳时点。修完不存在 outcome-driven tuning问题。

我不建议因此升级整个 lm-eval environment，因为那会顺带引入很多无关变更。更干净的是：

> 保持当前 pinned execution code/environment，但把 custom BBH task prompt semantics升级到 upstream已经修复的 BBH prompt版本，明确引用对应 upstream fix/commit，并重新生成 query prompts与 custom held-out configs。

要求每个 demonstration只出现一次 CoT cue。然后重新跑：

* prompt parity；
* 81-demo provenance；
* 192-record truncation；
* all hashes；
* launch receipt。

由于去掉重复文本只会让 prompt变短，3072 cutoff大概率仍然完全够，但必须让 machine gate重新确认，不能推测。

---

还有一个我会顺手补的 P1。`setup_draw_target.py` 里 target-gradient extraction仍然写：

```yaml
lora_dropout: 0.1
```

这里**不要直接改成 0.05**。历史 MMLU target-gradient config本来就是 0.1。 下游正式 SFT 才是你们后来审计出的 0.05。

所以 execution contract最好明确分开写：

```text
target-gradient extraction LoRA dropout = 0.1
selected-data SFT LoRA dropout = 0.05
```

这能防止新的 Claude窗口以后看到 provenance里“正确 dropout=0.05”，又误把 gradient extraction配置改掉。

而且 canary时我建议顺便记录运行时实际加载的 PEFT config和 `model.training` 状态。若 LoRA dropout实际处于 active training mode，再把 draw0 target extraction从 clean cache重复一次，检查 projected-gradient tensor hash是否可复现；64条 target的代价很低。若它本来被 loaded adapter config覆盖或处于 deterministic状态，这个检查也会直接告诉我们。

---

所以我现在的状态判断是：

> **3072 truncation fix：PASS。**
>
> **few-shot minimal-pair disclosure：PASS。**
>
> **frozen Adam-candidate / SGD-target contract：PASS。**
>
> **实际 BBH target-data wiring：FAIL，需修。**
>
> **BBH prompt fidelity：发现已知 v0.4.5 duplicate-CoT bug，建议 compute 前修。**
>
> 因此当前仍然是 **HOLD selection canary**。

可以直接回复 Claude：

> Hold the canary for one more patch. The two previous blockers are resolved correctly, but review found two new pre-compute issues.
>
> First, `setup_draw_target.py` is still wired to `data/target_draws/{draw}.jsonl`, while the frozen BBH queries live at `data/bbh_external/query_prompts/bbh_query_draw{d}_prompts.jsonl`; there is no committed `data/target_draws/bbhx_draw0.jsonl`. Parameterize the setup script with an explicit `--target_jsonl` (keeping the old MMLU path as the default), and for BBH fail-loud on the exact prompt-manifest SHA, 64-row count, ordered IDs, and emitted `cutoff_len: 3072`. Do not create an untracked copy/symlink as hidden provenance.
>
> Second, the currently pinned v0.4.5 BBH prompts visibly contain duplicated CoT triggers inside the few-shot demonstrations (`A: Let's think step by step.` followed by another `Let's think step by step.`). Upstream lm-eval later explicitly fixed this redundant BBH text. Because query-dependent selectors use these prompts while Random does not, do not knowingly launch the external comparison with the known buggy prompt. Keep the current execution environment if desired, but update the custom BBH prompt semantics to the upstream-corrected/official BBH CoT form, pin the exact upstream fix, regenerate both held-out eval configs and query prompts, and rerun parity, demo provenance, truncation, and hash gates. No accuracy has been observed, so this is a clean pre-compute protocol correction.
>
> Also record explicitly that target-gradient extraction retains the historical `lora_dropout=0.1`, while downstream SFT uses 0.05; do not conflate the two. In the selection canary, log the actually loaded PEFT dropout/training state and, if dropout is active during gradient extraction, repeat draw0 target extraction once to verify reproducibility.
>
> After those patches, regenerate the clean-head receipt. If all gates pass, proceed directly with the no-SFT + draw0 five-selector selection-only canary.

这次 review 也说明换窗口后加 machine-readable execution contract是非常值得的：Claude已经把前两个上下文错误修掉了，但**数据路径和已知 prompt bug 都是单靠“all gates green”仍然可能漏掉的执行细节**。现在把它们清掉以后，我对正式 BBH 实验会放心很多。

[1]: https://github.com/meta-llama/llama?utm_source=chatgpt.com "GitHub - meta-llama/llama: Inference code for Llama models · GitHub"
[2]: https://github.com/suzgunmirac/BIG-Bench-Hard?utm_source=chatgpt.com "GitHub - suzgunmirac/BIG-Bench-Hard: Challenging BIG-Bench Tasks and Whether Chain-of-Thought Can Solve Them · GitHub"
[3]: https://github.com/EleutherAI/lm-evaluation-harness/releases?utm_source=chatgpt.com "Releases · EleutherAI/lm-evaluation-harness · GitHub"

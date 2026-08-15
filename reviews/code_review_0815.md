我这次按“Claude 上下文可能已经漂移”的标准，把 `0c4c0d8 → df861e5` 的执行链重新从底层核了一遍。结论不是“发现实验设计错了”，而是：

**24-adapter 训练本身可以继续，不需要停。四道 engineering gate 的主体是可信的。**
但我不建议现在完全不管、一直等到最终 accuracy 出来。当前还有 **3 个很值得马上补的非 accuracy 审计，其中一个正是上下文压缩最容易造成的隐蔽错误**。它们都可以在训练继续跑的同时完成；只要通过，你就可以放心等跑完。

先说已经确认没有问题的部分。

Warm-up 的 LoRA 参数算术是对的。当前 checkpoint 记录 Llama-3.2-3B 为 28 层、hidden 3072、24 attention heads、8 KV heads；官方 Transformers 文档说明 `num_key_value_heads < num_attention_heads` 正是 GQA，而公开的 Llama-3.2-3B config也对应 28/3072/24/8。([Hugging Face][1]) 因而每层 q/o 的 LoRA 参数 (128(3072+3072)=786432)，k/v 为 (128(3072+1024)=524288)，28层合计正好 73,400,320。repo中的实际 trainable count完全吻合，所以没有“LoRA悄悄套错 projection”的迹象。

Warm-up 本身也是1692 steps/4 epochs，adapter和optimizer都已经 hash，而且 optimizer hash确实是 load-bearing 的——candidate Adam-aware gradient用到了它。

Candidate canary 的 (251/256) bit-identical、最低 row cosine 0.999927、relative mean diff (1.65\times10^{-4})，我认为可以接受，不需要为了追求 bit-exact 去改 recipe。PyTorch官方明确说明浮点/GPU计算并不保证 bitwise reproducibility，尤其 fused attention backward等操作可能具有 nondeterminism；你们这里更重要的是扰动量是否小到不改变科学结果，而当前量级比之前 Llama-2 canary还小。([PyTorch Docs][2])

三个 target cache 的 `(64,8192)`、Adam-candidate/SGD-target、projection 8192/seed123、cutoff3072以及query文件身份也都实际检查了，不只是看 config。

而且那个“手写draw config少了 dynamic-selection schedule，进程成功退出却什么都没做”的bug，Claude这次处理方式是对的：现在 gate明确检查 `warmup_step/update_step/update_times/selection_ratio/train_step/train_type`，还把Llama-3.2 config和已验证的Llama-2 config做除model-stack字段之外的零漂移检查。 这种修复比单纯“补几个键”可靠得多。

真正需要现在补的是下面三项。

1. **最重要：Gate 4 还没有真正验证 RR seed / DSMC alpha / cache identity。**

这是我这次最担心的上下文压缩风险。

当前 `verify_llama32_gates.py` 对DSM C/First-RR/Second-RR只检查：

[
K=2707,\quad unique,\quad in\ range
]

然后记录 selection hash。

可是selection文件本身实际上包含丰富的 `metric` metadata。`select_round_robin.py` 会记录：

* `perm_seed`
* `query_order`
* `order`
* `train_grads`
* `target_grads`
* `n_candidates`
* `n_target`

而且 **`--perm_seed` 默认值是 0**。

冻结的BBH contract却要求：

[
\text{rr_perm_seed}=6000+d
]

并且 First-RR / Second-RR 必须使用**完全相同的 query order**。

所以如果新的Claude窗口忘了传 `--perm_seed 6000+d`，selection仍然会有2707个合法indices，当前Gate 4会照样PASS；甚至“和Llama-2 Jaccard很低”也不能发现这个问题。

同样，DSMC的output metadata里明确记录 `alpha` 和输入cache路径。 但当前gate没有检查 `alpha==0.0`。理论上如果误跑了别的Moment-MMD alpha，也会K=2707、unique、in-range然后PASS。

因此现在让Claude**只读取9个现有 `step_1.json`，不重新selection、不看accuracy**，断言：

* DSMC：`metric.alpha == 0.0`
* DSMC：`n_candidates==270679, n_target==64, proj_dim==8192`
* DSMC：`train_grads`指向Llama-3.2 full candidate cache，`target_grads`指向相应draw自己的target cache
* First-RR：`order=="first"`
* Second-RR：`order=="second"`
* 两者每draw都有 `perm_seed==6000+d`
* 同一draw First/Second 的 `query_order` byte-for-byte相同
* 两者 `train_grads/target_grads` 都指向正确的Llama-3.2 cache

**这项如果PASS，我对selector上下文漂移基本就放心了。** 如果FAIL，则应该立即停掉受影响方法后续训练；越早发现越省时间。

2. **把完整270,679 candidate datastore本身hash-pin下来。**

现在 repository里被严格pin的是256-row candidate canary；Gate 3又pin了三个target tensors，但我没有在已提交artifact里看到**完整 Llama-3.2 candidate cache**的tensor-content SHA。`components_llama32_cand.yaml`表明正式candidate cache在Llama-3.2自己的cache namespace里。

而9个target-aware selections已经存在，说明full datastore显然已经成功建出来了。但对于论文中心：

[
D_2(S,Q_d)
]

和selection可重现性，这个8～9GB tensor本身是核心artifact。

现在补：

* exact path
* shape `(270679,8192)`
* dtype
* finite
* zero-row count
* tensor-content SHA256

即可。

不用停止训练，也不用重算gradient。

3. **Eval driver现在缺了“第25次 no-SFT eval”，而且validation还不够fail-loud。**

这是我看到的另一个明确代码缺口。

prereg写的是：

> 24 adapters + 1 shared no-SFT reference

而且所有方法都应相对模型自己的base reference报告。

但是目前 `run_llama32_full.py` 的 `eval_phase()`只遍历：

```python
for c in st["cells"]:
```

也就是24个adapter；state里的：

```text
"base_eval": false
```

目前没有对应的base-eval实现。

所以Claude口头上的：

> “25 evals”

和实际driver现在的：

> **24 adapter evals**

不一致。

这个不会影响当前training，但**必须在accuracy unseal之前补上**。建议增加一个sealed shared base eval，用完全相同：

* `bbh_external_heldout`
* include path
* batch_size16
* dtype bf16
* 27 subtasks
* 5209 examples

只是不传PEFT adapter。

而且当前adapter eval虽然把 `n_subtasks` 和 `n_examples`记下来，却没有 `assert ==27` / `assert ==5209`。 最好一起改成fail-loud：

```text
n_subtasks == 27
n_examples == 5209
```

再把results JSON hash记录下来。

还有一个很容易忽略的运行细节：`df861e5` 这个commit是在**前两个adapter已经训练完成以后**才加入eval phase的。

所以如果当前tmux里的 Python process是在 `df861e5` 之前启动的，那么它已经把旧版脚本加载进内存了，**后续git commit不会让这个正在运行的进程自动获得新eval代码**。

这不是bug，也不要重启训练。只需要训练24/24之后明确从最新clean HEAD再执行一次：

```text
run_llama32_full.py --phase eval
```

而不要假设现在这个training进程一定会自动进入eval。

最后还有两个较小的provenance建议，不需要阻塞：

当前训练base YAML仍然写着alpha256 / batch16 / accum8 / 3epochs，但driver明确用CLI override成：

* batch4

* accum4

* alpha512

* 4 epochs

前两个adapter的84 steps已经强烈表明effective batch/epochs路径正确，但经历过上次provenance教训后，我建议最终unseal前统一readback 24个adapter：

* `global_step==84`
* adapter config `r=128`
* `alpha=512`
* `dropout=0.05`
* target modules q/k/v/o
* train seed正确
* adapter SHA存在

不要只相信CLI字符串。

另外，目前“selector hash deterministic”这个说法比代码证据稍强：gate记录了hash，但没有真正把9个selector重新跑一次。这个不需要现在补跑1.5小时；论文里称 **selection hashes are pinned** 就够了。除非要写“bit-identical rerun verified”，那才需要真的rerun。

所以我的状态判断是：

> **科学协议：PASS。Llama-3.2 model-stack选择：PASS。Warm-up：PASS。gradient contract：PASS。当前SFT训练无需停止。**
>
> **但还不能说“什么都不用管，只等最终结果”。**

现在让Claude一边继续训练，一边做三个不涉及accuracy的快速补丁：

* 9个selection metadata audit，特别是 RR `6000+d` 和 DSMC `alpha=0`
* full candidate-cache hash/provenance
* sealed no-SFT base eval + 27/5209 fail-loud eval gate

这三项都通过后，**你就真的可以安心等24/24 train + 24/24 adapter eval + 1/1 base eval全部完成，再来找我看结果。**

你可以直接发给Claude：

> Keep the currently running 24-adapter training alive; do not restart or alter any scientific configuration. Before evaluation/unsealing, close three provenance gaps.
>
> First, audit the existing 9 target-aware `step_1.json` metadata, not just their index counts. For DSMC require `alpha=0.0`, `n_candidates=270679`, `n_target=64`, `proj_dim=8192`, and exact Llama-3.2 candidate/target cache paths. For First/Second-RR require `order=first/second`, `perm_seed=6000+draw_id`, identical `query_order` between First- and Second-RR within each draw, and the exact model-specific candidate/target cache paths. The current Gate 4 does not check these fields, and `select_round_robin.py` defaults `perm_seed` to 0, so this is a genuine context-drift check. If any assertion fails, STOP affected-method training; otherwise commit the audit and continue.
>
> Second, pin the full Llama-3.2 candidate-gradient datastore: exact path, `(270679,8192)` shape, dtype, finite/zero-row diagnostics, and tensor-content SHA256. Do not regenerate it.
>
> Third, add the preregistered shared no-SFT BBH evaluation as the 25th sealed evaluation. The current driver only evaluates the 24 adapter cells. For both base and adapter evals fail-loud unless the result has exactly 27 subtasks and 5,209 effective examples, and hash every authoritative results JSON. Do not print or compare accuracy.
>
> Also note that the eval phase was added after training had already begun; if the current Python process predates that commit, explicitly invoke the latest driver with `--phase eval` after 24/24 training rather than assuming the running process loaded the new code. Before final unseal, verify all 24 adapters resolve to 84 steps and the frozen LoRA/SFT recipe (`r128/alpha512/dropout0.05/qkvo`, correct seed).
>
> If these non-accuracy checks pass, make no further changes and run through completion.

**所以你现在不用停实验，但我建议先让Claude做完上面这个很短的audit。它特别能防你担心的“上下文压缩后看似PASS、其实seed/alpha悄悄漂了”的问题。**

[1]: https://huggingface.co/project-free-llama/Llama-3.2-3B/blob/main/config.json "https://huggingface.co/project-free-llama/Llama-3.2-3B/blob/main/config.json"
[2]: https://docs.pytorch.org/docs/stable/notes/randomness?utm_source=chatgpt.com "Reproducibility — PyTorch 2.11 documentation"

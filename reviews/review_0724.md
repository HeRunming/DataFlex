我直接检查了 `fa` 分支的算法实现、配置、测试、主实验、多随机种子结果、skew stress test 和理论 proposal。公开网页检索当前不可用，因此下面对代码和实验的判断是基于仓库本身；2026 年最新相关工作与最新 baseline 还需要另做一次文献刷新。

## 总体判断

以现在的形态投稿 ICLR，我认为大概率会落在 **weak reject**：工程量和实验量已经不小，但“通用 targeted data selection 方法”的中心结论还不够强。

核心原因不是实验做得少，而是：

1. 主实验中，MMD-GradCov 没有在任何 target 上显著击败 NICE、TSDS 等最强 baseline。你自己的五随机种子总结也很诚实地写明：它在 MMLU、TyDiQA 属于第一梯队，但没有统计显著优于最强 baseline，并且在 BBH 上落后。
2. 当前最有辨识度的结果其实不是平均性能，而是 **target 分布失衡时的稳定性**：GradCov-Adam 在 STEM-majority sweep 中 balanced accuracy 基本不随 skew 变化，并表现出稳定的高 effective rank。
3. 理论目前主要是“标准 MMD/RKHS 解释 + 一个新 kernel”，还不足以证明为什么它应当提高 SFT 后的目标性能。
4. 有若干实现、实验协议和理论表述问题，会让审稿人怀疑现有结果是否完全支撑论文 claim。

我的建议不是继续以“又一个 MMD selector”横向堆实验，而是把论文升级成：

> **Beyond Mean Gradient Matching: Gradient-Moment Coresets for Robust Targeted Instruction Tuning**

也就是从“泛化的 MMD 框架”收缩到一个更尖锐的问题：

> LESS 等方法主要匹配平均目标梯度；当目标能力是多模态、梯度相互抵消，或者小 target set 存在抽样偏斜时，平均梯度不稳定。我们联合匹配一阶梯度均值和二阶梯度方向矩，构造对 target heterogeneity/skew 更稳健的 coreset。

这条线比“GradCov 在三个 benchmark 上偶尔最好”更像 ICLR 论文。

---

## 一、当前 repo 最值得保留的贡献

目前最好的三部分是：

第一，**五随机种子完整 pipeline**。你不是只换 SFT seed，而是 warmup、重新计算梯度、重新选择、重新训练和评测，确实覆盖了 selection 与 training 两类方差。

第二，**NICE、LESS、TSDS 都已经纳入比较**。尤其 NICE 使用相同候选梯度缓存，只替换目标信号，这个对照很有解释力。

第三，**skew robustness 与 effective-rank mechanism**。GradCov-Adam 的 effective rank 不仅较高，而且跨 seed 和 skew 方向的方差显著更小；这是目前仓库里最接近“机制证据”的结果。

所以这项工作并不是推倒重做，而是需要重新确定中心命题，并围绕它补上理论和决定性实验。

---

## 二、理论上最需要修改的地方

### 1. 当前 stochastic greedy 的近似保证不能直接使用

代码写的是：每一步按照当前集合大小 (m+1) 选择使新 MMD 最小的样本，并声称随机子集贪心有 (1-1/e-\varepsilon) 保证。 配置文件也直接写了这一 approximation guarantee。

问题在于，当前 score

[
r_T(x)-\frac{1}{m+1}\left(r_S(x)+\frac{k(x,x)}2\right)
]

虽然确实最小化了“大小为 (m+1) 的新集合”的 MMD，但不同步骤优化的是不同归一化的目标。因此它不是对某个固定 set function 进行的标准贪心，经典 stochastic greedy 定理不能直接套用。

更干净的做法是明确最终预算 (K)。对于固定 (|S|=K)，最小化 MMD 等价于最大化

[
F_K(S)
======

## 2K\sum_{x\in S}r_T(x)

\sum_{x,x'\in S}k(x,x').
]

其边际增益是

[
\Delta_K(x\mid S)
=================

## 2K r_T(x)

## 2\sum_{s\in S}k(x,s)

k(x,x).
]

当 kernel 非负时，(F_K) 是 submodular，但未必 monotone。这样你有三个选择：

* 使用 budget-aware deterministic greedy，并只给出 submodularity 与经验近似质量；
* 使用适用于 non-monotone submodular maximization 的 RandomGreedy 类算法；
* 加一个不改变固定预算最优解和贪心排序的 modular shift，把结果写成 additive approximation guarantee，而不是当前不准确的乘法保证。

最重要的实验是比较：

[
\text{prefix-normalized greedy}
\quad\text{vs}\quad
\text{fixed-budget greedy}
\quad\text{vs}\quad
\text{non-monotone random greedy}.
]

在小规模 (N\le 20) 的数据上还能枚举最优集合，画出 objective gap。简单的合成例子就能构造出当前 prefix greedy 最终 MMD 明显差于 fixed-budget greedy 的情况。

### 2. “Gradient covariance”这个名称需要更精确

梯度在 merge 后被逐样本 L2 normalize。 因此目前

[
k(g,g')=\langle g,g'\rangle^2
]

匹配的不是原始梯度协方差，而是**投影后的单位梯度方向二阶矩**：

[
M_P=\mathbb E_{u\sim P}[uu^\top],
\qquad
u=\frac{\Pi g}{|\Pi g|}.
]

更准确的名称是：

* gradient directional second moment；
* gradient subspace kernel；
* normalized gradient moment matching。

这反而能得到一个非常漂亮的严格结论：

[
\operatorname{MMD}_{k_2}^2(P_S,P_T)
===================================

\left|
\mathbb E_S[uu^\top]
--------------------

\mathbb E_T[uu^\top]
\right|_F^2.
]

由此立刻得到两个 theorem。

对于任意对称矩阵 (A)，

[
\left|
\mathbb E_S[u^\top Au]
----------------------

\mathbb E_T[u^\top Au]
\right|
\le
|A|*F\operatorname{MMD}*{k_2}(P_S,P_T).
]

也就是说，它控制所有有界 quadratic gradient functionals 的误差。

若目标二阶矩存在 eigengap (\gamma)，再由 Davis–Kahan 可以得到目标更新子空间恢复误差：

[
|\widehat P_S-\widehat P_T|*F
\lesssim
\frac{\operatorname{MMD}*{k_2}(P_S,P_T)}{\gamma}.
]

这能把你目前的 effective-rank 观察升级为真正的理论贡献。

### 3. 必须处理二阶 kernel 的 sign invariance

因为

[
\langle g,-g\rangle^2=\langle g,g\rangle^2,
]

GradCov 会认为相反方向的梯度完全相似。它能够保存“子空间”，但无法区分协同更新和相互抵消的更新。

这既是优点，也是明确的 failure mode：

* 当目标梯度由 (+v) 和 (-v) 两个 mode 构成时，平均梯度为零，LESS 完全失去信号；
* GradCov 仍能识别正确子空间；
* 但如果训练集合同时选择大量 (+v) 和 (-v)，真实参数更新可能抵消。

因此最自然的新方法不是只用 GradCov，而是联合一阶、二阶矩：

[
k_{\text{moment}}(u,v)
======================

\alpha, k_{\text{lin}}(u,v)
+
(1-\alpha),k_{\text{quad}}(u,v).
]

例如使用非负的线性角度 kernel：

[
k_{\text{lin}}(u,v)=\frac{1+u^\top v}{2},
\qquad
k_{\text{quad}}(u,v)=(u^\top v)^2.
]

这样可以把 LESS 和 GradCov 统一为两个退化情形：

* (\alpha=1)：一阶平均梯度匹配；
* (\alpha=0)：二阶更新子空间匹配；
* (0<\alpha<1)：联合梯度矩匹配。

### 4. 建立与 downstream target loss 的真正联系

现在的 RKHS bound 只能说“匹配某个函数类中的期望”，不能直接解释为什么 SFT 后准确率更高。

可以利用目标损失的局部二阶展开。令从分布 (P) 抽取一个训练梯度 (g)，进行一步更新：

[
\theta'=\theta-\eta g.
]

则

[
\mathbb E_{g\sim P}L_T(\theta-\eta g)
\approx
L_T(\theta)
-\eta \mu_T^\top\mu_P
+\frac{\eta^2}{2}\operatorname{Tr}(H_T M_P),
]

其中

[
\mu_P=\mathbb E_P[g],
\qquad
M_P=\mathbb E_P[gg^\top].
]

于是 selected distribution 与真实 target-training distribution 的局部动力学差异满足

[
|\Delta L_S-\Delta L_T|
\lesssim
\eta|\mu_T|,|\mu_S-\mu_T|
+
\frac{\eta^2}{2}|H_T|_F,|M_S-M_T|_F
+
O(\eta^3).
]

这条定理非常关键：

* LESS 只控制第一项；
* GradCov 只控制第二项；
* Moment-MMD 同时控制两项。

这会使算法看起来不再是“随便选了一个 polynomial kernel”，而是从局部训练动力学推出来的。

### 5. Adam candidate 与 SGD target 不能作为主要 MMD 设定

当前 Adam 组件对 candidate 使用 Adam-preconditioned gradients，但对 target 使用 raw SGD gradients。 源码也明确支持这种不对称配置。

这对 LESS baseline 是复现官方协议，但对 MMD 理论不干净：MMD 要求 candidate 和 target 使用同一个 feature map。若两边使用不同变换，严格来说不再是在统一 RKHS 中匹配两个分布。

主方法应固定为：

* SGD–SGD；或者
* Adam–Adam。

Adam-candidate/SGD-target 可以保留为 “LESS-aligned asymmetric ablation”，但不宜成为主表中的默认 Ours。它也可能是当前 Adam/SGD 排名随任务剧烈变化的原因之一，需要专门分析。

---

## 三、实验上最关键的缺口

### P0：不补会直接影响可信度

**第一，必须加入 Random、Full data、Target-only 和 token-matched Random。**

当前五随机种子主表只有十种选择方法，没有最基础的 random/full/target-only。 而你自己的 proposal 已经把它们列为必要 sanity baselines。

需要同时报告两种 full-data 对照：

* full-data full epoch：比较最终性能上限；
* full-data equal-token/equal-step：比较相同训练计算量下的效果。

**第二，所有主结果必须 token-budget matched。**

仓库已有 diagnostics 暴露了明显风险：同样选择 50 条，MMD-Emb 大约 4717 tokens，而 random 大约 6883 tokens，另一些方法只有约 2280 tokens，差距可达两三倍。

这意味着“固定 5% 样本数、训练固定 epoch”并不等于固定训练计算量。主实验至少需要同时报告：

* 固定样本数；
* 固定总 response tokens；
* 固定 optimization steps。

否则审稿人很容易认为结果来自长度偏好而非选择质量。

**第三，修正测试和文档。**

实际构造函数参数是 `target_dataset`，但测试用的是 `eval_dataset`，因此相应测试会触发 `TypeError`，而不是测试期望的 `ValueError`。

另外，算法使用的是包含对角项的 biased empirical MMD，但现有测试计算的是去除对角项的 unbiased MMD；这个测试并没有验证实际被优化的目标。

repo 中还有大量“production-ready”“publication-ready”的自动生成文档和内部绝对路径。  投稿前应整理成干净的 paper artifact，而不是保留审计过程、Claude author、内部 `/jizhicfs/...` 路径。

### P1：决定论文能否成立的实验

| 实验                      | 最小设计                                                         | 回答的问题                         |
| ----------------------- | ------------------------------------------------------------ | ----------------------------- |
| Moment kernel           | linear、quadratic、linear+quadratic，统一 feature map             | 二阶矩是否真的带来超出 LESS 的信息          |
| Greedy optimizer        | prefix greedy、fixed-budget greedy、non-monotone random greedy | 性能来自 kernel 还是错误/近似 optimizer |
| Target heterogeneity    | mode 数量、mode 夹角、梯度 cancellation 程度                           | 何时平均梯度失效，Moment-MMD 何时有效      |
| Target skew             | 50/50、70/30、80/20、90/10，两个 skew 方向                           | 是否对 target sampling bias 稳健   |
| Target size             | (8,16,32,64,128,256)，每个大小多个 target draws                     | 小 target set 下的样本效率和方差        |
| Selection budget        | 1%、2%、5%、10%                                                 | 方法优势是否只存在于 5%                 |
| Projection dimension    | 512、1024、2048、4096、8192                                      | 随机投影误差与成本–性能关系                |
| Gradient representation | raw norm、unit-normalized、clipped norm、SGD、Adam               | 当前优势究竟来自 kernel 还是预处理         |
| Cost                    | cold-start 与 cache-reuse 两种成本                                | 比 LESS、NICE、TSDS 是否值得         |
| Model/pool transfer     | 至少两种模型家族、两个 candidate pools                                  | 是否只是 Llama-2/Tulu-V2 特例       |

其中最重要的是 **target draw 必须随机化**。目前 skew 多随机种子主要改变 warmup、selection 和 SFT seed，但 target set 本身似乎固定。若声称对“小 target set 的抽样偏斜”稳健，就应使用：

[
5\ \text{independent target draws}
\times
3\ \text{training/selection seeds}.
]

统计时采用 hierarchical bootstrap 或 mixed-effects model，把 target-draw variance 与 training variance分开。

### P2：增强说服力的 stress tests

建议再补三类：

1. **Target contamination**：加入 10%、20%、40% 无关或错误 target examples。
2. **Candidate duplicates/noise**：向 candidate pool 注入重复数据和无关域数据。
3. **OOD 与 general ability retention**：不仅报告 target benchmark，也报告相关 OOD task 和通用能力退化。

当前 proposal 已经提出 target-size、candidate-pool scaling、cost-performance 和 target multimodality，但大部分还没有形成正式主结果。

---

## 四、skew 实验怎样才能变成真正的论文贡献

现在的结果已经有潜力，但“为什么 target 是 90% STEM，方法反而应保持 humanities 性能”在问题定义上需要解释，否则审稿人会说：MMD 没有忠实匹配 target 分布。

应把问题明确成：

> 真实目标是潜在的多能力分布 (P^\star)，但我们只能获得一个很小的 empirical target set。target set 中的 mode proportion 可能因为有限采样而偏离 (P^\star)。我们研究 selector 对这种 target sampling shift 的鲁棒性。

然后加入一个 balanced oracle：

* Oracle：用真实 balanced target distribution 选择；
* Observed：用 skewed empirical target 选择；
* 测量 selected set 与 Oracle 的差异、worst-group accuracy 和 balanced accuracy。

真实实验之外，再做一个可控 toy model：

* 两个 target gradient modes (v_1,v_2)；
* 控制夹角从 (0^\circ) 到 (180^\circ)；
* 控制 mixture weight 从 50/50 到 95/5；
* candidate pool 含两个正确 mode 和 distractors；
* 指标为 mode coverage、subspace recovery、mean-gradient cancellation、一步 target loss。

理论预测应是：

* 当 (v_1\approx-v_2) 时，mean gradient 接近零，LESS 退化；
* 二阶矩仍恢复 (\operatorname{span}{v_1,v_2})；
* 纯二阶方法可能发生方向抵消；
* 一阶+二阶 Moment-MMD 同时获得方向性和 mode coverage。

这个实验会非常直接地把 theorem、算法和现实 MMLU skew 结果连起来。

---

## 五、统计报告也需要升级

当前“在 (2\times SE) 内视为 tie”可以用于内部分析，但不够正式。建议主论文使用：

* 同 seed、同 target draw 的 paired difference；
* benchmark item-level paired bootstrap；
* 95% confidence interval；
* 多方法比较时使用 Holm correction；
* 预先规定一个统一的 Ours 默认配置，不能 MMLU 选 SGD、TyDiQA 选 Adam、skew 又选 Adam。

特别是最后一点很重要。现在结果显示 Adam/SGD 的最优选择明显依赖任务。 主表必须使用一个在所有任务上固定的默认方法，其余版本放 ablation，否则容易被认为是按任务挑最好结果。

建议主指标包括：

* 平均 target score；
* worst-group score；
* balanced group score；
* general retention；
* target-draw robustness：不同 target draws 下性能标准差；
* selection stability：Jaccard、mode coverage、effective-rank variance。

---

## 六、我建议的最终论文结构

最可行的主线是：

**方法贡献**

1. 将 targeted selection 写成 gradient moment coreset。
2. 推导一阶+二阶矩对局部 target training dynamics 的控制。
3. 证明 quadratic gradient kernel 的 MMD 等价于 directional second-moment operator 差异。
4. 给出 eigenspace recovery bound。
5. 提出 fixed-budget submodular objective 和理论一致的近似算法。

**实验贡献**

1. 标准 targeted SFT 主表：至少两种模型、三到四个 target tasks。
2. Controlled heterogeneity/cancellation。
3. Target sampling skew 与 target-size scaling。
4. Mechanism：mean cancellation、second-moment error、subspace distance、effective rank、mode coverage。
5. 成本、projection dimension、token-budget 和 preprocessing ablations。

**中心 claim**

不要写：

> MMD-GradCov universally outperforms prior selectors.

应写：

> Mean-gradient selection is brittle when small target sets are heterogeneous or compositionally biased. Matching first- and second-order gradient moments yields consistently better target-distribution robustness while remaining competitive in standard targeted SFT.

---

## 最小可投稿版本

投稿前至少应完成以下六项：

1. 修正 greedy 理论与 stochastic guarantee。
2. 把 GradCov 改名并准确解释为 normalized gradient second-moment matching。
3. 实现 linear+quadratic Moment-MMD。
4. 加入 random/full/target-only/token-matched baselines。
5. 对 skew、target size 使用多个独立 target draws。
6. 在第二个模型家族上复现主结论，并报告成本。

做到这些以后，这会从“有不少实验但平均结果没有明确赢”的工作，转变成一篇理论命题、failure regime、算法设计和机制实验相互闭环的 ICLR 风格论文。

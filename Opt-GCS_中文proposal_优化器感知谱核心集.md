# Opt-GCS：面向无目标 SFT 数据选择的优化器感知谱核心集方法

> **建议投稿题目**  
> **Opt-GCS: Optimizer-Aware Spectral Coresets for Target-Free SFT Data Selection**  
>
> **中文题目**  
> **Opt-GCS：在 AdamW/Muon 诱导更新空间中构造谱核心集，用于无目标监督微调数据选择**
>
> **状态**：研究 proposal / 方法设计 / 理论与实验规划  
> **版本**：中文重写版  
> **核心关键词**：SFT data selection, optimizer-induced update space, AdamW, Muon, spectral coreset, logdet, partial whitening, target-free selection

---

## 摘要

监督微调（Supervised Fine-Tuning, SFT）数据选择通常被表述为“选择单个重要样本”的问题，例如选择高 loss 样本、高梯度范数样本、与目标验证集梯度相似的样本，或 Fisher 信息量较高的样本。本文提出另一种视角：**无目标 SFT 数据选择可以被看作是在优化器诱导的局部更新空间中构造谱核心集（spectral coreset）的问题**。一个有效的 SFT 子集不应只是包含若干高分样本，而应当共同覆盖当前模型在优化器作用下的主要训练几何结构。

我们提出 **Opt-GCS**，一种优化器感知的谱核心集构造方法。对于每个训练样本，我们在固定 checkpoint 与固定优化器状态下定义其局部更新特征。对于 AdamW，该特征是由冻结二阶矩状态给出的对角预条件梯度；对于 Muon，由于其更新包含对矩阵动量的 Newton-Schulz 正交化变换，我们将样本特征定义为冻结 Muon 更新映射对样本梯度扰动的 Fréchet 导数，即 **Muon-induced local response feature**。这样，AdamW 与 Muon 都可以被统一放入“优化器诱导局部更新空间”的框架中比较。

在获得样本更新特征后，Opt-GCS 估计其协方差的主导特征子空间，进行秩截断与部分白化，并在白化后的低维更新空间中使用 greedy log-determinant 目标选择子集。白化参数控制方法在“强调高方差主方向”和“覆盖低方差稀有方向”之间的权衡。因此，Opt-GCS 的选择逻辑不是 agreement，即反复选择与某个主方向或目标方向对齐的样本，而是 coverage，即选择能够共同覆盖更新空间的互补样本。

理论上，我们给出针对固定局部几何目标的保证：在适当 concentration 与 eigengap 条件下，主导更新子空间可由 probe set 估计；随机 sketch 可以近似保持更新空间的内积几何；在固定白化特征上，logdet 目标是单调次模函数，因此贪心算法具有经典的 $1-1/e$ 近似保证。需要强调的是，这些理论保证的是**局部优化器诱导更新空间中的谱覆盖质量**，而不是直接保证下游任务准确率。后者应当通过系统化 SFT 实验与负对照实验进行验证。

---

## 1. 研究动机

### 1.1 从“样本重要性”转向“几何覆盖”

现有 SFT 数据选择方法大多隐含地在回答一个问题：

> 哪些样本本身最重要？

因此常见指标包括：

- loss；
- gradient norm；
- target-gradient similarity；
- influence score；
- Fisher information / information gain；
- embedding diversity 或 gradient clustering。

这些方法有用，但可能存在一个共同问题：它们容易选择**冗余样本**。例如，如果大量样本都指向同一个强主导梯度方向，那么基于 agreement 或 norm 的方法可能会不断选择类似样本，而忽略其他较弱但互补的更新方向。

本文提出的问题是：

> 哪一组样本能够共同重构当前模型训练中最重要的优化器诱导更新几何？

也就是说，数据选择的目标不再只是单点重要性，而是**子集覆盖能力**。一个好的 SFT 子集应当保留主要更新方向，也应当覆盖对泛化、推理、指令遵循或专业能力有帮助的互补方向。

---

### 1.2 为什么要使用优化器诱导的更新空间？

现代 LLM 训练并不沿着原始 SGD 梯度更新。AdamW 会用二阶矩估计对梯度进行自适应缩放；Muon 则会对矩阵参数的动量更新进行近似正交化。因此，两个原始梯度相似的样本，在经过优化器变换后，可能产生不同的有效局部更新方向。

这意味着，用 raw gradient 定义训练几何可能并不充分。更合理的选择是：

> 用实际训练优化器诱导的局部更新方向来表示样本。

对于 AdamW，这对应冻结状态下的对角预条件梯度；对于 Muon，这对应冻结矩阵正交化更新映射的局部响应。这样，数据选择方法能够与真实训练动态更一致。

---

### 1.3 为什么 Muon 值得作为核心部分？

Muon 的几何结构比 AdamW 更丰富。公开资料中，Muon 通常被描述为一种面向神经网络 hidden layer 矩阵参数的优化器，其核心思想是对 momentum update 进行 Newton-Schulz 迭代，从而近似正交化更新矩阵。与 AdamW 的对角缩放不同，Muon 的更新是矩阵级的、非线性的、具有方向重排效果的。

因此，Muon 为本文提供了一个非常自然且有研究价值的问题：

> 如果优化器本身改变了样本更新几何，那么数据选择是否也应该随优化器改变？

更具体地说：

- AdamW-induced geometry 与 Muon-induced geometry 是否存在显著差异？
- 用 AdamW 几何选择的数据，是否仍然适合 Muon 训练？
- 用 Muon-induced local response feature 选择的数据，是否能更好地服务于 Muon SFT？
- Muon 的矩阵正交化是否会使谱结构更清晰、更稳定，或产生不同的长尾方向？

这使得 Muon 不只是工程实现上的附加实验，而是 Opt-GCS 的核心科学问题之一。

---

## 2. 问题定义

设 SFT 候选数据池为

$D=\{z_i\}_{i=1}^n,\qquad z_i=(x_i,y_i),$

其中 $x_i$ 是 instruction，$y_i$ 是 completion。在训练 checkpoint $\theta_t$ 下，每个样本的损失为

$$
\ell_i(\theta_t)=\ell(z_i;\theta_t),
$$

样本梯度为

$$
g_i^{(t)}=\nabla_\theta \ell_i(\theta_t).
$$

给定预算 $k$，目标是选择一个子集

$$
S\subset D,\qquad |S|=k,
$$

使得在 $S$ 上训练能够尽可能保留完整数据池在当前 checkpoint 与优化器状态下的主要训练几何。

本文关注 **target-free SFT data selection**。也就是说，选择过程中不依赖目标验证集、few-shot target examples 或人工指定的目标任务分布。

---

## 3. 优化器诱导的局部更新特征

### 3.1 统一定义

令 $\mathcal{A}_t$ 表示第 $t$ 步的优化器更新映射，$S_t$ 表示当前优化器状态。我们不直接把样本表示为 raw gradient $g_i$，而是定义其在冻结优化器状态下的局部更新响应：

$$
u_i^{(t)}
=
\mathcal{L}_{\mathcal{A}_t,S_t}(g_i^{(t)}),
$$

其中 $\mathcal{L}_{\mathcal{A}_t,S_t}$ 是由当前优化器状态诱导的局部线性化特征映射。

这个定义是局部的。我们不声称它模拟“只用样本 $z_i$ 训练”所得到的完整反事实轨迹，而是只问：

> 在当前 checkpoint 与当前优化器状态下，这个样本会诱导什么局部更新方向？

这是数据选择中更可扩展、更可实现的近似对象。

---

### 3.2 AdamW-induced feature

对于 AdamW，当前二阶矩估计会形成一个对角预条件器。冻结优化器状态后，定义样本特征为

$$
u_i^{AdamW}=D_tg_i,
$$

其中

$$
D_t
=
\operatorname{diag}
\left(
\frac{1}{\sqrt{\hat v_t}+\epsilon}
\right).
$$

这可以理解为冻结状态下的 AdamW 对角预条件梯度。

需要注意：

> $u_i^{AdamW}$ 不是严格意义上的“单样本 AdamW 更新”。真实 AdamW 更新依赖 batch gradient、bias correction、weight decay 与动态 optimizer state，不能简单分解为逐样本更新。这里的 $u_i^{AdamW}$ 是一个局部几何代理，用于刻画样本在当前优化器状态下的有效更新方向。

---

### 3.3 Muon-induced local response feature

Muon 更复杂，因为它的更新映射是非线性的。对于矩阵参数 $W$，Muon 通常维护一个 momentum-like matrix $M_t$，然后对该矩阵进行 Newton-Schulz 迭代近似正交化，得到最终更新方向。

令

$$
\Phi(\cdot)
$$

表示冻结 Muon 状态下的矩阵更新变换。对于样本 $i$ 的梯度矩阵 $G_i$，它对 momentum 的一阶扰动可以写为

$$
\delta M_i=(1-\mu)G_i,
$$

其中 $\mu$ 是 momentum 系数。

由于 $\Phi$ 是非线性映射，我们不定义“per-sample Muon update”，而是定义 **Muon-induced local response feature**：

$$
u_i^{Muon}
=
\operatorname{vec}
\left(
D\Phi_{M_t}[\delta M_i]
\right).
$$

其中：

- $M_t$ 是冻结的 Muon momentum state；
- $D\Phi_{M_t}[\cdot]$ 是 $\Phi$ 在 $M_t$ 处的 Fréchet 导数；
- $\delta M_i$ 是由样本梯度诱导的 momentum 扰动；
- $\operatorname{vec}(\cdot)$ 将矩阵响应展平为向量。

这一表述的优势是严格且清晰：我们没有声称 Muon 可以被简单逐样本分解，而是研究其冻结更新映射对单样本扰动的局部线性响应。

---

### 3.4 Muon 特征的三个实现层次

为了兼顾理论严谨性与工程可行性，可以实现三种 Muon feature：

#### 3.4.1 Muon-pre：正交化前的动量扰动特征

$$
u_i^{Muon-pre}
=
\operatorname{vec}(\delta M_i).
$$

这是最便宜的 Muon 版本，主要测试 momentum-space geometry 是否已经与 raw gradient / AdamW geometry 不同。

#### 3.4.2 Muon-JVP：unrolled Newton-Schulz 的 Jacobian-vector product

若 Muon 使用 $K$ 步 Newton-Schulz 迭代，则可以 unroll 该过程，并计算

$$
u_i^{Muon-JVP}
=
\operatorname{vec}
\left(
J_{\Phi_K}(M_t)\delta M_i
\right).
$$

这是本文推荐的主 Muon 特征，因为它真正刻画了正交化变换后的局部响应。

#### 3.4.3 Muon-FD：有限差分响应特征

用于 debug 和 correctness check：

$$
u_i^{Muon-FD}
=
\frac{
\operatorname{vec}(\Phi(M_t+\eta\delta M_i))
-
\operatorname{vec}(\Phi(M_t))
}{\eta}.
$$

Muon-FD 成本更高，不适合作为大规模主方法，但可用于验证 Muon-JVP 实现是否合理。

---

## 4. 谱假设：spiked-yet-heavy-tailed update covariance

给定优化器诱导特征 $u_i$，定义其协方差：

$$
\Sigma_u=\mathbb{E}[u_iu_i^\top].
$$

本文假设 $\Sigma_u$ 呈现 **spiked-yet-heavy-tailed** 结构：

$$
\Sigma_u
=
U_r\Lambda_rU_r^\top
+
\Sigma_{\text{tail}}.
$$

其中：

- $U_r$ 包含前 $r$ 个主特征向量；
- $\Lambda_r=\operatorname{diag}(\lambda_1,\ldots,\lambda_r)$；
- $\Sigma_{\text{tail}}$ 表示剩余较弱但不可忽略的方向；
- 在选择的 rank 附近存在可用 eigengap。

这一结构意味着：更新分布既不是完全 isotropic，也不是严格低秩。少数主方向非常重要，但尾部方向可能包含稀有能力、专业知识、长尾指令模式或有助于泛化的互补训练信号。

因此，数据选择不应只追逐最大方差方向，也不应完全平等地对待所有方向。我们需要一个可以在二者之间调节的机制，这就是 partial whitening。

---

## 5. Opt-GCS 方法

### 5.1 特征提取

对每个样本 $z_i$：

1. 计算 per-sample gradient：

$$
g_i=\nabla_\theta \ell(z_i;\theta_t).
$$

2. 转换为优化器诱导特征：

AdamW：

$$
u_i=D_tg_i.
$$

Muon-pre：

$$
u_i=\delta M_i.
$$

Muon-JVP：

$$
u_i=D\Phi_{M_t}[\delta M_i].
$$

3. 随机 sketch：

$$
\tilde u_i=Ru_i,
$$

其中 $R$ 是随机投影矩阵，例如 Rademacher projection 或 CountSketch。

4. 长度归一化与裁剪：

$$
\bar u_i
=
\frac{\tilde u_i}{L_i^\alpha}
\cdot
\min\left\{
1,
\frac{\tau}{\|\tilde u_i\|}
\right\},
$$

其中：

- $L_i$ 是 completion token length；
- $\alpha\in[0,1]$ 控制长度归一化强度；
- $\tau$ 是梯度范数裁剪阈值，例如 95 分位数。

5. L2 normalization：

$$
h_i=\frac{\bar u_i}{\|\bar u_i\|}.
$$

---

### 5.2 主子空间估计

计算经验协方差：

$$
\hat\Sigma
=
\frac{1}{n}\sum_{i=1}^n h_ih_i^\top.
$$

提取前 $r$ 个特征向量与特征值：

$$
\hat U_r,\hat\Lambda_r
=
\operatorname{TopEig}(\hat\Sigma,r).
$$

rank $r$ 可以通过以下方法选择：

- effective rank；
- entropy rank；
- eigengap；
- cumulative variance；
- probe-set 稳定性；
- 小规模 downstream diagnostic。

---

### 5.3 部分白化

先投影：

$$
p_i=\hat U_r^\top h_i.
$$

再做 partial whitening：

$$
x_i^{(\beta)}
=
\hat\Lambda_r^{-\beta/2}p_i,
\qquad
\beta\in[0,1].
$$

解释：

- $\beta=0$：不白化，保留原始谱尺度，偏向高方差主方向；
- $\beta=1$：完全白化，所有保留方向等权，强调稀有方向覆盖；
- $0<\beta<1$：在 dominant direction exploitation 与 rare direction exploration 之间折中。

实践中可以加 eigenvalue floor：

$$
\tilde\lambda_j=\max(\lambda_j,\lambda_{\min}),
$$

避免白化时放大噪声方向。

---

### 5.4 LogDet 核心集选择

定义目标函数：

$$
F(S)
=
\log\det
\left(
\epsilon I_r
+
\sum_{i\in S}
x_i^{(\beta)}x_i^{(\beta)\top}
\right).
$$

初始化：

$$
S_0=\varnothing,\qquad A_0=\epsilon I_r.
$$

第 $t$ 步选择：

$$
i_t
=
\arg\max_{i\notin S_{t-1}}
x_i^\top A_{t-1}^{-1}x_i.
$$

更新：

$$
S_t=S_{t-1}\cup\{i_t\},
$$

$$
A_t=A_{t-1}+x_{i_t}x_{i_t}^\top.
$$

该边际增益具有清晰几何含义：

- 若 $\|x_i\|$ 大，说明样本在更新子空间中信号强；
- 若样本方向尚未被当前子集覆盖，则 $A^{-1}$ 会给予更高权重；
- 若样本与已选样本冗余，则边际增益自动下降。

因此，Opt-GCS 不是简单选择“强样本”，而是选择“强且互补的样本”。

---

## 6. 理论框架

### 6.1 Claim 1：优化器诱导局部特征是良定义的

在固定 checkpoint 与固定优化器状态下，AdamW 与 Muon 都诱导良定义的局部样本特征：

AdamW：

$$
u_i=D_tg_i.
$$

Muon：

$$
u_i=D\Phi_{M_t}[\delta M_i].
$$

这一定义不试图预测完整训练轨迹，而是定义一个可计算、可比较、可用于数据选择的局部训练几何。

---

### 6.2 Claim 2：主导更新子空间可估计

假设归一化更新特征 $h_i$ 在裁剪后满足合适 concentration 条件，且 population covariance 存在 eigengap：

$$
\Delta_r=\lambda_r-\lambda_{r+1}>0.
$$

则经验协方差 $\hat\Sigma$ 会集中到 $\Sigma$ 附近。由 Davis-Kahan perturbation theorem，可得到经验主特征子空间对 population 主特征子空间的近似。

这说明：如果谱结构稳定，使用一个中等规模 probe set 来估计主导 update eigenspace 是合理的。

---

### 6.3 Claim 3：随机 sketch 近似保持更新几何

若 sketch dimension $p$ 足够大，随机投影 $R$ 可以近似保持样本更新特征之间的内积与距离：

$$
\langle Ru_i,Ru_j\rangle
\approx
\langle u_i,u_j\rangle.
$$

因此，方法可以在低维 sketch space 中执行，而不必存储完整参数空间梯度。

---

### 6.4 Claim 4：固定白化特征上的 LogDet 贪心具有近似保证

给定固定特征 $x_i^{(\beta)}$，目标函数

$$
F(S)
=
\log\det
\left(
\epsilon I+
\sum_{i\in S}x_ix_i^\top
\right)
$$

是单调次模函数。对于 cardinality constraint，贪心算法具有经典的

$$
1-\frac{1}{e}
$$

近似保证。

注意：该保证只针对固定 feature space 中的 logdet coverage objective，不是对下游任务准确率的保证。

---

### 6.5 理论边界

本文不声称：

1. 局部 update-space coverage 能形式化推出 downstream accuracy 提升；
2. 一阶局部几何能准确预测完整 SFT 轨迹；
3. Muon-JVP 特征等价于长程 Muon 训练动态；
4. 无目标选择必然优于有目标验证集时的 LESS 类方法；
5. 谱结构在所有 checkpoint、所有模型规模、所有数据域上都稳定。

本文真正声称的是：

> 在固定局部优化器诱导几何下，Opt-GCS 近似最大化一个可解释、可诊断、可计算的谱覆盖目标。

---

## 7. 与已有工作的关系

### 7.1 LESS

LESS 面向 targeted instruction tuning，通过构造低维梯度特征，并选择与目标样本梯度相似的训练样本。它是 optimizer-aware 的，并且适配 Adam 与 variable-length instruction data。

Opt-GCS 与 LESS 的区别：

| 维度 | LESS | Opt-GCS |
|---|---|---|
| 是否需要目标样本 | 需要 | 不需要 |
| 选择逻辑 | train-target gradient similarity | update-space spectral coverage |
| 主要目标 | targeted capability | target-free geometry preservation |
| 优化器感知 | Adam-aware | AdamW + Muon-aware |
| 子集结构 | 对齐目标方向 | 覆盖主导更新子空间 |

LESS 应作为有目标验证集场景下的强 baseline 或 upper baseline。

---

### 7.2 OPUS

OPUS 与 Opt-GCS 概念上最接近，因为它也强调 optimizer-induced update space。OPUS 关注大模型预训练中的动态数据选择，并将候选数据的 optimizer-induced effective update 投影到由 proxy 构造的目标方向上进行打分。

Opt-GCS 与 OPUS 的区别：

| 维度 | OPUS | Opt-GCS |
|---|---|---|
| 训练阶段 | pretraining | SFT |
| 选择目标 | projected utility | spectral coverage |
| 是否 target-free | 依赖 proxy target direction | target-free |
| 选择形式 | dynamic per-iteration scoring | coreset construction |
| 几何机制 | target direction projection | eigenspace + partial whitening + logdet |
| 优化器 | optimizer-induced update | AdamW + Muon local response |

OPUS 应被表述为“证明 optimizer-induced update space 是重要方向的相关工作”，而不是直接竞争者。

---

### 7.3 FisherSFT

FisherSFT 通过 last-layer linearization 近似 Fisher/Hessian 信息，并选择最大化 information gain 的样本。它的优势是计算高效、理论清晰、只需 forward 或较轻量计算。

Opt-GCS 的区别：

- 使用多层梯度/更新信息，而非 last-layer proxy；
- 显式考虑优化器诱导几何；
- 引入谱分解、秩截断与部分白化；
- 可以比较 AdamW 与 Muon 训练几何。

公平对比时必须注意：FisherSFT 计算成本更低，因此实验不仅要比较 sample budget，也要比较 wall-clock / GPU-hour。

---

### 7.4 TAGCOS

TAGCOS 使用样本梯度表示数据，通过 clustering 与 greedy coreset 进行 task-agnostic instruction tuning data selection。

Opt-GCS 的区别：

| 维度 | TAGCOS | Opt-GCS |
|---|---|---|
| 表示 | sample gradient | optimizer-induced local update |
| 子空间发现 | clustering 隐式发现 | eigendecomposition 显式发现 |
| 多样性机制 | cluster/coreset | logdet coverage |
| 谱控制 | 无显式 whitening | rank truncation + partial whitening |
| Muon 支持 | 无 | 有 |

---

### 7.5 SPICE

SPICE 从 logdet Fisher information 出发，并加入 gradient conflict penalty，以避免被选样本之间的冲突。它与 Opt-GCS 的关系是互补的：

- SPICE 改 objective：logdet + conflict penalty；
- Opt-GCS 改 representation：optimizer-induced spectral whitened space。

一个自然扩展是：

$$
\text{Opt-GCS features} + \text{SPICE conflict penalty}.
$$

---

## 8. 实验设计

### 8.1 核心问题

实验应回答五个问题：

1. optimizer-induced geometry 是否优于 raw-gradient geometry？
2. Muon-induced geometry 是否与 AdamW-induced geometry 显著不同？
3. spectral coverage 是否优于 score-based selection？
4. partial whitening 是否优于 no whitening 和 full whitening？
5. 局部 update-space coverage 是否能转化为下游 SFT 性能提升？

---

### 8.2 推荐设置

- 模型：Llama-3.1-8B / Qwen2.5-7B / 内部同规模 base model；
- 训练方式：LoRA 或较小规模 full fine-tuning；
- 优化器：AdamW 与 Muon；
- 数据：Open-Hermes-2.5 / 内部 SFT 数据池 / 数学或专业知识数据池；
- 框架：DataFlex + LLaMA-Factory；
- 评估：
  - MMLU：知识与综合能力；
  - BBH：复杂推理；
  - GSM8K / MATH：数学推理；
  - IFEval：指令遵循；
  - MT-Bench / AlpacaEval 类：开放式对话；
  - 内部专业数学/知识 benchmark。

---

### 8.3 主 baseline

| 方法 | Target-free | Optimizer-aware | Coverage-based | 说明 |
|---|---:|---:|---:|---|
| Random | 是 | 否 | 否 | sanity baseline |
| Loss top-k | 是 | 否 | 否 | 高 loss 启发式 |
| Grad norm top-k | 是 | 否 | 否 | 检验是否只是选大梯度 |
| Embedding k-center | 是 | 否 | 是 | 语义多样性 baseline |
| TAGCOS | 是 | 部分 | 部分 | gradient clustering coreset |
| FisherSFT | 是 | 否 | 是 | last-layer Fisher/logdet |
| LESS | 否 | 是 | 否 | targeted gradient similarity |
| OPUS-style projection | 视实现而定 | 是 | 否 | optimizer-induced target projection |
| Opt-GCS Raw | 是 | 否 | 是 | raw gradient ablation |
| Opt-GCS AdamW | 是 | 是 | 是 | 主 AdamW 版本 |
| Opt-GCS Muon-pre | 是 | 是 | 是 | 便宜 Muon 版本 |
| Opt-GCS Muon-JVP | 是 | 是 | 是 | 主 Muon 版本 |

---

### 8.4 优化器几何消融

| Feature | 含义 | 预期 |
|---|---|---|
| Raw gradient | SGD-like geometry | 较弱 |
| AdamW-preconditioned | 对角自适应几何 | 强于 raw |
| Muon-pre | Muon momentum-space geometry | 与 AdamW 不同 |
| Muon-JVP | 正交化后的局部响应 | 若 Muon 几何重要，应最强 |

关键诊断：

$$
\operatorname{Jaccard}(S_{AdamW},S_{Muon})
$$

以及 AdamW 与 Muon covariance eigenspace 的 principal angle / subspace similarity。

如果 AdamW 与 Muon 选择高度重合，那么 optimizer-specific claim 变弱；如果二者明显不同，且 Muon-JVP 选择更适合 Muon 训练，则本文贡献会明显增强。

---

### 8.5 白化消融

| $\beta$ | 含义 |
|---:|---|
| 0.0 | 不白化，强调主方向 |
| 0.25 | 轻度白化 |
| 0.5 | 平衡 |
| 0.75 | 强白化 |
| 1.0 | 完全白化，强调稀有方向 |

预期：

$$
\beta=0.25 \text{ 或 } 0.5
$$

可能优于两个极端。如果 $\beta=1$ 表现较差，说明尾部方向有噪声；如果 $\beta=0$ 表现较差，说明只覆盖主方向不够。

---

### 8.6 选择目标消融

| 方法 | 测试内容 |
|---|---|
| Score top-k | 是否只需要 magnitude |
| K-center in eigenspace | 是否只需要 diversity |
| LogDet greedy | magnitude + complementarity |
| Random subspace + LogDet | learned eigenspace 是否重要 |
| Shuffled eigenvalues + whitening | eigenvalue weighting 是否重要 |
| Random selected same length distribution | 是否只是长度分布差异 |

最关键负对照：

> Random subspace + LogDet。

如果它接近 Opt-GCS，则“主导谱子空间”的 claim 会明显变弱。

---

### 8.7 Rank 敏感性

测试：

$$
r\in\{5,10,20,50,100,\text{auto}\}.
$$

报告：

- cumulative variance；
- effective rank；
- entropy rank；
- downstream score；
- selected subset overlap；
- marginal logdet gain curve。

---

### 8.8 Checkpoint 稳定性

在不同 checkpoint 计算特征：

- 初始 checkpoint；
- warmup 后；
- SFT 中期；
- 多 checkpoint 平均 covariance。

如果 eigenspace 稳定，说明 one-shot selection 合理；如果不稳定，则需要 pathwise extension 或 dynamic re-selection。

---

## 9. 诊断图表

建议至少包含以下图表：

1. raw / AdamW / Muon 特征的 eigenvalue decay；
2. cumulative explained variance；
3. effective rank 与 entropy rank；
4. AdamW vs Muon eigenspace overlap；
5. 不同 checkpoint 的 eigenspace stability；
6. score vs token length，验证长度归一化；
7. selected subset 的 2D PCA / UMAP；
8. 不同方法选出数据的 domain composition；
9. 方法之间的 Jaccard overlap heatmap；
10. marginal logdet gain curve；
11. budget-performance curve；
12. compute-performance curve。

文章主线应当是：

> Opt-GCS 不只是提高分数，而是选出了几何上更互补、更能覆盖训练更新空间的数据。

---

## 10. 预期贡献

### 贡献 1：优化器诱导的 SFT 数据几何

将 SFT 数据选择从 raw gradient / embedding space 转移到 optimizer-induced local update space。

### 贡献 2：Muon-induced local response feature

通过冻结 Muon 更新映射的 Fréchet 导数，定义 Muon 下的样本局部响应特征。该定义避免了不严谨的“per-sample Muon update”说法，同时保留了 Muon 矩阵正交化带来的几何结构。

### 贡献 3：带 partial whitening 的谱核心集选择

提出 rank truncation + partial whitening + logdet greedy 的选择流程，在主方向利用与尾部方向探索之间建立可控折中。

### 贡献 4：无目标 coverage selection

区别于 LESS 类目标相似度方法，Opt-GCS 不需要 validation / target examples，而是选择能够覆盖内在训练几何的样本。

### 贡献 5：系统化 optimizer comparison

通过 AdamW / Muon / raw gradient 的消融，验证数据选择是否真的需要与优化器几何对齐。

---

## 11. 风险与应对

### 风险 1：Muon-JVP 成本过高

应对：

- 先实现 Muon-pre；
- 只在关键 matrix layers 上计算 JVP；
- 对 Newton-Schulz 迭代使用低精度或 checkpointed JVP；
- 使用 Muon-FD 只做小规模验证；
- 报告 cost-performance trade-off。

### 风险 2：optimizer-aware feature 没有 downstream 提升

应对：

- 诚实报告；
- 分析几何差异是否存在但不影响性能；
- 将贡献定位为 diagnostic tool 或理论框架；
- 检查是否评估任务不敏感。

### 风险 3：Random subspace + LogDet 表现接近

应对：

- 弱化 eigenspace recovery claim；
- 重新定位为 update sketch space 中的 logdet diversity；
- 分析随机投影是否已经保留足够几何信息。

### 风险 4：白化放大噪声

应对：

- 使用 partial whitening；
- 设置 eigenvalue floor；
- 调 $\beta$；
- 只保留稳定 eigen-directions。

### 风险 5：OPUS 看起来太相似

应对：

明确区分：

| 维度 | OPUS | Opt-GCS |
|---|---|---|
| 阶段 | pretraining | SFT |
| 目标 | projected utility | spectral coverage |
| 监督 | proxy target direction | target-free |
| 选择形式 | dynamic scoring | coreset construction |
| 几何 | optimizer-induced projection | eigenspace + whitening + logdet |
| Muon | optimizer-aware extension | central experimental axis |

---

## 12. 建议论文结构

### Section 1: Introduction

主线：

- SFT 数据选择通常是 importance-based；
- importance-based 容易冗余；
- 我们提出 optimizer-aware spectral coverage；
- AdamW 与 Muon 诱导不同训练几何；
- Opt-GCS 在无目标条件下构造谱核心集。

### Section 2: Related Work

分组：

1. targeted gradient selection：LESS；
2. unsupervised gradient coreset：TAGCOS；
3. information/logdet selection：FisherSFT、SPICE；
4. optimizer-induced data selection：OPUS；
5. Muon 与 matrix-aware optimizers。

### Section 3: Optimizer-Induced Local Update Space

严谨定义 AdamW 与 Muon features。

### Section 4: Spectral Coreset Construction

定义 covariance、eigenspace、partial whitening、logdet。

### Section 5: Theory

保持 modest：证明固定几何目标的可估计、可保持、可优化。

### Section 6: Experiments

主结果、消融、负对照、诊断图。

### Section 7: Discussion

讨论 optimizer-induced geometry 何时重要，何时不重要。

---

## 13. 最终摘要版表述

本文提出 Opt-GCS，一种用于无目标 SFT 数据选择的优化器感知谱核心集方法。不同于按 loss、gradient norm、target similarity 或 Fisher information 对样本单独排序，Opt-GCS 选择一组能够共同覆盖当前优化器诱导训练几何的样本。对于 AdamW，我们使用冻结二阶矩状态下的对角预条件梯度；对于 Muon，我们通过冻结矩阵正交化更新映射的 Fréchet 导数定义样本的局部响应特征。随后，我们估计更新协方差的主特征子空间，进行秩截断与部分白化，并使用 greedy logdet 目标选择互补样本。该方法在固定特征上具有经典的次模近似保证，并通过系统实验比较 raw gradient、AdamW-induced geometry 与 Muon-induced geometry，检验优化器感知谱覆盖是否能够提升 SFT 数据效率。

---

## 参考文献与相关资料

1. Xia et al. **LESS: Selecting Influential Data for Targeted Instruction Tuning.** ICML 2024.
2. Deb et al. **FisherSFT: Data-Efficient Supervised Fine-Tuning of Language Models Using Information Gain.** PMLR 2025.
3. Wang et al. **OPUS: Towards Efficient and Principled Data Selection in Large Language Model Pre-training in Every Iteration.** arXiv 2026.
4. Keller Jordan. **Muon: An optimizer for hidden layers in neural networks.** 2024.
5. Liu et al. **Scalable Muon: Scaling Laws for Massive LLM Training with Muon.** arXiv 2025.
6. TAGCOS authors. **TAGCOS: Task-Agnostic Gradient Clustered Coreset Selection for Instruction Tuning Data.** NAACL Findings 2025.
7. SPICE authors. **SPICE: Submodular Penalized Information-Conflict Selection for Efficient Instruction Tuning.** ICLR/OpenReview.
8. Nemhauser, Wolsey, Fisher. **An analysis of approximations for maximizing submodular set functions.** Mathematical Programming, 1978.
9. Davis and Kahan. **The rotation of eigenvectors by a perturbation. III.** SIAM Journal on Numerical Analysis, 1970.
10. Vershynin. **High-Dimensional Probability.** Cambridge University Press, 2018.
11. DataFlex authors. **DataFlex: A Unified Framework for Data-Centric Dynamic Training of Large Language Models.** arXiv 2026.

---

## 附录 A：最小可实现版本

如果希望先做一个最快 MVP，建议顺序如下：

1. 只做 AdamW-preconditioned feature；
2. 使用 TRAK / Rademacher random projection 到 4096 维；
3. 计算 normalized feature covariance；
4. 取 $r=20,50,100$ 三个 rank；
5. 测 $\beta=0,0.5,1$；
6. 比较 Random、GradNorm、Loss top-k、Opt-GCS-Score、Opt-GCS-LogDet；
7. 加 Random subspace + LogDet 负对照；
8. 跑一个小预算 SFT，例如 1k / 5k / 10k；
9. 再加入 Muon-pre；
10. 最后实现 Muon-JVP。

这样可以避免一开始就被 Muon-JVP 工程复杂度拖住，同时保留完整研究路线。

---

## 附录 B：最关键的 reviewer 防御句

> We do not claim that optimizer-induced spectral coverage formally guarantees downstream accuracy. Our theory guarantees only the estimation, preservation, and approximate maximization of a fixed local geometric objective. The empirical question is whether this optimizer-induced local geometry is predictive of useful SFT subsets.

> We do not define a per-sample Muon update. Since Muon applies a nonlinear matrix transformation to momentum updates, we define a sample-level local response feature through the Fréchet derivative of the frozen Muon update map.

> OPUS demonstrates the importance of optimizer-induced update space for dynamic pretraining selection. Opt-GCS studies a different regime: target-free SFT coreset construction via spectral coverage rather than proxy-target projection.

> The whitening parameter is not a heuristic post-processing trick; it is the mechanism that controls the exploitation-exploration trade-off across the recovered update spectrum.


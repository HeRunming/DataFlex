我觉得这个 topic 最漂亮的讲法是：**不要把 data selection 讲成“给每条样本打 heuristic 分数”，而要讲成“在高维随机梯度场中估计主训练信号，并用有限样本子集重构这个信号子空间”**。

也就是说，理论叙事不是从 influence function 开始，而是从高维概率里的 **spiked covariance / effective dimension / eigenspace recovery / sketching concentration** 开始。这样你可以自然导出一个可实现算法，而不是先拍脑袋设计 score，再事后找理论解释。

---

## 1. 先把 SFT 数据选择重新建模

设 SFT 数据池为：

[
\mathcal{D}={z_i}_{i=1}^n
]

每条样本在 checkpoint (\theta) 上诱导一个 per-sample gradient：

[
g_i=\nabla_\theta \ell(z_i;\theta)\in \mathbb{R}^d
]

这里 (d) 极大，可能是 LoRA 参数、最后几层参数、lm_head 参数，或者经过 sketch 后的 gradient embedding。核心观点是：

> SFT 数据不是一堆文本样本，而是一堆高维随机向量 (g_i)。
> data selection 的本质，是从这些随机向量中选一个子集，使它尽可能保留整体训练信号的主方向，同时减少冗余和噪声。

已有 LESS 其实已经隐含承认了这个设定：它把 instruction data 转成低维 gradient features，再用目标任务 few-shot examples 的 gradient similarity 来做 targeted instruction tuning。LESS 的贡献是 optimizer-aware、低秩梯度相似度、可迁移 gradient datastore。([arXiv][1]) 你的方法可以说：**LESS 是 supervised/targeted gradient matching；我们做 unsupervised/intrinsic gradient geometry discovery。**

你的文档里也已经把这个方向总结为：把 per-sample gradients 看作高维随机向量，选择能覆盖主梯度子空间、同时降低冗余和方差的数据。这个表述非常好，建议作为 paper 的理论核心。

---

## 2. 高维概率叙事：从“梯度有结构”到“子空间可恢复”

### 2.1 梯度不是各向同性噪声

如果 (g_i) 真的是 isotropic noise，那 data selection 很难比 random 强太多。但深度学习里的 stochastic gradients 已经被观察到有明显非高斯、heavy-tailed、结构化特征。比如 *On the Overlooked Structure of Stochastic Gradients* 报告了 dimension-wise gradients 往往呈 power-law heavy tails，而 iteration-wise gradients 更接近 light-tailed/Gaussian，这说明梯度不是简单的各向同性噪声。([arXiv][2])

因此可以提出第一个建模假设：

[
g_i = U a_i + \xi_i
]

其中：

* (U\in\mathbb{R}^{d\times r})：低维主训练信号子空间；
* (a_i\in\mathbb{R}^r)：样本在主信号方向上的系数；
* (\xi_i)：高维噪声、冗余、domain-specific nuisance direction。

于是总体 gradient covariance 是：

[
\Sigma = \mathbb{E}[g g^\top]
= U \Lambda U^\top + \sigma^2 I
]

这就是一个标准的 **spiked covariance model**。漂亮的地方在于：你不需要一开始知道 target/dev set，只要主信号子空间存在，就可以从数据自身估计出来。

---

### 2.2 主子空间代表“可复用训练信号”

为什么 top eigenspace 有意义？

因为在 SFT 中，很多样本的梯度方向高度重复：格式遵循、基础问答、安全拒答、简单 instruction-following 等，会形成大规模 common directions。真正有价值的样本则可能落在某些 outlier subspaces 上，例如：

* reasoning transformation；
* math symbolic manipulation；
* code execution pattern；
* professional knowledge recall；
* multi-step instruction decomposition；
* domain-specific answer style。

所以 gradient covariance 的 top eigenvectors 不只是“方差最大方向”，而是**训练动态中最可复用、最稳定、最能解释 batch update 的方向**。

这与 LESS/OPUS 形成自然对话。LESS 用目标 gradient 方向去找相似训练样本；OPUS 进一步强调 raw gradient 不够，data utility 应该定义在 optimizer-induced update space 里，而不是普通 gradient space。([arXiv][3]) 你的方法可以吸收这一点：不仅可以对 (g_i) 做 covariance，也可以对 optimizer-preconditioned update (u_i=P_t(g_i)) 做 covariance。

---

## 3. 理论主线可以讲成三个定理

你不需要一开始证明非常重的 generalization theorem。最适合这个 paper 的理论结构是三个层次：

---

### Theorem 1：主梯度子空间可被有限样本稳定估计

定义经验协方差：

[
\widehat{\Sigma}
================

\frac{1}{m}
\sum_{i=1}^{m}
g_i g_i^\top
]

如果 (g_i) 是 sub-Gaussian 或经过 clipping 后近似 sub-exponential，并且 population covariance 有 eigengap：

[
\lambda_r(\Sigma)-\lambda_{r+1}(\Sigma)=\Delta>0
]

那么由 matrix concentration + Davis-Kahan，可以得到：

[
|\sin\Theta(\widehat U_r, U_r)|
\lesssim
\frac{|\widehat{\Sigma}-\Sigma|}{\Delta}
]

而 covariance estimation 的误差可由高维概率工具控制。Vershynin 的高维概率教材把 concentration inequalities、random vectors/matrices、covariance estimation、dimension reduction 等作为核心工具体系，非常适合拿来支撑这个理论叙事。([math.uci.edu][4]) 近期 Davis-Kahan-type eigenspace perturbation 仍然是统计和理论 CS 中处理 eigenspace stability 的标准工具。([arXiv][5])

这条定理在 paper 里的意义是：

> 我们不是 heuristic 地做 SVD；在有 eigengap 的情况下，少量 gradient samples 足以恢复主训练信号子空间。

---

### Theorem 2：sketched gradient 仍然保持主子空间 score

实际 LLM 不可能保存 full gradient，所以要用 sketch：

[
\tilde g_i = R g_i,\quad R\in\mathbb{R}^{p\times d}
]

这里 (R) 可以是 random projection、CountSketch、LoRA-gradient projection、LESS-style gradient feature。高维概率叙事里可以用 Johnson-Lindenstrauss / subspace embedding / Hanson-Wright 来说明：

[
\langle g_i, g_j\rangle
\approx
\langle \tilde g_i, \tilde g_j\rangle
]

以及：

[
|U^\top g_i|^2
\approx
|\tilde U^\top \tilde g_i|^2
]

只要 sketch dimension (p) 足够大，约为：

[
p = O\left(\frac{r+\log n}{\varepsilon^2}\right)
]

这条定理在 paper 里的意义是：

> 我们的方法不是只在理论 full-gradient 空间成立，而是天然支持 scalable gradient sketching。

这也能和 LESS 对齐，因为 LESS 本身就是 reusable low-dimensional gradient datastore。([GitHub][6])

---

### Theorem 3：选中的子集能近似整体主梯度协方差

最终 data selection 不是只恢复 (U_r)，而是选子集 (S)。你希望：

[
\widehat{\Sigma}_S
==================

\frac{1}{|S|}
\sum_{i\in S}
g_i g_i^\top
]

在主子空间上接近全数据：

[
|U_r^\top(\widehat{\Sigma}_S-\widehat{\Sigma})U_r|
\leq \varepsilon
]

这就把算法自然导向两阶段：

1. 先找主子空间；
2. 再选能覆盖主子空间的样本，而不是简单选 gradient norm 最大的样本。

因此 score 不应该只是：

[
s_i = |\widehat U_r^\top g_i|^2
]

还要加入 diversity / coverage。否则容易选到一堆同方向高范数样本。

更好的目标函数是：

[
\max_{S:|S|=k}
\log\det
\left(
\epsilon I+
\sum_{i\in S}
x_i x_i^\top
\right)
]

其中：

[
x_i=\widehat U_r^\top \tilde g_i
]

这等价于在主梯度子空间内做 D-optimal design / volume maximization。它比 top-score 更漂亮，因为它直接对应“覆盖主子空间”。

---

## 4. 从理论导出算法：Spec-GCS

我建议算法不要叫简单的 GCS-Select，而叫：

> **Spec-GCS: Spectral Gradient Covariance Selection for SFT**

完整算法如下。

### Step 0：选择 gradient representation

为了可落地，先不要 full gradient。选以下之一：

* LoRA 参数 gradient；
* lm_head gradient；
* last transformer block gradient；
* LESS-style gradient sketch；
* optimizer-aware update vector。

如果你想和 OPUS 对齐，可以定义：

[
u_i = P_t(g_i)
]

其中 (P_t) 是 AdamW 或 Muon 诱导的 preconditioner/update map。OPUS 的关键观点就是数据效用应该定义在 optimizer-induced update space 中，而不是 raw gradient space。([arXiv][3])

---

### Step 1：采样一小批 probe data 估计协方差

随机抽 (m) 条样本，例如 5k–50k。

计算 gradient sketch：

[
\tilde g_i = R g_i
]

做 length normalization：

[
\bar g_i = \frac{\tilde g_i}{L_i^\alpha}
]

其中 (L_i) 是 completion token length，(\alpha\in[0.5,1])。LESS 也专门处理了 Adam optimizer 和 variable-length instruction data，这说明 length normalization 在 instruction tuning gradient selection 里不是细节，而是必须处理的问题。([arXiv][7])

---

### Step 2：估计主子空间

计算：

[
\widehat{\Sigma}
================

\frac{1}{m}
\sum_{i=1}^m
\bar g_i \bar g_i^\top
]

然后 randomized SVD：

[
\widehat{\Sigma}
\approx
\widehat U_r \widehat \Lambda_r \widehat U_r^\top
]

(r) 不要手动拍脑袋，可以用 effective rank 或 eigengap：

[
r_{\text{eff}}
==============

\frac{\operatorname{tr}(\widehat{\Sigma})}
{|\widehat{\Sigma}|_{\mathrm{op}}}
]

或者 entropy effective rank：

[
r_{\text{ent}}
==============

\exp
\left(
-\sum_j p_j\log p_j
\right),
\quad
p_j=\frac{\lambda_j}{\sum_k \lambda_k}
]

这就是你的“高维概率味道”：不是调参式 rank，而是由谱结构自适应决定。

---

### Step 3：给所有候选样本投影

对每个 candidate 计算：

[
x_i = \widehat U_r^\top \bar g_i
]

基础分数：

[
s_i = |x_i|_2^2
]

但这个只能表示“样本是否落在主信号子空间中”，不能保证去冗余。

---

### Step 4：主子空间内做 coverage selection

不要直接 top-k。建议用下面三种之一。

#### 版本 A：Score + k-center

先取 top (qk) 个样本，例如 (q=5)，再在 (x_i) 空间做 k-center / farthest-first。

简单、稳定、好实现。

#### 版本 B：Score + DPP

定义 kernel：

[
K_{ij}=x_i^\top x_j
]

用近似 DPP 选样本，兼顾质量和多样性。

更像论文方法。

#### 版本 C：LogDet greedy

最大化：

[
F(S)
====

\log\det
\left(
\epsilon I+
\sum_{i\in S}
x_i x_i^\top
\right)
]

这是我最推荐作为主算法的版本。因为它和理论目标最一致：**选择一个子集，使其在主梯度子空间里张成尽可能大的体积**。

最终算法可以写成：

[
i_t
===

\arg\max_i
\log
\left(
1+x_i^\top A_{t-1}^{-1}x_i
\right)
]

其中：

[
A_{t-1}
=======

\epsilon I+
\sum_{j\in S_{t-1}}x_jx_j^\top
]

这个贪心更新很快，因为 (r) 很小，(A^{-1}) 是 (r\times r) 矩阵。

---

## 5. 最终算法伪代码

```text
Algorithm: Spec-GCS for SFT Data Selection

Input:
  Data pool D = {z_i}_{i=1}^n
  Model checkpoint θ
  Budget k
  Probe size m
  Sketch dimension p
  Rank rule RANK()
  Length normalization α
  Regularization ε

1. Sample probe subset P ⊂ D, |P| = m.

2. For each z_i in P:
      compute per-sample gradient g_i = ∇θ ℓ(z_i; θ)
      sketch gradient: ĝ_i = Sketch(g_i) ∈ R^p
      normalize: h_i = ĝ_i / L_i^α

3. Estimate covariance:
      Σ_hat = (1/m) Σ_{i∈P} h_i h_i^T

4. Compute top-r eigenspace:
      U_r = top_eigenvectors(Σ_hat, r = RANK(Σ_hat))

5. For each z_i in D:
      compute/sketch/normalize h_i
      project to signal subspace: x_i = U_r^T h_i
      optional prefilter by score s_i = ||x_i||^2

6. Select subset S greedily:
      A = εI
      S = ∅
      repeat k times:
          choose i maximizing log(1 + x_i^T A^{-1} x_i)
          S = S ∪ {i}
          A = A + x_i x_i^T

7. Fine-tune model on S.
```

这个算法非常自然：**高维概率告诉你主子空间可恢复；sketching 告诉你低维梯度可用；experimental design/logdet 告诉你怎么选覆盖子空间的样本。**

---

## 6. 这篇 paper 的理论 claim 应该怎么写

我建议你不要过度承诺“选出来一定泛化更好”。更稳的 theoretical claim 是：

### Claim A：主梯度信号可恢复

在 spiked covariance + eigengap 假设下，少量 probe gradients 即可恢复主梯度子空间。

### Claim B：sketch 后的 score 保持

在 JL/subspace embedding 条件下，sketched gradients 保持主子空间投影分数和 logdet coverage objective。

### Claim C：所选子集是 spectral coreset

Spec-GCS 选出的子集 (S) 使得主子空间内的 covariance approximation error 小：

[
\left|
U_r^\top
(\widehat{\Sigma}_S-\widehat{\Sigma})
U_r
\right|
\leq \varepsilon
]

这比“我们选的样本 loss 更高”漂亮得多。它说明：

> 我们选出的不是高 loss 样本，而是一个能重构 SFT 主训练信号的 spectral coreset。

---

## 7. 和 LESS / OPUS 的关系要讲清楚

你可以这样定位：

| 方法       |         是否需要 target/dev set | 空间                                  | 核心信号                             | 主要问题                     |
| -------- | --------------------------: | ----------------------------------- | -------------------------------- | ------------------------ |
| LESS     | 需要 few-shot target examples | low-rank gradient space             | train-target gradient similarity | 依赖 target set            |
| OPUS     |   需要 proxy target direction | optimizer-induced update space      | update projection utility        | 主要面向 dynamic pretraining |
| Spec-GCS |                         不需要 | intrinsic gradient covariance space | 主训练信号子空间覆盖                       | 需要估计 per-sample gradient |

LESS 是你的最强 baseline，因为它已经在 targeted instruction tuning 中证明了 low-rank gradient similarity 的有效性。([arXiv][1]) OPUS 是你的理论邻居，因为它强调 optimizer-induced update geometry，你可以把它作为 optimizer-aware extension 的动机。([arXiv][3])

你最核心的差异化是：

> LESS asks: which training examples align with my target examples?
> Spec-GCS asks: which examples reconstruct the dominant gradient geometry of the SFT data distribution itself?

---

## 8. 实验上最关键的验证不是 benchmark，而是“谱现象”

如果没有谱现象，这篇就会很危险。所以第一阶段一定要先做 diagnostic。

你要画这些图：

1. **Eigenvalue decay**

   * 看 gradient covariance 是否明显 low-rank / power-law / spiked。

2. **Effective rank vs checkpoint**

   * 看 (r_{\text{eff}}) 是否稳定。

3. **Top eigenspace stability**

   * 比较不同 probe subset、不同 checkpoint 的 subspace overlap。

4. **Projection score distribution**

   * 看高分样本是不是只是长样本。
   * 必须做 length-controlled analysis。

5. **Domain composition**

   * 高 score 样本里 math/code/reasoning 是否更多？
   * 如果是专业数学 SFT，是否更偏定理调用、多步证明、抽象定义？

6. **Correlation with supervised methods**

   * 和 LESS / TracIn / IF 分数算 Spearman。
   * 理想结果不是完全相同，而是中等相关 + downstream performance 接近。

这部分会让文章非常有说服力，因为你不是直接说“我提出一个方法”，而是先揭示 SFT gradient field 的高维结构。

---

## 9. 最小实验路径

我建议你按这个最小路线走：

### Round 1：只做 diagnostics

* 模型：Qwen2.5-0.5B / 1.5B 或 Llama-3.2-1B。
* 参数：LoRA gradients。
* 数据：20k–50k SFT samples。
* 输出：谱图、effective rank、length bias、domain composition。

这一步 1 周内应该能看到方向是否靠谱。

### Round 2：做 selection

预算：

* 1%
* 5%
* 10%
* 20%

比较：

* Random
* Length-stratified random
* Embedding k-center
* Kernel herding
* Gradient norm top-k
* LESS
* Spec-GCS top-score
* Spec-GCS logdet

如果 Spec-GCS-logdet 明显优于 gradient norm top-k，说明你的“coverage not magnitude”叙事成立。

### Round 3：做 optimizer-aware extension

把 (g_i) 换成：

[
u_i = \text{AdamWUpdate}(g_i)
]

或者近似：

[
u_i = \frac{g_i}{\sqrt{v_t}+\epsilon}
]

然后比较：

* raw gradient covariance；
* AdamW-preconditioned covariance；
* LoRA-only covariance；
* last-layer covariance。

如果 optimizer-aware 版本更好，就能自然连接 OPUS。

---

## 10. 我认为最漂亮的最终 formulation

你可以把整篇文章浓缩成一句话：

> **We formulate SFT data selection as spectral coreset construction in the high-dimensional gradient covariance geometry.**

然后三句话展开：

1. Per-sample gradients in SFT exhibit low effective-rank and stable outlier eigenspaces.
2. These eigenspaces capture reusable training signals without requiring target/dev examples.
3. Selecting a logdet-diverse subset in the recovered eigenspace yields a compact spectral coreset for SFT.

这个 formulation 比“我们提出一个无监督 data selection score”强很多。

---

## 11. 你可以从理论导出的三个算法版本

我建议 paper 里放三个版本，形成递进：

### Spec-GCS-Score

[
s_i=|\widehat U_r^\top h_i|^2
]

最简单，用于证明主子空间投影有用。

### Spec-GCS-Diverse

先 top-(qk)，再 k-center / DPP。

证明“去冗余”有用。

### Spec-GCS-LogDet

[
\max_{|S|=k}
\log\det
\left(
\epsilon I+\sum_{i\in S}x_ix_i^\top
\right)
]

主方法，理论最漂亮。

最终主表里应该是：

* Random
* Embedding
* LESS
* GCS-Score
* GCS-Diverse
* **GCS-LogDet**

---

## 12. 这件事最可能失败在哪里

我觉得有三个风险。

第一，**top eigenspace 可能被 length / format / boilerplate dominated**。
解决：length normalization、format filtering、per-domain whitening。

第二，**主方差方向不等于有用方向**。
解决：不要只用 top eigenvectors，可以去掉 top-1/top-2 boilerplate directions，使用 middle-outlier subspace，或者用 whitened score：

[
s_i
===

\sum_{j=1}^{r}
\frac{\langle h_i,u_j\rangle^2}{\lambda_j^\beta}
]

其中 (\beta\in[0,1])。(\beta=0) 偏向主方差，(\beta=1) 偏向 whitening 后的覆盖。

第三，**无监督 selection 可能不如 targeted selection**。
这不是致命问题。你可以把 claim 改成：

> Spec-GCS is target-free and competitive when target examples are unavailable; when target examples are available, Spec-GCS can serve as an intrinsic prefilter before LESS.

也就是：

[
\text{Spec-GCS prefilter} \rightarrow \text{LESS rerank}
]

这甚至可能是最强 practical recipe。

---

## 最后给你一个可以直接写进 paper intro 的版本

> Modern SFT data selection methods often rely on target validation examples or influence estimates. We take a different view: the SFT data pool itself induces a high-dimensional random field of per-sample gradients. If useful training signals are reusable, then this field should not be isotropic; instead, its covariance should contain stable low-dimensional outlier eigenspaces. We therefore formulate data selection as spectral coreset construction: recover the dominant gradient covariance subspace from a small probe set, project all candidates into this subspace, and select a logdet-diverse subset that best covers the intrinsic training geometry. This yields an unsupervised, target-free, and sketchable data selection algorithm for LLM supervised fine-tuning.

中文讲法就是：

> 我们不再问“这条数据像不像目标 dev set”，而是问“这条数据是否参与构成 SFT 数据池自身的主训练信号”。如果 per-sample gradients 的高维协方差存在稳定 outlier 子空间，那么数据选择就可以被形式化为一个 spectral coreset 问题：恢复主梯度子空间，并选择一个能最大体积覆盖该子空间的样本子集。

我觉得这就是这个方向最漂亮、最有理论味、也最容易导出可实现算法的讲法。

[1]: https://arxiv.org/abs/2402.04333?utm_source=chatgpt.com "LESS: Selecting Influential Data for Targeted Instruction Tuning"
[2]: https://arxiv.org/html/2212.02083v3?utm_source=chatgpt.com "On the Overlooked Structure of Stochastic Gradients"
[3]: https://arxiv.org/abs/2602.05400?utm_source=chatgpt.com "OPUS: Towards Efficient and Principled Data Selection in Large Language Model Pre-training in Every Iteration"
[4]: https://www.math.uci.edu/~rvershyn/papers/HDP-book/HDP-2.pdf?utm_source=chatgpt.com "High-Dimensional Probability"
[5]: https://arxiv.org/pdf/2409.20207?utm_source=chatgpt.com "New matrix perturbation bounds with relative norm"
[6]: https://github.com/princeton-nlp/LESS?utm_source=chatgpt.com "[ICML 2024] LESS: Selecting Influential Data for Targeted ..."
[7]: https://arxiv.org/pdf/2402.04333?utm_source=chatgpt.com "Selecting Influential Data for Targeted Instruction Tuning"

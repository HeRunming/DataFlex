这个诊断已经把问题缩小得很清楚了，但结论要精确表述为：

> **它证明了原始 Moment-MMD 的 kernel mixture 严重失衡，尚未证明“一阶信息无害”，也尚未证明 scale-normalized joint 一定有效。**

因此，下一步确实应该实现 **scale-normalized Moment-MMD**。暂时不要继续用原始 (\alpha) 扫描，也不用立刻投入多随机种子训练或 toy model。

## 诊断结果究竟证明了什么

你测得候选间的区分度：

[
\sigma_{\mathrm{lin}}=0.0058,\qquad
\sigma_{\mathrm{quad}}=0.0004,
]

两者相差约 (14.5) 倍。

原始组合的有效相对强度近似为

[
R_\alpha
========

\frac{\alpha\sigma_{\mathrm{lin}}}
{(1-\alpha)\sigma_{\mathrm{quad}}}.
]

两者达到相近尺度的临界位置是

[
\alpha_\star
============

\frac{\sigma_{\mathrm{quad}}}
{\sigma_{\mathrm{lin}}+\sigma_{\mathrm{quad}}}
\approx
\frac{0.0004}{0.0062}
\approx 0.065.
]

所以：

* (\alpha=0.25) 时，一阶排名信号约为二阶的 (4.8) 倍；
* (\alpha=0.05) 时，两者才大致同量级；
* (\alpha=0.01) 时，二阶仍占主导，但已经足以改变约 17% 的集合。

这解释了为什么此前的 (\alpha={0,0.25,0.5,0.75,1}) sweep 几乎没有真正探索“均衡 joint”区域：除 (\alpha=0) 外，其他几个点大多已经被一阶分量支配。

这也和 MMD 文献中的基本认识一致：kernel 的选择和尺度会直接决定 MMD 对哪些分布差异敏感；多 kernel 方法通常需要选择或归一化各个 kernel，而不是直接按原始数值混合。([NeurIPS 会议论文集][1]) 例如 MMD-Fuse 明确采用归一化后的多个 MMD 统计量进行组合。([NeurIPS 会议论文集][2])

但是，有一点不能过度解读：

> 这个诊断解释了“为什么一个很小的一阶权重会大幅改变选择”，还没有因果证明“此前性能下降完全由尺度失配造成”。

尤其是你观察到：

[
D_2: 0.1456\rightarrow 0.1462
]

几乎不变，但下游准确率却下降了约 2.3 个点。这说明单一的全局 (D_2) 数值还不足以预测下游性能；两个集合可能具有几乎相同的二阶 MMD，却在任务内容、长度、局部梯度结构或更高阶性质上不同。

真正的因果检验是：**归一化后 joint 是否恢复。**

---

# 现在应该实现的目标

令

[
D_1(S,T)=\operatorname{MMD}*{k*{\mathrm{lin}}}^2(S,T),
]

[
D_2(S,T)=\operatorname{MMD}*{k*{\mathrm{quad}}}^2(S,T).
]

在相同候选池、相同 target 和相同预算 (K) 下，随机抽取 (B) 个子集：

[
R_b\subset C,\qquad |R_b|=K.
]

估计两个 reference scale：

[
s_1=
\frac1B\sum_{b=1}^B D_1(R_b,T),
\qquad
s_2=
\frac1B\sum_{b=1}^B D_2(R_b,T).
]

然后优化

[
\widetilde D_\beta(S,T)
=======================

\beta\frac{D_1(S,T)}{s_1+\varepsilon}
+
(1-\beta)\frac{D_2(S,T)}{s_2+\varepsilon}.
]

这等价于使用归一化 kernel：

[
\widetilde k_\beta(u,v)
=======================

\frac{\beta}{s_1+\varepsilon}k_{\mathrm{lin}}(u,v)
+
\frac{1-\beta}{s_2+\varepsilon}k_{\mathrm{quad}}(u,v).
]

这是一个很干净的方案：

* 两个分量变成“相对于同预算 random selection 的改善程度”；
* (\beta=0.5) 才真正接近等权；
* 正系数的 PSD kernel 组合仍然是 PSD；
* 保留标准 MMD/RKHS 解释；
* greedy 代码结构基本无需改变。

## 实现时最容易出错的地方

不能只是把 `alpha` 换成两个除法，然后仍然写：

```python
self_k = torch.ones(N)
```

归一化后：

[
\widetilde k_\beta(x,x)
=======================

\frac{\beta}{s_1+\varepsilon}
+
\frac{1-\beta}{s_2+\varepsilon},
]

它虽然仍然对所有单位向量是常数，但通常不再等于 1。

应直接计算：

```python
c1 = beta / (scale_lin + eps)
c2 = (1.0 - beta) / (scale_quad + eps)

def kmix(inner):
    return c1 * (1.0 + inner) / 2.0 + c2 * inner.square()

self_k_value = c1 + c2
self_k = torch.full(
    (N,),
    self_k_value,
    device=dev,
    dtype=X.dtype,
)
```

还要把 `scale_lin`、`scale_quad`、预算 (K)、随机子集数量和 calibration seed 写进 metadata。

## reference scale 怎么估计

建议先用：

```text
B = 256
```

个随机子集，无放回抽样，每个子集大小与真实 selection budget 完全相同。

先计算 256 个 (D_1,D_2)，检查分布：

* mean；
* median；
* standard deviation；
* 5%/95% quantile。

主版本可以使用 mean，因为它与 (\mathbb E_{\mathrm{rand}}[D_j]) 的定义一致。如果分布明显受少量异常 random subsets 支配，再把 median-normalized 作为鲁棒性消融，而不是悄悄替换。

这些尺度应当：

* 对每个 target set 单独估计；
* 对每个 selection budget 单独估计；
* 不使用下游准确率；
* 固定 calibration seed；
* 多个方法共用相同 random subsets。

不要在每一个 greedy step 根据当前 score 的标准差动态归一化。那样会使每一步的目标函数变化，理论解释会从一个固定 MMD objective 退化为自适应 heuristic。

---

# 下一轮实验分三阶段做

## 阶段一：只做 selection，不训练

运行：

[
\beta\in
{0,;0.1,;0.25,;0.5,;0.75,;0.9,;1}.
]

因为归一化后 (\beta) 才具有合理的可解释性，没必要再密集扫描极小的原始 (\alpha)。

每个 (\beta) 输出：

[
\frac{D_1}{s_1},\qquad
\frac{D_2}{s_2},
]

以及：

* 与 (\beta=0) 的 Jaccard overlap；
* 与 (\beta=1) 的 Jaccard overlap；
* 一阶与二阶 greedy marginal 的候选间标准差；
* effective rank；
* gradient cosine 分布；
* STEM/人文或其他数据领域比例；
* 总 token、平均长度和长度分布。

最重要的 sanity check 是：

> 在 (\beta=0.5) 附近，归一化后一阶与二阶 marginal 的区分度应处于相近数量级。

不必完全等于 1，但不能再相差 14 倍。若仍然相差超过约 3–5 倍，说明“random MMD normalization”和“greedy marginal scale”之间还有差异，需要进一步检查，而不是马上训练。

## 阶段二：只训练三个新 joint 点

选择诊断正常后，先跑：

[
\beta\in{0.25,0.5,0.75}
]

的单 seed SFT。

已有的两个 endpoint 不需要重跑：

* (\beta=0) 与原 GradCov 相同；
* (\beta=1) 乘以正标量后，选择排序与原 linear-MMD 相同。

所以只需增加 3 次训练。

这一阶段的判断标准：

### 情况 A：normalized joint 达到或超过 0.411

例如：

[
\beta=0.5:\quad 0.412\sim0.416.
]

这会强力支持：

> 原始 joint 失败是 kernel-scale mismatch，校准后的一阶和二阶矩具有互补性。

之后再做 3 target draws × 3 training seeds，并把 Moment-MMD 继续作为主方法。

### 情况 B：normalized joint 与 GradCov 接近

例如：

[
0.407\sim0.411.
]

说明校准修复了原方法，但一阶信息暂时没有明显额外价值。

此时 Moment-MMD 可以保留为理论统一框架，但 headline 仍应偏向二阶 directional moment robustness。

### 情况 C：normalized joint 仍明显低于 GradCov

例如三个 joint 都稳定低于 0.403。

这时就可以相当有把握地说：

> 尺度失配解释了原始混合为什么被一阶分量支配，但校准后的一阶 signed information 在该 regime 中仍未带来收益。

这时不应继续为 joint 调更多超参，应把 GradCov/Directional-Moment Coreset 设为主方法，Moment-MMD 作为完整且诚实的消融。

---

# 阶段三：只有 normalized joint 有希望时才扩大实验

若阶段二出现明显希望，再做：

* 3 个独立 `T_stem80` target draws；
* 每个 draw 3 个训练 seeds；
* 对照 (\beta=0)、最佳固定 joint、(\beta=1)；
* 再在 `T_hum80` 上镜像验证；
* 增加 1% selection budget。

最佳 (\beta) 不应在每个测试 target 上根据最终准确率分别调。论文中应：

* 预先固定 (\beta=0.5)，或者
* 用独立 development target 选择一个全局 (\beta)。

否则 reviewer 很容易认为 Moment-MMD 比 GradCov 多了一个 per-task tuning knob。

---

# Toy model 暂时放后面

现在已经有了一个比 toy 更紧迫、也更有说服力的真实数据检验：

> 用 scale-normalized objective 检验 joint 是否恢复。

如果 normalized joint 获胜，再设计 toy 来解释“为什么两个 moment 都必要”。

如果 normalized joint 仍然失败，强行设计一个 joint 必胜的 toy 反而容易显得是在证明一个真实实验不支持的故事。那时 toy 更适合展示两个 endpoint 的可辨识性边界，而不应作为主结果。

## 最终建议

现在直接实现 **random-subset MMD scale normalization**，但先不要一口气重跑完整 sweep。

执行顺序是：

1. 用 (B=256) 个同预算 random subsets 估计 (s_1,s_2)；
2. 实现固定的归一化 kernel，正确修改 self-kernel；
3. 跑 7 个 (\beta) 的 selection-only 诊断；
4. 检查 (\beta=0.5) 时两个 marginal 分量是否同量级；
5. 只训练 (\beta=0.25,0.5,0.75)；
6. 根据是否恢复到 0.411 左右，决定 Moment-MMD 还是 pure second-order 成为论文主线。

当前这个诊断是好消息：它说明上一轮负结果并不是对 Moment-MMD 的最终否定，而是发现了一个具体、可检验、理论上也合理的校准缺陷。

[1]: https://proceedings.neurips.cc/paper/2012/hash/dbe272bab69f8e13f14b405e038deb64-Abstract.html?utm_source=chatgpt.com "Optimal kernel choice for large-scale two-sample tests"
[2]: https://proceedings.neurips.cc/paper_files/paper/2023/hash/edd00cead3425393baf13004de993017-Abstract-Conference.html?utm_source=chatgpt.com "MMD-Fuse: Learning and Combining Kernels for Two-Sample Testing Without Data Splitting"

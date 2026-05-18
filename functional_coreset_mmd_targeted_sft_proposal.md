# Proposal: Functional Coreset Selection via RKHS/MMD for Targeted Instruction Tuning

## 0. Working Title

**Functional Coreset Selection for Targeted Instruction Tuning**

Alternative titles:

- **MMD-Select: Functional Distribution Matching for Targeted Instruction Tuning**
- **Beyond Pointwise Influence: RKHS Coresets for Instruction Data Selection**
- **Matching Target Training-Signal Distributions for Instruction Data Selection**

---

## 1. Motivation

### 1.1 Background

Instruction tuning data selection is often formulated as a **targeted selection** problem:

> Given a large instruction-tuning candidate pool and a small set of target examples representing a desired capability, select a small subset of candidate data that best improves downstream performance on the target task.

Representative existing methods include:

- **Influence / gradient-based methods**, such as LESS, which select training examples based on gradient similarity to a few target examples.
- **Distribution matching methods**, such as DSIR and TSDS, which select data by aligning the selected distribution with a target distribution.
- **Quality / diversity / heuristic methods**, which use quality scorers, embedding similarity, data complexity, or diversity objectives.

The existing paper pool we previously organized already shows that many data attribution and data selection methods are built around **loss**, **loss decrease**, **gradient similarity**, **influence approximation**, or **cheap proxies of these quantities**. In that list, LESS is a representative targeted SFT method based on gradient similarity, while DSIR and TSDS are the closest distribution-matching baselines.

### 1.2 Core Observation

Pointwise attribution methods ask:

\[
\text{How useful is this single example for reducing target loss?}
\]

This can lead to a narrow ranking view:

\[
\text{score}(x_i) = \text{relevance}(x_i, T)
\]

where \(T\) is the target set.

However, for targeted instruction tuning, a selected subset should not only contain examples individually similar to the target. It should also **represent the target capability distribution** while avoiding excessive redundancy.

The proposed view asks instead:

> Can the selected subset approximate the target distribution in a capability-relevant function space?

This leads to an RKHS / MMD / mean-embedding formulation.

---

## 2. Problem Formulation

Let:

\[
C = \{x_i\}_{i=1}^{N}
\]

be a large candidate instruction pool.

Let:

\[
T = \{z_j\}_{j=1}^{n}
\]

be a small target set representing a downstream task or capability.

Let:

\[
S \subset C,\quad |S| = m
\]

be the selected subset, where \(m = \rho N\) and \(\rho\) is the selection ratio.

The goal is to select \(S\) such that fine-tuning on \(S\) improves target performance:

\[
\theta_S^\star = \operatorname{SFT}(\theta_0, S)
\]

\[
\operatorname{Eval}(\theta_S^\star, \mathcal{B}_{target})
\]

is maximized.

Instead of directly estimating individual sample utility, we define a kernel \(k\) over instruction examples and select \(S\) by minimizing the Maximum Mean Discrepancy between the selected empirical distribution and the target empirical distribution:

\[
S^\star =
\arg\min_{S\subset C,\ |S|=m}
\operatorname{MMD}_k^2(\widehat{P}_S,\widehat{P}_T)
\]

where:

\[
\widehat{P}_S = \frac{1}{m}\sum_{x_i\in S}\delta_{x_i}
\]

\[
\widehat{P}_T = \frac{1}{n}\sum_{z_j\in T}\delta_{z_j}
\]

---

## 3. RKHS and Mean Embedding View

Let \(k(x,x')\) be a positive definite kernel, with corresponding RKHS \(\mathcal{H}_k\) and feature map:

\[
\phi(x)=k(x,\cdot)\in \mathcal{H}_k
\]

The mean embedding of a distribution \(P\) is:

\[
\mu_P = \mathbb{E}_{x\sim P}[\phi(x)]
\]

For any \(f\in \mathcal{H}_k\), by the reproducing property:

\[
f(x) = \langle f,\phi(x)\rangle_{\mathcal{H}_k}
\]

Therefore:

\[
\mathbb{E}_{x\sim P}[f(x)]
=
\langle f,\mu_P\rangle_{\mathcal{H}_k}
\]

This means \(\mu_P\) encodes the expectation of every function in the RKHS under distribution \(P\).

---

## 4. MMD as Functional Discrepancy

The Maximum Mean Discrepancy between \(P\) and \(Q\) is:

\[
\operatorname{MMD}_k(P,Q)
=
\|\mu_P-\mu_Q\|_{\mathcal{H}_k}
\]

Equivalently:

\[
\operatorname{MMD}_k(P,Q)
=
\sup_{\|f\|_{\mathcal{H}_k}\le 1}
\left|
\mathbb{E}_{x\sim P}[f(x)]
-
\mathbb{E}_{x\sim Q}[f(x)]
\right|
\]

This gives the main theoretical interpretation:

> MMD is the worst-case expectation discrepancy over all unit-norm functions in the RKHS.

For data selection, this means:

> If target-relevant capability functions are well represented in \(\mathcal{H}_k\), then minimizing MMD between selected and target data minimizes the worst-case discrepancy of those capability functions.

For any \(f\in \mathcal{H}_k\), we have:

\[
\left|
\mathbb{E}_{\widehat{P}_S}f
-
\mathbb{E}_{\widehat{P}_T}f
\right|
=
\left|
\langle f,\mu_S-\mu_T\rangle_{\mathcal{H}_k}
\right|
\le
\|f\|_{\mathcal{H}_k}
\operatorname{MMD}_k(\widehat{P}_S,\widehat{P}_T)
\]

This is the cleanest theoretical guarantee of the framework.

---

## 5. Empirical MMD Decomposition

The empirical squared MMD between selected data \(S\) and target data \(T\) is:

\[
\operatorname{MMD}_k^2(S,T)
=
\frac{1}{m^2}
\sum_{i,i'\in S} k(x_i,x_{i'})
+
\frac{1}{n^2}
\sum_{j,j'\in T} k(z_j,z_{j'})
-
\frac{2}{mn}
\sum_{i\in S,j\in T} k(x_i,z_j)
\]

The second term depends only on \(T\) and is constant during selection.

Thus the effective objective is:

\[
\frac{1}{m^2}
\sum_{i,i'\in S} k(x_i,x_{i'})
-
\frac{2}{mn}
\sum_{i\in S,j\in T} k(x_i,z_j)
\]

This has a natural interpretation:

- The cross term:

\[
-\frac{2}{mn}
\sum_{i\in S,j\in T} k(x_i,z_j)
\]

encourages **target relevance**.

- The selected-selected term:

\[
\frac{1}{m^2}
\sum_{i,i'\in S} k(x_i,x_{i'})
\]

penalizes **selected-set redundancy**.

Therefore MMD selection is not merely nearest-neighbor retrieval. It is a principled relevance-diversity objective derived from functional distribution matching.

---

## 6. Greedy MMD Selection

The exact combinatorial optimization is expensive:

\[
\arg\min_{S:|S|=m} \operatorname{MMD}_k^2(S,T)
\]

A practical greedy approximation selects one example at a time.

Let:

\[
r_T(x)
=
\frac{1}{|T|}
\sum_{z\in T}k(x,z)
\]

be the average target relevance.

Let:

\[
r_S(x)
=
\frac{1}{|S|+\epsilon}
\sum_{s\in S}k(x,s)
\]

be the average redundancy against the current selected set.

Then a simple greedy score is:

\[
\operatorname{score}(x\mid S)
=
r_T(x)
-
\lambda r_S(x)
\]

At each step:

\[
x^\star =
\arg\max_{x\in C\setminus S}
\operatorname{score}(x\mid S)
\]

This is the most direct implementation of MMD-based coreset selection.

---

## 7. Relationship to Existing Methods

### 7.1 LESS

LESS selects examples by low-rank gradient similarity to target examples. In a simplified form, its score is approximately:

\[
\operatorname{score}_{LESS}(x)
=
\langle g(x), \bar{g}_T\rangle
\]

where:

\[
\bar{g}_T = \frac{1}{|T|}\sum_{z\in T}g(z)
\]

If we use a linear gradient kernel:

\[
k_{\text{grad-lin}}(x,x')
=
\langle g(x),g(x')\rangle
\]

then MMD greedy selection becomes:

\[
\operatorname{score}_{MMD}(x\mid S)
=
\langle g(x), \bar{g}_T\rangle
-
\lambda
\langle g(x), \bar{g}_S\rangle
\]

Thus, linear gradient MMD can be interpreted as:

\[
\text{LESS-style target gradient relevance}
+
\text{selected gradient redundancy penalty}
\]

This is close to LESS, so it should not be the only novelty. It is useful as an ablation.

The more novel extensions are:

- gradient RBF MMD;
- gradient angular kernel;
- gradient covariance / second-order kernel.

These move beyond matching the target mean gradient and instead match the target gradient distribution or update subspace.

### 7.2 DSIR

DSIR formalizes data selection as matching a target distribution using importance resampling. It estimates importance weights in a reduced feature space and samples data accordingly.

MMD selection differs in that it constructs a coreset by directly minimizing a functional discrepancy:

\[
\operatorname{MMD}_k(\widehat{P}_S,\widehat{P}_T)
\]

rather than estimating density ratios and resampling.

DSIR is closer to:

\[
\text{density-ratio matching}
\]

whereas MMD coreset is closer to:

\[
\text{functional expectation matching}
\]

MMD also contains an explicit selected-set self-similarity penalty, which can help avoid over-sampling near-duplicate high-density regions.

### 7.3 TSDS

TSDS uses optimal transport to align selected data with a small target set, with an additional diversity regularizer and KDE correction for near-duplicates.

MMD selection differs in the theoretical object:

- TSDS / OT: geometric transport cost.
- MMD: worst-case function expectation discrepancy in an RKHS.

A concise comparison:

| Aspect | TSDS / OT | MMD / RKHS |
|---|---|---|
| Alignment object | Transport cost between selected and target examples | Mean embedding discrepancy |
| Theoretical view | Geometric distribution alignment | Functional discrepancy / IPM |
| Diversity | Added via regularizer | Appears naturally in MMD expansion |
| Efficient algorithm | Approximate nearest neighbor / transport solution | Greedy MMD / kernel herding / random features |
| Strong regime | Geometric matching and near-duplicate robust selection | Functional expectation matching and multimodal target coverage |

---

## 8. Kernel Design

Kernel choice is the central modeling decision.

A kernel defines the function class over which distributional discrepancy is controlled. Therefore, the goal is not simply to make the method look different from existing work, but to choose kernels that imply meaningful new inductive biases.

---

### 8.1 Linear Embedding Kernel

Let \(h(x)\) be an embedding of the instruction example.

\[
k_{\text{emb-lin}}(x,x')
=
\langle h(x),h(x')\rangle
\]

Then:

\[
\operatorname{MMD}^2(S,T)
=
\left\|
\frac{1}{|S|}\sum_{s\in S}h(s)
-
\frac{1}{|T|}\sum_{t\in T}h(t)
\right\|^2
\]

This is essentially centroid matching.

#### Value

- Useful as a sanity baseline.
- Shows whether mean matching plus redundancy penalty helps.

#### Risk

- Too close to embedding centroid similarity.
- Weak novelty.

#### Recommendation

Use as an ablation, not as the main contribution.

---

### 8.2 RBF / Laplace Embedding Kernel

\[
k_{\text{emb-rbf}}(x,x')
=
\exp\left(
-\frac{\|h(x)-h(x')\|^2}{2\sigma^2}
\right)
\]

or:

\[
k_{\text{emb-lap}}(x,x')
=
\exp\left(
-\frac{\|h(x)-h(x')\|}{\sigma}
\right)
\]

#### Insight

Unlike linear MMD, RBF MMD does not merely match the centroid. It can capture nonlinear and multimodal distribution differences in embedding space.

This is useful when the target set contains multiple capability modes.

#### Risk

- Bandwidth sensitivity.
- In high dimensions, poor bandwidth choice may collapse to near-neighbor behavior or near-constant similarity.

#### Experiments

- Median heuristic bandwidth.
- Multi-bandwidth kernel:

\[
k = \sum_{\ell}\alpha_\ell k_{\sigma_\ell}
\]

- Target multimodality stress test.

---

### 8.3 Linear Gradient Kernel

Let:

\[
g(x)=\nabla_\theta \ell(x;\theta_0)
\]

or a low-dimensional projected gradient feature, e.g. LoRA gradient, last-layer gradient, or LESS-style projected gradient.

\[
k_{\text{grad-lin}}(x,x')
=
\langle \tilde{g}(x),\tilde{g}(x')\rangle
\]

#### Insight

This matches training-signal distributions rather than semantic distributions.

#### Relation to LESS

This is close to LESS if the redundancy term is removed. With MMD, it becomes:

\[
\langle g(x),\bar{g}_T\rangle
-
\lambda\langle g(x),\bar{g}_S\rangle
\]

#### Recommendation

Use as an important ablation against LESS, but not as the only kernel.

---

### 8.4 Gradient RBF / Angular Kernel

\[
k_{\text{grad-rbf}}(x,x')
=
\exp\left(
-\frac{\|\tilde{g}(x)-\tilde{g}(x')\|^2}{2\sigma^2}
\right)
\]

or:

\[
k_{\text{grad-ang}}(x,x')
=
\exp\left(
\frac{\cos(\tilde{g}(x),\tilde{g}(x'))}{\tau}
\right)
\]

#### Core Insight

LESS and linear gradient similarity mainly align candidate examples with the **mean target gradient**.

Gradient RBF MMD instead matches the **target gradient distribution**.

This matters when target examples are heterogeneous or multimodal.

#### Claim

> For complex target capabilities, matching the target gradient distribution is more robust than aligning all selected examples with a single average gradient direction.

#### Recommended Status

This should be one of the main kernels.

---

### 8.5 Gradient Covariance / Polynomial-2 Kernel

Define:

\[
A(x)=\tilde{g}(x)\tilde{g}(x)^\top
\]

and:

\[
k_{\text{cov}}(x,x')
=
\langle A(x),A(x')\rangle_F
=
\langle \tilde{g}(x),\tilde{g}(x')\rangle^2
\]

This is equivalent to a degree-2 polynomial kernel over gradient features.

#### Core Insight

This kernel matches second-order training-signal structure:

\[
\mathbb{E}_{S}[gg^\top]
\approx
\mathbb{E}_{T}[gg^\top]
\]

It does not merely align with the average gradient direction. It tries to preserve the update subspace or empirical Fisher-like structure induced by the target task.

#### Why This Is Valuable

For heterogeneous target sets, the average gradient may be small or misleading due to cancellation. The covariance operator can still reveal the target task's relevant update subspace.

#### Claim

> The selected subset should preserve the target gradient covariance operator, not merely the target mean gradient.

#### Recommended Status

This is the most theoretically distinctive kernel and should be a main contribution.

---

### 8.6 NTK / Local Function-Dynamics Kernel

A more function-space version uses a local Jacobian or NTK-like kernel:

\[
k_{\text{NTK}}(x,x')
=
\nabla_\theta f_\theta(x)^\top
\nabla_\theta f_\theta(x')
\]

For language models, approximations may include:

- LoRA adapter Jacobian;
- final-layer hidden-state kernel;
- token-level logit gradient approximation;
- response-token loss gradient.

#### Insight

This shifts the view from loss-gradient similarity to local function-space training dynamics.

#### Risk

Implementation cost may be high.

#### Recommendation

Keep as a theory/appendix direction unless compute allows.

---

### 8.7 Loss Trajectory Kernel

For checkpoints \(\theta_1,\dots,\theta_K\), define:

\[
\ell(x) =
[
\ell_{\theta_1}(x),
\ell_{\theta_2}(x),
\dots,
\ell_{\theta_K}(x)
]
\]

Then:

\[
k_{\text{traj}}(x,x')
=
\exp\left(
-\frac{\|\ell(x)-\ell(x')\|^2}{2\sigma^2}
\right)
\]

#### Insight

This matches learning-dynamics patterns rather than single-point loss.

#### Risk

It may be perceived as another loss proxy.

#### Recommendation

Use only as an exploratory ablation.

---

### 8.8 Instruction-Response Decomposition Kernel

Each SFT sample has structure:

\[
x=(q,a)
\]

where \(q\) is the instruction/query and \(a\) is the response.

Define:

\[
k(x,x')
=
\alpha k_q(q,q')
+
\beta k_a(a,a')
+
\gamma k_{qa}((q,a),(q',a'))
\]

or a product kernel:

\[
k(x,x')
=
k_q(q,q')\cdot k_a(a,a')
\]

#### Insight

Instruction tuning data is structured. Query similarity and response similarity mean different things.

- Query kernel: task distribution coverage.
- Response kernel: demonstration behavior / style / quality.
- Product kernel: both query and response must match.

#### Claim

> Targeted SFT selection should distinguish task coverage from demonstration behavior.

#### Recommendation

Use as an important ablation, especially for instruction tuning.

---

### 8.9 Quality-Gated MMD Kernel

Let \(q(x)\in[0,1]\) be a quality score.

Define:

\[
k_{\text{qMMD}}(x,x')
=
q(x)q(x')k_{\text{base}}(x,x')
\]

or:

\[
\min_S
\operatorname{MMD}^2(S,T)
-
\eta\frac{1}{|S|}\sum_{s\in S}q(s)
\]

#### Insight

Low-quality examples should contribute less to functional distribution matching even if they are semantically close to target.

#### Risk

If not careful, this becomes ordinary quality filtering.

#### Recommendation

Use as a practical recipe, not the main theoretical contribution.

---

### 8.10 Mixture-of-Kernels

\[
k_\alpha(x,x')
=
\sum_{r=1}^{R}\alpha_r k_r(x,x'),
\quad \alpha_r\ge 0
\]

Example:

\[
k=
\alpha k_{\text{emb-rbf}}
+
\beta k_{\text{grad-rbf}}
+
\gamma k_{\text{grad-cov}}
+
\delta k_{\text{instr-resp}}
\]

#### Insight

Different kernels correspond to different function classes:

- embedding kernel: semantic distribution;
- gradient kernel: training-signal distribution;
- covariance kernel: update-subspace distribution;
- instruction-response kernel: SFT structural behavior;
- quality gate: robustness to noisy demonstrations.

#### Risk

Too many weights may look ad hoc.

#### Recommendation

Use a small number of carefully motivated kernels. Avoid over-engineering the first version.

---

## 9. Recommended Method Variants

The main paper can define a general framework:

\[
\text{MMD-Select}(k)
\]

and instantiate it with three main kernels:

### 9.1 MMD-Emb-RBF

Purpose: compete with representation and distribution-matching baselines.

Claim:

> Nonlinear MMD captures target distribution multimodality better than centroid similarity or nearest-neighbor retrieval.

### 9.2 MMD-Grad-RBF

Purpose: compete directly with LESS.

Claim:

> Instead of aligning to the mean target gradient, match the full target gradient distribution.

### 9.3 MMD-GradCov

Purpose: provide the strongest theoretical novelty.

Claim:

> Preserve the target task's gradient covariance operator / update subspace rather than pointwise or mean-gradient influence.

---

## 10. Baselines

A strong paper should compare against several classes of baselines.

### 10.1 Sanity Baselines

| Baseline | Purpose |
|---|---|
| Random | Basic lower bound |
| Full data | Check whether small selected subsets outperform or approach full SFT |
| Target set only | Verify target set alone is insufficient |
| Length-matched random | Control sample length effects |
| Domain/task-balanced random | Control high-level mixture effects if labels exist |

### 10.2 Quality / Heuristic Baselines

| Baseline | Purpose |
|---|---|
| Quality-score top-k | Control whether the method simply selects high-quality data |
| Perplexity/loss filtering | Control easy/hard-example effects |
| Complexity-based selection | Compare against instruction complexity heuristics |
| Diversity-based heuristic | Compare against pure diversity selection |

### 10.3 Representation / Distribution Matching Baselines

| Baseline | Purpose |
|---|---|
| Embedding nearest neighbor to target | Direct target similarity |
| Target centroid similarity | Closest to linear embedding MMD without redundancy |
| k-center | Coverage/diversity baseline |
| Facility location | Submodular representation selection |
| DPP / log-det | Diversity-heavy selection |
| DSIR-style resampling | Importance-resampling distribution matching |
| TSDS | Strong OT distribution-alignment baseline |
| TAROT if applicable | More recent OT-based targeted selection baseline |

### 10.4 Influence / Gradient Baselines

| Baseline | Purpose |
|---|---|
| LESS | Main targeted SFT baseline |
| Mean gradient similarity | Simpler LESS-like baseline |
| TracIn-style gradient inner product | Classical influence-style baseline |
| Gradient DPP | Diversity in gradient space |
| Gradient clustering + top relevance | Check whether MMD just approximates clustering |

### 10.5 Ablation Baselines

| Ablation | Purpose |
|---|---|
| Target relevance only | Remove redundancy term |
| Diversity only | Remove target relevance |
| Full MMD | Test full relevance-diversity objective |
| Linear vs RBF kernel | Test whether nonlinear distribution matching matters |
| Embedding vs gradient vs covariance kernel | Test different function spaces |
| With vs without quality gate | Test noise robustness |

---

## 11. Experimental Design

### 11.1 Main Targeted SFT Experiment

#### Candidate Pool

Use a large instruction tuning pool.

Possible choices:

- FLAN-style instruction data;
- OpenAssistant-style data;
- Alpaca/Tulu-style mixture;
- other open instruction pools used in prior targeted SFT work.

#### Target Set

For each target task, collect a small representative set:

\[
|T|\in\{16,32,64,128,256\}
\]

Target sets should be separated from evaluation data to avoid leakage.

#### Selection Ratios

Use:

\[
\rho\in\{1\%,2\%,5\%,10\%\}
\]

At minimum:

\[
1\%,5\%,10\%
\]

#### Training

All selected subsets should be trained with the same:

- base model;
- optimizer;
- learning rate;
- batch size;
- max length;
- epochs or token budget;
- random seeds.

Important: control not only sample count but also token count.

#### Evaluation

Evaluate on:

- target benchmark;
- related OOD benchmark;
- general ability retention benchmark;
- safety/helpfulness if relevant.

---

### 11.2 Direct Comparison with LESS

Use the same gradient feature infrastructure where possible.

Compare:

1. LESS:

\[
\langle g(x),\bar{g}_T\rangle
\]

2. Gradient linear MMD:

\[
\langle g(x),\bar{g}_T\rangle
-
\lambda\langle g(x),\bar{g}_S\rangle
\]

3. Gradient RBF MMD.

4. Gradient covariance MMD:

\[
k(x,x')=(g(x)^\top g(x'))^2
\]

5. Gradient DPP.

Measure:

- downstream score;
- selected-set pairwise gradient cosine;
- selected gradient Gram matrix spectrum;
- effective rank;
- target gradient MMD;
- overlap with LESS-selected samples.

Expected story:

> LESS is strong when target examples are homogeneous. MMD-Grad-RBF and MMD-GradCov should be stronger when the target set is heterogeneous or multimodal.

---

### 11.3 Comparison with TSDS and DSIR

Compare MMD with OT and importance-resampling methods.

#### Against DSIR

Use the same feature space if possible:

- embedding feature;
- gradient feature;
- random projected feature.

Compare:

- selected-target discrepancy;
- duplicate rate;
- diversity;
- downstream performance.

Expected story:

> DSIR is scalable and density-ratio based, but MMD explicitly constructs a coreset with selected-set repulsion.

#### Against TSDS

Compare:

- OT alignment + diversity regularizer;
- MMD functional discrepancy;
- MMD with RBF / covariance kernels.

Test under:

- target multimodality;
- near-duplicate candidate pool;
- small target set;
- noisy target examples.

Expected story:

> TSDS is strong for geometric alignment; MMD is strong when functional coverage and selected-set redundancy matter.

---

### 11.4 Target Multimodality Stress Test

Construct target sets with controlled heterogeneity:

1. Single-mode target:
   - examples from one subtask or capability pattern.

2. Multi-mode target:
   - examples from several subskills.

3. Conflicting-gradient target:
   - target examples whose gradients are not well represented by one mean direction.

Compare:

- LESS;
- embedding NN;
- TSDS;
- MMD-Emb-RBF;
- MMD-Grad-RBF;
- MMD-GradCov.

Expected result:

> Mean-gradient methods degrade as target complexity increases; distributional MMD kernels remain more stable.

---

### 11.5 Target Set Size Scaling

Evaluate:

\[
|T|\in\{8,16,32,64,128,256\}
\]

Questions:

- Does MMD require more target examples than LESS?
- Is MMD more stable when target set is small?
- Which kernel is most sample-efficient?

---

### 11.6 Candidate Pool Scaling

Evaluate:

\[
N\in\{50k,100k,500k,1M\}
\]

under fixed selected budget:

\[
m=5k \text{ or } 10k
\]

Questions:

- Does the method benefit from a larger pool?
- Does performance degrade when the pool contains more irrelevant examples?
- Is MMD more robust to large heterogeneous pools?

---

### 11.7 Cost-Performance Analysis

Report:

- feature extraction cost;
- selection cost;
- memory footprint;
- training cost;
- downstream performance per GPU-hour.

Compare:

- embedding MMD: lower selection cost;
- gradient MMD: higher cost but better task alignment;
- covariance MMD: stronger but potentially more expensive;
- LESS: gradient extraction cost;
- TSDS: ANN / OT assignment cost.

---

## 12. Analysis and Diagnostics

The paper should not only report benchmark scores. It should also analyze what selected data looks like.

### 12.1 Relevance and Redundancy

Report:

\[
\frac{1}{|S||T|}\sum_{s\in S,t\in T}k(s,t)
\]

and:

\[
\frac{1}{|S|^2}\sum_{s,s'\in S}k(s,s')
\]

This directly corresponds to the MMD decomposition.

### 12.2 Effective Rank

For selected feature matrix \(H_S\), compute the spectrum of:

\[
K_S = H_SH_S^\top
\]

or:

\[
\Sigma_S = \frac{1}{|S|}\sum_{s\in S}h(s)h(s)^\top
\]

Effective rank:

\[
r_{\text{eff}}(\Sigma_S)
=
\exp\left(
-\sum_i p_i\log p_i
\right)
\]

where:

\[
p_i=\frac{\lambda_i}{\sum_j\lambda_j}
\]

Expected:

- LESS may produce stronger target relevance but lower effective rank.
- MMD should produce higher effective rank and less redundancy.

### 12.3 Target Coverage

If target examples can be clustered into modes, measure:

- selected examples per mode;
- entropy over target modes;
- mode coverage;
- worst-mode performance.

### 12.4 Duplicate / Near-Duplicate Rate

Measure:

- exact duplicate rate;
- high embedding similarity pairs;
- high response overlap;
- high gradient similarity duplicates.

This is important because TSDS explicitly addresses near-duplicates. MMD should also reduce them via self-similarity penalty.

### 12.5 MMD-Performance Correlation

For each method and kernel, compute selected-target MMD and downstream score.

Question:

> Does lower MMD correlate with better performance?

This parallels DSIR's KL reduction analysis.

---

## 13. Expected Contributions

The paper can claim four contributions.

### Contribution 1: Functional Coreset Formulation

We formulate targeted instruction data selection as RKHS mean embedding approximation:

\[
\min_{S:|S|=m}
\operatorname{MMD}_k^2(\widehat{P}_S,\widehat{P}_T)
\]

This shifts the view from pointwise utility estimation to functional distribution matching.

### Contribution 2: Relevance-Diversity Decomposition

We show that empirical MMD naturally decomposes into:

- target relevance;
- selected-set redundancy penalty.

This provides a principled alternative to heuristic similarity-diversity combinations.

### Contribution 3: Training-Signal Kernels

We design gradient-distribution and gradient-covariance kernels for targeted SFT:

- MMD-Grad-RBF matches target gradient distributions.
- MMD-GradCov matches target gradient covariance / update subspace.

These go beyond mean-gradient alignment used by influence-style selection.

### Contribution 4: Systematic Evaluation

We compare against:

- LESS;
- TSDS;
- DSIR-style resampling;
- embedding nearest-neighbor;
- diversity baselines;
- quality/loss heuristics;
- random/full data.

We also provide mechanistic diagnostics: redundancy, effective rank, target coverage, and MMD-performance correlation.

---

## 14. Key Hypotheses

### Hypothesis 1

MMD-Select improves over pure target similarity because it penalizes selected-set redundancy.

### Hypothesis 2

MMD-Grad-RBF improves over LESS when the target set is heterogeneous or multimodal, because it matches a gradient distribution instead of one mean gradient direction.

### Hypothesis 3

MMD-GradCov is more robust than first-order gradient similarity because it preserves the target task's update subspace:

\[
\mathbb{E}_S[gg^\top]\approx \mathbb{E}_T[gg^\top]
\]

### Hypothesis 4

Lower selected-target MMD in an appropriate kernel space correlates with better downstream target performance.

---

## 15. Minimal Viable Prototype

A practical first version should avoid over-complexity.

### Candidate Pool

- 100k–300k instruction examples.

### Target Tasks

- 3–5 targeted tasks.

### Target Set Size

- 64 or 128 examples per task.

### Selection Ratios

- 1%, 5%, 10%.

### Baselines

Minimum baseline set:

- Random;
- Full data;
- Embedding nearest neighbor;
- DPP / diversity;
- TSDS;
- LESS.

### Proposed Methods

- MMD-Emb-RBF;
- MMD-Grad-RBF;
- MMD-GradCov.

### Diagnostics

- target relevance;
- selected redundancy;
- selected-target MMD;
- gradient effective rank;
- selected set overlap with LESS;
- downstream score.

If this prototype shows that MMD-Grad-RBF or MMD-GradCov improves over LESS in heterogeneous target settings, the project is worth scaling.

---

## 16. Risks and Mitigations

### Risk 1: MMD is just similarity + diversity

Mitigation:

- Derive the objective from RKHS functional discrepancy.
- Compare against heuristic relevance-diversity baselines.
- Show MMD-performance correlation.
- Show nonlinear / gradient covariance kernels produce different selected sets and better behavior.

### Risk 2: Linear kernels collapse to existing methods

Mitigation:

- Treat linear embedding and linear gradient kernels as ablations.
- Main novelty should rely on RBF gradient distribution and gradient covariance kernels.

### Risk 3: Kernel choice is ad hoc

Mitigation:

- Explain each kernel as a different function class.
- Use ablations.
- Use target multimodality experiments.
- Avoid too many manually tuned mixture weights.

### Risk 4: Gradient kernels are expensive

Mitigation:

- Use projected gradients.
- Use LoRA gradients.
- Use random projections.
- Compare cost-performance tradeoffs.
- Provide embedding MMD as a cheaper variant.

### Risk 5: TSDS may be very strong

Mitigation:

- Do not claim universal dominance.
- Emphasize functional discrepancy vs transport geometry.
- Test regimes where MMD should help: multimodal target, gradient conflict, near-duplicate candidate pool, small target set.

---

## 17. Suggested Paper Abstract Draft

Targeted instruction tuning aims to select a small subset of training examples from a large instruction pool to induce a desired downstream capability. Existing methods often rank examples by pointwise relevance or influence to a few target examples, which can over-select redundant data and fail to represent heterogeneous target capabilities. We propose **Functional Coreset Selection**, a framework that views targeted data selection as approximating the target distribution in a reproducing kernel Hilbert space. Our method selects examples by minimizing the Maximum Mean Discrepancy between the selected subset and the target set, yielding an objective that naturally decomposes into target relevance and selected-set redundancy. We instantiate this framework with semantic, gradient-distribution, and gradient-covariance kernels. In particular, our gradient covariance kernel preserves the target task's update subspace rather than merely aligning with an average target gradient. Experiments on targeted instruction tuning compare against LESS, TSDS, DSIR-style resampling, representation retrieval, and diversity baselines. We show that functional coreset selection improves robustness under low selection ratios and heterogeneous target sets, while providing interpretable diagnostics through selected-set redundancy, effective rank, and target distribution discrepancy.

---

## 18. Recommended First Experiment Plan

### Phase 1: Implement Selection Methods

- Embedding feature extraction.
- Gradient feature extraction.
- Greedy MMD selection.
- MMD-Emb-RBF.
- MMD-Grad-RBF.
- MMD-GradCov.

### Phase 2: Reproduce Core Baselines

- Random.
- Full data.
- Embedding nearest neighbor.
- DPP / diversity.
- LESS.
- TSDS or a faithful approximation.

### Phase 3: Small-Scale Validation

- Candidate pool: 100k.
- Select: 1k / 5k / 10k.
- Base model: one small-to-medium LLM.
- Target tasks: 3 tasks.
- Target set: 64 examples.

### Phase 4: Mechanistic Analysis

- MMD decomposition.
- Effective rank.
- Pairwise similarity.
- Target mode coverage.
- Selected set overlap.

### Phase 5: Scale Up

- Larger candidate pool.
- More target tasks.
- More base models.
- Small-to-large transfer.

---

## 19. One-Sentence Summary

This project proposes to move targeted instruction data selection from **pointwise influence ranking** to **functional coreset construction**, using MMD in carefully chosen RKHSs to select data that is both target-relevant and non-redundant, with gradient-distribution and gradient-covariance kernels providing the clearest theoretical and empirical differentiation from LESS-style methods.

---

## 20. References to Compare Against

The most important related works to position against are:

1. **LESS: Selecting Influential Data for Targeted Instruction Tuning**
   - Targeted SFT.
   - Low-rank gradient similarity search.
   - Optimizer-aware influence.
   - Main influence-style baseline.

2. **DSIR: Data Selection with Importance Resampling**
   - Distribution matching via importance resampling.
   - Reduced feature space.
   - KL reduction as distribution proximity metric.

3. **TSDS: Data Selection for Task-Specific Model Finetuning**
   - Target-set-guided selection.
   - Optimal transport distribution alignment.
   - Diversity regularization and KDE correction.

4. **Kernel Two-Sample Test / MMD**
   - Theoretical foundation for MMD as RKHS worst-case expectation discrepancy.

5. **Kernel Herding / Frank-Wolfe Quadrature**
   - Theoretical foundation for greedy mean-embedding coreset construction.


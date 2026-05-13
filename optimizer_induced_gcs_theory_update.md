# Optimizer-Induced Gradient Covariance Spectrum for SFT Data Selection

> Working title: **Optimizer-Induced Gradient Covariance Spectrum as an Unsupervised Data Selection Criterion for LLM Supervised Fine-Tuning**  
> Short name: **Opt-GCS**  
> Purpose: update the earlier “Gradient Covariance Spectrum” theory so that it no longer over-relies on vanilla SGD or overly strong first-order loss-decrease claims.

---

## 0. Motivation: why the theory needs this update

The earlier version of the theory was based on the following local SGD-style argument:

\[
\theta^+ = \theta - \eta g_i,
\]

and used the first-order Taylor approximation

\[
L_{\mathrm{target}}(\theta-\eta g_i)
\approx
L_{\mathrm{target}}(\theta)
-
\eta \langle \nabla L_{\mathrm{target}}(\theta), g_i\rangle.
\]

This is useful as intuition, but it is too fragile as the main theoretical basis for LLM SFT data selection.

The main concerns are:

1. **Real SFT often uses AdamW or Muon, not vanilla SGD.**  
   The model does not follow raw gradient directions \(g_i\). It follows optimizer-induced update directions.

2. **LLM loss landscapes are highly nonconvex and high-dimensional.**  
   A first-order Taylor approximation should not be claimed to accurately predict full retraining effects.

3. **Influence-style arguments are known to be fragile in deep learning.**  
   They may work as ranking heuristics or local surrogates, but not as exact counterfactual retraining approximations.

4. **Data selection is a multi-step training problem.**  
   A single-step local loss decrease is not equivalent to final downstream benchmark improvement.

Therefore, the theory should be reframed.

The revised position is:

> We do not claim that first-order Taylor expansion accurately predicts the final effect of adding or removing a training example after full SFT. Instead, we use optimizer-induced local updates as a tractable representation of short-horizon training signals. The theoretical object of interest is not raw gradient covariance, but the covariance spectrum of the update directions that the optimizer actually follows.

---

## 1. New theoretical object: optimizer-induced update

Let the SFT data pool be

\[
\mathcal D = \{z_i\}_{i=1}^n,
\]

where each sample \(z_i = (x_i, y_i)\) contains an instruction/input and a completion.

At checkpoint \(\theta_t\), define the per-sample loss

\[
\ell_i(\theta_t)
=
\ell(z_i;\theta_t),
\]

and the raw per-sample gradient

\[
g_i^{(t)}
=
\nabla_\theta \ell_i(\theta_t).
\]

In vanilla SGD, the effective update direction is simply

\[
u_i^{(t)} = g_i^{(t)}.
\]

However, for modern optimizers, the model does not update along \(g_i^{(t)}\). Instead, define a general optimizer-induced update map

\[
u_i^{(t)}
=
\mathcal A_t\!\left(g_i^{(t)};\mathcal S_t\right),
\]

where:

- \(\mathcal A_t\) is the update map induced by the optimizer at step \(t\);
- \(\mathcal S_t\) is the optimizer state, such as AdamW's moment estimates or Muon's momentum/matrix-normalization state.

The central covariance object becomes

\[
\Sigma_u^{(t)}
=
\mathbb E_i
\left[
u_i^{(t)}(u_i^{(t)})^\top
\right].
\]

The earlier raw-gradient covariance

\[
\Sigma_g^{(t)}
=
\mathbb E_i
\left[
g_i^{(t)}(g_i^{(t)})^\top
\right]
\]

is only a special case.

---

## 2. Why this fixes the SGD mismatch

The revised framework treats SGD, AdamW, and Muon uniformly.

### 2.1 SGD as a special case

For SGD,

\[
\mathcal A_t(g) = g.
\]

Therefore,

\[
u_i = g_i,
\qquad
\Sigma_u = \Sigma_g.
\]

So the original gradient covariance theory is recovered as a special case.

---

### 2.2 AdamW-induced update space

AdamW uses momentum, diagonal preconditioning, and weight decay. A simplified AdamW update is

\[
m_t
=
\beta_1 m_{t-1}
+
(1-\beta_1)g_t,
\]

\[
v_t
=
\beta_2 v_{t-1}
+
(1-\beta_2)g_t^2,
\]

\[
\Delta \theta_t
=
-\eta
\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}
-
\eta \lambda \theta_t.
\]

For data selection at a fixed checkpoint, we can locally freeze the optimizer state and approximate the per-sample effective update by

\[
u_i^{\mathrm{AdamW}}
\approx
D_t g_i,
\]

where

\[
D_t
=
\operatorname{diag}
\left(
\frac{1}{\sqrt{\hat v_t}+\epsilon}
\right).
\]

If we include momentum explicitly,

\[
u_i^{\mathrm{AdamW}}
\approx
D_t
\left(
\beta_1 m_t + (1-\beta_1) g_i
\right).
\]

For ranking candidates, the common term \(D_t\beta_1 m_t\) is shared across samples, so the sample-distinguishing component is approximately

\[
(1-\beta_1)D_tg_i.
\]

Thus the relevant covariance is not

\[
\Sigma_g = \mathbb E[g_i g_i^\top],
\]

but

\[
\Sigma_{\mathrm{AdamW}}
=
\mathbb E[D_t g_i g_i^\top D_t].
\]

This gives a principled AdamW-aware version of GCS.

---

### 2.3 Muon-induced update space

Muon is not merely a diagonal preconditioner. It applies a matrix-aware transformation to momentum-like updates, typically involving Newton–Schulz orthogonalization for matrix parameters.

For a matrix parameter \(W\), write the per-sample gradient as

\[
G_i^W = \nabla_W \ell_i(\theta).
\]

The Muon-induced per-sample update can be represented abstractly as

\[
U_i^W
=
\mathcal M_t(G_i^W),
\]

where \(\mathcal M_t\) is the Muon update transformation at the current optimizer state.

Vectorizing across selected matrix parameters,

\[
u_i^{\mathrm{Muon}}
=
\operatorname{vec}
\left(
\{\mathcal M_t(G_i^W)\}_{W\in\mathcal P}
\right).
\]

A local linearization gives

\[
u_i^{\mathrm{Muon}}
\approx
J_{\mathcal M,t} g_i,
\]

where \(J_{\mathcal M,t}\) is the Jacobian or local linear surrogate of the Muon update map. The corresponding covariance is

\[
\Sigma_{\mathrm{Muon}}
\approx
J_{\mathcal M,t}
\Sigma_g
J_{\mathcal M,t}^{\top}.
\]

This makes clear that Muon changes the geometry. A method that only studies raw-gradient covariance may recover the wrong principal directions if the optimizer significantly reshapes the update space.

---

## 3. Revised main hypothesis

The old hypothesis was:

> Useful SFT signals are contained in the low effective-rank structure of the raw gradient covariance.

The new hypothesis is:

> Useful SFT signals are contained in the low effective-rank structure of the optimizer-induced update covariance.

Formally, at a checkpoint \(\theta_t\), assume the update covariance has a spiked or low effective-rank structure:

\[
\Sigma_u^{(t)}
=
U_r \Lambda_r U_r^\top
+
\Sigma_{\mathrm{noise}},
\]

with eigengap

\[
\Delta_r
=
\lambda_r(\Sigma_u)
-
\lambda_{r+1}(\Sigma_u)
>
0.
\]

The principal update subspace

\[
\operatorname{span}(U_r)
\]

captures the dominant local training geometry under the actual optimizer.

---

## 4. What the theory should and should not claim

### 4.1 Strong claims we should avoid

We should **not** claim:

1. First-order Taylor expansion accurately predicts the final full-SFT retraining effect.
2. Raw gradients are the correct geometry under AdamW or Muon.
3. Influence scores are reliable pointwise counterfactual estimates in deep nonconvex LLMs.
4. Top raw-gradient eigenvectors necessarily correspond to useful downstream capabilities.
5. Unsupervised gradient spectrum selection must outperform targeted selection such as LESS.

These claims are too strong.

---

### 4.2 Claims we can defend

We can defend the following:

1. At a fixed checkpoint and optimizer state, each sample induces a local update direction \(u_i\).
2. If the update distribution has low effective-rank structure, its principal subspace can be estimated from a probe set.
3. Random sketching can preserve the relevant update-space geometry.
4. Selecting a logdet-diverse subset in the recovered update subspace constructs a spectral coreset.
5. This coreset preserves dominant local optimizer-induced training geometry.
6. If downstream capabilities align with these dominant update directions, the selected subset should be useful for SFT.

The theoretical guarantee is therefore a **geometry-preservation guarantee**, not a global nonconvex training guarantee.

---

## 5. The revised theory

### 5.1 Empirical update covariance

Sample a probe subset

\[
P \subset \mathcal D,
\qquad
|P| = m.
\]

For each \(z_i \in P\), compute or approximate the optimizer-induced update

\[
u_i
=
\mathcal A_t(g_i;\mathcal S_t).
\]

Apply length normalization and clipping:

\[
\bar u_i
=
\frac{u_i}{L_i^\alpha}
\cdot
\min\left\{
1,\frac{\tau}{\|u_i\|}
\right\},
\]

where:

- \(L_i\) is the completion length;
- \(\alpha\in[0,1]\) controls length normalization;
- \(\tau\) is the clipping threshold.

Then estimate

\[
\widehat \Sigma_u
=
\frac{1}{m}
\sum_{i\in P}
\bar u_i \bar u_i^\top.
\]

---

### 5.2 Eigenspace recovery

Let \(U_r\) be the top-\(r\) eigenspace of the population update covariance \(\Sigma_u\), and let \(\widehat U_r\) be the top-\(r\) eigenspace of \(\widehat\Sigma_u\).

If the update vectors are sub-Gaussian, bounded, or clipped so that matrix concentration applies, then with high probability,

\[
\|\widehat\Sigma_u-\Sigma_u\|_{\mathrm{op}}
\]

is small.

By Davis–Kahan,

\[
\|\sin\Theta(\widehat U_r,U_r)\|_{\mathrm{op}}
\leq
\frac{
\|\widehat\Sigma_u-\Sigma_u\|_{\mathrm{op}}
}
{\Delta_r}.
\]

Thus, if the update covariance has a clear eigengap and low effective rank, a small probe set can recover the dominant optimizer-induced update subspace.

---

### 5.3 Sketching

Full update vectors are too large to store. Let

\[
R\in\mathbb R^{p\times d}
\]

be a sketching matrix, and define

\[
\tilde u_i = R\bar u_i.
\]

If \(R\) is a suitable Johnson–Lindenstrauss or subspace embedding, then for the relevant finite set or low-dimensional subspace,

\[
\langle \tilde u_i,\tilde u_j\rangle
\approx
\langle \bar u_i,\bar u_j\rangle.
\]

Therefore, the covariance spectrum and projected coordinates can be estimated in sketch space.

In practice, \(R\) can be:

- random Gaussian projection;
- CountSketch;
- blockwise random projection;
- LoRA-gradient sketch;
- last-layer update feature;
- LESS-style low-dimensional gradient feature.

---

## 6. From update subspace to data selection

After estimating the top update eigenspace \(\widehat U_r\), project every candidate sample into this subspace:

\[
x_i
=
\widehat U_r^\top \tilde u_i
\in\mathbb R^r.
\]

A naive score is

\[
s_i
=
\|x_i\|_2^2.
\]

But top-score selection can be redundant: it may pick many samples pointing in the same direction.

Therefore, the better objective is logdet coverage:

\[
F(S)
=
\log\det
\left(
\epsilon I_r+
\sum_{i\in S}
x_i x_i^\top
\right).
\]

This selects samples that jointly span the dominant update subspace.

---

## 7. Greedy logdet selection

Let

\[
A_S
=
\epsilon I_r+
\sum_{j\in S}
x_jx_j^\top.
\]

Adding candidate \(i\) gives

\[
A_{S\cup\{i\}}
=
A_S+x_ix_i^\top.
\]

By the matrix determinant lemma,

\[
\det(A_S+x_ix_i^\top)
=
\det(A_S)
\left(
1+x_i^\top A_S^{-1}x_i
\right).
\]

Therefore, the marginal gain is

\[
\Delta(i\mid S)
=
F(S\cup\{i\})-F(S)
=
\log
\left(
1+x_i^\top A_S^{-1}x_i
\right).
\]

The greedy rule is

\[
i_t
=
\arg\max_{i\notin S_{t-1}}
x_i^\top A_{S_{t-1}}^{-1}x_i.
\]

This has a clear interpretation:

- \(\|x_i\|\) large means the sample has strong signal in the principal update subspace.
- \(A_S^{-1}\) is large along directions not yet covered by the selected subset.
- Therefore, \(x_i^\top A_S^{-1}x_i\) selects samples that are both strong and complementary.

---

## 8. Submodularity guarantee

The logdet objective is monotone and submodular.

If \(S\subseteq T\), then

\[
A_T\succeq A_S.
\]

Thus,

\[
A_T^{-1}\preceq A_S^{-1}.
\]

Therefore,

\[
x_i^\top A_T^{-1}x_i
\leq
x_i^\top A_S^{-1}x_i,
\]

and

\[
\Delta(i\mid T)
\leq
\Delta(i\mid S).
\]

So \(F\) satisfies diminishing returns.

By the classical greedy guarantee for monotone submodular maximization under a cardinality constraint,

\[
F(S_{\mathrm{greedy}})
\geq
(1-1/e)F(S^\star),
\]

where

\[
S^\star
=
\arg\max_{|S|\leq k}F(S).
\]

Thus, Opt-GCS-LogDet constructs an approximate spectral coreset in optimizer-induced update space.

---

## 9. Revised algorithm: Opt-GCS-LogDet

```text
Algorithm: Opt-GCS-LogDet

Input:
  SFT data pool D = {z_i}_{i=1}^n
  checkpoint θ_t
  optimizer state S_t
  budget k
  probe size m
  sketch dimension p
  length exponent α
  clipping threshold τ
  rank rule RANK
  ridge ε

1. Sample probe subset P ⊂ D, |P| = m.

2. For each z_i ∈ P:
      compute per-sample gradient:
          g_i = ∇_θ ℓ(z_i; θ_t)

      compute optimizer-induced update:
          u_i = A_t(g_i; S_t)

      length-normalize and clip:
          ū_i = (u_i / L_i^α) * min{1, τ / ||u_i||}

      sketch:
          ṽ_i = R ū_i ∈ R^p

3. Estimate update covariance:
      Σ_hat_u = (1/m) Σ_{i∈P} ṽ_i ṽ_i^T

4. Compute top-r eigenspace:
      U_hat_r = TopEig(Σ_hat_u, r = RANK(Σ_hat_u))

5. For each candidate z_i ∈ D:
      compute/sketch/normalize its optimizer-induced update ṽ_i
      project:
          x_i = U_hat_r^T ṽ_i ∈ R^r

6. Greedy logdet selection:
      S = ∅
      A = ε I_r

      repeat k times:
          choose i maximizing x_i^T A^{-1} x_i
          S ← S ∪ {i}
          A ← A + x_i x_i^T

7. Fine-tune θ_t on selected subset S.
```

---

## 10. Pathwise extension

A single checkpoint may still be noisy. A stronger version uses multiple checkpoints:

\[
\theta_{t_1},\ldots,\theta_{t_T}.
\]

For each checkpoint,

\[
u_i^{(t)}
=
\mathcal A_t(g_i^{(t)};\mathcal S_t).
\]

Define the pathwise update covariance

\[
\Sigma_{\mathrm{path}}
=
\frac{1}{T}
\sum_{t=1}^T
\mathbb E_i
\left[
u_i^{(t)}(u_i^{(t)})^\top
\right].
\]

This reduces dependence on one checkpoint and connects the method to trajectory-based attribution.

A pathwise score can be

\[
s_i
=
\sum_{t=1}^T
\|
(U_r^{(t)})^\top u_i^{(t)}
\|^2.
\]

Or we can concatenate/update-average the projected coordinates before logdet selection.

---

## 11. How to position this against LESS and OPUS

### 11.1 LESS

LESS asks:

\[
\text{Which training examples align with target examples?}
\]

Its score is based on gradient similarity to few-shot target examples. It is targeted and supervised by a small target set.

Opt-GCS asks:

\[
\text{Which training examples cover the dominant optimizer-induced update geometry of the SFT data pool itself?}
\]

It is unsupervised and does not require target examples.

Thus, the two methods are complementary:

- If target examples are available, LESS is a strong targeted selector.
- If target examples are unavailable, Opt-GCS is a target-free intrinsic selector.
- If both are available, use Opt-GCS as a prefilter and LESS as a reranker.

---

### 11.2 OPUS

OPUS argues that data utility should be defined in the optimizer-induced update space, especially under AdamW and Muon.

Opt-GCS adopts the same high-level lesson but uses it differently:

- OPUS: dynamic utility scoring by projection onto a target direction.
- Opt-GCS: unsupervised spectral coreset construction in optimizer-induced update space.

So Opt-GCS can be positioned as:

> An unsupervised, spectral, coreset-style analogue of optimizer-induced data selection.

---

## 12. Reviewer-facing response

Potential reviewer criticism:

> Your theoretical derivation relies on SGD and first-order Taylor approximation, but your experiments use AdamW or Muon. Also, LLM loss is highly nonconvex, so why should one-step first-order approximation predict data utility?

Suggested response:

> We agree that a raw-SGD first-order retraining approximation would be too strong for LLM SFT. Our method does not rely on such a claim. We formulate each training example by its optimizer-induced local update direction \(u_i=\mathcal A_t(g_i;\mathcal S_t)\), where \(\mathcal A_t\) can represent SGD, AdamW, or Muon. Our theory studies the covariance spectrum of these effective update directions and proves that, under standard concentration and eigengap assumptions, the dominant update subspace can be recovered and covered by a small logdet-diverse subset. Thus, our guarantee is a local geometry-preservation guarantee in the actual optimizer update space, not a global guarantee on nonconvex retraining trajectories. The first-order view is used only as a short-horizon surrogate explaining why local update geometry can be predictive of SFT behavior.

---

## 13. Key ablations to support the theory

The paper should include the following variants.

| Variant | Representation | Optimizer matched? | Purpose |
|---|---|---:|---|
| Raw-GCS | \(g_i\) | No | Tests naive raw-gradient spectrum |
| AdamW-GCS | \(D_tg_i\) | Yes | Tests optimizer-induced diagonal preconditioning |
| Muon-GCS | \(\mathcal M_t(g_i)\) | Yes | Tests matrix-aware update geometry |
| Path-GCS | \(\{u_i^{(t)}\}_{t=1}^T\) | Yes | Tests trajectory stability |
| GCS-Score | top \(\|x_i\|^2\) | Depends | Tests magnitude-only selection |
| GCS-LogDet | logdet coverage | Depends | Tests coverage over magnitude |
| LESS | target gradient similarity | Partly/Yes | Strong supervised baseline |
| Random | none | N/A | Lower bound |
| Full data | all data | N/A | Upper/reference baseline |

The most important result would be:

\[
\text{AdamW-GCS or Muon-GCS} > \text{Raw-GCS}.
\]

This would empirically support the claim that optimizer-induced geometry matters.

The second most important result would be:

\[
\text{GCS-LogDet} > \text{GCS-Score}.
\]

This would support the claim that coverage is better than magnitude-only selection.

---

## 14. Recommended rewritten paper claim

Instead of:

> We use first-order loss decrease to select influential SFT samples.

Use:

> We formulate unsupervised SFT data selection as spectral coreset construction in optimizer-induced update space. Each example is represented by the local update direction induced by the actual optimizer. We recover the dominant update covariance subspace from a small probe set and select a logdet-diverse subset that covers this subspace. This gives a target-free selection criterion with a geometry-preservation guarantee, avoiding overly strong assumptions about global nonconvex retraining effects.

---

## 15. Summary

The new theory should be centered on three principles:

1. **Optimizer-induced geometry, not raw SGD geometry.**

   \[
   u_i = \mathcal A_t(g_i;\mathcal S_t)
   \]

2. **Spectral coreset, not exact influence.**

   \[
   \max_{|S|=k}
   \log\det
   \left(
   \epsilon I+
   \sum_{i\in S}x_ix_i^\top
   \right)
   \]

3. **Local update-space guarantee, not global nonconvex retraining guarantee.**

   The theory proves recoverability and coverage of dominant update geometry, while downstream performance is validated empirically.

This reframing directly addresses the main objections to the earlier version and makes the paper much more defensible against criticisms about SGD mismatch, AdamW/Muon usage, and the fragility of first-order influence approximations.

---

## References

- Mengzhou Xia, Sadhika Malladi, Suchin Gururangan, Sanjeev Arora, Danqi Chen. **LESS: Selecting Influential Data for Targeted Instruction Tuning.** ICML 2024.  
  https://arxiv.org/abs/2402.04333

- S. Wang et al. **OPUS: Towards Efficient and Principled Data Selection in LLM Pre-training.** 2026.  
  https://arxiv.org/abs/2602.05400

- Samyadeep Basu, Philip Pope, Soheil Feizi. **Influence Functions in Deep Learning Are Fragile.** ICLR 2021.  
  https://arxiv.org/abs/2006.14651

- Keller Jordan. **Muon: An optimizer for hidden layers in neural networks.** 2024.  
  https://kellerjordan.github.io/posts/muon/

- Jiacheng Liu et al. **Muon is Scalable for LLM Training.** 2025.  
  https://arxiv.org/pdf/2502.16982

# Paper framing: "Matching the Target Is Not Enough"

**Status: framing document, not the paper.** This fixes the claim structure, the evidence chain, and —
most importantly — the **boundaries** of each claim, so that drafting cannot silently re-inflate anything
we have already retracted. Target venue ICLR 2027 (deadline **2026-09-16**, 9-page main text).

Written per `advice_0814_3` ("stop looking for results, start writing the conclusion") and `choice_0814_3`
("the paper can start in parallel; the second model is now confirmation, not a precondition for having a
story").

---

## 1. The one sentence

> **Target-aware selection can improve the differentiable target surrogate on the very examples used to
> define the target, while simultaneously degrading the task metric on those same examples.**

The organizing principle is a **phenomenon**, not our method. DSMC is the *instrument* that makes the test
sharp — it is the method that most successfully minimizes the geometry it optimizes, which is precisely
what makes its downstream failure informative. The paper is **not** "our method wins."

## 2. The honest logical chain

$$\text{geometric alignment} \;+\; \text{surrogate improvement} \;\;\not\Longrightarrow\;\; \text{task improvement}$$

Both antecedents are **measured and confirmed**, not assumed. That is the contribution: not "CE and reward
can disagree" (ROSE already argued that), and not "Random is a strong baseline" (already known), but that
**successfully optimizing set-level gradient geometry — verifiably — still fails to deliver utility.**

## 3. Evidence chain, with each link's scope

| # | claim | evidence | scope limit |
|---|---|---|---|
| 1 | DSMC really does minimize the geometry it targets | lowest $D_2(S,Q_d)$ in **3/3** draws | $D_2$ is *our* objective; not a claim about all geometries |
| 2 | Better geometry tracks **worse** accuracy | Spearman($D_2$, acc) = **+0.771 / +0.829 / +0.886** (all-6); **+0.700 / +0.800 / +0.900** (primary-5); **+0.400 / +0.600 / +0.800** (targeted-4) | 6 methods × 3 draws is a small ranking sample; **descriptive only, no p-values** |
| 3 | The operational surrogate genuinely improves | all 4 targeted methods reduce wrapped query CE (**−0.685 … −1.400**); both Random arms *increase* it (**+0.329 / +0.351**) — a perfect sign split | this is the *operational* wrapped surrogate |
| 4 | Task metric falls on the **same 64 items** | query CoT EM drops for every method; targeted **−0.037 … −0.065** | rules out *pure* query→held-out shift; does **not** exclude every notion of overfitting |
| 5 | The CE gain is serialization-dependent | Llama-2: bare-context CE improves for **no** method (dsmc **+0.276** … randk **+1.007**). Llama-3.2: it *does* improve for targeted arms (**−0.41 … −0.57**) | **retracted and stays retracted**: we do *not* claim CE per se is misaligned with CoT. The cross-stack flip shows serialization sensitivity is itself stack-dependent |
| 6 | Not simple task specialization | exposure correlations **−0.210 / −0.280 / −0.224** | absence of one mechanism ≠ evidence for another |

**Headline BBH result** (micro EM, no-SFT base **0.396429**): randk 0.3925, seqlabelmatch 0.3805,
first_rr 0.3715, second_rr 0.3687, dsmc 0.3631, less 0.3601. **Every method is below base**;
DSMC − Random = **−2.94 pp** (0/3 draw blocks favour DSMC).

## 4. Retractions that must not creep back

Each of these was claimed by us at some point and **falsified by our own diagnostics**:

- ❌ "format explains 41%" — unsupported decomposition.
- ❌ "we found the mechanism" — no mechanism is identified.
- ❌ "the competing explanation is ruled out" → ✅ *"query→held-out generalization failure cannot by
  itself explain the result, because the task metric also fails on the very items defining the signal."*
- ❌ "cross-entropy is misaligned with CoT generation" → ✅ scoped to the **operational** surrogate (D2c).
- ❌ "it's not dropout" — never established.
- ❌ "DSMC is the best targeted selector" — true on MMLU, **does not generalize** to BBH.
- ⚠️ D3 stays **exploratory, appendix**. Main text gets at most one sentence: *"We find no evidence that
  greater per-task query exposure protects that task, arguing against a simple task-frequency
  specialization account."*

## 5. Contributions, in order

1. **Directional second-moment matching (DSMC)** — exact marginal greedy on $k(u,v)=\langle u,v\rangle^2$.
   On MMLU with **Llama-2-7B** it beat first-order round-robin in 10/10 replicates (+1.55 pp) and
   second-order in 10/10 (+0.88 pp). **That advantage does NOT transfer to Llama-3.2-3B** (−0.18 pp, 2/5
   blocks; −0.31 pp, 1/5) — so the method-level benefit is **model-stack dependent** and must always be
   stated with its stack. DSMC is the paper's *instrument*, not a contribution to be defended.
2. That advantage is **not** a target-awareness advantage: Random is competitive on MMLU (DSMC − Random
   ≈ 0 on **both** stacks: −0.01 pp on Llama-2, +0.12 pp on Llama-3.2) and **consistently better in the
   observed BBH draw-level comparisons**. (Not "significantly" — no inferential test was pre-registered
   or run.)
3. **Core**: better target-gradient geometry does not guarantee better downstream utility; on BBH the
   ranking is *reversed*, with the geometry–accuracy association pointing the wrong way in 3/3 draws.
4. **Same-item dissociation**: on the identical 64 query items, targeted methods lower final-answer CE yet
   lower CoT exact-match — so this is not merely distribution shift from query to test.
5. **Serialization is load-bearing, but its effect is model-stack dependent**: bare-context CE reverses
   the wrapped-CE result on Llama-2 but not on Llama-3.2. A target-gradient sensitivity further shows that
   among the frozen compared subsets DSMC remains the closest under the evaluation-matched serialization,
   while the finer non-DSMC ranking changes on Llama-3.2.
6. Base/no-SFT and Seq×Label controls show targeted SFT can **negatively transfer**. On format: *the
   Seq×Label-matched Random control shifts performance partway toward the targeted subsets, making
   instruction-format/provenance composition a plausible contributor, but it does **not** identify a
   causal decomposition.* (The earlier "format explains part but not all" phrasing had quietly
   reintroduced the retracted causal reading.)

## 5b. Cross-stack status (Llama-3.2-3B, outcome A — complete)

The second model stack **replicated the direction** of the geometry→utility reversal.

**Claim wording, fixed (advice_0817).** Use:

> **Across the two tested model stacks, better target-gradient alignment does not reliably translate into
> better downstream utility.**

or, equivalently safe but stronger:

> **Better target-gradient alignment is not sufficient for downstream improvement: we observe the same
> geometry–utility reversal on two model stacks.**

Do **not** write "unreliable in general" — we ran two stacks with n=3 draw blocks each and no inferential
test, so a universal claim outruns the evidence.

| | Llama-2-7B | Llama-3.2-3B |
|---|---|---|
| base (no-SFT) micro EM | 0.396429 | 0.471108 |
| best method | Random-K | Random-K |
| worst method | (LESS) then DSMC | DSMC |
| DSMC − Random | **−2.94 pp** | **−0.89 pp** |
| draw blocks favouring DSMC | 0/3 | 0/3 |
| DSMC lowest D2 | 3/3 | 3/3 |
| all methods below base | yes | yes |

**Report honestly: the direction replicates, the magnitude is attenuated** (−0.89 vs −2.94 pp; on the 3B
stack every method sits within 1 pp of base). Do not present the two stacks as equally dramatic.

Outcomes B/C/D are excluded and **all three Outcome A conditions hold** on this stack: DSMC lowest D2 3/3,
DSMC last downstream, and wrapped query CE **−3.26** nats for the targeted selectors. Claim 3 therefore
stands as: better target-gradient geometry does not guarantee better downstream utility, demonstrated on
two model stacks — and on both, the *surrogate the pipeline optimizes* improved while utility fell.

Two secondary sub-checks differ from Llama-2 and belong in the appendix, stated plainly:
Random-K's wrapped CE now **falls** (−1.37) rather than rising, so on this stack it is a **magnitude**
split (Random improves the surrogate far less, 3/3 draws) rather than Llama-2's **sign** split; and
bare-context CE now improves for the targeted arms. The latter may **not** be used to un-retract
"cross-entropy per se is misaligned with CoT" — per the frozen prereg bare CE stays a
serialization-sensitivity diagnostic, and the cross-stack difference instead shows that *how much
serialization matters is itself model-stack dependent*.

## 5c. The evidence structure is now symmetric (MMLU 5% cross-stack, Outcome 3)

The final scoped arm (stop-rule amendment #1) tested the *one* place our evidence was lopsided, and the
result went **against our own method**:

| axis | survives a model-stack change? |
|---|---|
| geometry→utility failure (BBH) | **yes** — two stacks, full Outcome A |
| DSMC's own method advantage (MMLU) | **no** — Llama-2 only |

This is the paper's most credible single fact about its own construction: *what replicates robustly is the
failure of geometric target alignment to guarantee utility — not the method that best achieves that
alignment.* Reporting a result adverse to our own method is the strongest available evidence that the
central claim was not reverse-engineered from a favourable setting. **Main text**, not appendix.

## 6. Position relative to neighbours

- ***A Critical Look at Targeted Instruction Selection*** (2026): gradient distance is the most predictive
  representation, but lower query loss does not always give the best downstream result, and trends differ
  across models. **We differ**: they measure predictiveness of a distance; we *successfully optimize* a
  set-level geometry and show the optimization itself does not transfer.
- **ROSE**: CE is an unreliable surrogate, hence reward-oriented selection. **We differ**: we don't propose
  a new objective; we show the failure persists even when the matching objective is optimized as intended.
- **Large-scale data selection for instruction tuning**: selector gains are setting-dependent and Random is
  often hard to beat. We supply an *instrumented empirical counterexample chain*, not another leaderboard.

## 7. Structure for 9 pages

1. Intro — the one sentence, the chain, what is *not* claimed.
2. DSMC and the $D_2$ objective (compact; it is the instrument).
3. MMLU controlled attribution — where matching helps **on Llama-2**, plus the 5% cross-stack check
   showing that this method-level help does **not** transfer.
4. External BBH — the reversal. **Main-text table.**
5. Geometry/outcome dissociation ($D_2$ vs accuracy, 3/3).
6. Same-item surrogate/task dissociation, **including** the negative bare-CE result.
7. Second model stack (Llama-3.2-3B) — **main text, per `choice_0814`**, not appendix.
8. Limitations, then conclusion.

Appendix: D3 exposure, Seq×Label, contamination audits, prompt-parity gates, all pins/hashes.

## 8. Limitations we state ourselves

- Two model stacks (Llama-2-7B, Llama-3.2-3B) — better than one, still not a broad sweep; and the second
  is a **model-stack** move (model + tokenizer + template together), not an architecture-only ablation.
- The MMLU method-level differences are **small in absolute terms** (all four methods within 0.43 pp of one
  another, all 1.5–2.0 pp below base on Llama-3.2). The *consistency* contrast (10/10 on Llama-2 vs 1–2/5
  on Llama-3.2) carries the reading, not the pp gap. Descriptive only; n=5 blocks.
- The 3B effect is **small in absolute terms** (all methods within 1 pp of base) and draw 2 has Random-K
  and Second-RR essentially tied, so that draw's ranking is fragile. Descriptive statistics only.
- One candidate pool (Tulu V2), two target tasks (MMLU, BBH), $K$ fixed at 2707 for BBH.
- Rank statistics are descriptive; 6 methods × 3 draws does not support inference.
- No mechanism is identified. We show *when* the surrogate assumption fails, not *why*.
- Second-model framing note: the draw ($n=3$) is the statistical unit; the 24 adapter cells are **not** 24
  independent replicates.

## 9. Second model: resolved

**Outcome A fired** (see §5b). Of the four pre-registered outcomes this was the strongest replication, so
claim 3 keeps its general form rather than retreating to "model-dependent". Nothing about the protocol
changed in response to the result, and the stop rule now binds: no third model, no third task, no LR
sweep, no new selector. Remaining work is writing.

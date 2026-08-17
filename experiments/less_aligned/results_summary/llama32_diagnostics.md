# Llama-3.2-3B: the three pre-registered diagnostics

**Evaluation only** — no training, no selection, no protocol change. Shared no-SFT base plus the 24
existing adapters. This is the pre-registered closure that commit `e0ec06d` claimed prematurely.

Seeds are averaged **within** a draw; the draw (n=3) is the unit. **No significance is claimed.**

## Headline: Outcome A's third condition IS met

> **Full pre-registered Outcome A replicates.** The targeted selectors move the **operational surrogate**
> toward the target (wrapped query CE **−3.19 … −3.40** nats) while their **downstream utility falls** and
> DSMC — the method with the lowest $D_2$ in 3/3 draws — is **last**.

So the earlier retraction is now resolved in the affirmative, but *on evidence*, not assertion.

| method | Δ wrapped CE | Δ bare CE | Δ CoT EM (same 64 items) | held-out EM Δ |
|---|---|---|---|---|
| **DSMC** | **−3.255** | −0.573 | **−0.065** | **−0.97 pp** |
| First-RR | −3.398 | −0.460 | −0.010 | −0.55 pp |
| Second-RR | −3.185 | −0.408 | −0.010 | −0.44 pp |
| **Random-K** | **−1.374** | **+0.195** | **+0.013** | **−0.08 pp** |

Base: wrapped CE 10.346 / 10.296 / 9.837, bare CE 6.687 / 7.467 / 6.426, CoT EM 0.531 / 0.484 / 0.531.

**The ordering is the cleanest statement of the phenomenon on this stack:** DSMC improves the surrogate
*most* among the RR/DSMC arms on the metric that matters least, and is *worst* downstream; Random-K
improves the surrogate *least* (−1.37, under half the targeted methods, in **3/3 draws**) and is *best*
downstream. Sign-separated on CoT EM too: every targeted method's same-item CoT EM falls, Random-K's rises.

## What replicates, and what differs from Llama-2

**Replicates (the load-bearing parts):**

| | Llama-2-7B | Llama-3.2-3B |
|---|---|---|
| targeted reduce wrapped CE | ✅ (−0.685 … −1.400) | ✅ (−3.185 … −3.398) |
| targeted same-item CoT EM falls | ✅ (−0.037 … −0.065) | ✅ (−0.010 … −0.065) |
| DSMC lowest $D_2$ | 3/3 | 3/3 |
| DSMC last / Random best downstream | ✅ | ✅ |
| Random improves the surrogate least | ✅ | ✅ 3/3 draws |

**Differs — must be reported, not smoothed:**

1. **Random-K's wrapped CE now *decreases* (−1.374) instead of increasing (+0.329).** On Llama-2 the sign
   split was *perfect*: targeted down, Random up. Here it is a **magnitude** split, not a sign split —
   every arm improves the surrogate, Random just improves it far less (still 3/3 draws behind every
   targeted method). The pre-registered `random_increases_wrapped_ce` check therefore reads **false**.
   The paper must say *"Random improves the operational surrogate substantially less than the targeted
   selectors"* on this stack, **not** *"Random moves away from the target."*

2. **Bare-context CE now *improves* for the targeted methods (−0.41 … −0.57), where on Llama-2 no method
   improved it.** This must **not** be used to un-retract anything. Per the frozen prereg, bare CE is a
   **serialization-sensitivity diagnostic only** and may not be promoted to a primary criterion whatever
   it shows. Its correct use is the reverse: it shows that **how much serialization matters is itself
   model-stack dependent** — which *strengthens* the D2c-motivated caution rather than removing it. We
   still do **not** claim "cross-entropy per se is misaligned with CoT generation."

3. **Absolute CE scales are not comparable across stacks.** Base wrapped CE is ~10.3 here vs ~4.6 on
   Llama-2, because the `llama3` template wraps with **special tokens** (`<|start_header_id|>` = 128006)
   that a base checkpoint has never seen in that role, whereas llama2's `[INST]` is ordinary text. Only
   **deltas against the same base** are interpretable. The wrapper was taken from LlamaFactory's own
   `llama3` template encoder — the same serialization used for both target-gradient extraction and SFT on
   this stack — so "operational" means what this pipeline actually fed the model.

4. **CoT EM is noisier here: 2/3 draws negative, not 3/3.** Draw 1 is positive for every method
   (DSMC +0.016). On 64 items one flipped item is ±0.016, so these are small counts. DSMC still has the
   most negative mean (−0.065, 6× First-/Second-RR) and is the only method whose drop exceeds −0.07 in
   any draw.

## The chain, on this stack

$$D_2 \downarrow \;(3/3)\qquad L_Q^{\text{wrapped}} \downarrow \;(-3.26)\qquad\text{CoT EM} \downarrow\;(-0.065)\qquad\text{held-out EM} \downarrow\;(-0.97\text{ pp})$$

Geometry improves, the operational surrogate improves, and **both** task metrics fall — on the same 64
items that *defined* the targeting signal, and on the frozen 5,209-example held-out set. This is the
double dissociation, on a second model stack.

## Verdict record

| pre-registered check | result |
|---|---|
| `targeted_reduce_wrapped_ce` | **true** ← Outcome A's third condition |
| `targeted_reduce_cot_em` | **true** |
| `random_increases_wrapped_ce` | **false** (differs from Llama-2; see difference 1) |
| `no_method_improves_bare_ce` | **false** (differs from Llama-2; see difference 2) |

The two `false` results are **not** failures of the central claim — neither is part of Outcome A. They are
recorded because they are genuine cross-stack differences, and because the second one is precisely the
kind of result that could be misused to re-inflate a retracted claim.

## Stop rule

Experiments end here. No third model, no third task, no LR/epoch sweep, no reward-aware or CoT-gradient
DSMC, no new matched Random, no further search for a setting where DSMC wins. Remaining work is writing.

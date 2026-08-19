# Llama-3.2-3B × MMLU 5%: RESULTS — pre-registered **Outcome 3**

**The positive MMLU method result does NOT transfer.** On Llama-2-7B, DSMC beat First-RR in **10/10**
replicates (+1.55 pp) and Second-RR in **10/10** (+0.88 pp). On Llama-3.2-3B both reverse: DSMC is
**−0.18 pp** behind First-RR (2/5 blocks) and **−0.31 pp** behind Second-RR (1/5 blocks).

This is pre-registered **Outcome 3**, and the prereg said in advance it is *not a failure*: it makes the
paper **more unified**, not weaker. Nothing was tuned in response, and the prohibited 1% follow-up will not
be run.

## Primary result

Balanced MMLU = (STEM + HUM)/2, 5-shot, 57 subtasks. Unit = the **five draw-index blocks**, with the
stem-majority and hum-majority directions averaged **within** an index first.

**no-SFT reference: balanced 0.4947** (STEM 0.4742, HUM 0.5152, all-MMLU 0.5609)

| method | blk0 | blk1 | blk2 | blk3 | blk4 | mean | Δ vs base |
|---|---|---|---|---|---|---|---|
| **Second-RR** | 0.4791 | 0.4779 | 0.4822 | 0.4771 | 0.4813 | **0.4795** | −1.51 pp |
| First-RR | 0.4841 | 0.4720 | 0.4796 | 0.4772 | 0.4781 | 0.4782 | −1.65 pp |
| **DSMC** | 0.4757 | 0.4733 | 0.4769 | 0.4771 | 0.4789 | **0.4764** | −1.83 pp |
| Random-K | 0.4746 | 0.4750 | 0.4722 | 0.4779 | 0.4760 | 0.4752 | −1.95 pp |

**Every method is below the no-SFT base**, as on both BBH stacks.

## The three pre-registered comparisons

| comparison | mean | blocks favouring DSMC | per-block (pp) |
|---|---|---|---|
| $\Delta_{\text{rep}}$ = DSMC − First-RR | **−0.182 pp** | **2/5** | −0.85, +0.13, −0.27, −0.01, +0.08 |
| $\Delta_{\text{MMD}}$ = DSMC − Second-RR | **−0.314 pp** | **1/5** | −0.34, −0.46, −0.53, +0.01, −0.24 |
| $\Delta_{\text{rand}}$ = DSMC − Random-K | **+0.124 pp** | 3/5 | +0.11, −0.17, +0.47, −0.08, +0.29 |

## Cross-stack contrast — this is the headline

| | Llama-2-7B (10 replicates) | Llama-3.2-3B (5 blocks) |
|---|---|---|
| DSMC − First-RR | **+1.55 pp, 10/10 wins** | **−0.18 pp, 2/5** |
| DSMC − Second-RR | **+0.88 pp, 10/10 wins** | **−0.31 pp, 1/5** |
| DSMC − Random-K | −0.01 pp, 4/10 | +0.12 pp, 3/5 |
| all methods vs base | — | all below base |

On Llama-2 the DSMC advantage over both RR arms was **perfectly consistent** (10/10 and 10/10). On
Llama-3.2 it is **absent and slightly negative**. Meanwhile DSMC − Random stays near zero on *both*
stacks (−0.01 → +0.12 pp), which is exactly the target-awareness null we already reported.

## What must now be said, and what must not

**Must be said — the method claim is scoped down:**

> Directional second-moment matching produced a consistent advantage over first- and second-order
> round-robin selection on MMLU **with Llama-2-7B**, but that advantage **did not transfer** to
> Llama-3.2-3B. The method-level benefit of second-moment matching is therefore **model-stack dependent**.

**Must not be said any more:**

- ❌ "DSMC beats first-order targeted selection on MMLU" — as an unqualified claim. Always attach the stack.
- ❌ Any framing where DSMC is the paper's *contribution to be defended*. It is the **instrument**.

**Why this strengthens rather than weakens the paper.** The evidence structure is now symmetric instead of
lopsided:

| axis | status |
|---|---|
| geometry→utility failure | replicates on **two** model stacks (BBH), full Outcome A |
| positive DSMC method advantage | **single-stack**, does not replicate |

So the thing that survives model change is the **negative** result, and the thing that does not survive is
our **own method's** advantage. That is the honest and more unified story: *what replicates robustly is the
failure of geometric target alignment to guarantee utility — not the method that best achieves that
alignment.* We report a result against our own method's interest, which is the strongest possible evidence
that the central claim was not reverse-engineered from a favourable setting.

## Reading rule declared, not tuned

The prereg wrote Outcome 2 as "DSMC ≈ Second-RR" without a numeric band. The analysis script declares a
symmetric **0.2 pp** tie threshold. Outcome 3 fires on either reading here: DSMC is behind Second-RR by
0.314 pp (outside the band) **and** behind First-RR, so no choice of threshold in any plausible range
turns this into Outcome 1 or 2.

## Honest limits

- **Small absolute differences.** All four methods sit within 0.43 pp of each other and 1.5–2.0 pp below
  base. **No significance is claimed**; n=5 blocks, descriptive only.
- The Llama-2 arm's unit was 10 replicates; this arm's is 5 direction-averaged blocks. The *consistency*
  contrast (10/10 vs 1–2/5) is what carries the reading, not the pp gap alone.
- **n=10 is not claimed here.** The two directions of a block share a training seed, and Random-K shares
  one adapter between them.
- One model stack per task-budget cell; no mechanism identified.

## Stop rule — now fully binding

Amendment #1 is spent. **No Llama-3.2 MMLU 1% follow-up** (explicitly prohibited in advance, and the
prohibition holds precisely because the outcome was unfavourable to our method), no third model, no third
task, no LR/epoch sweep, no new selector, no re-tuning. **All experiments end here.** Remaining work is
writing.

## Provenance

35 unique adapters (30 targeted + 5 shared Random-K) backing 40 analysis cells; 420 steps each
(= ⌊13533/128⌋×4, effective batch 128 × 4 epochs), epoch 3.993, r128/α512/dropout 0.05, {q,k,v,o}_proj,
all 35 hashes distinct; 36/36 evaluations at exactly 57 MMLU subtasks, all results JSONs hashed. DSMC
`alpha=0.0`; RR `perm_seed=3000+index` (the historical MMLU seed, recovered from the frozen Llama-2
selections — **not** the BBH arm's 6000+d); target cutoff 2048 (the historical MMLU value, not BBH's 3072);
candidate datastore reused and hash-pinned (`bcbb3a0f2f2b371f…`); Random-K subsets byte-identical to the
frozen Llama-2 5% subsets.

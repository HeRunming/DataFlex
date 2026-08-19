# Eval-matched target-gradient $D_2$ sensitivity (post-hoc)

**Status: POST-HOC SENSITIVITY ANALYSIS.** Target-side gradients only. No new adapters, no new
selections, no SFT. It does **not** redefine any primary result, and per `writing_advice_0819` no further
experiment follows from it whatever it shows. This is the **last** diagnostic.

## The objection it addresses

An adversarial reviewer's strongest remaining move: *"DSMC has the lowest $D_2$ — but you extracted target
gradients through the chat wrapper, while the task metric reads a bare prompt. Is the geometry result an
artifact of that serialization mismatch?"*

So the frozen subsets were re-scored against target gradients extracted under the **bare,
evaluation-matched** serialization (LlamaFactory `empty` template — no wrapper tokens), holding everything
else fixed: same query files, cutoff 3072, projection 8192/seed 123, candidate = Adam-aware, target = SGD.

$$D_2^{\text{evalctx}}(S,Q_d) = \lVert M_S - M_{Q_d}^{\text{bare}} \rVert_F^2$$

## Non-vacuity first

The bare serialization changes the target gradients **substantially**, so this is a real test rather than a
formality:

| stack | mean row cosine, operational vs eval-matched |
|---|---|
| Llama-2-7B | 0.335 / 0.342 / 0.363 |
| Llama-3.2-3B | 0.249 / 0.270 / 0.272 |

Mean row cosines of 0.25–0.36 mean the two target representations are far from interchangeable. The script
**refuses to report** if the two tensors come out identical, since a silently ignored template override
would make the diagnostic vacuous.

## Result: the DSMC $D_2$ minimum is robust

| stack | DSMC lowest $D_2$ (operational) | DSMC lowest $D_2$ (**eval-matched**) |
|---|---|---|
| Llama-2-7B | 3/3 | **3/3** |
| Llama-3.2-3B | 3/3 | **3/3** |

Eval-matched $D_2$ (lower = better geometry):

| stack / draw | DSMC | Second-RR | First-RR | Random-K |
|---|---|---|---|---|
| L2 draw0 | **0.06675** | 0.07102 | 0.06830 | 0.08875 |
| L2 draw1 | **0.08230** | 0.08586 | 0.08393 | 0.10358 |
| L2 draw2 | **0.05655** | 0.05989 | 0.05800 | 0.07831 |
| L3.2 draw0 | **0.07985** | 0.08921 | 0.09134 | 0.08847 |
| L3.2 draw1 | **0.07660** | 0.08619 | 0.08809 | 0.08598 |
| L3.2 draw2 | **0.07124** | 0.08203 | 0.08414 | 0.08051 |

**Reading.** DSMC still minimizes the target discrepancy in **3/3 draws on both model stacks** under the
*bare serialization the task metric actually uses*, while still being **worse than Random downstream in 3/3
draws on both stacks**. The geometry–utility counterexample therefore does **not** depend on the chat
wrapper used to extract target gradients.

## An honest nuance that must be reported

The **full ranking is not identical** — on Llama-3.2 it changes in **3/3 draws**, because **Random-K moves
from last to second** on eval-matched $D_2$ (e.g. draw0: 0.08847 vs Second-RR 0.08921, First-RR 0.09134).
On Llama-2 the ranking is unchanged in 3/3.

Two consequences, both stated rather than smoothed:

1. **The load-bearing claim survives**: DSMC is the $D_2$ minimizer under both serializations, on both
   stacks. The counterexample only needs *DSMC minimizes the geometry yet loses downstream*, and that holds
   in 6/6 stack-draw cells under both serializations.
2. **The finer ordering among the non-DSMC arms is serialization-sensitive**, and on Llama-3.2 the
   *magnitude* of Random's geometric disadvantage largely disappears under the bare serialization. So we do
   **not** claim a serialization-invariant ordering of all four methods. This is consistent with the D2c and
   cross-stack bare-CE findings: **serialization is load-bearing**, just not for the DSMC minimum.

## What this licenses

> The counterexample — a selector that verifiably minimizes the target second-moment discrepancy yet
> underperforms a target-independent baseline — holds under **both** the operational and the
> evaluation-matched target serialization, on **both** model stacks.

It does **not** license: any claim that all $D_2$ rankings are serialization-invariant; any re-opening of
the method claim; or any further experiment. Amendment #1 is spent and the stop rule is fully binding.

## Provenance

Six target-only extractions (2 stacks × 3 draws, 64 queries each). Configs derived structurally from the
frozen BBH draw configs with a four-key allowlist (`template`, `component_name`, `components_cfg_file`,
`output_dir`) plus a key-set assertion, so **only** the template differs. Pinned candidate datastores
symlinked into each cache namespace so no candidate gradients were recomputed. All six eval-matched caches
are (64, 8192), finite, zero zero-rows.

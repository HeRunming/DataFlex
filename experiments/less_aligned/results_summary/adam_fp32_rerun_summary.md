# Adam-aware Selection: fp32 Fix Re-run (LESS-Adam vs MMD-Adam)

**Date:** 2026-06-07
**Goal:** After the fp32 Adam-preconditioning fix (commit `1bc65c9`), re-run the
Adam-aware selectors so MMD-Adam variants can be compared fairly against
LESS-Adam. The prior Adam-MMD numbers were computed under the bf16 bug (target
gradients 100% zero → MMD degenerated to diversity-only selection) and were not
a valid comparison.

## Setup (LESS-aligned)

- **Base:** Llama-2-7B; **pool:** 270,679 LESS Tulu-V2 (flan_v2 100k + cot 100k + dolly 15011 + oasst1 55668)
- **Warmup:** 5% random subset (13533), 4 epochs → `checkpoint-1692` (AdamW optimizer state)
- **Selection:** candidate grads = Adam-preconditioned (fp32) TRAK proj (8192-dim, seed 123); target grads = SGD. 5% budget = 13533.
- **SFT:** from base, LoRA r=128 α=512, 4 epochs, effective batch 128.
- **Eval:** BBH CoT 3-shot exact_match; MMLU 5-shot acc; TyDiQA 1-shot GoldP macro-F1/EM (with `Answer:` trigger).

## Results (fp32-fixed)

| Method | BBH | MMLU | TyDiQA-F1 | TyDiQA-EM |
|---|---|---|---|---|
| less_adam | 0.3726 | **0.4574** | 0.5418 | 0.4317 |
| mmd_grad_rbf_adam | 0.3744 | 0.4542 | 0.5610 | 0.4269 |
| **mmd_grad_cov_adam** | **0.3821** | 0.4527 | **0.5735** | **0.4426** |

### Per-target best
- **BBH:** mmd_grad_cov_adam (0.3821) — beats less_adam (0.3726)
- **MMLU:** less_adam (0.4574) — three within 0.005
- **TyDiQA:** mmd_grad_cov_adam (0.5735 F1) — beats less_adam (0.5418)

## fp32 fix: before vs after (Adam variants)

BBH exact_match, the cleanest signal (MMLU/TyDiQA selections also changed, so
those aren't strictly the same comparison axis):

| Method | BBH (bug/bf16) | BBH (fp32) | Δ |
|---|---|---|---|
| less_adam | 0.4016 | 0.3726 | −0.029 |
| mmd_grad_rbf_adam | 0.3832 | 0.3744 | −0.009 |
| mmd_grad_cov_adam | 0.3667 | **0.3821** | **+0.015** |

Under the bug, `grad_cov` was the *worst* Adam method on BBH (target grads were
zero → kernel collapsed to pure diversity). After the fp32 fix it becomes the
*best* and overtakes LESS-Adam — exactly the predicted behavior: the gradient
covariance kernel only works once the target gradient distribution is real.

## Validation that the fix is real

- Target gradient zero-rows: **100/100 (bug) → 0/100 (fixed)**.
- grad_rbf vs grad_cov selection overlap: **0.772 (degenerate) → 0.59–0.68** across targets (kernels now produce genuinely different coresets).
- Cross-method overlaps (rbf/cov/less) all 0.52–0.68 — no collapse.

## Notes / caveats

- **TyDiQA prompt fix:** transformers 4.54.1 emits EOS as the first token on
  bare-question prompts for non-Latin scripts (bengali/telugu/arabic → F1≈0).
  Adding an explicit `Answer:` trigger (applied identically to all methods)
  restores normal answering. No-trigger outputs archived under
  `eval_results/tydiqa_notrigger/`.
- Candidate Adam gradients were computed once on the 270k pool (8×H20, ~3h) and
  reused across all 9 caches via symlink; per-target runs only recompute the
  small target grads.

## Takeaway

With the fp32 fix, **MMD-GradCov-Adam is the strongest Adam method on 2 of 3
targets (BBH, TyDiQA)** and MMD variants match LESS on MMLU. This is the first
*valid* Adam-aware comparison and it supports the central hypothesis: matching
the target gradient covariance/distribution beats mean-gradient (LESS) influence
when the target is heterogeneous.

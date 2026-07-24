# Adding NICE: baseline vs LESS / MMD / TSDS (LESS-aligned setting)

**Date:** 2026-07-02

NICE (Wang et al. ICML 2025, "Data Selection for Instruction Tuning in LLMs
with Non-differentiable Evaluation Metric") added as a baseline. NICE = LESS
skeleton with the validation signal replaced: instead of the NTP-loss gradient
of the gold target answer, it uses a **reward-weighted policy gradient**
(vanilla REINFORCE) over MC generations, where the reward is a **task metric**.

## Setup (identical to the existing gradient methods)
- Base Llama-2-7B, warmup LoRA (r128 α512, 4ep on 5% random of the 270k LESS
  Tulu-V2 pool) → checkpoint-1692 (adapter + AdamW state).
- **Candidate gradients: the SAME adam-preconditioned TRAK-8192 cache LESS-adam
  uses** (`less_output/train/1`). NICE therefore differs from LESS-adam ONLY in
  the validation signal → the contribution is cleanly isolated.
- NICE policy model = warmup checkpoint-1692. MC sampling mc=16, temp=1.0.
  Reward per target: MMLU = answer-letter exact-match; BBH = CoT answer
  exact-match; TyDiQA = SQuAD-F1. Vanilla REINFORCE (summed-NLL × reward,
  no baseline), row-normalized, projected 8192 (seed 123), scored by mean
  ⟨g_cand, g_val⟩, top-5% (13,533).
- SFT: from base, LoRA r128 α512, 4 epochs, eff batch 128. Eval: BBH CoT
  3-shot exact_match, MMLU 5-shot acc, TyDiQA 1-shot GoldP macro-F1.

## ⚠️ Cross-pipeline caveat
NICE was run in a freshly rebuilt env (**transformers 4.50.0 / llamafactory
0.9.3**, py3.10). The existing LESS/MMD/TSDS numbers in the table below are the
**git values from the older pipeline (transformers 4.54.1)**. Per the repo's own
warning ("never compare across pipelines"), the NICE column is not strictly
apples-to-apples with the others — especially TyDiQA, whose EOS-first-token
behavior is transformers-version-sensitive. Treat NICE vs the rest as
**indicative**; a same-env LESS anchor re-run is the clean follow-up (deferred
by decision — reusing old numbers for now).

## Results

| Method | grad/emb | BBH | MMLU | TyDiQA-F1 |
|---|---|---|---|---|
| less_sgd | sgd | 0.3847 | 0.4642 | 0.5433 |
| mmd_grad_rbf_sgd | sgd | 0.3649 | 0.4648 | 0.5387 |
| mmd_grad_cov_sgd | sgd | 0.3873 | **0.4704** | 0.5572 |
| less_adam | adam | 0.3726 | 0.4574 | 0.5418 |
| mmd_grad_rbf_adam | adam | 0.3744 | 0.4542 | 0.5610 |
| mmd_grad_cov_adam | adam | 0.3821 | 0.4527 | **0.5735** |
| mmd_emb_rbf | emb | 0.3916 | 0.4495 | 0.5468 |
| mmd_emb_rbf_stochastic | emb | 0.3930 | 0.4516 | 0.5535 |
| tsds | emb (OT) | **0.3944** | 0.4587 | 0.5635 |
| **nice** † | adam+policy | 0.3849 | 0.4547 | 0.5720 |

† NICE run in the transformers-4.50 pipeline (see caveat). BBH shown as macro
over 27 subtasks (aggregate exact_match 0.3830).

### Per-target
- **BBH:** tsds 0.3944 leads; nice 0.3849 mid-pack (above both LESS, below emb/tsds).
- **MMLU:** mmd_grad_cov_sgd 0.4704 leads; nice 0.4547 ≈ less_adam (0.4574), below sgd methods.
- **TyDiQA:** mmd_grad_cov_adam 0.5735 leads; **nice 0.5720 is 2nd, essentially tied**, above less_adam (+0.030), tsds (+0.009), all MMD-grad-rbf.

## Reading

**NICE lands competitive but does not top any column.** Its strongest showing is
TyDiQA (0.5720, statistically tied with the best MMD-GradCov-adam 0.5735 and
clearly above LESS-adam 0.5418) — the heterogeneous multilingual target where a
reward-driven signal helps most. On MMLU it matches LESS-adam; on BBH it beats
both LESS variants but trails the semantic (emb/TSDS) methods, consistent with
the existing finding that BBH's reasoning coverage is better captured
semantically.

**NICE ≠ LESS despite shared candidate gradients.** NICE-bbh and LESS-adam-bbh
select from the identical adam TRAK cache with the same BBH target, yet their
selections overlap only **1.8%**. The reward-weighted policy gradient is a
genuinely different validation signal, not a reparametrization of LESS — so NICE
is a substantive, non-redundant baseline.

**For the MMD story:** NICE does not displace MMD-GradCov as the top method on
MMLU/TyDiQA. On TyDiQA it comes within noise of GradCov-adam, so the headline
"GradCov wins on the heterogeneous target" now needs the multi-seed test to
separate GradCov, NICE, and TSDS (all within ~0.01 there). NICE strengthens the
paper's baseline suite (adds a reward/policy-gradient influence method alongside
LESS's mean-gradient influence) without threatening the core GradCov contribution
on MMLU (GradCov-sgd 0.4704 vs NICE 0.4547, a clear 0.016 gap).

## Signal-coverage note
NICE reward signal reached (mc=16): MMLU 266/285 targets, BBH 77/100, TyDiQA
8/9 languages contributed a non-zero policy gradient (zero-signal targets =
those the warmup model never solves in 16 samples; they contribute nothing to
the vanilla-REINFORCE sum). Raw eval outputs under `eval_results/{bbh,mmlu,tydiqa}/nice_*`.

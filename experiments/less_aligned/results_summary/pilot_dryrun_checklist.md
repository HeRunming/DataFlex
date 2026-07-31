# Target-draw pilot — dry-run pass/fail checklist + run plan (stem80_draw0)

Selection-only dry run of all 8 methods on `stem80_draw0`. NO SFT. review_0729/_0730/_0731 + choice_0731.

## Pass/fail checklist

| check | status | detail |
|-------|--------|--------|
| target grads extracted, candidate cache reused (no 270k re-extraction) | PASS | (64,8192), ~6 min via symlink to shared Adam cache |
| preflight manifest asserts target JSONL hash == frozen meta | PASS | matches `stem80_draw0.meta.json` |
| candidate symlink resolves to frozen cache (`readlink -f`) | PASS | == less_output/train/1 |
| all 8 selections exactly 13,533 unique in-range | PASS | dsmc/less/first_rr/second_rr/gist/nice/randk/randk_lenmatch |
| NICE off-by-one fixed (was 13,534) | PASS | `--num_select` exact-K |
| NICE reward = frozen NICE-MMLU-EM (first-letter EM) | PASS | doc/code reconciled; no gold-prob path |
| NICE reward diagnostics emitted | PASS | mean 0.34, 14/64 zero-signal (IDs saved), 50 retained, histogram |
| randk_lenmatch length-hist == dsmc per bucket | PASS | both `[8998,2009,1228,493,805]` |
| dsmc / second_rr / gist bit-identical on rerun | PASS | identical sorted-index sha |
| **NICE bit-reproducible** | PASS (strict mode) | strict-deterministic: val-grad sha + selection sha identical across 2 runs |
| env recorded for NICE strict determinism | PASS | torch 2.10.0+cu128, transformers 4.50.0, cuDNN 91002, H20 |

**NICE reproducibility gate (choice_0731 option 4)**: strict-determinism mode SUCCEEDED — no
unsupported-op errors; `torch.use_deterministic_algorithms(True)` + `CUBLAS_WORKSPACE_CONFIG=:4096:8`
+ math SDPA + no TF32 gives bit-identical val-gradients and selection across runs. Frozen as the NICE
config for all draws (option-1 artifact fallback NOT needed).

## Frozen selection hashes (stem80_draw0)

See `pilot_draw0_selection_manifest.json`. Pairwise Jaccard (informative sanity only, not a result):
NICE highly distinct (0.02–0.13 vs all); randk ≈ budget ratio 0.024 vs all; 2nd-order (dsmc/second_rr)
cluster; 1st-order/relevance (less/first_rr/gist) cluster ~0.44.

## Run plan for the 2×2 pilot (NOT yet run — SFT is gated)

Statistical unit = target draw. Pilot = 2 directions × 2 draws = {stem80_draw0, stem80_draw1,
hum80_draw0, hum80_draw1}. Per review_0731, Random-K is reused across same-draw-index directions
(same subset seed 2000+idx + same train seed), so the pilot needs **fewer than 8×4 adapters**:

- 7 draw-specific methods (dsmc, less, first_rr, second_rr, gist, nice, randk_lenmatch) × 4 draws = 28
- Random-K: shared per draw-index across directions → 2 adapters (draw0 seed2000, draw1 seed2001)
- **Total = 30 unique SFT adapters** (not 32).

Full 5-draws-per-direction expansion (later, only if pilot clean): 7×10 + 5 = 75 adapters.

Each SFT: seed = the draw's `train_seed` (draw0→42, draw1→1), eff-batch 128, 4 epochs, then eval
mmlu_stem+mmlu_humanities; per-draw paired Δ vs DSMC (balanced + target-weighted). No p<0.05 claims.

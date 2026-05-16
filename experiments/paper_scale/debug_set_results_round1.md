# Debug Set Benchmark Results (Round 1)

**Setup**: 10 methods + base, budget=5000, Llama-3.1-8B + LoRA (rank=16), Open-Hermes-2.5 training data.

## Complete Results Table

| Method                 | eval_loss | MMLU   | GSM8K  | MMLU-Sub | IFEval | Avg Rank |
|------------------------|-----------|--------|--------|----------|--------|----------|
| random_subspace_logdet | 0.936     | 0.6561 | 0.5519 | 0.5926   | 0.1904 | 2.5      |
| loss                   | 0.881     | 0.6536 | 0.5603 | 0.5913   | 0.1719 | 3.8      |
| fisher_sft             | 0.859     | 0.6536 | 0.5171 | 0.5918   | 0.2551 | 4.2      |
| less                   | 0.822     | 0.6546 | 0.5292 | 0.5868   | 0.2015 | 5.2      |
| **opt_gcs_logdet**     | 0.861     | 0.6511 | 0.5451 | 0.5898   | 0.2033 | 5.5      |
| random                 | 0.874     | 0.6524 | 0.5504 | 0.5911   | 0.1479 | 6.2      |
| opt_gcs_score          | 0.869     | 0.6496 | 0.5633 | 0.5864   | 0.1701 | 6.5      |
| base (no SFT)          | —         | 0.6531 | 0.5398 | 0.5886   | 0.0961 | 7.2      |
| opt_gcs_unwhitened     | 0.880     | 0.6481 | 0.5625 | 0.5851   | 0.1701 | 7.8      |
| opt_gcs_rank50         | 0.873     | 0.6512 | 0.5330 | 0.5864   | 0.1645 | 8.2      |
| grad_norm_topk         | 0.813     | 0.6512 | 0.5140 | 0.5840   | 0.1793 | 8.8      |

## Per-Benchmark Rankings

### MMLU (5-shot accuracy, 57 subjects)
Spread: 0.6481–0.6561 (0.8%). **Not sensitive** to SFT data selection at this scale.

### GSM8K (8-shot CoT, exact match)
Spread: 0.5140–0.5633 (4.9%). **Most differentiated** for math reasoning.
1. opt_gcs_score (0.5633)
2. opt_gcs_unwhitened (0.5625)
3. loss (0.5603)

### MMLU-Subset (10261 harder questions, 7 categories, generation)
Spread: 0.5840–0.5926 (0.9%). Similar to standard MMLU — low sensitivity.

### IFEval (instruction following, prompt-level strict accuracy)
Spread: 0.0961–0.2551 (15.9%). **Highly differentiated** for instruction following.
1. fisher_sft (0.2551)
2. opt_gcs_logdet (0.2033)
3. less (0.2015)

## Key Observations

### 1. eval_loss is anti-correlated with downstream benchmarks
`grad_norm_topk` has the best eval_loss (0.813) but worst average rank (8.8/11).
`random_subspace_logdet` has the worst eval_loss (0.936) but best average rank (2.5/11).

### 2. Different selection criteria specialize in different abilities
- **FisherSFT**: Best at IFEval (instruction following), worst at GSM8K (math)
- **OptGCS-Score/Unwhitened**: Best at GSM8K (math reasoning), mid on IFEval
- **RandomSubspace-LogDet**: Most robust across all benchmarks
- **GradNorm-TopK**: Best at reducing validation loss, worst everywhere else

### 3. Spectral score is promising for math transfer
OptGCS-Score and Unwhitened rank #1 and #2 on GSM8K. The spectral representation
captures signal relevant for mathematical reasoning transfer.

### 4. LogDet coverage alone is not enough
OptGCS-LogDet (#5.5 avg) doesn't dominate OptGCS-Score (#6.5) or random-subspace (#2.5).
Coverage may need to be combined with importance scoring.

### 5. Target-aware selection is not universally beneficial
LESS has good eval_loss and MMLU, but poor GSM8K (below base). It over-specializes
to the target validation distribution.

## Implications for Next Round

1. **Need hybrid method**: Combine OptGCS spectral score (good for GSM8K) with LogDet coverage (good for IFEval/robustness).
2. **Need multi-seed random_subspace**: Verify if seed=9999 was lucky or if random-subspace logdet is genuinely strong.
3. **eval_loss should not be used as primary metric** for method selection.

## Evaluation Details

- MMLU: lm_eval v0.4.12, 5-shot, accuracy
- GSM8K: lm_eval v0.4.12, gsm8k_cot, 8-shot, exact_match strict
- MMLU-Subset: Custom script (generation + A/B/C/D extraction), 10261 questions from OpenDCAI/dataflex-selector-MMLUSubset-test
- IFEval: lm_eval v0.4.12, 0-shot, prompt_level_strict_acc
- All evaluations on single H20 GPU per model, bfloat16

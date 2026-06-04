# Three-Target Data Selection Results

**Setup:** Llama-2-7B + LoRA (r=128, α=512), 5% selection from 270k LESS pool, 4-epoch SFT.

**Metrics:** BBH = macro-avg exact_match (CoT 3-shot, 27 subtasks); MMLU = acc (5-shot, 57 subjects); TyDiQA = macro-F1 over languages (macro-EM in parens).

| Method | BBH | MMLU | TyDiQA-F1 | TyDiQA-EM |
|---|---|---|---|---|
| less_sgd | 0.3676 | 0.4633 | 0.4029 | 0.3037 |
| less_adam | 0.4016 | 0.4699 | 0.4869 | 0.3574 |
| mmd_grad_rbf_sgd | 0.3717 | 0.4741 | 0.4512 | 0.3250 |
| mmd_grad_rbf_adam | 0.3832 | 0.4647 | 0.4339 | 0.3211 |
| mmd_grad_cov_sgd | 0.3861 | 0.4748 | 0.4517 | 0.3256 |
| mmd_grad_cov_adam | 0.3667 | 0.4626 | 0.4336 | 0.3104 |
| mmd_emb_rbf | 0.4015 | 0.4484 | 0.2773 | 0.2002 |
| mmd_emb_rbf_stochastic | 0.3948 | 0.4507 | 0.3208 | 0.2335 |

## Per-target best method

- **BBH**: less_adam (0.4016)
- **MMLU**: mmd_grad_cov_sgd (0.4748)
- **TyDiQA-F1**: less_adam (0.4869)

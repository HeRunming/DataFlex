# BBH parser-validity audit

**Status: post-hoc analysis of frozen generation samples.** No generation, selection, or training was rerun. The conservative recovery rule was written before full aggregation and is not a replacement primary metric.

| stack | method | standard valid | conditional EM | conservative EM |
|---|---|---:|---:|---:|
| llama2 | dsmc | 93.46% | 38.85% | 36.42% |
| llama2 | randk | 93.50% | 41.98% | 39.50% |
| llama32 | dsmc | 94.42% | 48.87% | 46.77% |
| llama32 | randk | 93.64% | 50.23% | 47.61% |

## DSMC minus Random

| stack | standard EM | invalid rate | conditional EM | conservative EM |
|---|---:|---:|---:|---:|
| llama2 | -2.94 pp | +0.03 pp | -3.13 pp | -3.08 pp |
| llama32 | -0.89 pp | -0.78 pp | -1.36 pp | -0.84 pp |

The standard-parser invalid-rate difference is negligible on Llama-2 (+0.03 percentage points for DSMC) and favours DSMC on Llama-3.2 (-0.78 points). More importantly, DSMC remains below Random both conditional on standard-valid outputs and under the frozen conservative recovery rule on both stacks. The observed utility gap is therefore not explained by DSMC merely producing more unparseable answers. This audit does not establish semantic correctness beyond the tested extraction rules.

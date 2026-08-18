# ICLR 2027 paper draft

This directory contains the anonymous ICLR 2027 draft for the DataFlex-FA
experiments.

## Build

Use a TeX Live installation with `latexmk`:

```bash
make
```

The submission source is `main.tex`. Keep `\iclrfinalcopy` commented for the
anonymous submission.

## Current status

- The paper uses the official ICLR 2027 style downloaded on 2026-08-18.
- The central BBH and Llama-2 MMLU results are filled from committed result
  artifacts.
- One explicit placeholder remains for the pre-registered
  Llama-3.2-3B × MMLU 5% experiment.
- Appendix sections are structured but still contain red TODO markers.
- Bibliographic metadata should be rechecked against proceedings immediately
  before submission, especially for 2026 preprints that may acquire venue
  citations.

## Source-of-truth result files

- `experiments/less_aligned/results_summary/full5draw_5pct_aggregate.csv`
- `experiments/less_aligned/results_summary/full1pct_aggregate.csv`
- `experiments/less_aligned/results_summary/attribution_2x2_results.csv`
- `experiments/less_aligned/results_summary/bbh_forensic_geometry.json`
- `experiments/less_aligned/results_summary/bbh_forensic_query_loss.json`
- `experiments/less_aligned/results_summary/bbh_forensic_query_cot.json`
- `experiments/less_aligned/results_summary/llama32_results.json`
- `experiments/less_aligned/results_summary/llama32_diagnostics.json`

Do not replace qualified baseline names such as `LESS-style`,
`GIST-SharedProj`, or `NICE-MMLU-EM` with claims of exact official
reproduction.

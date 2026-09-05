# ICLR 2027 paper draft

This directory contains the anonymous ICLR 2027 draft and its paper-facing
artifact manifest.

## Build

Use a TeX Live installation with `latexmk`:

```bash
make
```

The submission source is `main.tex`. Keep `\iclrfinalcopy` commented for the
anonymous submission.

## Current status

- The paper uses the official ICLR 2027 style downloaded on 2026-08-18.
- The MMLU and BBH results, cross-stack checks, and post-hoc diagnostics are
  filled from committed result artifacts.
- `artifact_manifest.md` maps each major figure/table to result files,
  analysis code, protocols, and configurations; the JSON companion records
  hashes and byte sizes.
- `scripts/validate_bbh_run_state.py` verifies that the complete 36-cell
  Llama-2 BBH run state includes the two pre-launch canary cells.
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
- `experiments/less_aligned/results_summary/bbh_draw_uncertainty.json`
- `experiments/less_aligned/results_summary/bbh_run_state_validation.json`

Do not replace qualified baseline names such as `LESS-style`,
`GIST-SharedProj`, or `NICE-MMLU-EM` with claims of exact official
reproduction.

# BBH signed-first-moment and task-level diagnostics

**Status: offline analysis of frozen artifacts.** No selection, training, or evaluation was rerun. Draw remains the primary experimental unit; the subtask bootstrap is a secondary descriptive heterogeneity analysis.

## Signed first-moment discrepancy

$D_1(S,Q)=\|\mathbb E_S[u]-\mathbb E_Q[u]\|_2$ is sign-sensitive.

| stack | draw | DSMC | First-RR | Random | First-RR < Random | First-RR − Random EM |
|---|---:|---:|---:|---:|:---:|---:|
| llama2 | 0 | 0.4886 | 0.4763 | 0.5957 | yes | -3.06 pp |
| llama2 | 1 | 0.4972 | 0.4861 | 0.6032 | yes | -1.17 pp |
| llama2 | 2 | 0.4610 | 0.4516 | 0.5741 | yes | -2.06 pp |
| llama32 | 0 | 0.4404 | 0.4611 | 0.5012 | yes | -1.21 pp |
| llama32 | 1 | 0.4599 | 0.4789 | 0.5179 | yes | -0.10 pp |
| llama32 | 2 | 0.4081 | 0.4374 | 0.4736 | yes | -0.12 pp |

First-RR is closer than Random under signed $D_1$ in **3/3** Llama-2 draws and **3/3** Llama-3.2 draws, while its downstream BBH mean is lower than Random on both stacks (-2.10 and -0.48 points). Thus the DSMC--Random reversal is not the only observed alignment--utility counterexample; a sign-sensitive first-moment selector shows the same ordering. This does not establish failure for every sign-sensitive representation.

## Task-level DSMC minus Random

| stack | wins / ties / losses | macro mean | median | task-bootstrap 95% interval | micro delta |
|---|---:|---:|---:|---:|---:|
| llama2 | 4 / 0 / 23 | -2.89 pp | -1.67 pp | [-4.68, -1.41] pp | -2.94 pp |
| llama32 | 10 / 0 / 17 | -1.10 pp | -0.92 pp | [-2.57, +0.27] pp | -0.89 pp |

The task-level view is descriptive rather than a replacement for the three draw-level primary analysis. It reports how broadly the aggregate gap is distributed across the 27 harness subtasks.

Regrouping size variants into the 23 conceptual BBH families gives:

| stack | family wins / ties / losses | family macro mean | family median | family-bootstrap 95% interval |
|---|---:|---:|---:|---:|
| llama2 | 3 / 0 / 20 | -2.94 pp | -1.64 pp | [-4.94, -1.25] pp |
| llama32 | 9 / 0 / 14 | -1.23 pp | -0.83 pp | [-2.91, +0.26] pp |

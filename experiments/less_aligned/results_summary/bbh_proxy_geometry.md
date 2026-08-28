# Held-out BBH proxy geometry

**Status: post-hoc target-only analysis under a protocol frozen before the proxy gradients were extracted.** Frozen selections and downstream results are reused; no reselection or SFT is performed.

The proxy set contains 256 examples sampled from the query reservoir after excluding every example used by any of the three selection draws.

## Result

| stack | draw | metric | DSMC | First-RR | Second-RR | Random | closest |
|---|---:|---|---:|---:|---:|---:|---|
| llama2 | 0 | D1_proxy | 0.46684 | 0.45508 | 0.50589 | 0.57679 | first_rr |
| llama2 | 0 | D2_proxy | 0.07067 | 0.07250 | 0.07482 | 0.09445 | dsmc |
| llama2 | 1 | D1_proxy | 0.46540 | 0.45579 | 0.50108 | 0.57522 | first_rr |
| llama2 | 1 | D2_proxy | 0.07069 | 0.07256 | 0.07409 | 0.09360 | dsmc |
| llama2 | 2 | D1_proxy | 0.46524 | 0.45562 | 0.50093 | 0.57786 | first_rr |
| llama2 | 2 | D2_proxy | 0.07067 | 0.07237 | 0.07392 | 0.09412 | dsmc |
| llama32 | 0 | D1_proxy | 0.41998 | 0.44544 | 0.43849 | 0.48214 | dsmc |
| llama32 | 0 | D2_proxy | 0.05035 | 0.06087 | 0.05868 | 0.06056 | dsmc |
| llama32 | 1 | D1_proxy | 0.42084 | 0.44568 | 0.43871 | 0.48190 | dsmc |
| llama32 | 1 | D2_proxy | 0.05037 | 0.06083 | 0.05886 | 0.06132 | dsmc |
| llama32 | 2 | D1_proxy | 0.41935 | 0.44450 | 0.44258 | 0.48403 | dsmc |
| llama32 | 2 | D2_proxy | 0.05034 | 0.06230 | 0.06010 | 0.06103 | dsmc |

## Interpretation

On a proxy-test set disjoint from every selection query, the frozen DSMC subsets remain closer than Random under D2 in every draw on both stacks, while the frozen First-RR subsets remain closer under signed D1. Both targeted subsets nevertheless have lower held-out BBH utility than Random. The ordering reversal therefore is not limited to measuring geometry on the examples used for selection.

The proxy set is sampled from the held-out query reservoir rather than the 5,209-example task-evaluation partition. This post-hoc analysis establishes frozen-subset geometry ordering only; it does not evaluate a selector trained against the proxy set.

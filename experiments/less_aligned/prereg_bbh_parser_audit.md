# Frozen post-hoc protocol: BBH parser-validity audit

**Written before aggregating the full set of generation outputs.**

This analysis uses saved `lm-eval` sample JSONL files only. It does not rerun
generation, selection, or training.

## Primary parser diagnostics

For every Llama-2 and Llama-3.2 BBH DSMC/Random cell:

1. `standard_accuracy`: the frozen lm-eval exact-match field.
2. `standard_valid`: the frozen `filtered_resps` value is neither empty nor
   `[invalid]`.
3. `conditional_accuracy`: exact matches divided by standard-valid outputs.

The main question is whether DSMC has a higher invalid-output rate than Random.
If it does not, the DSMC--Random exact-match gap cannot be attributed to DSMC
simply producing more unparseable responses.

## Conservative recovery diagnostic

For outputs invalid under the frozen parser only, apply a fixed
target-type-aware recovery rule to the raw response:

- multiple-choice target `(X)`: take the last parenthesized capital letter in
  the final 500 characters;
- Boolean/yes-no target: take the last standalone `True`, `False`, `Yes`, or
  `No` in the final 500 characters;
- numeric target: take the last standalone signed integer/decimal in the final
  500 characters;
- free-text target: accept only text following a case-insensitive
  `the answer is` or `answer:` marker, or the final non-empty line.

All comparisons use lower-casing, whitespace collapse, and stripping of outer
quotes and terminal punctuation. A recovered answer only changes the
diagnostic if the standard parser marked the example invalid; standard-valid
predictions are never overwritten.

Report the conservative recovered accuracy and the number of recovered invalid
outputs. This is a sensitivity analysis, not a replacement primary metric.

## Aggregation and scope

- Average examples within a run.
- Average the two SFT seeds within each query draw.
- Report three draw-level DSMC--Random differences and their mean.
- No significance testing.
- No result may trigger parser tuning, prompt changes, regeneration, or new
  training.

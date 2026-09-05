#!/usr/bin/env python3
"""Descriptive draw-level uncertainty for the primary BBH DSMC--Random contrast."""

from __future__ import annotations

import argparse
import json
import math
import statistics as st
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "experiments" / "less_aligned" / "results_summary"
T95_DF2 = 4.302652729911275


def summarize(values):
    mean = st.mean(values)
    sd = st.stdev(values)
    half_width = T95_DF2 * sd / math.sqrt(len(values))
    return {
        "per_draw_difference_pp": values,
        "mean_difference_pp": mean,
        "sample_sd_pp": sd,
        "descriptive_t_interval_95_pp": [mean - half_width, mean + half_width],
        "same_sign_draws": sum(x < 0 for x in values),
        "n_draws": len(values),
        "minimum_two_sided_sign_test_p": 0.25,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out",
        default=str(SUMMARY / "bbh_draw_uncertainty.json"),
    )
    args = ap.parse_args()

    l2 = json.loads((SUMMARY / "bbh_forensic_geometry.json").read_text())
    l32 = json.loads((SUMMARY / "llama32_results.json").read_text())
    l2_diff = [
        (
            l2["per_draw"][str(d)]["methods"]["dsmc"]["acc_seed_avg"]
            - l2["per_draw"][str(d)]["methods"]["randk"]["acc_seed_avg"]
        )
        * 100
        for d in range(3)
    ]
    l32_diff = [
        (
            l32["per_draw"][str(d)]["dsmc"]["draw_mean"]
            - l32["per_draw"][str(d)]["randk"]["draw_mean"]
        )
        * 100
        for d in range(3)
    ]

    report = {
        "analysis": "descriptive uncertainty for paired BBH DSMC-minus-Random draw effects",
        "statistical_unit": (
            "query/selection draw; two SFT seeds are averaged within each draw"
        ),
        "scope": (
            "The t intervals summarize dispersion over only three draws and "
            "are not treated as confirmatory population inference."
        ),
        "stacks": {
            "llama2": summarize(l2_diff),
            "llama32": summarize(l32_diff),
        },
        "cross_stack_note": (
            "Both stacks reuse the same three query draws, so the six "
            "stack-draw cells are not six independent selection draws."
        ),
    }
    Path(args.out).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

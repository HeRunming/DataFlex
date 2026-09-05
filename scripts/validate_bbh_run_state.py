#!/usr/bin/env python3
"""Validate completeness and canary integration for the frozen Llama-2 BBH run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments" / "less_aligned"
SUMMARY = EXP / "results_summary"
EXPECTED_STEPS = 84
EXPECTED_SUBTASKS = 27
EXPECTED_EXAMPLES = 5209
N_TRAIN_GPUS = 8


def load(path: Path):
    with path.open() as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out",
        default=str(SUMMARY / "bbh_run_state_validation.json"),
    )
    args = ap.parse_args()

    plan = load(EXP / "bbh_external_run_plan.json")
    state = load(EXP / "bbh_full_run_state.json")
    canary = load(EXP / "bbh_sft_canary_report.json")

    expected_ids = {x["adapter_id"] for x in plan["cells"]}
    state_cells = state["cells"]
    state_ids = set(state_cells)
    missing = sorted(expected_ids - state_ids)
    extra = sorted(state_ids - expected_ids)

    malformed = {}
    for aid, row in state_cells.items():
        problems = []
        if row.get("train_ok") is not True:
            problems.append("train_ok")
        if row.get("eval_ok") is not True:
            problems.append("eval_ok")
        if row.get("global_step") != EXPECTED_STEPS:
            problems.append("global_step")
        if row.get("n_subtasks") != EXPECTED_SUBTASKS:
            problems.append("n_subtasks")
        if row.get("n_examples") != EXPECTED_EXAMPLES:
            problems.append("n_examples")
        for key in ("adapter_sha256", "results_sha256", "_micro_sealed"):
            if key not in row:
                problems.append(key)
        if problems:
            malformed[aid] = problems

    canary_checks = {}
    for aid, crow in canary["cells"].items():
        srow = state_cells.get(aid)
        checks = {
            "present_in_state": srow is not None,
            "adapter_sha256_matches": (
                srow is not None
                and srow.get("adapter_sha256") == crow.get("adapter_sha256")
            ),
            "results_sha256_matches": (
                srow is not None
                and srow.get("results_sha256") == crow.get("eval_results_sha256")
            ),
            "micro_matches": (
                srow is not None
                and srow.get("_micro_sealed") == crow.get("_raw_micro_stored")
            ),
            "global_step_matches": (
                srow is not None and srow.get("global_step") == crow.get("global_step")
            ),
        }
        canary_checks[aid] = checks

    progress = state.get("progress", {})
    progress_matches = (
        progress.get("trained") == len(expected_ids)
        and progress.get("evaluated") == len(expected_ids)
        and progress.get("total") == len(expected_ids)
    )
    canary_matches = all(all(x.values()) for x in canary_checks.values())
    total_minutes = sum(float(x["train_minutes"]) for x in state_cells.values())

    report = {
        "analysis": "Llama-2 BBH run-state completeness validation",
        "expected_cell_count": len(expected_ids),
        "state_cell_count": len(state_ids),
        "missing_cell_ids": missing,
        "extra_cell_ids": extra,
        "malformed_cells": malformed,
        "progress_matches_plan": progress_matches,
        "canary_cells": canary_checks,
        "canary_metadata_matches_state": canary_matches,
        "training_time": {
            "sum_per_adapter_elapsed_minutes": total_minutes,
            "sum_per_adapter_elapsed_hours": total_minutes / 60,
            "aggregate_gpu_hours_at_8_gpus_per_job": total_minutes / 60 * N_TRAIN_GPUS,
        },
    }
    report["PASS"] = not missing and not extra and not malformed and progress_matches and canary_matches
    Path(args.out).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not report["PASS"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

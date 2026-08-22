#!/usr/bin/env python3
"""Offline diagnostics for the paper revision.

Uses only frozen artifacts:
  1. signed first-moment discrepancy D1 from existing geometry JSONs;
  2. paired DSMC-Random BBH subtask differences from existing eval JSONs;
  3. deterministic bootstrap over the 27 harness subtasks.

No model loading, selection, or training is performed.
"""

from __future__ import annotations

import argparse
import glob
import json
import random
import statistics as st
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments" / "less_aligned"
SUMMARY = EXP / "results_summary"
METRIC = "exact_match,get-answer"
METHODS = ("dsmc", "randk")
DRAWS = (0, 1, 2)
SEEDS = (42, 1)


def load_json(path: Path):
    with path.open() as f:
        return json.load(f)


def cells_from_state(path: Path):
    state = load_json(path)
    cells = state["cells"]
    if isinstance(cells, dict):
        return {k: v for k, v in cells.items()}
    return {x["adapter_id"]: x for x in cells}


def load_subtasks(path: str):
    obj = load_json(Path(path))
    out = {}
    counts = {}
    for name, row in obj["results"].items():
        prefix = "bbh_external_heldout_"
        if not name.startswith(prefix):
            continue
        task = name[len(prefix) :]
        out[task] = float(row[METRIC])
        counts[task] = int(obj["n-samples"][name]["effective"])
    if len(out) != 27:
        raise ValueError(f"{path}: expected 27 subtasks, got {len(out)}")
    return out, counts


def aggregate_task_differences(state_path: Path, id_prefix: str):
    cells = cells_from_state(state_path)
    per_task = {}
    counts_ref = None
    for draw in DRAWS:
        scores = {}
        for method in METHODS:
            seed_scores = []
            for seed in SEEDS:
                aid = f"{id_prefix}_draw{draw}_{method}_seed{seed}"
                row = cells.get(aid)
                if row is not None:
                    result_path = row["results_json"]
                elif id_prefix == "bbhx":
                    # Two draw-0 seed-42 cells were completed during the
                    # pre-launch canary and are not duplicated in the final
                    # run-state file. They are nevertheless frozen result
                    # artifacts used by the committed aggregate.
                    pattern = (
                        f"/jizhicfs/karonhe/dataflex_saves/eval_results/"
                        f"bbh_external/{aid}/**/results_*.json"
                    )
                    hits = sorted(glob.glob(pattern, recursive=True))
                    if len(hits) != 1:
                        raise KeyError(f"{aid}: expected one canary result, got {hits}")
                    result_path = hits[0]
                else:
                    raise KeyError(aid)
                scores_i, counts = load_subtasks(result_path)
                if counts_ref is None:
                    counts_ref = counts
                elif counts != counts_ref:
                    raise ValueError("subtask counts differ across result files")
                seed_scores.append(scores_i)
            scores[method] = {
                task: st.mean(x[task] for x in seed_scores)
                for task in seed_scores[0]
            }
        per_task[str(draw)] = {
            task: scores["dsmc"][task] - scores["randk"][task]
            for task in sorted(scores["dsmc"])
        }
    mean_delta = {
        task: st.mean(per_task[str(d)][task] for d in DRAWS)
        for task in sorted(per_task["0"])
    }
    return per_task, mean_delta, counts_ref


def bootstrap_mean(values, seed=20260822, n_boot=20000):
    rng = random.Random(seed)
    n = len(values)
    reps = [
        st.mean(values[rng.randrange(n)] for _ in range(n))
        for _ in range(n_boot)
    ]
    reps.sort()
    lo = reps[int(0.025 * n_boot)]
    hi = reps[int(0.975 * n_boot) - 1]
    return lo, hi


def signed_d1_summary(path: Path, methods):
    obj = load_json(path)
    per_draw = {}
    for draw, row in obj["per_draw"].items():
        vals = {m: float(row["methods"][m]["D1_to_Q"]) for m in methods}
        acc = {
            m: float(row["methods"][m]["acc_seed_avg"])
            for m in methods
        }
        per_draw[draw] = {
            "D1": vals,
            "accuracy": acc,
            "first_rr_minus_random": vals["first_rr"] - vals["randk"],
            "dsmc_minus_random": vals["dsmc"] - vals["randk"],
            "first_rr_minus_random_accuracy": (
                acc["first_rr"] - acc["randk"]
            ),
            "first_rr_closer_than_random": vals["first_rr"] < vals["randk"],
            "dsmc_closer_than_random": vals["dsmc"] < vals["randk"],
        }
    return {
        "definition": obj["definition"]["D1"],
        "per_draw": per_draw,
        "first_rr_closer_than_random": (
            f"{sum(x['first_rr_closer_than_random'] for x in per_draw.values())}"
            f"/{len(per_draw)}"
        ),
        "dsmc_closer_than_random": (
            f"{sum(x['dsmc_closer_than_random'] for x in per_draw.values())}"
            f"/{len(per_draw)}"
        ),
        "mean_first_rr_minus_random": st.mean(
            x["first_rr_minus_random"] for x in per_draw.values()
        ),
        "mean_dsmc_minus_random": st.mean(
            x["dsmc_minus_random"] for x in per_draw.values()
        ),
        "mean_first_rr_minus_random_accuracy": st.mean(
            x["first_rr_minus_random_accuracy"] for x in per_draw.values()
        ),
    }


def task_summary(per_task, mean_delta, counts):
    values = list(mean_delta.values())
    eps = 1e-12
    wins = sum(x > eps for x in values)
    losses = sum(x < -eps for x in values)
    ties = len(values) - wins - losses
    lo, hi = bootstrap_mean(values)
    total = sum(counts.values())
    micro = sum(mean_delta[t] * counts[t] for t in mean_delta) / total
    ordered = sorted(mean_delta.items(), key=lambda kv: kv[1])
    family_values = {}
    for task, value in mean_delta.items():
        if task.startswith("logical_deduction_"):
            family = "logical_deduction"
        elif task.startswith("tracking_shuffled_objects_"):
            family = "tracking_shuffled_objects"
        else:
            family = task
        family_values.setdefault(family, []).append(value)
    family_delta = {
        family: st.mean(xs) for family, xs in sorted(family_values.items())
    }
    family_list = list(family_delta.values())
    flo, fhi = bootstrap_mean(family_list)
    return {
        "per_draw": per_task,
        "mean_delta_by_subtask": mean_delta,
        "n_subtasks": len(values),
        "win_tie_loss_for_dsmc": {"win": wins, "tie": ties, "loss": losses},
        "macro_mean_delta": st.mean(values),
        "macro_median_delta": st.median(values),
        "bootstrap_over_subtasks": {
            "seed": 20260822,
            "n_boot": 20000,
            "percentile_95_interval": [lo, hi],
            "scope": (
                "descriptive heterogeneity interval over the 27 harness subtasks; "
                "subtasks are not the pre-registered experimental unit"
            ),
        },
        "micro_delta_reconstructed": micro,
        "family_regroup": {
            "n_families": len(family_list),
            "mean_delta_by_family": family_delta,
            "win_tie_loss_for_dsmc": {
                "win": sum(x > eps for x in family_list),
                "tie": sum(abs(x) <= eps for x in family_list),
                "loss": sum(x < -eps for x in family_list),
            },
            "macro_mean_delta": st.mean(family_list),
            "macro_median_delta": st.median(family_list),
            "bootstrap_95_interval": [flo, fhi],
        },
        "largest_dsmc_deficits": [
            {"subtask": t, "delta": x} for t, x in ordered[:7]
        ],
        "largest_dsmc_gains": [
            {"subtask": t, "delta": x} for t, x in reversed(ordered[-7:])
        ],
    }


def render_markdown(rep):
    lines = [
        "# BBH signed-first-moment and task-level diagnostics",
        "",
        "**Status: offline analysis of frozen artifacts.** No selection, training, or "
        "evaluation was rerun. Draw remains the primary experimental unit; the "
        "subtask bootstrap is a secondary descriptive heterogeneity analysis.",
        "",
        "## Signed first-moment discrepancy",
        "",
        r"$D_1(S,Q)=\|\mathbb E_S[u]-\mathbb E_Q[u]\|_2$ is sign-sensitive.",
        "",
        "| stack | draw | DSMC | First-RR | Random | First-RR < Random | First-RR − Random EM |",
        "|---|---:|---:|---:|---:|:---:|---:|",
    ]
    for stack in ("llama2", "llama32"):
        for draw, row in rep["signed_D1"][stack]["per_draw"].items():
            d = row["D1"]
            lines.append(
                f"| {stack} | {draw} | {d['dsmc']:.4f} | {d['first_rr']:.4f} | "
                f"{d['randk']:.4f} | {'yes' if row['first_rr_closer_than_random'] else 'no'} | "
                f"{row['first_rr_minus_random_accuracy']*100:+.2f} pp |"
            )
    lines += [
        "",
        "First-RR is closer than Random under signed $D_1$ in "
        f"**{rep['signed_D1']['llama2']['first_rr_closer_than_random']}** "
        "Llama-2 draws and "
        f"**{rep['signed_D1']['llama32']['first_rr_closer_than_random']}** "
        "Llama-3.2 draws, while its downstream BBH mean is lower than Random on "
        "both stacks ("
        f"{rep['signed_D1']['llama2']['mean_first_rr_minus_random_accuracy']*100:+.2f} "
        "and "
        f"{rep['signed_D1']['llama32']['mean_first_rr_minus_random_accuracy']*100:+.2f} "
        "points). Thus the DSMC--Random reversal is not the only observed "
        "alignment--utility counterexample; a sign-sensitive first-moment selector "
        "shows the same ordering. This does not establish failure for every "
        "sign-sensitive representation.",
        "",
        "## Task-level DSMC minus Random",
        "",
        "| stack | wins / ties / losses | macro mean | median | task-bootstrap 95% interval | micro delta |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for stack in ("llama2", "llama32"):
        x = rep["task_level"][stack]
        w = x["win_tie_loss_for_dsmc"]
        ci = x["bootstrap_over_subtasks"]["percentile_95_interval"]
        lines.append(
            f"| {stack} | {w['win']} / {w['tie']} / {w['loss']} | "
            f"{x['macro_mean_delta']*100:+.2f} pp | "
            f"{x['macro_median_delta']*100:+.2f} pp | "
            f"[{ci[0]*100:+.2f}, {ci[1]*100:+.2f}] pp | "
            f"{x['micro_delta_reconstructed']*100:+.2f} pp |"
        )
    lines += [
        "",
        "The task-level view is descriptive rather than a replacement for the "
        "three draw-level primary analysis. It reports how broadly the aggregate "
        "gap is distributed across the 27 harness subtasks.",
        "",
        "Regrouping size variants into the 23 conceptual BBH families gives:",
        "",
        "| stack | family wins / ties / losses | family macro mean | family median | family-bootstrap 95% interval |",
        "|---|---:|---:|---:|---:|",
    ]
    for stack in ("llama2", "llama32"):
        x = rep["task_level"][stack]["family_regroup"]
        w = x["win_tie_loss_for_dsmc"]
        ci = x["bootstrap_95_interval"]
        lines.append(
            f"| {stack} | {w['win']} / {w['tie']} / {w['loss']} | "
            f"{x['macro_mean_delta']*100:+.2f} pp | "
            f"{x['macro_median_delta']*100:+.2f} pp | "
            f"[{ci[0]*100:+.2f}, {ci[1]*100:+.2f}] pp |"
        )
    lines += [
        "",
    ]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out-json",
        default=str(SUMMARY / "bbh_signed_d1_taskwise.json"),
    )
    ap.add_argument(
        "--out-md",
        default=str(SUMMARY / "bbh_signed_d1_taskwise.md"),
    )
    args = ap.parse_args()

    l2_tasks = aggregate_task_differences(
        EXP / "bbh_full_run_state.json", "bbhx"
    )
    l32_tasks = aggregate_task_differences(
        EXP / "llama32_full_run_state.json", "l32"
    )
    rep = {
        "analysis": "signed D1 plus BBH task-level paired DSMC-Random diagnostics",
        "status": "offline frozen-artifact analysis; no new selection/training/evaluation",
        "signed_D1": {
            "llama2": signed_d1_summary(
                SUMMARY / "bbh_forensic_geometry.json",
                ["dsmc", "first_rr", "randk"],
            ),
            "llama32": signed_d1_summary(
                SUMMARY / "llama32_forensic_geometry.json",
                ["dsmc", "first_rr", "randk"],
            ),
        },
        "task_level": {
            "llama2": task_summary(*l2_tasks),
            "llama32": task_summary(*l32_tasks),
        },
        "interpretation": (
            "The observed alignment-utility reversal is not unique to the "
            "sign-invariant second moment: First-RR is closer than Random under "
            "signed D1 in every draw on both stacks, yet has lower held-out "
            "utility. Task-level summaries describe breadth but do not replace "
            "the draw-level primary unit."
        ),
    }
    Path(args.out_json).write_text(json.dumps(rep, indent=2) + "\n")
    Path(args.out_md).write_text(render_markdown(rep) + "\n")
    print(f"wrote {args.out_json}")
    print(f"wrote {args.out_md}")


if __name__ == "__main__":
    main()

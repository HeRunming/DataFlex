#!/usr/bin/env python3
"""Score frozen BBH subsets against the held-out proxy-test gradient set."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments" / "less_aligned"
SUMMARY = EXP / "results_summary"
SAVES = Path(os.environ.get("DATAFLEX_SAVES", ROOT.parent / "dataflex_saves"))
METHODS = ("dsmc", "first_rr", "second_rr", "randk")
DRAWS = (0, 1, 2)

STACKS = {
    "llama2": {
        "candidate": SAVES / "less_output/train/1/all_projected_grads.pt",
        "proxy": SAVES / "proxy_l2_output/target/1/all_projected_grads.pt",
        "selection": lambda d, m: SAVES / f"selbbhx_draw{d}_{m}/step_1.json",
        "accuracy": SUMMARY / "bbh_forensic_geometry.json",
    },
    "llama32": {
        "candidate": SAVES / "llama32_less_output/train/1/all_projected_grads.pt",
        "proxy": SAVES / "proxy_l32_output/target/1/all_projected_grads.pt",
        "selection": lambda d, m: (
            SAVES / f"selbbhx_draw{d}_randk/step_1.json"
            if m == "randk"
            else SAVES / f"sel_llama32_draw{d}_{m}/step_1.json"
        ),
        "accuracy": SUMMARY / "llama32_forensic_geometry.json",
    },
}


def tensor_hash(path: Path):
    x = torch.load(path, map_location="cpu")
    if not torch.is_tensor(x):
        x = torch.as_tensor(x)
    return hashlib.sha256(x.contiguous().numpy().tobytes()).hexdigest()


def unit(x):
    x = x.float()
    return x / x.norm(dim=1, keepdim=True).clamp_min(1e-12)


def moments(u):
    return u.mean(0), (u.T @ u) / u.shape[0]


def render(rep):
    lines = [
        "# Held-out BBH proxy geometry",
        "",
        "**Status: post-hoc target-only analysis under a protocol frozen before "
        "the proxy gradients were extracted.** Frozen selections and downstream "
        "results are reused; no reselection or SFT is performed.",
        "",
        "The proxy set contains 256 examples sampled from the query reservoir "
        "after excluding every example used by any of the three selection draws.",
        "",
        "## Result",
        "",
        "| stack | draw | metric | DSMC | First-RR | Second-RR | Random | closest |",
        "|---|---:|---|---:|---:|---:|---:|---|",
    ]
    for stack in STACKS:
        for draw in map(str, DRAWS):
            row = rep["stacks"][stack]["per_draw"][draw]
            for metric in ("D1_proxy", "D2_proxy"):
                vals = row["methods"]
                lines.append(
                    f"| {stack} | {draw} | {metric} | "
                    f"{vals['dsmc'][metric]:.5f} | "
                    f"{vals['first_rr'][metric]:.5f} | "
                    f"{vals['second_rr'][metric]:.5f} | "
                    f"{vals['randk'][metric]:.5f} | "
                    f"{row[f'ranking_{metric}_best_first'][0]} |"
                )
    lines += [
        "",
        "## Interpretation",
        "",
        rep["VERDICT"]["reading"],
        "",
        rep["VERDICT"]["scope"],
        "",
    ]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out",
        default=str(SUMMARY / "bbh_proxy_geometry.json"),
    )
    ap.add_argument(
        "--markdown",
        default=str(SUMMARY / "bbh_proxy_geometry.md"),
    )
    args = ap.parse_args()

    manifest = json.load(
        open(ROOT / "data/bbh_external/bbh_proxy_test_manifest.json")
    )
    rep = {
        "analysis": "held-out BBH proxy geometry on frozen selected subsets",
        "status": (
            "post-hoc target-only analysis; proxy protocol frozen before "
            "gradient extraction; no reselection or SFT"
        ),
        "proxy_manifest": "data/bbh_external/bbh_proxy_test_manifest.json",
        "proxy_size": manifest["size"],
        "proxy_selected_ids_sha256": manifest["selected_ids_sha256"],
        "definition": {
            "D1_proxy": "||mean(U_S)-mean(U_Qproxy)||_2",
            "D2_proxy": "||M_S-M_Qproxy||_F^2",
        },
        "stacks": {},
    }

    for stack, cfg in STACKS.items():
        x = unit(torch.load(cfg["candidate"], map_location="cpu"))
        q = unit(torch.load(cfg["proxy"], map_location="cpu"))
        if q.shape != (256, 8192) or not torch.isfinite(q).all():
            raise ValueError(f"{stack}: invalid proxy tensor {q.shape}")
        q1, q2 = moments(q)
        acc_json = json.load(open(cfg["accuracy"]))
        stack_rep = {
            "candidate_tensor_sha256": tensor_hash(cfg["candidate"]),
            "proxy_tensor_sha256": tensor_hash(cfg["proxy"]),
            "proxy_shape": list(q.shape),
            "per_draw": {},
        }
        for draw in DRAWS:
            methods = {}
            for method in METHODS:
                selection_path = cfg["selection"](draw, method)
                idx = json.load(open(selection_path))["indices"]
                u = x[torch.tensor(idx)]
                m1, m2 = moments(u)
                methods[method] = {
                    "D1_proxy": float((m1 - q1).norm()),
                    "D2_proxy": float(((m2 - q2) ** 2).sum()),
                    "selection_cache": str(selection_path.relative_to(SAVES)),
                    "n_selected": len(idx),
                    "heldout_accuracy": float(
                        acc_json["per_draw"][str(draw)]["methods"][method][
                            "acc_seed_avg"
                        ]
                    ),
                }
            row = {"methods": methods}
            for metric in ("D1_proxy", "D2_proxy"):
                order = sorted(METHODS, key=lambda m: methods[m][metric])
                row[f"ranking_{metric}_best_first"] = order
                row[f"dsmc_closer_than_random_{metric}"] = (
                    methods["dsmc"][metric] < methods["randk"][metric]
                )
                row[f"first_rr_closer_than_random_{metric}"] = (
                    methods["first_rr"][metric] < methods["randk"][metric]
                )
            stack_rep["per_draw"][str(draw)] = row
        stack_rep["summary"] = {
            metric: {
                "dsmc_closer_than_random": (
                    f"{sum(r[f'dsmc_closer_than_random_{metric}'] for r in stack_rep['per_draw'].values())}/3"
                ),
                "first_rr_closer_than_random": (
                    f"{sum(r[f'first_rr_closer_than_random_{metric}'] for r in stack_rep['per_draw'].values())}/3"
                ),
                "dsmc_mean_minus_random": sum(
                    r["methods"]["dsmc"][metric] - r["methods"]["randk"][metric]
                    for r in stack_rep["per_draw"].values()
                )
                / 3,
                "first_rr_mean_minus_random": sum(
                    r["methods"]["first_rr"][metric]
                    - r["methods"]["randk"][metric]
                    for r in stack_rep["per_draw"].values()
                )
                / 3,
            }
            for metric in ("D1_proxy", "D2_proxy")
        }
        rep["stacks"][stack] = stack_rep

    robust_d2 = all(
        rep["stacks"][s]["summary"]["D2_proxy"]["dsmc_closer_than_random"]
        == "3/3"
        for s in STACKS
    )
    robust_d1 = all(
        rep["stacks"][s]["summary"]["D1_proxy"][
            "first_rr_closer_than_random"
        ]
        == "3/3"
        for s in STACKS
    )
    rep["VERDICT"] = {
        "dsmc_D2_closer_than_random_both_stacks": robust_d2,
        "first_rr_D1_closer_than_random_both_stacks": robust_d1,
        "reading": (
            "On a proxy-test set disjoint from every selection query, the frozen "
            "DSMC subsets remain closer than Random under D2 in every draw on "
            "both stacks, while the frozen First-RR subsets remain closer under "
            "signed D1. Both targeted subsets nevertheless have lower held-out "
            "BBH utility than Random. The ordering reversal therefore is not "
            "limited to measuring geometry on the examples used for selection."
            if robust_d2 and robust_d1
            else
            "Held-out proxy geometry does not preserve all primary target-geometry "
            "orderings. Claims must be restricted to the orderings reported here."
        ),
        "scope": (
            "The proxy set is sampled from the held-out query reservoir rather "
            "than the 5,209-example task-evaluation partition. This post-hoc "
            "analysis establishes frozen-subset geometry ordering only; it does "
            "not evaluate a selector trained against the proxy set."
        ),
    }
    Path(args.out).write_text(json.dumps(rep, indent=2) + "\n")
    Path(args.markdown).write_text(render(rep).rstrip() + "\n")
    print(f"wrote {args.out}")
    print(f"wrote {args.markdown}")


if __name__ == "__main__":
    main()

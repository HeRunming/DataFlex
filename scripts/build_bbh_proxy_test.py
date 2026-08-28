#!/usr/bin/env python3
"""Build and render a held-out BBH proxy-test set under a frozen protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
from collections import Counter, defaultdict
from pathlib import Path

from render_bbh_query_prompts import load_pinned_tasks


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "bbh_external"
PROMPTS = DATA / "query_prompts"
EXP = ROOT / "experiments" / "less_aligned"


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def sha_ids(ids) -> str:
    return hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.open() if line.strip()]


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def allocate(counts, total):
    n = sum(counts.values())
    quota = {k: total * v / n for k, v in counts.items()}
    out = {k: math.floor(x) for k, x in quota.items()}
    remaining = total - sum(out.values())
    order = sorted(counts, key=lambda k: (-(quota[k] - out[k]), k))
    for k in order[:remaining]:
        out[k] += 1
    if sum(out.values()) != total or any(out[k] > counts[k] for k in out):
        raise ValueError("invalid allocation")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--seed", type=int, default=20260827)
    ap.add_argument("--raw-out", default=str(DATA / "bbh_proxy_test.jsonl"))
    ap.add_argument(
        "--prompt-out",
        default=str(PROMPTS / "bbh_proxy_test_prompts.jsonl"),
    )
    ap.add_argument(
        "--manifest",
        default=str(DATA / "bbh_proxy_test_manifest.json"),
    )
    args = ap.parse_args()

    reservoir_path = DATA / "bbh_query_reservoir.jsonl"
    reservoir = read_jsonl(reservoir_path)
    draws = {
        d: read_jsonl(DATA / f"bbh_query_draw{d}.jsonl")
        for d in range(3)
    }
    excluded = {x["id"] for rows in draws.values() for x in rows}
    remaining = [x for x in reservoir if x["id"] not in excluded]
    if len(reservoir) != 1302 or len(excluded) != 185 or len(remaining) != 1117:
        raise ValueError(
            f"unexpected population sizes: {len(reservoir)}, {len(excluded)}, "
            f"{len(remaining)}"
        )

    by_task = defaultdict(list)
    for row in remaining:
        by_task[row["file_task"]].append(row)
    counts = {k: len(v) for k, v in by_task.items()}
    allocation = allocate(counts, args.size)

    rng = random.Random(args.seed)
    selected = []
    for task in sorted(by_task):
        rows = sorted(by_task[task], key=lambda x: x["id"])
        rng.shuffle(rows)
        selected.extend(rows[: allocation[task]])
    selected = sorted(selected, key=lambda x: (x["file_task"], x["id"]))

    raw_out = Path(args.raw_out)
    prompt_out = Path(args.prompt_out)
    manifest = Path(args.manifest)
    write_jsonl(raw_out, selected)

    tasks = load_pinned_tasks(sorted(by_task))
    rendered = []
    prompt_meta = []
    for row in selected:
        task = tasks[row["file_task"]]
        nshot = task.config.num_fewshot
        context = task.fewshot_context(
            {"input": row["input"], "target": row["target"]},
            nshot,
        )
        if row["input"] not in context:
            raise ValueError(f"{row['id']}: input absent from rendered context")
        rendered.append(
            {
                "dataset": "bbh_proxy_test",
                "id": row["id"],
                "messages": [
                    {"role": "user", "content": context},
                    {"role": "assistant", "content": row["target"]},
                ],
            }
        )
        prompt_meta.append(
            {
                "id": row["id"],
                "file_task": row["file_task"],
                "prompt_sha256": hashlib.sha256(context.encode()).hexdigest(),
                "target_sha256": hashlib.sha256(row["target"].encode()).hexdigest(),
            }
        )
    write_jsonl(prompt_out, rendered)

    report = {
        "analysis": "held-out BBH proxy-test construction",
        "protocol": "experiments/less_aligned/prereg_bbh_proxy_geometry.md",
        "seed": args.seed,
        "size": args.size,
        "population": {
            "reservoir": len(reservoir),
            "selection_draw_union": len(excluded),
            "remaining_after_exclusion": len(remaining),
        },
        "sampling": (
            "deterministic proportional stratification over 27 harness "
            "subtasks using largest-remainder allocation"
        ),
        "remaining_counts": counts,
        "allocation": allocation,
        "selected_counts": dict(Counter(x["file_task"] for x in selected)),
        "selected_ids_sha256": sha_ids([x["id"] for x in selected]),
        "source_hashes": {
            "reservoir": sha_file(reservoir_path),
            **{
                f"draw{d}": sha_file(DATA / f"bbh_query_draw{d}.jsonl")
                for d in draws
            },
        },
        "outputs": {
            "raw_jsonl": str(raw_out.relative_to(ROOT)),
            "raw_sha256": sha_file(raw_out),
            "prompt_jsonl": str(prompt_out.relative_to(ROOT)),
            "prompt_sha256": sha_file(prompt_out),
        },
        "prompt_renderer": (
            "pinned custom BBH lm-eval task objects via fewshot_context()"
        ),
        "per_example_prompts": prompt_meta,
        "assertions": {
            "selected_disjoint_from_all_selection_draws": not (
                {x["id"] for x in selected} & excluded
            ),
            "all_27_subtasks_covered": len(
                {x["file_task"] for x in selected}
            )
            == 27,
        },
    }
    manifest.write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {raw_out} ({len(selected)} rows)")
    print(f"wrote {prompt_out} ({len(rendered)} rows)")
    print(f"wrote {manifest}")


if __name__ == "__main__":
    main()

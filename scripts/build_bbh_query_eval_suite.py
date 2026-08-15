#!/usr/bin/env python3
"""Build frozen lm-eval task configs for the 64-example QUERY draws (advice_0814_2). Artifacts only.

Purpose: evaluate CoT exact-match on the *exact same items* whose final-answer cross-entropy the
selection signal is built from. Comparing query CE (64 items) against held-out EM (5,209 items) changes
BOTH the metric AND the examples, so it cannot separate

    (a) surrogate-metric misalignment  -- final-answer CE improves, CoT EM does not, ON THE SAME ITEMS
    (b) finite-query overfitting       -- query CoT EM improves but held-out EM degrades

This suite closes that gap: same items, both measurements.

Emitted configs are byte-identical to the held-out suite except for the dataset source (the draw's 64
query items) and the task name — same description, same de-duplicated 3-shot CoT demonstrations, same
generate_until / greedy / max_gen_toks=1024 / until stops / get-answer regex / exact_match / micro
aggregation. The demonstrations are inherited from the already-frozen held-out configs, so the
CoT-cue de-duplication carries over automatically.
"""
import argparse, hashlib, json, os

ROOT = "/jizhicfs/karonhe/DataFlex_fa"
EXP = f"{ROOT}/experiments/less_aligned"
HELDOUT_TASKS = f"{EXP}/bbh_external_tasks"
SPLIT = f"{ROOT}/data/bbh_external"
OUT = f"{EXP}/bbh_query_tasks"


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--draws", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--manifest", default=f"{EXP}/bbh_query_eval_pin_manifest.json")
    args = ap.parse_args()
    import yaml

    man = {"purpose": ("CoT exact-match on the SAME 64 query items whose final-answer CE defines the "
                       "targeting signal, so metric and examples are no longer confounded"),
           "derived_from": "the frozen bbh_external_tasks held-out suite",
           "only_difference": "dataset source (the draw's query items) and the task name",
           "draws": {}}

    for d in args.draws:
        rows = [json.loads(l) for l in open(f"{SPLIT}/bbh_query_draw{d}.jsonl") if l.strip()]
        by_task = {}
        for r in rows:
            by_task.setdefault(r["file_task"], []).append(r)
        outd = f"{OUT}/draw{d}"
        os.makedirs(f"{outd}/data", exist_ok=True)
        emitted = {}
        for task, rs in sorted(by_task.items()):
            src_cfg = f"{HELDOUT_TASKS}/bbh_external_heldout_{task}.yaml"
            cfg = yaml.safe_load(open(src_cfg))
            dp = f"{outd}/data/{task}_query.jsonl"
            with open(dp, "w") as f:
                for r in rs:
                    f.write(json.dumps({"input": r["input"], "target": r["target"]},
                                       ensure_ascii=False) + "\n")
            cfg["task"] = f"bbh_query_d{d}_{task}"
            cfg["dataset_kwargs"] = {"data_files": {"test": dp}}
            op = f"{outd}/bbh_query_d{d}_{task}.yaml"
            with open(op, "w") as f:
                yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True, width=10 ** 6)
            emitted[task] = {"n": len(rs), "config_sha256": sha_file(op), "data_sha256": sha_file(dp),
                             "inherited_from_sha256": sha_file(src_cfg)}
        grp = {"group": f"bbh_query_d{d}",
               "task": [f"bbh_query_d{d}_{t}" for t in sorted(emitted)],
               "aggregate_metric_list": [{"metric": "exact_match", "aggregation": "mean",
                                          "weight_by_size": True, "filter_list": "get-answer"}],
               "metadata": {"version": 1.0,
                            "derived_from": "bbh_external_heldout (frozen); dataset source only"}}
        gp = f"{outd}/_bbh_query_d{d}.yaml"
        with open(gp, "w") as f:
            yaml.safe_dump(grp, f, sort_keys=False, allow_unicode=True, width=10 ** 6)
        man["draws"][str(d)] = {"group": f"bbh_query_d{d}", "include_path": os.path.relpath(outd, ROOT),
                                "n_subtasks": len(emitted), "n_examples": sum(v["n"] for v in emitted.values()),
                                "group_sha256": sha_file(gp), "subtasks": emitted}
        print(f"[query-suite] draw{d}: {len(emitted)} subtasks, "
              f"{sum(v['n'] for v in emitted.values())} examples -> {outd}")

    json.dump(man, open(args.manifest, "w"), indent=2)
    print(f"wrote {args.manifest}")


if __name__ == "__main__":
    main()

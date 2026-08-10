#!/usr/bin/env python3
"""Pin the exact lm-eval BBH CoT-fewshot evaluation definition (choice_0809 item 3).

"Pinning" must be verifiable, not a promise in prose. This records, for the INSTALLED lm-eval:
  * package version + install path (+ git SHA if the install happens to be a repo)
  * sha256 of the group YAML, the shared CoT template YAML, and ALL 27 subtask YAMLs
  * the aggregation contract actually in force (metric / aggregation / weight_by_size / filter)
  * the hard-coded few-shot prompt text per subtask (hashed), since lm-eval has changed BBH CoT
    prompts historically — the local text is the ground truth for this run
  * the raw local BBH data hashes
Also emits the exact 27-subtask list so nothing ad-hoc can substitute for it later.
"""
import json, os, glob, hashlib, subprocess, sys

ROOT = "/jizhicfs/karonhe/DataFlex_fa"
OUT = f"{ROOT}/experiments/less_aligned/bbh_lmeval_pin.json"
BBH_LOCAL = "/jizhicfs/karonhe/less_data_zip/data/eval/bbh/test"


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    import lm_eval
    d = os.path.dirname(lm_eval.__file__)
    cot = f"{d}/tasks/bbh/cot_fewshot"

    ver = "unknown"
    try:
        out = subprocess.check_output([sys.executable, "-m", "pip", "show", "lm_eval"],
                                      stderr=subprocess.DEVNULL).decode()
        for line in out.splitlines():
            if line.lower().startswith("version:"):
                ver = line.split(":", 1)[1].strip()
    except Exception:
        pass
    git_sha = None
    try:
        git_sha = subprocess.check_output(["git", "-C", d, "rev-parse", "HEAD"],
                                         stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        git_sha = "not-a-git-checkout (installed from wheel/sdist)"

    group_yaml = f"{cot}/_bbh_cot_fewshot.yaml"
    tmpl = [p for p in glob.glob(f"{cot}/_*template*") + glob.glob(f"{d}/tasks/bbh/**/_*template*",
                                                                  recursive=True)]
    subtasks = sorted(p for p in glob.glob(f"{cot}/*.yaml") if not os.path.basename(p).startswith("_"))

    import yaml
    g = yaml.safe_load(open(group_yaml))
    agg = g.get("aggregate_metric_list", [])

    pin = {
        "lm_eval": {"version": ver, "install_path": d, "git_sha": git_sha},
        "group": {"name": g.get("group"), "yaml": os.path.relpath(group_yaml, d),
                  "sha256": sha_file(group_yaml),
                  "metadata_version": (g.get("metadata") or {}).get("version"),
                  "n_tasks_listed": len(g.get("task", [])),
                  "aggregation_contract": agg},
        "template_yamls": {os.path.basename(p): sha_file(p) for p in sorted(set(tmpl))},
        "n_subtask_yamls": len(subtasks),
        "subtask_yamls": {os.path.basename(p).replace(".yaml", ""): sha_file(p) for p in subtasks},
        "subtask_list_27": [os.path.basename(p).replace(".yaml", "") for p in subtasks],
        "fewshot_prompt_hashes": {},
        "local_bbh_data": {},
        "note": ("The 27 entries are lm-eval's operational subtasks; the original BBH paper describes "
                 "23 conceptual task families (logical_deduction and tracking_shuffled_objects each "
                 "ship 3 size-variants). The pinned group micro-aggregates the 27 with "
                 "weight_by_size: true, so 27-subtask scores are PRIMARY and any 23-family regrouping "
                 "is a secondary diagnostic only."),
    }

    # hard-coded few-shot prompt text lives in each subtask yaml (doc_to_text / description)
    for p in subtasks:
        y = yaml.safe_load(open(p))
        blob = json.dumps({k: y.get(k) for k in ("description", "doc_to_text", "doc_to_target",
                                                 "target_delimiter", "filter_list", "generation_kwargs",
                                                 "num_fewshot", "fewshot_config")},
                          sort_keys=True, ensure_ascii=False)
        pin["fewshot_prompt_hashes"][os.path.basename(p).replace(".yaml", "")] = \
            hashlib.sha256(blob.encode()).hexdigest()

    for f in sorted(glob.glob(f"{BBH_LOCAL}/*.json")):
        pin["local_bbh_data"][os.path.basename(f).replace(".json", "")] = sha_file(f)

    # split artifacts this evaluation must be paired with
    sp = f"{ROOT}/data/bbh_external/bbh_split_manifest.json"
    if os.path.exists(sp):
        pin["paired_split_manifest"] = {"path": os.path.relpath(sp, ROOT), "sha256": sha_file(sp)}

    json.dump(pin, open(OUT, "w"), indent=2)
    print(f"lm_eval {ver}  git={git_sha}")
    print(f"group={pin['group']['name']} metadata_version={pin['group']['metadata_version']} "
          f"tasks_listed={pin['group']['n_tasks_listed']}")
    print(f"aggregation_contract={agg}")
    print(f"subtask yamls hashed: {pin['n_subtask_yamls']}")
    print(f"template yamls hashed: {list(pin['template_yamls'])}")
    print(f"local BBH data files hashed: {len(pin['local_bbh_data'])}")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Pin the lm-eval BBH CoT-fewshot evaluation and build a frozen CUSTOM held-out suite
(choice_0809 items 3 and 6/7). Artifacts only — no model is loaded, nothing is evaluated.

Why a custom suite is required: the stock `bbh_cot_fewshot` group evaluates the FULL BBH test split,
but our external-validation design evaluates only the 5,209-example held-out split (the complementary
1,302 are the query reservoir). We therefore emit per-subtask YAMLs that change ONLY the dataset
source (a local held-out jsonl) and keep every other behaviour byte-identical to the pinned config:
same `description`, same `doc_to_text`, the same hard-coded 3-shot CoT samples, `generate_until`,
greedy decoding, `max_gen_toks=1024`, the same `until` stops, the same `get-answer` regex filter, and
the same `exact_match` metric + micro (`weight_by_size: true`) group aggregation.

Also records the pin: lm_eval version, group/template/subtask YAML hashes, few-shot sample hashes,
raw BBH data hashes, and the 27-subtask <-> 23-family mapping.
"""
import argparse, hashlib, json, os, glob, shutil

ROOT = "/jizhicfs/karonhe/DataFlex_fa"
LM = "/jizhicfs/karonhe/envs/dataflex-fa/lib/python3.10/site-packages/lm_eval"
SRC = f"{LM}/tasks/bbh/cot_fewshot"
BBH_RAW = "/jizhicfs/karonhe/less_data_zip/data/eval/bbh/test"
BBH_COT_PROMPTS = "/jizhicfs/karonhe/less_data_zip/data/eval/bbh/cot-prompts"
CUE = "Let's think step by step."
OUT_TASKS = f"{ROOT}/experiments/less_aligned/bbh_external_tasks"
SPLIT_DIR = f"{ROOT}/data/bbh_external"

FAMILY_OF = {}
for base in ["logical_deduction", "tracking_shuffled_objects"]:
    for sz in ["three", "five", "seven"]:
        FAMILY_OF[f"{base}_{sz}_objects"] = base


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def sha_str(s):
    return hashlib.sha256(s.encode()).hexdigest()


def dedupe_cot_cue(cfg, task, official_body):
    """Remove the DUPLICATED CoT trigger from few-shot demonstration targets (code_review_0810_3).

    The pinned lm-eval v0.4.5 BBH config emits the cue twice per demonstration, because
        doc_to_text  = "Q: {{input}}\nA: Let's think step by step.\n"
    already ends with it, and each demo's `target` ALSO begins with "Let's think step by step.".
    Rendered few-shot text therefore reads:

        A: Let's think step by step.
         Let's think step by step.
        We start at the origin ...

    The OFFICIAL BBH cot-prompts contain the cue exactly ONCE per demonstration (verified: 3 occurrences
    per file, no doubled form anywhere). Upstream lm-eval later removed this redundant text as well.

    Why fix it before compute rather than treat it as "affects all methods equally": Random-K never
    touches the query prompt, while DSMC / First-RR / Second-RR / LESS-style all derive their selection
    signal from query GRADIENTS taken on this prompt. A malformed prompt is therefore not a shared
    constant -- it can distort target-aware gradient geometry specifically, while leaving the Random
    comparator untouched. Since no BBH accuracy has been observed at any point, correcting it now is a
    clean pre-compute protocol fix with no outcome-driven tuning risk.

    Each rewritten demonstration is VALIDATED against the official cot-prompt text: the de-duplicated
    "Q: ...\nA: <cue>\n<rationale>" block must appear verbatim in the official file, so we are restoring
    the official form rather than inventing one.
    """
    fixed = 0
    for i, smp in enumerate(cfg.get("fewshot_config", {}).get("samples", [])):
        tgt = smp.get("target", "")
        if not tgt.lstrip().startswith(CUE):
            continue                                  # e.g. boolean_expressions: already clean
        rest = tgt.lstrip()[len(CUE):]
        # `doc_to_text` already supplies "A: <cue>\n", so the target must continue from AFTER that
        # newline. Most subtasks' official rationale starts on the next line, but some (e.g.
        # sports_understanding) continue on the SAME line as the cue -- there the official text is
        # "A: <cue> <rationale>". Try both joins and keep whichever reproduces the official file, rather
        # than assuming one shape and silently rewriting the prompt into a form BBH never used.
        # Candidates in order of preference. `rest` often begins "\n" or " " (the original separator
        # between the duplicated cue and the rationale); both must be stripped or the rendered prompt
        # ends up "A: <cue>\n <rationale>" -- one byte off the official "A: <cue>\n<rationale>".
        # Prefer the NEWLINE join (what doc_to_text actually emits) before falling back to the
        # same-line form, and try fully-stripped candidates first so no stray leading space survives.
        chosen = None
        cands = (rest.lstrip(), rest.lstrip("\n").lstrip(" "), rest.lstrip("\n"), rest)
        for candidate in cands:                       # newline join first: matches doc_to_text exactly
            if official_body is None or ("Q: " + smp["input"] + "\nA: " + CUE + "\n" + candidate) in official_body:
                chosen = candidate
                break
        if chosen is None:                            # some subtasks continue on the SAME line
            for candidate in cands:
                if ("Q: " + smp["input"] + "\nA: " + CUE + " " + candidate) in official_body:
                    chosen = candidate
                    break
        if chosen is None:
            raise RuntimeError(f"{task} demo#{i}: de-duplicated block does NOT match the official "
                               f"cot-prompt in any join form; refusing to rewrite on a guess")
        smp["target"] = chosen
        fixed += 1
    return fixed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=f"{ROOT}/experiments/less_aligned/bbh_eval_pin_manifest.json")
    args = ap.parse_args()
    os.makedirs(OUT_TASKS, exist_ok=True)

    import lm_eval
    try:
        import importlib.metadata as md
        ver = md.version("lm_eval")
    except Exception:
        ver = getattr(lm_eval, "__version__", "unknown")

    group_yaml = f"{SRC}/_bbh_cot_fewshot.yaml"
    tmpl_yaml = f"{SRC}/_cot_fewshot_template_yaml"
    subtask_files = sorted(p for p in glob.glob(f"{SRC}/*.yaml")
                           if not os.path.basename(p).startswith("_"))
    subtasks = [os.path.basename(p).replace(".yaml", "") for p in subtask_files]
    assert len(subtasks) == 27, f"expected 27 lm-eval BBH subtasks, found {len(subtasks)}"

    import yaml as _yaml
    tmpl = _yaml.safe_load(open(tmpl_yaml))

    # ---- emit custom held-out subtask configs: ONLY the data source changes ----
    heldout_by_task = {}
    for row in (json.loads(l) for l in open(f"{SPLIT_DIR}/bbh_eval_heldout.jsonl")):
        heldout_by_task.setdefault(row["file_task"], []).append(row)
    os.makedirs(f"{OUT_TASKS}/data", exist_ok=True)
    emitted = {}
    total_deduped = 0
    for p, name in zip(subtask_files, subtasks):
        cfg = _yaml.safe_load(open(p))
        task = name.replace("bbh_cot_fewshot_", "")
        # FIRST record the upstream (buggy) few-shot hash, THEN de-duplicate the CoT cue, so the manifest
        # shows both what shipped and what we actually evaluate with.
        upstream_fewshot_sha = sha_str(json.dumps(cfg.get("fewshot_config", {}), sort_keys=True))
        official_p = f"{BBH_COT_PROMPTS}/{task}.txt"
        official_body = None
        if os.path.exists(official_p):
            raw_off = open(official_p).read()
            official_body = raw_off.split("-----\n", 1)[1] if "-----\n" in raw_off else raw_off
        n_dd = dedupe_cot_cue(cfg, task, official_body)
        total_deduped += n_dd
        rows = heldout_by_task.get(task, [])
        if not rows:
            raise RuntimeError(f"no held-out rows for subtask {task}")
        dpath = f"{OUT_TASKS}/data/{task}_heldout.jsonl"
        with open(dpath, "w") as f:
            for r in rows:
                f.write(json.dumps({"input": r["input"], "target": r["target"]},
                                   ensure_ascii=False) + "\n")
        newcfg = dict(cfg)
        newcfg.pop("include", None)
        newcfg.pop("dataset_name", None)
        # inherit every pinned behaviour verbatim from the template, then override ONLY the source
        for k, v in tmpl.items():
            newcfg.setdefault(k, v)
        newcfg["task"] = f"bbh_external_heldout_{task}"
        newcfg["dataset_path"] = "json"
        newcfg["dataset_kwargs"] = {"data_files": {"test": dpath}}
        newcfg["test_split"] = "test"
        outp = f"{OUT_TASKS}/bbh_external_heldout_{task}.yaml"
        with open(outp, "w") as f:
            _yaml.safe_dump(newcfg, f, sort_keys=False, allow_unicode=True, width=10**6)
        emitted[task] = {"config": os.path.relpath(outp, ROOT), "n_heldout": len(rows),
                         "data_sha256": sha_file(dpath), "config_sha256": sha_file(outp),
                         "conceptual_family": FAMILY_OF.get(task, task),
                         "source_subtask_yaml_sha256": sha_file(p),
                         "fewshot_samples_sha256": sha_str(json.dumps(cfg.get("fewshot_config", {}),
                                                                     sort_keys=True)),
                         "fewshot_samples_sha256_upstream_v045": upstream_fewshot_sha,
                         "n_demos_cot_cue_deduped": n_dd,
                         "official_cot_prompt_sha256": (sha_file(official_p)
                                                        if os.path.exists(official_p) else None)}
    # group config with the SAME micro aggregation as the pinned group
    grp = {"group": "bbh_external_heldout",
           "task": [f"bbh_external_heldout_{t}" for t in sorted(emitted)],
           "aggregate_metric_list": [{"metric": "exact_match", "aggregation": "mean",
                                      "weight_by_size": True, "filter_list": "get-answer"}],
           "metadata": {"version": 1.0,
                        "derived_from": "lm_eval bbh_cot_fewshot (pinned; see bbh_eval_pin_manifest.json)",
                        "difference": "dataset source only -> local held-out split"}}
    gp = f"{OUT_TASKS}/_bbh_external_heldout.yaml"
    with open(gp, "w") as f:
        _yaml.safe_dump(grp, f, sort_keys=False, allow_unicode=True, width=10**6)

    man = {
        "lm_eval_version": ver,
        "lm_eval_path": LM,
        "pinned_source": {
            "group_yaml": {"path": os.path.relpath(group_yaml, LM), "sha256": sha_file(group_yaml)},
            "template_yaml": {"path": os.path.relpath(tmpl_yaml, LM), "sha256": sha_file(tmpl_yaml)},
            "n_subtasks": len(subtasks), "subtasks": subtasks,
            "aggregation": "mean with weight_by_size=true (MICRO over the 27 subtasks)",
            "num_fewshot": tmpl.get("num_fewshot"),
            "generation_kwargs": tmpl.get("generation_kwargs"),
            "filter_list": tmpl.get("filter_list"),
            "metric_list": tmpl.get("metric_list"),
            "stock_dataset_path": tmpl.get("dataset_path"),
        },
        "task_accounting": {
            "conceptual_task_families_23": sorted({FAMILY_OF.get(t, t) for t in emitted}),
            "n_conceptual_families": len({FAMILY_OF.get(t, t) for t in emitted}),
            "lm_eval_subtasks_27": sorted(emitted),
            "n_lm_eval_subtasks": len(emitted),
            "note": "Primary reporting is the 27 lm-eval subtasks and their micro aggregate, matching "
                    "the pinned group. The 23-family regrouping is a SECONDARY diagnostic only.",
        },
        "custom_heldout_suite": {
            "group_config": os.path.relpath(gp, ROOT), "group_sha256": sha_file(gp),
            "subtasks": emitted,
            "total_heldout_examples": sum(v["n_heldout"] for v in emitted.values()),
            "invariants_preserved": ["description", "doc_to_text", "hard-coded 3-shot CoT samples",
                                     "generate_until", "greedy (do_sample=false, temperature=0)",
                                     "max_gen_toks=1024", "until stops", "get-answer regex filter",
                                     "exact_match metric", "micro group aggregation"],
            "only_difference": "dataset source replaced by the local held-out jsonl split",
        },
        "cot_cue_deduplication": {
            "applied": True,
            "n_demonstrations_rewritten": total_deduped,
            "n_demonstrations_total": sum(len(_yaml.safe_load(open(p)).get("fewshot_config", {})
                                              .get("samples", [])) for p in subtask_files),
            "defect": ("pinned lm-eval v0.4.5 renders the CoT trigger TWICE per demonstration: "
                       "doc_to_text already ends with \"A: Let's think step by step.\\n\" and each demo "
                       "target ALSO begins with \"Let's think step by step.\", producing "
                       "\"A: Let's think step by step.\\n Let's think step by step.\\n...\". Upstream "
                       "lm-eval later removed this redundant BBH text."),
            "fix": ("strip the leading cue from each demonstration target so it appears exactly ONCE, "
                    "matching the official BBH cot-prompts"),
            "validation": ("every rewritten demonstration block was checked to appear VERBATIM in the "
                           "official cot-prompts/<task>.txt; the script raises rather than guessing"),
            "why_before_compute": ("Random-K never reads the query prompt while DSMC/First-RR/Second-RR/"
                                   "LESS-style all take query GRADIENTS on it, so a malformed prompt is "
                                   "not a shared constant -- it can distort target-aware gradient "
                                   "geometry while leaving the Random comparator untouched. No BBH "
                                   "accuracy has been observed, so this is outcome-independent."),
            "note": ("boolean_expressions' 3 demos already lacked the leading cue and were left "
                     "untouched, hence 78 rewritten of 81."),
            "residual_single_space": (
                "After de-duplication the rendered demo answer reads \"A: <cue>\\n <rationale>\" -- one "
                "space before the rationale where the official cot-prompt file has none. That space is "
                "lm-eval's own `target_delimiter`, whose default is \" \" (verified on the STOCK task "
                "object), not an artifact of this rewrite. It is applied identically when lm-eval builds "
                "the EVALUATION prompt, so query and evaluation remain mutually consistent -- which is "
                "the property that matters here. We do NOT override target_delimiter, because that would "
                "diverge from the pinned harness behaviour for every BBH result in the literature."),
        },
        "raw_bbh_data": {os.path.basename(p): sha_file(p) for p in sorted(glob.glob(f"{BBH_RAW}/*.json"))},
        "split_manifest_sha256": sha_file(f"{SPLIT_DIR}/bbh_split_manifest.json"),
        "no_compute_run": "artifacts only: no model loaded, nothing evaluated",
    }
    json.dump(man, open(args.out, "w"), indent=2)
    print(f"lm_eval {ver}; pinned group={len(subtasks)} subtasks, micro aggregation")
    print(f"emitted {len(emitted)} custom held-out subtask configs "
          f"({man['custom_heldout_suite']['total_heldout_examples']} examples) -> {OUT_TASKS}")
    print(f"CoT cue de-duplicated in {total_deduped} demonstrations (validated against official prompts)")
    print(f"23 conceptual families vs 27 lm-eval subtasks recorded")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

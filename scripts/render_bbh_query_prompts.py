#!/usr/bin/env python3
"""Render BBH query-gradient training prompts from the PINNED lm-eval CoT few-shot context
(choice_0809 item 6). Artifacts only — no gradients, no selection, no SFT.

Why this script exists
----------------------
In the MMLU experiments target gradients were built from 0-shot single-example prompts while the
evaluation was 5-shot, so the query gradient was measured in a different prompt distribution than the
one the model was scored in. For BBH we remove that mismatch by NOT re-implementing the prompt at all:
we call lm-eval's own `Task.fewshot_context()` on the *pinned* custom held-out task objects, so the
prompt prefix a query gradient is taken at is byte-identical to the prefix lm-eval will build at
evaluation time. Any future change to the pinned YAMLs propagates to both sides simultaneously; there
is no second copy of the prompt logic to drift.

What is and is NOT aligned (stated honestly, see prereg "Prompt alignment")
-------------------------------------------------------------------------
ALIGNED:     the prompt CONTEXT — task `description`, the 3 hard-coded CoT few-shot exemplars (which do
             contain full rationales), the `Q:`/`A:` delimiters, and the trailing
             "A: Let's think step by step.\n" cue.
NOT ALIGNED: the supervised CONTINUATION. BBH ships only a final answer per item (`(C)`, `14`, `Yes`),
             never a gold rationale for test items, so the loss supervises the bare final target while
             evaluation scores a generated rationale followed by "So the answer is X". Fabricating
             teacher rationales would inject a much larger confound (a second model's reasoning style)
             into the target signal, so we do not. This is prompt-context alignment, NOT
             reasoning-trace alignment.

Output: one llamafactory-style messages jsonl per draw, plus a manifest recording, for every rendered
example, the sha256 of the prompt and of the lm-eval-generated context so the parity audit
(`audit_bbh_prompt_parity.py`) can re-verify the two sides independently.
"""
import argparse, hashlib, json, os, warnings

warnings.filterwarnings("ignore")

ROOT = "/jizhicfs/karonhe/DataFlex_fa"
SPLIT_DIR = f"{ROOT}/data/bbh_external"
TASKS_DIR = f"{ROOT}/experiments/less_aligned/bbh_external_tasks"
OUT_DIR = f"{ROOT}/data/bbh_external/query_prompts"


def sha_str(s):
    return hashlib.sha256(s.encode()).hexdigest()


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load_pinned_tasks(subtasks):
    """Instantiate the pinned custom held-out lm-eval task objects (one per subtask).

    We use the CUSTOM held-out tasks rather than the stock ones only because they are the configs this
    experiment actually evaluates with; the parity audit separately proves the two produce identical
    prompts for the same doc, since the only difference between them is the dataset source.
    """
    from lm_eval.tasks import TaskManager
    tm = TaskManager(include_path=TASKS_DIR, verbosity="ERROR")
    names = [f"bbh_external_heldout_{t}" for t in subtasks]
    loaded = tm.load_task_or_group(names)
    out = {}
    for t in subtasks:
        obj = loaded[f"bbh_external_heldout_{t}"]
        out[t] = obj
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--draws", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--out_dir", default=OUT_DIR)
    ap.add_argument("--manifest",
                    default=f"{ROOT}/experiments/less_aligned/bbh_query_prompt_manifest.json")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # every subtask appearing in ANY draw must have a pinned task object
    draw_rows = {}
    needed = set()
    for d in args.draws:
        p = f"{SPLIT_DIR}/bbh_query_draw{d}.jsonl"
        rows = [json.loads(l) for l in open(p) if l.strip()]
        draw_rows[d] = rows
        needed |= {r["file_task"] for r in rows}
    tasks = load_pinned_tasks(sorted(needed))

    # the pinned num_fewshot must come from the config, never be hard-coded here
    nfs = {t: tasks[t].config.num_fewshot for t in tasks}
    if len(set(nfs.values())) != 1:
        raise RuntimeError(f"inconsistent num_fewshot across subtasks: {nfs}")
    num_fewshot = next(iter(nfs.values()))

    man = {"purpose": "BBH query-gradient prompts rendered from the pinned lm-eval CoT few-shot context",
           "renderer": "lm_eval Task.fewshot_context() on the pinned custom held-out task configs",
           "target_num_fewshot": num_fewshot,
           "evaluation_num_fewshot": num_fewshot,
           "alignment": {
               "aligned": ["task description", f"{num_fewshot} hard-coded CoT few-shot exemplars",
                           "Q:/A: delimiters", "trailing \"A: Let's think step by step.\" cue"],
               "not_aligned": ["supervised continuation: BBH provides only a final answer per item, so "
                               "the loss supervises the bare final target while evaluation scores a "
                               "generated rationale ending in \"So the answer is X\""],
               "claim": "prompt-context alignment, NOT reasoning-trace alignment"},
           "draws": {}, "no_compute_run": "artifacts only: no gradients, no selection, no SFT"}

    for d, rows in draw_rows.items():
        out_p = f"{args.out_dir}/bbh_query_draw{d}_prompts.jsonl"
        per_ex, comp27 = [], {}
        with open(out_p, "w") as f:
            for r in rows:
                t = tasks[r["file_task"]]
                doc = {"input": r["input"], "target": r["target"]}
                ctx = t.fewshot_context(doc, num_fewshot)
                # sanity: the rendered context must END with this doc's own question block, i.e. the
                # few-shot exemplars must be followed by the query itself and nothing else.
                if r["input"] not in ctx:
                    raise RuntimeError(f"{r['id']}: query input missing from rendered context")
                if not ctx.rstrip().endswith("Let's think step by step."):
                    raise RuntimeError(f"{r['id']}: context does not end with the CoT cue:\n"
                                       f"{ctx[-120:]!r}")
                rec = {"dataset": f"bbh_query_draw{d}", "id": r["id"],
                       "messages": [{"role": "user", "content": ctx},
                                    {"role": "assistant", "content": r["target"]}]}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                per_ex.append({"id": r["id"], "file_task": r["file_task"],
                               "prompt_sha256": sha_str(ctx), "target_sha256": sha_str(r["target"]),
                               "prompt_chars": len(ctx)})
                comp27[r["file_task"]] = comp27.get(r["file_task"], 0) + 1
        man["draws"][str(d)] = {
            "prompts_jsonl": os.path.relpath(out_p, ROOT), "file_sha256": sha_file(out_p),
            "n": len(rows), "subtask_composition_27": comp27,
            "source_draw_jsonl_sha256": sha_file(f"{SPLIT_DIR}/bbh_query_draw{d}.jsonl"),
            "per_example": per_ex}
        print(f"[render] draw{d}: {len(rows)} prompts over {len(comp27)} subtasks -> {out_p}")
        print(f"          mean prompt chars {sum(e['prompt_chars'] for e in per_ex)/len(per_ex):.0f}")

    json.dump(man, open(args.manifest, "w"), indent=2)
    print(f"\nwrote {args.manifest}")


if __name__ == "__main__":
    main()

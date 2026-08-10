#!/usr/bin/env python3
"""27-subtask PROMPT PARITY AUDIT (choice_0809 item 7). Artifacts only — no model, nothing evaluated.

Three independent gates, all byte-for-byte, all 27 lm-eval subtasks:

  GATE A  custom held-out task  ==  stock pinned `bbh_cot_fewshot` task, for the SAME doc, MODULO ONE
          DECLARED EXCEPTION. We hold the doc fixed and compare the full generation request string plus
          the config fields that govern scoring (description, doc_to_text, num_fewshot,
          generation_kwargs, until stops, filters, metrics, output type, doc_to_target). A single canary
          doc suffices per subtask because the prompt is a pure function of (config, doc).

          THE DECLARED EXCEPTION (code_review_0810_3): stock v0.4.5 renders the CoT trigger TWICE per
          demonstration ("A: Let's think step by step.\n Let's think step by step.\n..."), because
          `doc_to_text` already ends with the cue and each demo `target` restates it. Upstream lm-eval
          later removed that redundant text. Our custom suite restores the OFFICIAL single-cue form, so
          demanding byte-equality with stock would now demand reproducing a known bug. Therefore:
            * the stock rendering is normalized by collapsing the duplicated cue, and the remainder must
              match byte for byte -- so any OTHER difference is still caught;
            * `fewshot_config` is validated against the OFFICIAL BBH cot-prompts file instead of stock:
              every demonstration must carry the cue exactly once AND appear verbatim in
              cot-prompts/<task>.txt.
          The official file is the higher authority here, and it is checked rather than assumed.

  GATE B  query-gradient prompt (from render_bbh_query_prompts.py, i.e. the file the SFT target dataset
          is actually built from)  ==  the evaluation prompt prefix lm-eval builds for that same query
          example. Verified against the STOCK task object, so gate B does not inherit any custom-config
          assumption from gate A.
          SCOPE, stated honestly: both sides ultimately call lm-eval's own `fewshot_context()`, so this
          is (i) an integrity check that the stored artifact was not corrupted after rendering and
          (ii) a stock-vs-custom agreement check on those docs. It does NOT independently re-derive
          lm-eval's prompt-construction logic. It is FAIL-CLOSED: a missing prompt file or a subtask with
          zero checked prompts is a FAILURE, not a skip (an earlier version `continue`d, which made the
          gate silently vacuous because `all([]) is True`).

  GATE C  the held-out data actually loaded by each custom task is exactly the intended split: right
          example count, ids/rows matching the split jsonl, and disjoint from the query reservoir.

  GATE D  DISCLOSURE, not a pass/fail: quantifies the one place "byte-identical" stops holding. The
          training-time sequence carries the llamafactory llama2 `<s>[INST] ... [/INST]` wrapper, while
          lm-eval (invoked without --apply_chat_template) sends the context verbatim. Identical to the
          MMLU arm, so not a new confound — but recorded rather than left implicit.

Also re-verifies the documented asymmetry rather than glossing it: the supervised continuation is the
bare BBH final target and is NOT the evaluation's CoT trajectory. That is reported as a KNOWN,
INTENDED difference (BBH has no gold rationales for test items) — not silently ignored.

`--tamper_check` runs a built-in negative control: it perturbs one subtask's description in memory and
confirms gate A actually fails, so a green report is demonstrably not vacuous.
"""
import argparse, difflib, hashlib, json, os, warnings

warnings.filterwarnings("ignore")

ROOT = "/jizhicfs/karonhe/DataFlex_fa"
SPLIT_DIR = f"{ROOT}/data/bbh_external"
TASKS_DIR = f"{ROOT}/experiments/less_aligned/bbh_external_tasks"
PROMPTS_DIR = f"{SPLIT_DIR}/query_prompts"

# config fields that must match between stock and custom (i.e. everything except the data source)
COMPARED_FIELDS = ["description", "doc_to_text", "doc_to_target", "num_fewshot",
                   "generation_kwargs", "filter_list", "metric_list", "output_type", "test_split"]
# `fewshot_config` is compared SEPARATELY, because our custom suite deliberately differs from stock
# v0.4.5 there: the pinned stock config duplicates the CoT trigger inside each demonstration, and we
# restore the OFFICIAL BBH single-cue form. So the correct reference for the demonstrations is the
# official BBH cot-prompts file, not the buggy stock YAML. Everything else must still match stock byte
# for byte. See bbh_eval_pin_manifest.json -> cot_cue_deduplication.
CUE = "Let's think step by step."
BBH_COT_PROMPTS = "/jizhicfs/karonhe/less_data_zip/data/eval/bbh/cot-prompts"
# Fields allowed to differ. The data-source fields differ because that is the intended change;
# `fewshot_config` differs because of the DECLARED CoT-dedup exception -- and it is NOT simply waived:
# it is validated separately and more strictly, against the official BBH cot-prompts (see gate A).
EXEMPT_FIELDS = ["task", "dataset_path", "dataset_name", "dataset_kwargs", "include", "metadata",
                 "fewshot_config"]


def sha(s):
    return hashlib.sha256(s.encode()).hexdigest()


def request_string(task, doc, num_fewshot):
    """The exact context lm-eval would send to the model for `doc` under `task`."""
    return task.fewshot_context(doc, num_fewshot)


def norm(v):
    """Canonical JSON so dict ordering never masquerades as a difference."""
    return json.dumps(v, sort_keys=True, ensure_ascii=False, default=str)


def gate_d_chat_wrapper(report, sample_ids=3):
    """GATE D: quantify the ONE place where 'byte-identical' stops being true.

    The rendered prompt is stored in a llamafactory sharegpt `messages` record and consumed with
    `template: llama2`, whose user slot is `{bos_token}[INST] {{content}} [/INST]`. lm-eval, invoked
    WITHOUT --apply_chat_template, sends the context verbatim. So the token sequence a query gradient is
    taken on is the evaluation context wrapped in `<s>[INST] ... [/INST]`, not the bare context.

    This is consistent with the MMLU arm (same wrapper there), so it is not a new confound and not a
    launch blocker — but it is a real limit on the phrase "byte-identical", so it is measured and
    recorded rather than left implicit. Gates A/B compare the STORED STRING; gate D reports the delta
    between that string and the actual training-time sequence.
    """
    try:
        from llamafactory.data.template import TEMPLATES
        slots = [str(s) for s in TEMPLATES["llama2"].format_user.slots]
    except Exception as e:                       # llamafactory absent -> report, never silently pass
        report["gate_d_chat_template_wrapper"] = {"resolved": False, "error": repr(e)}
        return
    example = None
    p = f"{PROMPTS_DIR}/bbh_query_draw0_prompts.jsonl"
    if os.path.exists(p):
        rec = json.loads(open(p).readline())
        ctx = rec["messages"][0]["content"]
        example = {"id": rec["id"],
                   "stored_prompt_sha256": sha(ctx),
                   "training_sequence_sha256": sha(f"<s>[INST] {ctx} [/INST]"),
                   "identical": False}
    report["gate_d_chat_template_wrapper"] = {
        "resolved": True,
        "llamafactory_llama2_format_user": slots,
        "training_time_wrapper": "<s>[INST] {context} [/INST]",
        "lm_eval_wrapper": "none (invoked without --apply_chat_template; context sent verbatim)",
        "stored_string_is_byte_identical_to_eval_context": True,
        "tokenized_training_sequence_is_byte_identical": False,
        "example": example,
        "verdict": ("Prompt CONTEXT is byte-identical (gates A/B). The training-time token sequence "
                    "additionally carries the llama2 [INST] wrapper, exactly as in the MMLU arm, so the "
                    "correct claim is 'byte-identical up to the llama2 chat wrapper applied at "
                    "gradient-extraction time'. Not a new confound; disclosed, not silently assumed."),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=f"{ROOT}/experiments/less_aligned/bbh_prompt_parity_audit.json")
    ap.add_argument("--per_subtask_queries", type=int, default=1,
                    help="min query examples per subtask required in gate B; fewer is a FAILURE, so the "
                         "gate cannot pass vacuously")
    ap.add_argument("--tamper_check", action="store_true",
                    help="negative control: perturb one subtask's description IN MEMORY and confirm "
                         "gate A actually fails, proving a green report is not vacuous")
    args = ap.parse_args()

    from lm_eval.tasks import TaskManager

    # stock tasks: default search path. custom tasks: our include_path.
    tm_stock = TaskManager(verbosity="ERROR")
    tm_cust = TaskManager(include_path=TASKS_DIR, verbosity="ERROR")

    split = json.load(open(f"{SPLIT_DIR}/bbh_split_manifest.json"))
    subtasks = sorted(split["per_task_split"])
    assert len(subtasks) == 27, f"expected 27 subtasks, found {len(subtasks)}"

    heldout_by_task, reservoir_ids = {}, set()
    for r in (json.loads(l) for l in open(f"{SPLIT_DIR}/bbh_eval_heldout.jsonl") if l.strip()):
        heldout_by_task.setdefault(r["file_task"], []).append(r)
    for r in (json.loads(l) for l in open(f"{SPLIT_DIR}/bbh_query_reservoir.jsonl") if l.strip()):
        reservoir_ids.add(r["id"])

    report = {"gate_a_config_and_prompt_parity": {}, "gate_b_query_prompt_parity": {},
              "gate_c_heldout_data": {}, "failures": []}

    def fail(gate, subtask, detail):
        report["failures"].append({"gate": gate, "subtask": subtask, "detail": detail})

    # ---------------- GATE A: custom == stock, same doc ----------------
    for t in subtasks:
        stock = tm_stock.load_task_or_group([f"bbh_cot_fewshot_{t}"])[f"bbh_cot_fewshot_{t}"]
        cust = tm_cust.load_task_or_group([f"bbh_external_heldout_{t}"])[f"bbh_external_heldout_{t}"]
        sdump, cdump = stock.config.to_dict(), cust.config.to_dict()

        diffs = {f: {"stock": sdump.get(f), "custom": cdump.get(f)}
                 for f in COMPARED_FIELDS if norm(sdump.get(f)) != norm(cdump.get(f))}

        # ---- demonstrations: validate against the OFFICIAL BBH prompt, not the buggy stock YAML ----
        off_p = f"{BBH_COT_PROMPTS}/{t}.txt"
        off_body = None
        if os.path.exists(off_p):
            raw_off = open(off_p).read()
            off_body = raw_off.split("-----\n", 1)[1] if "-----\n" in raw_off else raw_off
        demo_checks = []
        for di, smp in enumerate((cdump.get("fewshot_config") or {}).get("samples", [])):
            tgt = smp.get("target", "")
            cue_once = not tgt.lstrip().startswith(CUE)      # doc_to_text already supplies the cue
            block_nl = "Q: " + smp["input"] + "\nA: " + CUE + "\n" + tgt
            block_sp = "Q: " + smp["input"] + "\nA: " + CUE + " " + tgt
            in_official = off_body is not None and (block_nl in off_body or block_sp in off_body)
            demo_checks.append({"idx": di, "cue_appears_once": cue_once,
                                "matches_official_cot_prompt": in_official})
            if not (cue_once and in_official):
                fail("A", t, {"demo": di, "cue_appears_once": cue_once,
                              "matches_official_cot_prompt": in_official,
                              "why": "demonstration must carry the cue exactly once AND match the "
                                     "official BBH cot-prompt text"})
        # everything not compared and not exempt would be an unnoticed behavioural difference
        unexpected = sorted((set(sdump) | set(cdump)) - set(COMPARED_FIELDS) - set(EXEMPT_FIELDS))
        unexpected = [f for f in unexpected if norm(sdump.get(f)) != norm(cdump.get(f))]

        nf_s, nf_c = stock.config.num_fewshot, cust.config.num_fewshot
        # `fewshot_config` is exempt from the raw stock comparison because of the CoT-dedup exception,
        # so its NON-demonstration keys must be checked explicitly -- otherwise they are unguarded.
        # `sampler` matters most: flipping first_n -> default makes lm-eval use an UNSEEDED
        # random.Random(), so demo order becomes nondeterministic and gate A would only catch it
        # probabilistically.
        s_fs = (sdump.get("fewshot_config") or {})
        c_fs = (cdump.get("fewshot_config") or {})
        fs_meta_diffs = {k: {"stock": s_fs.get(k), "custom": c_fs.get(k)}
                         for k in set(s_fs) | set(c_fs)
                         if k != "samples" and norm(s_fs.get(k)) != norm(c_fs.get(k))}
        n_demos_ok = len(c_fs.get("samples", [])) == len(s_fs.get("samples", [])) == nf_c
        if fs_meta_diffs:
            fail("A", t, {"fewshot_config_non_sample_diffs": fs_meta_diffs,
                          "why": "sampler/config keys must match stock exactly; only `samples` may "
                                 "differ, and only by the declared CoT de-duplication"})
        if not n_demos_ok:
            fail("A", t, {"n_demos_stock": len(s_fs.get("samples", [])),
                          "n_demos_custom": len(c_fs.get("samples", [])), "num_fewshot": nf_c,
                          "why": "demonstration count must equal stock and num_fewshot"})
        demos_ok = (all(d["cue_appears_once"] and d["matches_official_cot_prompt"] for d in demo_checks)
                    and not fs_meta_diffs and n_demos_ok)
        # canary doc: a fixed synthetic doc isolates the PROMPT from any data difference
        canary = {"input": "__PARITY_CANARY_INPUT__", "target": "__PARITY_CANARY_TARGET__"}
        ps, pc = request_string(stock, canary, nf_s), request_string(cust, canary, nf_c)
        # The ONLY intended prompt difference is the removal of the duplicated CoT trigger. Normalize
        # that specific defect out of the STOCK rendering; anything else that still differs is a real
        # finding. `dedup_stock` collapses "A: <cue>\n <cue>" / "A: <cue>\n<cue>" back to one cue.
        # Stock renders "A: <cue>\n <cue>\n<rationale>" (target_delimiter " " then the restated cue).
        # Collapsing to "A: <cue>\n <rationale>" removes ONLY the duplicate cue, keeping lm-eval's own
        # target_delimiter space -- which is exactly what the corrected custom config renders.
        dedup_stock = (ps.replace(f"A: {CUE}\n {CUE}\n", f"A: {CUE}\n ")
                         .replace(f"A: {CUE}\n {CUE} ", f"A: {CUE}\n ")
                         .replace(f"A: {CUE}\n{CUE}\n", f"A: {CUE}\n"))
        prompt_equal_modulo_dedup = dedup_stock == pc
        ok = prompt_equal_modulo_dedup and demos_ok and not diffs and not unexpected
        report["gate_a_config_and_prompt_parity"][t] = {
            "identical_prompt_raw": ps == pc,
            "identical_prompt_modulo_cot_dedup": prompt_equal_modulo_dedup,
            "demonstrations_match_official_cot_prompt": demos_ok,
            "fewshot_config_sampler": c_fs.get("sampler"),
            "fewshot_config_non_sample_diffs": fs_meta_diffs,
            "n_demonstrations": len(c_fs.get("samples", [])),
            "demonstration_checks": demo_checks,
            "prompt_sha256": sha(pc),
            "num_fewshot_stock": nf_s, "num_fewshot_custom": nf_c,
            "compared_field_diffs": diffs, "unexpected_field_diffs": unexpected,
            "fewshot_samples_sha256": sha(norm(cdump.get("fewshot_config"))),
            "pass": ok}
        if not ok:
            d = "" if prompt_equal_modulo_dedup else "\n".join(list(difflib.unified_diff(
                dedup_stock.splitlines(), pc.splitlines(), "stock_dedup", "custom", lineterm=""))[:40])
            fail("A", t, {"prompt_diff": d, "config_diffs": diffs, "unexpected": unexpected})

        # ---------------- GATE C: the loaded data is the intended split ----------------
        docs = list(cust.test_docs())
        want = heldout_by_task[t]
        n_ok = len(docs) == len(want) == split["per_task_split"][t]["n_heldout"]
        rows_ok = all(docs[i]["input"] == want[i]["input"] and docs[i]["target"] == want[i]["target"]
                      for i in range(min(len(docs), len(want))))
        leak = sum(1 for w in want if w["id"] in reservoir_ids)
        cok = n_ok and rows_ok and leak == 0
        report["gate_c_heldout_data"][t] = {
            "n_loaded": len(docs), "n_expected": split["per_task_split"][t]["n_heldout"],
            "rows_match_split_jsonl": rows_ok, "n_ids_also_in_query_reservoir": leak, "pass": cok}
        if not cok:
            fail("C", t, {"n_loaded": len(docs), "n_expected": split["per_task_split"][t]["n_heldout"],
                          "rows_match": rows_ok, "leak": leak})

        # ---------------- GATE B: rendered query prompt == stock eval prefix ----------------
        # compared against STOCK, so gate B is independent of the custom config.
        checked = []
        for d in (0, 1, 2):
            mp = f"{PROMPTS_DIR}/bbh_query_draw{d}_prompts.jsonl"
            if not os.path.exists(mp):
                # FAIL-CLOSED. A missing prompt file previously `continue`d, which silently turned the
                # most load-bearing gate into a no-op (all([]) is True). Never skip.
                fail("B", t, {"missing_prompts_file": mp,
                              "why": "render_bbh_query_prompts.py must be run before this audit"})
                continue
            src = {r["id"]: r for r in (json.loads(l)
                                        for l in open(f"{SPLIT_DIR}/bbh_query_draw{d}.jsonl") if l.strip())}
            for rec in (json.loads(l) for l in open(mp) if l.strip()):
                q = src[rec["id"]]
                if q["file_task"] != t:
                    continue
                rendered = rec["messages"][0]["content"]
                expected_raw = request_string(stock, {"input": q["input"], "target": q["target"]}, nf_s)
                # normalize the SAME known defect out of the stock side (see gate A)
                expected = (expected_raw.replace(f"A: {CUE}\n {CUE}\n", f"A: {CUE}\n ")
                                        .replace(f"A: {CUE}\n {CUE} ", f"A: {CUE}\n ")
                                        .replace(f"A: {CUE}\n{CUE}\n", f"A: {CUE}\n"))
                same = rendered == expected
                checked.append({"draw": d, "id": rec["id"], "identical": same,
                                "prompt_sha256": sha(rendered),
                                "supervised_continuation": rec["messages"][1]["content"]})
                if not same:
                    fail("B", t, {"id": rec["id"], "diff": "\n".join(list(difflib.unified_diff(
                        expected.splitlines(), rendered.splitlines(),
                        "lm_eval_stock", "rendered", lineterm=""))[:40])})
        # every subtask must contribute at least --per_subtask_queries checked prompts, else the gate
        # is vacuous for it (all([]) is True). This is the check the CLI flag always promised.
        if len(checked) < args.per_subtask_queries:
            fail("B", t, {"n_checked": len(checked), "required": args.per_subtask_queries,
                          "why": "gate B would be vacuous for this subtask"})
        report["gate_b_query_prompt_parity"][t] = {
            "n_query_examples_checked": len(checked),
            "n_distinct_ids": len({c["id"] for c in checked}),
            "all_identical": bool(checked) and all(c["identical"] for c in checked),
            "meets_min_coverage": len(checked) >= args.per_subtask_queries,
            "examples": checked}
        print(f"[parity] {t:45s} A={'ok' if ok else 'FAIL'} "
              f"C={'ok' if cok else 'FAIL'} B={len(checked)} queries "
              f"{'ok' if all(c['identical'] for c in checked) else 'FAIL'}")

    gate_d_chat_wrapper(report)

    # ---------------- negative control (in memory; never writes to disk) ----------------
    if args.tamper_check:
        t0 = subtasks[0]
        stock = tm_stock.load_task_or_group([f"bbh_cot_fewshot_{t0}"])[f"bbh_cot_fewshot_{t0}"]
        cust = tm_cust.load_task_or_group([f"bbh_external_heldout_{t0}"])[f"bbh_external_heldout_{t0}"]
        canary = {"input": "__PARITY_CANARY_INPUT__", "target": "__PARITY_CANARY_TARGET__"}
        clean_equal = request_string(stock, canary, stock.config.num_fewshot) == \
            request_string(cust, canary, cust.config.num_fewshot)
        cust.config.description = (cust.config.description or "") + "__TAMPER__"
        tampered_equal = request_string(stock, canary, stock.config.num_fewshot) == \
            request_string(cust, canary, cust.config.num_fewshot)
        detected = clean_equal and not tampered_equal
        report["negative_control"] = {
            "subtask": t0, "method": "in-memory perturbation of config.description",
            "clean_prompts_equal": clean_equal, "tampered_prompts_equal": tampered_equal,
            "tamper_detected": detected,
            "verdict": ("gate A demonstrably distinguishes a tampered config from a clean one, so a "
                        "green gate A is informative, not vacuous" if detected else
                        "NEGATIVE CONTROL FAILED — gate A did not detect a tampered description"),
            "note": "purely in memory; no file on disk was modified",
        }
        print(f"[negctl] {t0}: clean_equal={clean_equal} tampered_equal={tampered_equal} "
              f"-> tamper {'DETECTED' if detected else 'NOT DETECTED'}")
        if not detected:
            fail("A", t0, {"negative_control": "gate A failed to detect an in-memory tamper"})

    nb = sum(v["n_query_examples_checked"] for v in report["gate_b_query_prompt_parity"].values())
    covered = sum(1 for v in report["gate_b_query_prompt_parity"].values()
                  if v["n_query_examples_checked"] > 0)
    distinct = sum(v["n_distinct_ids"] for v in report["gate_b_query_prompt_parity"].values())
    # non-vacuity is part of the verdict, not a footnote: a gate that checked nothing is NOT a pass.
    gate_b_pass = (nb > 0 and covered == len(subtasks)
                   and all(v["all_identical"] and v["meets_min_coverage"]
                           for v in report["gate_b_query_prompt_parity"].values()))
    report["summary"] = {
        "n_subtasks": len(subtasks),
        "gate_a_pass": all(v["pass"] for v in report["gate_a_config_and_prompt_parity"].values()),
        "gate_b_pass": gate_b_pass,
        "gate_b_n_query_prompts_checked": nb,
        "gate_b_n_distinct_query_examples": distinct,
        "gate_b_subtasks_with_queries": covered,
        "gate_b_subtasks_without_queries": len(subtasks) - covered,
        "gate_b_min_per_subtask_required": args.per_subtask_queries,
        "gate_b_row_vs_distinct_note": (
            f"{nb} prompt ROWS were checked over {distinct} distinct query examples; ids recurring "
            f"across draws are counted once per draw, consistent with the reported draw overlap."),
        "gate_c_pass": all(v["pass"] for v in report["gate_c_heldout_data"].values()),
        "total_heldout_loaded": sum(v["n_loaded"] for v in report["gate_c_heldout_data"].values()),
        "n_failures": len(report["failures"]),
        "known_intended_differences": [
            ("supervised continuation: query gradients supervise the bare BBH final target, while "
             "evaluation scores a generated CoT trajectory ending in 'So the answer is X'. BBH ships no "
             "gold rationales for test items, so this cannot be removed without fabricating teacher "
             "rationales — a strictly larger confound."),
            ("chat wrapper: the training-time sequence carries the llama2 '<s>[INST] ... [/INST]' "
             "wrapper while lm-eval sends the context verbatim. Same in the MMLU arm, so not a new "
             "confound. Quantified in gate D."),
        ],
        "scope_of_gate_b": (
            "Gate B verifies that the STORED prompt artifact equals what lm-eval builds for the same "
            "doc, using the STOCK task object (so it does not inherit gate A's assumption). Both sides "
            "call lm-eval's own fewshot_context(), so gate B is an integrity check on the artifact and a "
            "stock-vs-custom agreement check — it does NOT independently re-derive lm-eval's prompt "
            "construction logic, and is not claimed to."),
        "no_compute_run": "artifacts only: no model loaded, nothing evaluated"}
    report["summary"]["ALL_GATES_PASS"] = (report["summary"]["gate_a_pass"]
                                          and report["summary"]["gate_b_pass"]
                                          and report["summary"]["gate_c_pass"])
    json.dump(report, open(args.out, "w"), indent=2)
    s = report["summary"]
    print(f"\ngate A (custom==stock prompt+config, 27 subtasks): {'PASS' if s['gate_a_pass'] else 'FAIL'}")
    print(f"gate B (query prompt==eval prefix, {nb} prompts over {covered} subtasks): "
          f"{'PASS' if s['gate_b_pass'] else 'FAIL'}")
    print(f"gate C (held-out data == split, {s['total_heldout_loaded']} examples): "
          f"{'PASS' if s['gate_c_pass'] else 'FAIL'}")
    gd = report.get("gate_d_chat_template_wrapper", {})
    print(f"gate D (chat-wrapper delta, disclosure only): "
          f"{'resolved' if gd.get('resolved') else 'UNRESOLVED'} — stored string byte-identical, "
          f"training sequence adds {gd.get('training_time_wrapper', '?')}")
    print(f"ALL_GATES_PASS = {s['ALL_GATES_PASS']}   failures={s['n_failures']}")
    print(f"wrote {args.out}")
    return 0 if s["ALL_GATES_PASS"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

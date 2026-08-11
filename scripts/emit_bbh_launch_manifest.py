#!/usr/bin/env python3
"""Emit the BBH external-validation run plan + launch manifest (choice_0809 item 8).
Artifacts only — no gradients, no selection, no SFT.

This is the single launch record for the round: it pins every upstream artifact by sha256, restates the
frozen design, and asserts the invariant that makes the crossed design clean, namely that the 15 subsets
(3 draws x 5 methods) are functions of the draw only and are therefore reused BIT-IDENTICALLY by both
SFT seeds. `--verify` re-checks every referenced hash and every gate verdict, and exits non-zero if any
artifact has drifted, so the manifest cannot silently describe a stale tree.
"""
import argparse, hashlib, json, os, subprocess

ROOT = "/jizhicfs/karonhe/DataFlex_fa"
EXP = f"{ROOT}/experiments/less_aligned"
SPLIT_DIR = f"{ROOT}/data/bbh_external"
SAVES = "/jizhicfs/karonhe/dataflex_saves"

DRAWS = [0, 1, 2]
SFT_SEEDS = [42, 1]
# PRIMARY comparison set (unchanged by advice_0810_2)
METHODS = ["dsmc", "second_rr", "first_rr", "less", "randk"]
# SECONDARY sensitivity control, added pre-accuracy: matches DSMC's post-template SEQUENCE-length
# distribution at fixed K. Not a sixth primary selector; may never alter the primary comparisons.
# REPLACED the sequence-only control with the JOINT (sequence, label) matched control per
# code_review_0811: the sequence-only version still carried 7.14x DSMC's loss-bearing label positions, so
# it could only rule out coarse input length, not the response-length / classification-vs-generation
# format axis. Same 6-adapter budget -> total stays 36, not 42.
SECONDARY = ["randk_seqlabelmatch"]
LENMATCH_SEED_BASE = 7000
BUDGET = 2707
RANDOM_SEED_BASE, RR_PERM_SEED_BASE = 5000, 6000

# hashed artifacts: path -> why it is pinned
PINNED = {
    f"{SPLIT_DIR}/bbh_split_manifest.json": "20/80 split + 3 draws + frozen selection seeds",
    f"{SPLIT_DIR}/bbh_eval_heldout.jsonl": "the 5,209 held-out evaluation examples",
    f"{SPLIT_DIR}/bbh_query_reservoir.jsonl": "the 1,302 query reservoir (disjoint from eval)",
    f"{EXP}/bbh_lmeval_pin.json": "lm-eval version + upstream YAML/data hashes + Python runtime code "
                                  "(api/task.py, evaluator.py, models/huggingface.py, filters) + "
                                  "tokenizer + pip environment",
    f"{EXP}/bbh_eval_pin_manifest.json": "frozen custom held-out suite (27 subtask configs)",
    f"{EXP}/bbh_query_prompt_manifest.json": "query prompts rendered from lm-eval fewshot_context()",
    f"{EXP}/bbh_prompt_parity_audit.json": "27-subtask byte-for-byte parity gates A/B/C/D",
    f"{EXP}/bbh_external_tasks/_bbh_external_heldout.yaml": "custom group config (micro aggregation)",
    f"{EXP}/results_summary/contamination_global_lexical_bbh_heldout.json":
        "pool-wide lexical screen vs the held-out split",
    f"{EXP}/bbh_execution_contract.json": "P0-1 frozen feature/selector contract (Adam cand / SGD target)",
    f"{EXP}/bbh_token_truncation_audit.json": "P0-2 execution-level token/truncation gate",
    f"{EXP}/bbh_fewshot_leakage_audit.json": "81 CoT demos vs eval/query populations",
}


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def suite_hashes():
    """Hash the 27 custom subtask YAMLs and their 27 held-out data files INDIVIDUALLY.

    Hashing only the group YAML and `bbh_eval_pin_manifest.json` is not enough: the group file is just a
    list of task names, and the pin manifest's own hash does not change when the files it describes
    change. So editing a subtask's prompt or deleting rows from a held-out jsonl would leave a
    coarse-grained check green while both the evaluation prompt AND the evaluation set were corrupted.
    These 54 hashes are the frozen evaluation, so they are pinned individually.
    """
    pin = json.load(open(f"{EXP}/bbh_eval_pin_manifest.json"))
    out = {}
    for task, v in sorted(pin["custom_heldout_suite"]["subtasks"].items()):
        cfg = f"{ROOT}/{v['config']}"
        data = f"{EXP}/bbh_external_tasks/data/{task}_heldout.jsonl"
        out[f"subtask_config:{task}"] = {"sha256": sha_file(cfg),
                                         "expected_in_pin_manifest": v["config_sha256"],
                                         "why": "frozen evaluation prompt for this subtask"}
        out[f"subtask_data:{task}"] = {"sha256": sha_file(data),
                                       "expected_in_pin_manifest": v["data_sha256"],
                                       "why": f"the {v['n_heldout']} held-out examples for this subtask"}
    return out


def suite_drift():
    """Any subtask config/data file whose hash disagrees with the pin manifest."""
    return sorted(k for k, v in suite_hashes().items() if v["sha256"] != v["expected_in_pin_manifest"])


def git_commit():
    try:
        return subprocess.check_output(["git", "-C", ROOT, "rev-parse", "HEAD"],
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def diff_vs(ref):
    """Files differing between `ref` and the current HEAD (empty list if none/unavailable)."""
    try:
        out = subprocess.check_output(["git", "-C", ROOT, "diff", "--name-only", f"{ref}..HEAD"],
                                      stderr=subprocess.DEVNULL).decode().strip()
        return [l for l in out.split("\n") if l]
    except Exception:
        return None


def git_clean():
    """True iff the working tree has no uncommitted changes."""
    try:
        out = subprocess.check_output(["git", "-C", ROOT, "status", "--porcelain"],
                                      stderr=subprocess.DEVNULL).decode().strip()
        return out == ""
    except Exception:
        return None


def build_run_plan():
    """36 cells = 3 draws x (5 primary + 1 secondary control) x 2 SFT seeds, over 18 subsets.

    Subsets are frozen BY RULE here (seeds and matching targets fixed); the draw1/draw2 artifacts are
    generated only once those draws' target gradients exist.
    """
    cells = []
    for d in DRAWS:
        for m in METHODS + SECONDARY:
            subset_id = f"bbhx_draw{d}_{m}"          # the FROZEN subset: no seed in the name, by design
            for s in SFT_SEEDS:
                adapter_id = f"{subset_id}_seed{s}"
                cells.append({
                    "draw": d, "method": m, "train_seed": s,
                    "subset_id": subset_id,
                    "adapter_id": adapter_id,
                    "budget": BUDGET,
                    "dataset_key": f"bbhx_draw{d}_{m}",
                    "selection_dir": f"{SAVES}/selbbhx_draw{d}_{m}",
                    "sft_out": f"{SAVES}/sft_results/{adapter_id}",
                    "eval_out": f"{SAVES}/eval_results/bbh_external/{adapter_id}",
                    "random_k_seed": RANDOM_SEED_BASE + d if m == "randk" else None,
                    "rr_perm_seed": RR_PERM_SEED_BASE + d if m in ("first_rr", "second_rr") else None,
                    "lenmatch_seed": LENMATCH_SEED_BASE + d if m == "randk_seqlabelmatch" else None,
                    "role": "secondary_sensitivity_control" if m in SECONDARY else "primary",
                })
    return cells


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_plan", default=f"{EXP}/bbh_external_run_plan.json")
    ap.add_argument("--out", default=f"{EXP}/bbh_external_launch_manifest.json")
    ap.add_argument("--verify", action="store_true",
                    help="re-check every pinned hash and gate verdict against an existing manifest")
    ap.add_argument("--receipt", action="store_true",
                    help="emit the canary launch receipt at the CURRENT head; refuses on a dirty tree "
                         "or with any open blocker, so a receipt always points at a real clean commit")
    ap.add_argument("--receipt_out", default=f"{EXP}/bbh_canary_launch_receipt.json")
    args = ap.parse_args()

    missing = [p for p in PINNED if not os.path.exists(p)]
    if missing:
        raise SystemExit("missing pinned artifacts:\n  " + "\n  ".join(missing))

    hashes = {os.path.relpath(p, ROOT): {"sha256": sha_file(p), "why": why}
              for p, why in sorted(PINNED.items())}
    hashes.update(suite_hashes())      # + the 27 subtask configs and 27 held-out data files

    # ---- gate verdicts are READ from the artifacts, never re-asserted by hand ----
    parity = json.load(open(f"{EXP}/bbh_prompt_parity_audit.json"))["summary"]
    contam = json.load(open(f"{EXP}/results_summary/contamination_global_lexical_bbh_heldout.json"))
    split = json.load(open(f"{SPLIT_DIR}/bbh_split_manifest.json"))
    prompts = json.load(open(f"{EXP}/bbh_query_prompt_manifest.json"))
    pin = json.load(open(f"{EXP}/bbh_eval_pin_manifest.json"))

    trunc = json.load(open(f"{EXP}/bbh_token_truncation_audit.json"))
    leak = json.load(open(f"{EXP}/bbh_fewshot_leakage_audit.json"))
    contract = json.load(open(f"{EXP}/bbh_execution_contract.json"))

    n_prompts = parity["gate_b_n_query_prompts_checked"]
    n_covered = parity["gate_b_subtasks_with_queries"]
    n_subtasks = pin["task_accounting"]["n_lm_eval_subtasks"]
    drift = suite_drift()

    gates = {
        "parity_all_gates_pass": parity["ALL_GATES_PASS"],
        "parity_gate_a_27_subtasks": parity["gate_a_pass"],
        "parity_gate_b_query_prompts": {"pass": parity["gate_b_pass"], "n_prompts": n_prompts,
                                        "subtasks_covered": n_covered,
                                        "n_distinct_examples":
                                            parity.get("gate_b_n_distinct_query_examples")},
        "parity_gate_c_heldout_rows": {"pass": parity["gate_c_pass"],
                                       "n_examples": parity["total_heldout_loaded"]},
        "contamination_strong_J0.5": contam["pool_strong_count"],
        "contamination_weak_J0.3": contam["pool_weak_count"],
        "contamination_lsh_recall_nominal": contam.get("lsh_recall_nominal") or contam.get("lsh_recall"),
        "query_eval_disjoint": split["query_eval_disjoint"],
        "custom_suite_hash_drift": drift or None,
        "token_truncation_verdict": trunc["verdict"],
        "token_truncation_n_affected": trunc["n_materially_truncated"],
        "token_truncation_detail": trunc.get("truncated_by_subtask") or None,
        "fewshot_leakage_verdict": leak["verdict"],
        "fewshot_leakage_exact": leak["n_exact_duplicates"],
        "fewshot_leakage_needing_human_clearance": leak.get("n_needing_human_clearance"),
        "fewshot_leakage_answer_flip_pairs": len(leak.get("answer_flip_pairs") or []),
        "audit_coverage": {"truncation_records": trunc.get("n_records"),
                           "leakage_demonstrations": leak.get("n_demonstrations")},
        "execution_contract_candidate_cache_verified": contract["candidate_cache"].get("sha256_matches"),
        "warmup_adapter_sha256_matches": contract["warmup_checkpoint"].get("adapter_sha256_matches"),
        "warmup_optimizer_sha256_matches": contract["warmup_checkpoint"].get("optimizer_sha256_matches"),
        "llamafactory_preprocessing_pinned": sum(
            1 for v in (contract.get("llamafactory_preprocessing_pin", {}).get("files") or {}).values()
            if v),
        "target_grad_cutoff_len": contract["target_gradient_extraction"]["cutoff_len"],
        "sft_cutoff_len": contract["target_gradient_extraction"]["sft_cutoff_len"],
        "execution_contract_gradient_types": {
            "candidate": contract["frozen_feature_contract"]["gradient_type_candidate"],
            "target": contract["frozen_feature_contract"]["gradient_type_target"]},
    }
    blockers = []
    if not parity["ALL_GATES_PASS"]:
        blockers.append("prompt parity audit did not pass")
    # non-vacuity: a gate that checked nothing must never read as a pass
    if not n_prompts or n_covered != n_subtasks:
        blockers.append(f"gate B coverage vacuous: {n_prompts} prompts over {n_covered}/{n_subtasks} "
                        f"subtasks")
    if parity["total_heldout_loaded"] != pin["custom_heldout_suite"]["total_heldout_examples"]:
        blockers.append("gate C example count disagrees with the pinned suite")
    if drift:
        blockers.append(f"custom suite drifted from the pin manifest: {drift}")
    # P0-2: the execution-level gate. String parity is NOT sufficient to launch on.
    if trunc["verdict"] != "PASS":
        blockers.append(f"token/truncation audit verdict={trunc['verdict']}: "
                        f"{trunc['n_materially_truncated']}/{trunc['n_records']} query records lose "
                        f"their own query tail at cutoff_len (see bbh_token_truncation_audit.json)")
    # PASS and PASS_WITH_DISCLOSURE are both launchable; REVIEW/FAIL are not. PASS_WITH_DISCLOSURE means
    # near-verbatim pairs exist but every one is inherent to the OFFICIAL BBH construction (see the
    # audit's `disclosure` field) -- editing the benchmark to remove them would be worse.
    if not str(leak["verdict"]).startswith("PASS"):
        blockers.append(f"few-shot demonstration leakage audit verdict={leak['verdict']}: "
                        f"{len(leak.get('near_duplicate_pairs_escalated') or [])} escalated pair(s) need "
                        f"explicit human clearance")
    # generalize the gate-B vacuity fix to the two new audits: an audit that inspected nothing must
    # never reach the manifest as clean.
    if trunc.get("n_records") != 192:
        blockers.append(f"truncation audit covered {trunc.get('n_records')} records, expected 192")
    if leak.get("n_demonstrations") != 81:
        blockers.append(f"leakage audit covered {leak.get('n_demonstrations')} demos, expected 81")
    # P0-1: never launch on an unverified candidate cache or a flipped gradient-type contract
    if contract["candidate_cache"].get("sha256_matches") is not True:
        blockers.append("candidate cache tensor-content sha256 not verified against the master manifest "
                        "(re-run emit_bbh_execution_contract.py --verify_candidate_cache)")
    # BOTH warm-up hashes must be true: the optimizer state feeds Adam preconditioning of the candidate
    # gradients, so a drifted optimizer.pt changes the feature space with no config change.
    w = contract["warmup_checkpoint"]
    for k in ("adapter_sha256_matches", "optimizer_sha256_matches"):
        if w.get(k) is not True:
            blockers.append(f"warm-up checkpoint {k}={w.get(k)} (expected True)")
    lf = contract.get("llamafactory_preprocessing_pin", {})
    if not lf or any(v is None for v in lf.get("files", {"x": None}).values()):
        blockers.append("LlamaFactory preprocessing/template sources not fully pinned "
                        "(processor_utils.py, supervised.py, template.py)")
    if (contract["frozen_feature_contract"]["gradient_type_candidate"] != "adam"
            or contract["frozen_feature_contract"]["gradient_type_target"] != "sgd"):
        blockers.append("frozen gradient-type contract violated (must be candidate=adam, target=sgd)")
    if contam["pool_strong_count"] or contam["pool_weak_count"]:
        blockers.append("lexical contamination hits against the held-out split")
    if not split["query_eval_disjoint"]:
        blockers.append("query reservoir and held-out eval are not disjoint")

    cells = build_run_plan()
    subsets = sorted({c["subset_id"] for c in cells})
    n_m = len(METHODS) + len(SECONDARY)
    assert len(cells) == len(DRAWS) * n_m * len(SFT_SEEDS), len(cells)
    assert len(subsets) == len(DRAWS) * n_m, len(subsets)
    # the invariant, checked structurally: each subset must appear exactly once per SFT seed
    for sid in subsets:
        seeds = sorted(c["train_seed"] for c in cells if c["subset_id"] == sid)
        assert seeds == sorted(SFT_SEEDS), f"{sid} has seeds {seeds}"
    if not args.verify:                      # --verify must be read-only; never mutate the tree
        json.dump({"experiment": "bbh_external_validation", "budget": BUDGET,
                   "n_cells": len(cells), "n_frozen_subsets": len(subsets), "cells": cells},
                  open(args.run_plan, "w"), indent=2)

    man = {
        "experiment": "bbh_external_validation",
        "purpose": "test whether the MMLU finding (DSMC beats targeted selectors, ties/loses to Random) "
                   "is an MMLU-family artifact, on a different benchmark family",
        "status": "PRE-COMPUTE CHECKPOINT — artifacts + gates only. NO gradients, NO selection, NO SFT.",
        "prereg": "experiments/less_aligned/prereg_bbh_external.md",
        "run_plan": os.path.relpath(args.run_plan, ROOT),
        # A manifest cannot contain its own commit, so do not pretend to. This records the HEAD the
        # artifacts were emitted AGAINST plus whether the tree was clean; the authoritative
        # execution-time receipt is emitted separately by --receipt at the real launch HEAD.
        "emitted_against_head": git_commit(),
        "emitted_against_tree_clean": git_clean(),
        "launch_receipt": ("run `emit_bbh_launch_manifest.py --receipt` on a CLEAN tree at the launch "
                           "commit; it writes bbh_canary_launch_receipt.json recording the true "
                           "executing HEAD. This manifest deliberately does not claim to be it."),

        "design": {
            "draws": DRAWS, "methods_primary": METHODS,
            "methods_secondary_control": SECONDARY,
            "secondary_control_note": (
                "randk_seqlabelmatch jointly matches DSMC's 2D (post-cutoff sequence length, loss-bearing "
                "label positions) histogram at fixed K over 5x5 pre-fixed bins, seed 7000+draw_id. It "
                "REPLACES the earlier sequence-only randk_lenmatch, which matched sequence tokens to "
                "0.986x but still carried 7.14x DSMC's label positions and so could not rule out the "
                "response-length / classification-vs-generation format axis. Sensitivity control ONLY: "
                "the primary comparison set is unchanged and the primary analysis may not be altered by "
                "it. Source composition is a diagnostic, NOT matched."),
            "superseded_secondary_control": ("randk_lenmatch (sequence-only) is retained as an UNTRAINED "
                                             "diagnostic artifact and is not in the run plan"),
            "sft_seeds": SFT_SEEDS,
            "n_frozen_subsets": len(subsets), "n_adapters": len(cells),
            "shared_no_sft_reference": 1,
            "crossed": "seeds fully CROSSED with draws: 3 draws x 2 seeds, every draw under both seeds",
            "budget": BUDGET, "epochs": 4, "optimizer_steps_per_adapter": 84,
            "effective_batch": 128,
            "model": "Llama-2-7B + LoRA r128/alpha512/dropout0.05 on q,k,v,o; lr 2e-5 linear, "
                     "warmup_ratio 0.03, bf16, cutoff 2048",
            "warmup_checkpoint": "warmup_seed42/checkpoint-1692 (hash-pinned)",
            "candidate_pool": "Tulu 270,679 (unchanged, so only the target/eval axis moves)",
        },

        "frozen_selection_seeds": {
            "random_k_seed": f"{RANDOM_SEED_BASE} + draw_id",
            "rr_perm_seed": f"{RR_PERM_SEED_BASE} + draw_id",
            "rr_seed_shared_by": ["first_rr", "second_rr"],
            "per_draw": {str(d): {"random_k_seed": RANDOM_SEED_BASE + d,
                                  "rr_perm_seed": RR_PERM_SEED_BASE + d} for d in DRAWS},
            "INVARIANT": "selection seeds are functions of draw_id ONLY, never of the SFT seed. The 15 "
                         "subsets are therefore reused bit-identically by both SFT seeds, so the "
                         "training-seed axis is clean. To be re-verified by subset hash after selection.",
            "NOT_eliminated": "Because the seeds are 5000+d / 6000+d, each of the three blocks carries a "
                              "DIFFERENT Random-subset realization and a different RR visiting order. "
                              "Selection randomness is not removed, it is BLOCKED WITH THE DRAW INDEX. "
                              "This is deliberate (three independent Random realizations are more "
                              "informative than reusing one), but it means the design must be described "
                              "as 'three draw/selection-realization blocks crossed with two SFT seeds', "
                              "NOT as having no third randomness source. Block spread is therefore not "
                              "pure query-realization variance: for targeted methods it is driven mainly "
                              "by query realization, for Random by the Random-subset realization.",
        },

        "task_accounting": {
            "primary": "27 lm-eval subtasks, micro-aggregated (weight_by_size=true) exactly as the "
                       "pinned bbh_cot_fewshot group does",
            "secondary_diagnostic_only": "23 conceptual BBH task families",
            "n_subtasks": pin["task_accounting"]["n_lm_eval_subtasks"],
            "n_families": pin["task_accounting"]["n_conceptual_families"],
        },

        "evaluation": {
            "lm_eval_version": pin["lm_eval_version"],
            "custom_group": "bbh_external_heldout",
            "include_path": "experiments/less_aligned/bbh_external_tasks",
            "n_heldout_examples": pin["custom_heldout_suite"]["total_heldout_examples"],
            "differences_vs_stock": 'TWO preregistered changes vs stock lm-eval v0.4.5: (1) dataset source -> the frozen 5,209-example held-out split; (2) removal of the redundant CoT trigger that installed v0.4.5 renders twice per demonstration, validated against the official BBH CoT prompts (upstream lm-eval later shipped the same fix). Generation settings, filtering, metric, num_fewshot, sampler and all other task semantics remain pinned.',
            "audited_by": ("gate A: stock-vs-custom byte comparison with ONLY the duplicated CoT cue "
                           "normalized away, plus per-demonstration validation against the official BBH "
                           "cot-prompts and an exact match of all non-sample fewshot_config keys"),
            "num_fewshot": prompts["target_num_fewshot"],
        },

        "prompt_alignment": {
            "claim": prompts["alignment"]["claim"],
            "aligned": prompts["alignment"]["aligned"],
            "not_aligned": prompts["alignment"]["not_aligned"],
            "renderer": prompts["renderer"],
        },

        "execution_contract": {"artifact": "experiments/less_aligned/bbh_execution_contract.json",
                               "note": "AUTHORITATIVE method/selector mapping; supersedes prose summaries"},
        "gates": gates,
        "GO_FOR_SELECTION_CANARY": not blockers,
        "blockers": blockers or None,
        "next_gate": "selection-only canary: no-SFT held-out evaluation + draw0 target-gradient "
                     "extraction + all 5 selectors to K=2707, verifying target cache, selection sizes, "
                     "hashes, determinism, RR order, Random seed, Jaccard/source/token diagnostics. "
                     "NO training. Only after that: the 30-adapter run.",

        "analysis_prereg": {
            "primary": "pinned micro group metric on the 5,209-example held-out split",
            "paired": "DSMC - method within each (draw, seed) cell => 6 paired observations",
            "summaries": ["per-cell values", "mean", "median", "win counts out of 6",
                          "query-draw spread (average over seeds within each draw)",
                          "seed sensitivity (average over draws within each seed)",
                          "optional descriptive two-way draw x seed table"],
            "explicitly_not_done": ["variance-component inference", "p-value thresholds",
                                    "significance claims", "any custom aggregate metric",
                                    "LR/LoRA/epoch/budget sweeps", "source-balanced DSMC variant",
                                    "D2(S, P_heldout) — dropped outright, see d2_reference"],
            "d2_reference": "D2(S, Q_d) ONLY. D2(S, P_heldout) is NOT part of this experiment: a "
                            "diagnostic whose run/skip decision could be made after seeing results is "
                            "not pre-registered, so it was dropped rather than left 'optional'. It would "
                            "require its own pre-registration. No 'balanced BBH reference' is invented.",
            "reporting": "both outcomes informative; reported as a held-out BBH external-validation "
                         "split, never as an official full-BBH leaderboard score",
        },

        "pinned_artifacts": hashes,
        "no_compute_run": "artifacts only: no gradients, no selection, no SFT",
    }

    if args.receipt:
        problems = list(blockers)
        if not git_clean():
            problems.append("working tree is DIRTY — a launch receipt must point at a clean commit")
        if drift:
            problems.append(f"custom suite drift: {drift}")
        if problems:
            print("cannot emit a launch receipt:\n  " + "\n  ".join(problems))
            return 1
        receipt = {
            "receipt": "BBH selection-canary launch receipt",
            "executing_head": git_commit(),
            "tree_clean": True,
            "gates": gates,
            "n_cells": len(cells), "n_frozen_subsets": len(subsets), "budget": BUDGET,
            "execution_contract_sha256": sha_file(f"{EXP}/bbh_execution_contract.json"),
            "pinned_artifact_count": len(hashes),
            "pinned_artifacts": hashes,
            "scope": ("authorises ONLY the no-SFT held-out evaluation + draw0 five-selector "
                      "selection-only canary. It does NOT authorise the 30-adapter run."),
        }
        json.dump(receipt, open(args.receipt_out, "w"), indent=2)
        print(f"launch receipt at CLEAN head {receipt['executing_head'][:12]} -> {args.receipt_out}")
        return 0

    if args.verify:
        if not os.path.exists(args.out):
            raise SystemExit(f"--verify: no existing manifest at {args.out}")
        old = json.load(open(args.out))
        problems = []
        oldh = old.get("pinned_artifacts", {})
        for p, v in oldh.items():
            cur = hashes.get(p)
            if cur is None:
                problems.append(f"{p}: no longer pinned")
            elif cur["sha256"] != v["sha256"]:
                problems.append(f"{p}: sha256 drifted")
        for p in sorted(set(hashes) - set(oldh)):
            problems.append(f"{p}: newly pinned but absent from the manifest")
        problems += [f"{k}: disagrees with bbh_eval_pin_manifest.json" for k in drift]
        if blockers:
            problems += [f"gate blocker: {b}" for b in blockers]
        if not old.get("GO_FOR_SELECTION_CANARY"):
            problems.append("manifest records blockers")
        if problems:
            raise SystemExit("--verify FAILED:\n  " + "\n  ".join(problems))
        print(f"--verify OK: {len(hashes)} pinned artifacts unchanged "
              f"({len(suite_hashes())} of them the 27 subtask configs + 27 held-out data files), "
              f"gates still green, run plan NOT rewritten")
        return 0

    json.dump(man, open(args.out, "w"), indent=2)
    print(f"run plan : {len(cells)} cells over {len(subsets)} frozen subsets -> {args.run_plan}")
    print(f"gates    : parity={gates['parity_all_gates_pass']} "
          f"(A={gates['parity_gate_a_27_subtasks']}, "
          f"B={gates['parity_gate_b_query_prompts']['n_prompts']} prompts/"
          f"{gates['parity_gate_b_query_prompts']['subtasks_covered']} subtasks, "
          f"C={gates['parity_gate_c_heldout_rows']['n_examples']} examples)  "
          f"contamination={gates['contamination_strong_J0.5']}@J0.5/"
          f"{gates['contamination_weak_J0.3']}@J0.3")
    print(f"GO_FOR_SELECTION_CANARY = {man['GO_FOR_SELECTION_CANARY']}  blockers={blockers or 'none'}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

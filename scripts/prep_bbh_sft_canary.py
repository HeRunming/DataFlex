#!/usr/bin/env python3
"""Pre-SFT gate (advice_0811 items 2 and 3). Artifacts only — no training, no evaluation.

Two jobs, both fail-loud:

  --materialize   export each draw0 selection to a `sft_subsets/*.jsonl` and register it in
                  data/dataset_info.json, following the convention the MMLU arm used.

  --check         load ALL SIX draw0 dataset keys through the ACTUAL LlamaFactory SFT data path and
                  require exactly K=2707 examples each. This matters because the two-adapter canary
                  trains only DSMC and Random, so it would NOT otherwise exercise the newly added
                  `bbhx_draw0_randk_seqlabelmatch` registration path. No third adapter is trained.

  --receipt       emit `bbh_sft_canary_launch_receipt.json` approving EXACTLY two cells:
                  bbhx_draw0_dsmc_seed42 and bbhx_draw0_randk_seed42. This is a SEPARATE gate from the
                  selection-canary receipt: that one approved a selection/artifact state, this one
                  approves the first SFT.
"""
import argparse, hashlib, json, os, subprocess

ROOT = "/jizhicfs/karonhe/DataFlex_fa"
EXP = f"{ROOT}/experiments/less_aligned"
SAVES = "/jizhicfs/karonhe/dataflex_saves"
SUBSETS = f"{SAVES}/sft_subsets"
CAND_JSONL = f"{ROOT}/data/less_train_all.jsonl"
DRAW_ID, K = 0, 2707
METHODS = ["dsmc", "second_rr", "first_rr", "less", "randk", "randk_seqlabelmatch"]
CANARY_CELLS = ["bbhx_draw0_dsmc_seed42", "bbhx_draw0_randk_seed42"]


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def sha_idx(idx):
    return hashlib.sha256(json.dumps(sorted(idx)).encode()).hexdigest()


def git(*a):
    try:
        return subprocess.check_output(["git", "-C", ROOT, *a],
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return None


def key_of(m):
    return f"bbhx_draw{DRAW_ID}_{m}"


def materialize():
    """Write each selection out as a sharegpt jsonl and register it (MMLU convention)."""
    os.makedirs(SUBSETS, exist_ok=True)
    rows = None
    info_p = f"{ROOT}/data/dataset_info.json"
    info = json.load(open(info_p))
    out = {}
    for m in METHODS:
        sel_p = f"{SAVES}/selbbhx_draw{DRAW_ID}_{m}/step_1.json"
        if not os.path.exists(sel_p):
            raise SystemExit(f"missing selection {sel_p}")
        idx = json.load(open(sel_p))["indices"]
        if len(idx) != K or len(set(idx)) != K:
            raise SystemExit(f"{m}: {len(idx)} indices ({len(set(idx))} unique), expected {K}")
        if rows is None:
            # NOTE: use readlines(), NOT read().splitlines(). str.splitlines() also breaks on \x0b \x0c
            # \x1c \x1d \x1e \x85 \u2028 \u2029, and this pool contains those characters inside JSON
            # string values -- splitlines() yields 274,187 "lines" for 270,679 records, which silently
            # shifts every index. The fail-loud row check below caught this.
            rows = open(CAND_JSONL).readlines()
        jl = f"{SUBSETS}/bbhx_draw{DRAW_ID}_{m}_sel.jsonl"
        # write in SELECTION ORDER so the file is a faithful record of the selector's output
        with open(jl, "w") as f:
            for i in idx:
                f.write(rows[i].rstrip("\n") + "\n")
        n = sum(1 for l in open(jl) if l.strip())
        if n != K:
            raise SystemExit(f"{m}: wrote {n} rows, expected {K}")
        info[key_of(m)] = {"file_name": jl, "formatting": "sharegpt",
                           "columns": {"messages": "messages"},
                           "tags": {"role_tag": "role", "content_tag": "content",
                                    "user_tag": "user", "assistant_tag": "assistant"}}
        out[m] = {"dataset_key": key_of(m), "jsonl": jl, "n": n,
                  "subset_sha256": sha_idx(idx), "jsonl_sha256": sha_file(jl)}
        print(f"[materialize] {key_of(m):34s} {n} rows  subset={out[m]['subset_sha256'][:12]}")
    json.dump(info, open(info_p, "w"), indent=2, ensure_ascii=False)
    return out


def loader_check():
    """Load all six keys through the REAL LlamaFactory SFT data path; require exactly K each."""
    import warnings
    warnings.filterwarnings("ignore")
    from transformers import AutoTokenizer
    from llamafactory.data.loader import get_dataset
    from llamafactory.data.template import get_template_and_fix_tokenizer
    from llamafactory.hparams import DataArguments, ModelArguments
    from llamafactory.model import load_tokenizer

    base = "/jizhicfs/karonhe/models/shakechen/Llama-2-7b-hf"
    res = {}
    for m in METHODS:
        k = key_of(m)
        model_args = ModelArguments(model_name_or_path=base)
        tok_mod = load_tokenizer(model_args)
        data_args = DataArguments(dataset=[k], dataset_dir=f"{ROOT}/data", template="llama2",
                                 cutoff_len=2048, overwrite_cache=True,
                                 preprocessing_num_workers=8, train_on_prompt=False,
                                 mask_history=False)
        template = get_template_and_fix_tokenizer(tok_mod["tokenizer"], data_args)
        # stage="sft" is the same path the real SFT run takes
        from llamafactory.hparams import TrainingArguments as _TA
        import transformers
        targs = transformers.Seq2SeqTrainingArguments(output_dir="/tmp/_loadchk", do_train=True,
                                                      report_to=[])
        ds = get_dataset(template, model_args, data_args, targs, stage="sft", **tok_mod)
        train = ds["train_dataset"]
        n = len(train)
        ok = n == K
        ex = train[0]
        res[m] = {"dataset_key": k, "n_examples": n, "expected": K, "ok": ok,
                  "first_example_input_ids_len": len(ex["input_ids"]),
                  "first_example_label_positions": int(sum(1 for x in ex["labels"] if x != -100))}
        print(f"[loader] {k:34s} n={n:5d} expected={K}  {'OK' if ok else 'FAIL'}")
        if not ok:
            raise SystemExit(f"loader check FAILED for {k}: {n} != {K}")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--materialize", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--receipt", action="store_true")
    ap.add_argument("--out", default=f"{EXP}/bbh_sft_canary_launch_receipt.json")
    args = ap.parse_args()

    rep = json.load(open(args.out)) if os.path.exists(args.out) else {}
    if args.materialize:
        rep["materialized_subsets"] = materialize()
    if args.check:
        rep["loader_check"] = loader_check()
        rep["loader_check_note"] = (
            "all SIX draw0 dataset keys resolve through the real LlamaFactory SFT data path to exactly "
            "K=2707. The new randk_seqlabelmatch key is included even though the two-adapter canary does "
            "not train it, precisely so its registration path is exercised before the full run.")
    if args.receipt:
        pin = json.load(open(f"{EXP}/bbh_eval_pin_manifest.json"))
        base_eval = json.load(open(f"{EXP}/bbh_canary_report.json"))["phase_baseeval"]
        contract = json.load(open(f"{EXP}/bbh_execution_contract.json"))
        subs = {}
        for m in ("dsmc", "randk"):
            sel = json.load(open(f"{SAVES}/selbbhx_draw{DRAW_ID}_{m}/step_1.json"))
            subs[m] = {"subset_sha256": sha_idx(sel["indices"]), "n": len(sel["indices"])}
        clean = git("status", "--porcelain") == ""
        rep.update({
            "receipt": "BBH SFT CANARY launch receipt",
            "scope": ("authorises EXACTLY the two engineering adapters below. It does NOT authorise the "
                      "remaining 34, and it is a SEPARATE gate from bbh_canary_launch_receipt.json, which "
                      "approved a selection/artifact state rather than any training."),
            "approved_cells": CANARY_CELLS,
            "approved_subsets": subs,
            "runtime_head": git("rev-parse", "HEAD"),
            "tree_clean": clean,
            "resolved_sft_recipe": {
                "base_model": contract["base_model"],
                "warmup_checkpoint": contract["warmup_checkpoint"]["path"],
                "budget_K": contract["budget_K"],
                "epochs": 4, "expected_optimizer_steps": 84, "effective_batch": 128,
                "lora": "r128 / alpha512 / dropout 0.05 on q,k,v,o",
                "lr": "2e-5 linear, warmup_ratio 0.03", "bf16": True,
                "cutoff_len": contract["target_gradient_extraction"]["sft_cutoff_len"],
                "note": ("SFT cutoff is 2048; the 3072 cutoff applies ONLY to target-gradient extraction. "
                         "SFT LoRA dropout is 0.05; extraction uses 0.1."),
            },
            "held_out_eval_suite": {
                "group": "bbh_external_heldout", "n_subtasks": 27,
                "n_examples": pin["custom_heldout_suite"]["total_heldout_examples"],
                "group_config_sha256": pin["custom_heldout_suite"]["group_sha256"],
            },
            "base_reference": {
                "micro_exact_match": base_eval["micro_aggregate_exact_match"],
                "results_json_sha256": base_eval["results_json_sha256"],
                "role": "shared no-SFT reference; all 36 adapters are reported as deltas against it",
            },
            "ENGINEERING_ONLY": (
                "Interim accuracy from these two adapters CANNOT alter the protocol. It is not a stopping "
                "condition even if the DSMC-vs-Random gap is large. Raw outputs are stored and hashed but "
                "must not be compared in any summary. On engineering failure only infrastructure fixes are "
                "permitted (paths, manifests, offline eval, resume, disk, hash bookkeeping) -- never "
                "method, SFT hyperparameters, prompts, subsets, or eval definitions."),
            "stop_rule": ("advice_0811: no further matched Random, source control, LR control, epoch "
                          "control, or method variant will be added."),
        })
        if not clean:
            print("WARNING: tree is dirty; commit before treating this receipt as authoritative")
    json.dump(rep, open(args.out, "w"), indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()

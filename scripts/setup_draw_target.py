#!/usr/bin/env python3
"""
Set up per-draw target-gradient extraction config + dataset registration for the target-draw pilot.
Emits, for one draw (e.g. stem80_draw0):
  - registers the draw jsonl as a llamafactory dataset key `<draw>_target`
  - writes a select YAML  experiments/less_aligned/configs/draws/select_<draw>.yaml
  - appends a components entry `<draw>` to src/dataflex/configs/components_draws.yaml
Target grads then come from:  dataflex-cli train <select yaml>  ->  <cache>/target/1/all_projected_grads.pt
Candidate grads are REUSED from the shared less_output cache (Adam), target uses SGD (LESS-aligned),
matching the frozen protocol / hum80_mirror_manifest provenance (warmup_seed42/checkpoint-1692).

This only writes config/registration files — it runs no gradients. Idempotent.
"""
import argparse, hashlib, json, os, re

WARMUP = "/jizhicfs/karonhe/dataflex_saves/sft_results/warmup_seed42/checkpoint-1692"
# Two DIFFERENT dropouts, deliberately (code_review_0810_3): target-gradient extraction keeps the
# historical 0.1 used by every completed MMLU target-grad config; downstream selected-data SFT uses the
# audited 0.05. Unifying them would make BBH target gradients incomparable to the MMLU arm.
TARGET_GRAD_LORA_DROPOUT = 0.1
SFT_LORA_DROPOUT = 0.05
BASE = "/jizhicfs/karonhe/models/shakechen/Llama-2-7b-hf"
SAVES = "/jizhicfs/karonhe/dataflex_saves"
ROOT = "/jizhicfs/karonhe/DataFlex_fa"

SELECT_YAML = """model_name_or_path: {base}
adapter_name_or_path: {warmup}
trust_remote_code: true
stage: sft
do_train: true
finetuning_type: lora
lora_rank: 128
lora_alpha: 512
lora_target: q_proj,k_proj,v_proj,o_proj
lora_dropout: 0.1
dataset: less_train_all
template: llama2
cutoff_len: {cutoff_len}
overwrite_cache: true
preprocessing_num_workers: 16
output_dir: {saves}/less_aligned/draw_{draw}
logging_steps: 10
overwrite_output_dir: true
save_steps: 99999
report_to: none
per_device_train_batch_size: 1
gradient_accumulation_steps: 1
learning_rate: 2.0e-05
num_train_epochs: 1.0
bf16: true
ddp_timeout: 180000000
train_step: 2
train_type: dynamic_select
components_cfg_file: src/dataflex/configs/components_draws.yaml
component_name: {draw}
warmup_step: 1
update_step: 1
update_times: 1
selection_ratio: 0.05
optimizer_state_path: {warmup}
target_dataset: {draw}_target
eval_dataset: {draw}_target
eval_strategy: 'no'
"""

COMPONENT = """  {draw}:
    name: mmd
    params:
      cache_dir: {saves}/draw_{draw}_output
      proj_dim: 8192
      save_interval: 16
      seed: 123
      candidate_subsample: -1
      greedy_device: auto
      gradient_type: adam
      sigma: null
      kernel_type: grad_cov
      target_gradient_type: sgd
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--draw", required=True, help="e.g. stem80_draw0")
    ap.add_argument("--cutoff_len", type=int, default=2048,
                    help="cutoff for TARGET-GRADIENT extraction. MMLU draws used 2048. BBH draws MUST "
                         "use 3072: at 2048, 7/192 BBH 3-shot CoT query prompts are tail-truncated so "
                         "badly the query itself is deleted (see bbh_token_truncation_audit.json and "
                         "code_review_0810_2). This does NOT change the downstream SFT cutoff.")
    ap.add_argument("--target_jsonl", default=None,
                    help="explicit path to the target/query jsonl. Default is the MMLU layout "
                         "data/target_draws/<draw>.jsonl. BBH draws MUST pass the frozen rendered "
                         "prompts, e.g. "
                         "data/bbh_external/query_prompts/bbh_query_draw0_prompts.jsonl -- there is no "
                         "data/target_draws/bbhx_draw0.jsonl, and creating an untracked copy or symlink "
                         "would put the real input outside provenance.")
    ap.add_argument("--verify_manifest", default=None,
                    help="path to bbh_query_prompt_manifest.json; fail-loud check that --target_jsonl "
                         "matches its recorded sha256, row count and ordered-id hash")
    ap.add_argument("--expect_rows", type=int, default=None,
                    help="required row count (BBH: 64). Mismatch is a hard failure.")
    args = ap.parse_args()
    draw = args.draw

    # FAIL-CLOSED for BBH, checked FIRST so the error is actionable: --target_jsonl / --verify_manifest /
    # --expect_rows all default to the MMLU behaviour, so a bare BBH invocation would either die on a
    # nonexistent data/target_draws/bbhx_draw*.jsonl or skip ALL provenance checking.
    if draw.startswith("bbhx") and not (args.target_jsonl and args.verify_manifest and args.expect_rows):
        _d = re.search(r"draw(\d+)$", draw)
        _d = _d.group(1) if _d else "0"
        raise SystemExit(
            "BBH draws require --target_jsonl, --verify_manifest AND --expect_rows.\n"
            "Correct invocation:\n"
            f"  scripts/setup_draw_target.py --draw {draw} --cutoff_len 3072 \\\n"
            f"      --target_jsonl data/bbh_external/query_prompts/bbh_query_draw{_d}_prompts.jsonl \\\n"
            "      --verify_manifest experiments/less_aligned/bbh_query_prompt_manifest.json \\\n"
            "      --expect_rows 64")

    jsonl = args.target_jsonl or f"{ROOT}/data/target_draws/{draw}.jsonl"
    if not os.path.isabs(jsonl):
        jsonl = f"{ROOT}/{jsonl}"
    if not os.path.exists(jsonl):
        raise FileNotFoundError(jsonl)
    if os.path.islink(jsonl):
        raise RuntimeError(f"{jsonl} is a SYMLINK. The target data must be a committed real file so the "
                           f"gradient extraction input is inside provenance.")

    rows = [l for l in open(jsonl) if l.strip()]
    if args.expect_rows is not None and len(rows) != args.expect_rows:
        raise RuntimeError(f"{jsonl}: {len(rows)} rows, expected {args.expect_rows}")


    # ---- fail-loud provenance: the file we register MUST be the audited artifact ----
    if args.verify_manifest:
        man = json.load(open(args.verify_manifest if os.path.isabs(args.verify_manifest)
                             else f"{ROOT}/{args.verify_manifest}"))
        # Parse the FULL trailing integer, not the last character. `draw[-1]` made "bbhx_draw10" parse
        # as draw 0 and then pass every check against draw0's manifest entry -- silently wiring draw10 to
        # draw0's data, i.e. exactly the mis-wiring this verification exists to prevent.
        m = re.search(r"draw(\d+)$", draw)
        did = m.group(1) if m else None
        entry = man["draws"].get(str(did))
        if entry is None:
            raise RuntimeError(f"no draw '{did}' in {args.verify_manifest}")
        want_path = f"{ROOT}/{entry['prompts_jsonl']}"
        if os.path.realpath(want_path) != os.path.realpath(jsonl):
            raise RuntimeError(f"--target_jsonl {jsonl} is not the manifest's file {want_path}")
        h = hashlib.sha256(open(jsonl, "rb").read()).hexdigest()
        if h != entry["file_sha256"]:
            raise RuntimeError(f"{jsonl} sha256 {h[:12]} != manifest {entry['file_sha256'][:12]} — the "
                               f"audited prompts and the extraction input have DIVERGED")
        if len(rows) != entry["n"]:
            raise RuntimeError(f"{jsonl}: {len(rows)} rows, manifest says {entry['n']}")
        ids = [json.loads(l)["id"] for l in rows]
        src_meta = json.load(open(f"{ROOT}/data/bbh_external/bbh_query_draw{did}.meta.json"))
        oh = hashlib.sha256(json.dumps(ids).encode()).hexdigest()
        if oh != src_meta["ordered_ids_sha256"]:
            raise RuntimeError(f"ordered-id hash mismatch for draw{did}: the prompt file is not the "
                               f"frozen draw ordering")
        print(f"[verify] {os.path.relpath(jsonl, ROOT)}")
        print(f"[verify]   sha256 matches prompt manifest : {h[:16]}")
        print(f"[verify]   rows                           : {len(rows)} (== manifest n)")
        print(f"[verify]   ordered-id hash matches draw{did} : {oh[:16]}")

    # 1. register dataset key
    info = json.load(open(f"{ROOT}/data/dataset_info.json"))
    key = f"{draw}_target"
    info[key] = {"file_name": jsonl, "formatting": "sharegpt",
                 "columns": {"messages": "messages"},
                 "tags": {"role_tag": "role", "content_tag": "content",
                          "user_tag": "user", "assistant_tag": "assistant"}}
    json.dump(info, open(f"{ROOT}/data/dataset_info.json", "w"), indent=2, ensure_ascii=False)

    # 2. select yaml
    os.makedirs(f"{ROOT}/experiments/less_aligned/configs/draws", exist_ok=True)
    yaml_p = f"{ROOT}/experiments/less_aligned/configs/draws/select_{draw}.yaml"
    open(yaml_p, "w").write(
        SELECT_YAML.format(base=BASE, warmup=WARMUP, saves=SAVES, draw=draw,
                           cutoff_len=args.cutoff_len))
    # read the emitted YAML back and assert the two settings that silently break the experiment if wrong
    emitted = open(yaml_p).read()
    if f"cutoff_len: {args.cutoff_len}" not in emitted:
        raise RuntimeError(f"emitted YAML does not contain cutoff_len: {args.cutoff_len}")
    # READ BACK the dropout too. Printing a hardcoded "0.1" while the template said something else is
    # precisely the target-vs-SFT dropout confusion this is meant to prevent.
    import yaml as _y
    _cfg = _y.safe_load(emitted)
    if float(_cfg.get("lora_dropout", -1)) != TARGET_GRAD_LORA_DROPOUT:
        raise RuntimeError(f"emitted YAML lora_dropout={_cfg.get('lora_dropout')} != "
                           f"{TARGET_GRAD_LORA_DROPOUT} (target-gradient extraction value). Downstream "
                           f"SFT uses {SFT_LORA_DROPOUT}; do NOT unify them.")
    if int(_cfg.get("cutoff_len", -1)) != args.cutoff_len:
        raise RuntimeError(f"emitted YAML cutoff_len={_cfg.get('cutoff_len')} != {args.cutoff_len}")
    print(f"[emit] {os.path.relpath(yaml_p, ROOT)}  cutoff_len={_cfg['cutoff_len']}  "
          f"lora_dropout={_cfg['lora_dropout']} (target-extraction value; "
          f"downstream SFT uses {SFT_LORA_DROPOUT}) [both READ BACK from the file]")
    print(f"[emit] dataset key '{key}' -> {os.path.relpath(jsonl, ROOT)}")

    # 3. components file (create header once, append component if missing)
    comp_path = f"{ROOT}/src/dataflex/configs/components_draws.yaml"
    if not os.path.exists(comp_path):
        open(comp_path, "w").write("selectors:\n")
    body = open(comp_path).read()
    if f"\n  {draw}:" not in body and not body.rstrip().endswith(f"{draw}:"):
        with open(comp_path, "a") as f:
            f.write(COMPONENT.format(draw=draw, saves=SAVES))
    print(f"[setup] {draw}: registered dataset {key}, wrote select_{draw}.yaml, "
          f"component -> components_draws.yaml, target cache -> {SAVES}/draw_{draw}_output")


if __name__ == "__main__":
    main()
